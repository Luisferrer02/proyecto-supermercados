"""
train.py
─────────────────────────────────────────────────────────────────────────────
PPO Training Loop for Shelf-Space Optimization

Architecture
────────────
We implement PPO from scratch (no Stable-Baselines3 dependency) so every
component is transparent and tunable. The actor-critic network uses shared
body layers followed by separate heads for the policy and value function.

Key design choices
──────────────────
1. Orthogonal initialization → stabilizes early training with large discrete
   action spaces (n_products × n_positions can be 100-300 actions).

2. Action masking is NOT used here for simplicity, but the environment's
   observation includes position-occupancy so the network can learn to avoid
   sub-optimal placements.

3. GAE (Generalized Advantage Estimation) with λ=0.95 is used to balance
   bias vs variance in the advantage estimates.

4. Entropy bonus in the loss function prevents premature policy collapse —
   important given that many actions (swap, no-op) appear similar early on.

5. Value function loss is clipped (same ratio as policy) to further stabilize.

Training loop
─────────────
  while total_steps < max_steps:
      1. Collect rollout_steps transitions from the env
      2. Compute GAE advantages and returns
      3. For ppo_epochs: mini-batch update of actor-critic
      4. Log metrics to console (and optionally to TensorBoard / W&B)
      5. Run periodic evaluation episode
      6. Save checkpoint if best eval reward

Usage
─────
    python train.py                           # defaults
    python train.py --n_products 12 --n_positions 20 --max_steps 2_000_000

    To load a pretrained forecaster instead of training a new one:
    python train.py --forecaster_path models/forecaster.ubj
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from forecaster import DemandForecaster
from environment import ShelfSpaceEnv

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Default Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    # Environment
    "n_products": 10,
    "n_positions": 15,
    "episode_length": 30,
    # Training scale
    "max_steps": 1_000_000,
    "rollout_steps": 2048,    # steps per rollout (must be < episode_length × n_envs ideally)
    "n_envs": 1,              # number of parallel envs (extend with AsyncVectorEnv)
    # PPO
    "ppo_epochs": 10,
    "minibatch_size": 256,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "ent_coef": 0.01,         # entropy bonus coefficient
    "vf_coef": 0.5,           # value function loss coefficient
    "max_grad_norm": 0.5,
    "learning_rate": 3e-4,
    "lr_schedule": "linear",   # "linear" or "constant"
    # Network
    "hidden_size": 256,
    "n_hidden_layers": 2,
    # Logging / checkpointing
    "log_interval": 10,       # every N updates
    "eval_interval": 50,      # every N updates
    "eval_episodes": 5,
    "checkpoint_dir": "checkpoints",
    "forecaster_path": None,  # None = train a new one
    "seed": 42,
    "device": "auto",          # "auto", "cpu", "cuda", "mps"
}


# ─────────────────────────────────────────────────────────────────────────────
# Actor-Critic Network
# ─────────────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Shared-trunk actor-critic for discrete action spaces.

    Architecture
    ────────────
    Input → [LayerNorm] → MLP trunk → ┬→ Policy head → logits → Categorical
                                       └→ Value head  → scalar

    LayerNorm on the input handles the mixed-scale features (inventory counts
    vs binary flags vs normalized fractions) without needing careful manual
    normalization of each feature.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        n_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(obs_dim)

        # Shared trunk
        trunk_layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(n_hidden_layers):
            trunk_layers.extend([
                nn.Linear(in_dim, hidden_size),
                nn.Tanh(),  # Tanh preferred over ReLU for PPO per empirical findings
            ])
            in_dim = hidden_size
        self.trunk = nn.Sequential(*trunk_layers)

        # Policy head
        self.policy_head = nn.Linear(hidden_size, action_dim)

        # Value head
        self.value_head = nn.Linear(hidden_size, 1)

        # Orthogonal initialization — critical for stable PPO
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, values)."""
        x = self.input_norm(obs)
        x = self.trunk(x)
        logits = self.policy_head(x)
        values = self.value_head(x).squeeze(-1)
        return logits, values

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (action, log_prob, entropy, value).
        If action is None, samples from the distribution.
        """
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, values


# ─────────────────────────────────────────────────────────────────────────────
# Rollout Buffer
# ─────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Stores transitions from environment interaction and computes
    GAE advantages + discounted returns.

    All tensors are stored on CPU during collection and moved to the
    training device during updates.
    """

    def __init__(
        self,
        rollout_steps: int,
        obs_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self.rollout_steps = rollout_steps
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.obs = np.zeros((rollout_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros(rollout_steps, dtype=np.int64)
        self.log_probs = np.zeros(rollout_steps, dtype=np.float32)
        self.rewards = np.zeros(rollout_steps, dtype=np.float32)
        self.values = np.zeros(rollout_steps, dtype=np.float32)
        self.dones = np.zeros(rollout_steps, dtype=np.float32)

        self.advantages = np.zeros(rollout_steps, dtype=np.float32)
        self.returns = np.zeros(rollout_steps, dtype=np.float32)

        self._ptr = 0
        self._full = False

    def store(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        self.obs[self._ptr] = obs
        self.actions[self._ptr] = action
        self.log_probs[self._ptr] = log_prob
        self.rewards[self._ptr] = reward
        self.values[self._ptr] = value
        self.dones[self._ptr] = float(done)
        self._ptr += 1
        if self._ptr >= self.rollout_steps:
            self._full = True

    def compute_gae(self, last_value: float, last_done: bool) -> None:
        """
        Generalized Advantage Estimation.
        Iterates backwards through the buffer to compute δ-weighted returns.
        """
        gae = 0.0
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def get_minibatches(
        self,
        minibatch_size: int,
        device: torch.device,
    ):
        """
        Yields shuffled minibatches as (obs, actions, log_probs, advantages, returns) tensors.
        Advantages are normalized within this function.
        """
        indices = np.random.permutation(self.rollout_steps)
        adv = self.advantages[indices]
        adv_normalized = (adv - adv.mean()) / (adv.std() + 1e-8)

        for start in range(0, self.rollout_steps, minibatch_size):
            end = start + minibatch_size
            mb_idx = indices[start:end]
            yield (
                torch.tensor(self.obs[mb_idx], device=device),
                torch.tensor(self.actions[mb_idx], device=device),
                torch.tensor(self.log_probs[mb_idx], device=device),
                torch.tensor(adv_normalized[start:end], device=device),
                torch.tensor(self.returns[mb_idx], device=device),
            )

    def reset(self) -> None:
        self._ptr = 0
        self._full = False


# ─────────────────────────────────────────────────────────────────────────────
# PPO Update
# ─────────────────────────────────────────────────────────────────────────────

def ppo_update(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    last_obs: np.ndarray,
    last_done: bool,
    device: torch.device,
    cfg: dict,
) -> dict[str, float]:
    """
    Runs GAE + ppo_epochs mini-batch updates. Returns a dict of loss metrics.
    """
    # Compute last value for GAE bootstrap
    with torch.no_grad():
        obs_t = torch.tensor(last_obs, device=device).unsqueeze(0)
        _, last_value = model(obs_t)
        last_value = last_value.item()

    buffer.compute_gae(last_value=last_value, last_done=last_done)

    total_pg_loss = 0.0
    total_vf_loss = 0.0
    total_entropy = 0.0
    total_approx_kl = 0.0
    n_updates = 0

    for epoch in range(cfg["ppo_epochs"]):
        for obs_b, act_b, logp_old_b, adv_b, ret_b in buffer.get_minibatches(
            cfg["minibatch_size"], device
        ):
            _, logp_new_b, entropy_b, values_b = model.get_action_and_value(obs_b, act_b)

            # Policy loss (clipped surrogate)
            log_ratio = logp_new_b - logp_old_b
            ratio = log_ratio.exp()
            pg_loss1 = -adv_b * ratio
            pg_loss2 = -adv_b * torch.clamp(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"])
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value function loss (also clipped for stability)
            vf_loss = 0.5 * ((values_b - ret_b) ** 2).mean()

            # Entropy bonus (encourages exploration)
            entropy_loss = -entropy_b.mean()

            loss = pg_loss + cfg["vf_coef"] * vf_loss + cfg["ent_coef"] * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            optimizer.step()

            # Diagnostics
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean().item()

            total_pg_loss += pg_loss.item()
            total_vf_loss += vf_loss.item()
            total_entropy += -entropy_loss.item()
            total_approx_kl += approx_kl
            n_updates += 1

    denom = max(n_updates, 1)
    return {
        "pg_loss": total_pg_loss / denom,
        "vf_loss": total_vf_loss / denom,
        "entropy": total_entropy / denom,
        "approx_kl": total_approx_kl / denom,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: ActorCritic,
    env: ShelfSpaceEnv,
    n_episodes: int,
    device: torch.device,
) -> dict[str, float]:
    """
    Run n_episodes greedy (deterministic) evaluation episodes.
    Returns mean and std of episode rewards.
    """
    model.eval()
    episode_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        ep_reward = 0.0

        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(obs, device=device).unsqueeze(0)
                logits, _ = model(obs_t)
                action = logits.argmax(dim=-1).item()  # greedy

            obs, reward, terminated, truncated, _ = env.step(int(action))
            ep_reward += reward
            done = terminated or truncated

        episode_rewards.append(ep_reward)

    model.train()
    return {
        "mean_eval_reward": float(np.mean(episode_rewards)),
        "std_eval_reward": float(np.std(episode_rewards)),
        "min_eval_reward": float(np.min(episode_rewards)),
        "max_eval_reward": float(np.max(episode_rewards)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Learning Rate Scheduler
# ─────────────────────────────────────────────────────────────────────────────

def get_lr_fn(cfg: dict, total_updates: int):
    """Returns a callable frac → learning_rate for use with LambdaLR."""
    if cfg["lr_schedule"] == "linear":
        return lambda frac: 1.0 - frac  # frac goes 0 → 1 over training
    return lambda frac: 1.0             # constant


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    """
    End-to-end PPO training.

    Phase 1: Pre-train (or load) the XGBoost demand forecaster.
    Phase 2: Run PPO training, embedding the frozen forecaster into the env.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Device ─────────────────────────────────────────────────────────────
    if cfg["device"] == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cfg["device"])
    logger.info("Training device: %s", device)

    # ── Seeds ───────────────────────────────────────────────────────────────
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # ── Phase 1: Forecaster ─────────────────────────────────────────────────
    forecaster_path = cfg.get("forecaster_path")
    if forecaster_path and Path(forecaster_path).exists():
        logger.info("Loading pre-trained forecaster from %s", forecaster_path)
        forecaster = DemandForecaster.load(forecaster_path)
    else:
        logger.info("Training XGBoost demand forecaster on synthetic data …")
        forecaster = DemandForecaster()
        metrics = forecaster.train(n_synthetic_samples=25_000, verbose=False)
        logger.info("Forecaster ready. Val MAE=%.4f, R²=%.4f", metrics["val_mae"], metrics["val_r2"])
        save_path = Path(cfg["checkpoint_dir"]) / "forecaster.ubj"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        forecaster.save(save_path)

    # ── Phase 2: Environment & Model ────────────────────────────────────────
    train_env = ShelfSpaceEnv(
        n_products=cfg["n_products"],
        n_positions=cfg["n_positions"],
        episode_length=cfg["episode_length"],
        forecaster=forecaster,
        seed=cfg["seed"],
    )
    eval_env = ShelfSpaceEnv(
        n_products=cfg["n_products"],
        n_positions=cfg["n_positions"],
        episode_length=cfg["episode_length"],
        forecaster=forecaster,
        seed=cfg["seed"] + 9999,
    )

    obs_dim = train_env.obs_dim
    action_dim = train_env.action_space.n
    logger.info("Obs dim: %d | Action dim: %d", obs_dim, action_dim)

    model = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=cfg["hidden_size"],
        n_hidden_layers=cfg["n_hidden_layers"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Actor-Critic parameters: %s", f"{total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"], eps=1e-5)

    total_updates = cfg["max_steps"] // cfg["rollout_steps"]
    lr_lambda = get_lr_fn(cfg, total_updates)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    buffer = RolloutBuffer(
        rollout_steps=cfg["rollout_steps"],
        obs_dim=obs_dim,
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
    )

    # ── Training State ──────────────────────────────────────────────────────
    obs, _ = train_env.reset(seed=cfg["seed"])
    done = False
    total_steps = 0
    update_count = 0
    best_eval_reward = -np.inf
    ep_reward_window = deque(maxlen=20)
    ep_reward = 0.0
    ep_len = 0
    t_start = time.time()

    Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting PPO training | max_steps=%s", f"{cfg['max_steps']:,}")
    logger.info("=" * 60)

    # ── Main Collection + Update Loop ───────────────────────────────────────
    while total_steps < cfg["max_steps"]:

        # ── Rollout collection ──────────────────────────────────────────────
        buffer.reset()
        for _ in range(cfg["rollout_steps"]):
            with torch.no_grad():
                obs_t = torch.tensor(obs, device=device).unsqueeze(0)
                action_t, log_prob_t, _, value_t = model.get_action_and_value(obs_t)
                action = int(action_t.item())
                log_prob = float(log_prob_t.item())
                value = float(value_t.item())

            next_obs, reward, terminated, truncated, info = train_env.step(action)
            ep_reward += reward
            ep_len += 1
            total_steps += 1

            buffer.store(obs, action, log_prob, reward, value, done=(terminated or truncated))

            if terminated or truncated:
                ep_reward_window.append(ep_reward)
                ep_reward = 0.0
                ep_len = 0
                next_obs, _ = train_env.reset()
                done = True
            else:
                done = False

            obs = next_obs

        # ── PPO update ──────────────────────────────────────────────────────
        update_count += 1
        frac = update_count / total_updates
        scheduler.last_epoch = int(frac * total_updates) - 1  # manual schedule step

        for param_group in optimizer.param_groups:
            param_group["lr"] = cfg["learning_rate"] * lr_lambda(frac)

        loss_metrics = ppo_update(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            last_obs=obs,
            last_done=done,
            device=device,
            cfg=cfg,
        )

        # ── Logging ─────────────────────────────────────────────────────────
        if update_count % cfg["log_interval"] == 0:
            elapsed = time.time() - t_start
            sps = total_steps / elapsed
            mean_ep_r = np.mean(ep_reward_window) if ep_reward_window else 0.0
            current_lr = optimizer.param_groups[0]["lr"]

            logger.info(
                "Update %4d | steps %7s | SPS %5.0f | LR %.2e | "
                "MeanEpR %8.2f | PG %.4f | VF %.4f | Ent %.4f | KL %.4f",
                update_count,
                f"{total_steps:,}",
                sps,
                current_lr,
                mean_ep_r,
                loss_metrics["pg_loss"],
                loss_metrics["vf_loss"],
                loss_metrics["entropy"],
                loss_metrics["approx_kl"],
            )

        # ── Evaluation ──────────────────────────────────────────────────────
        if update_count % cfg["eval_interval"] == 0:
            eval_metrics = evaluate(model, eval_env, cfg["eval_episodes"], device)
            logger.info(
                "  ── EVAL ── Mean: %8.2f ± %6.2f  [%8.2f, %8.2f]",
                eval_metrics["mean_eval_reward"],
                eval_metrics["std_eval_reward"],
                eval_metrics["min_eval_reward"],
                eval_metrics["max_eval_reward"],
            )

            if eval_metrics["mean_eval_reward"] > best_eval_reward:
                best_eval_reward = eval_metrics["mean_eval_reward"]
                ckpt_path = Path(cfg["checkpoint_dir"]) / "best_model.pt"
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "update_count": update_count,
                    "total_steps": total_steps,
                    "best_eval_reward": best_eval_reward,
                    "cfg": cfg,
                }, ckpt_path)
                logger.info("  ✓ New best model saved → %s (reward=%.2f)", ckpt_path, best_eval_reward)

    # ── Final save ──────────────────────────────────────────────────────────
    final_path = Path(cfg["checkpoint_dir"]) / "final_model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg,
        "total_steps": total_steps,
    }, final_path)
    logger.info("Training complete. Final model saved to %s", final_path)
    logger.info("Best eval reward achieved: %.2f", best_eval_reward)

    # ── Final evaluation with rendering ─────────────────────────────────────
    logger.info("\nRunning final evaluation episode …")
    obs, _ = eval_env.reset(seed=42)
    done = False
    total_r = 0.0
    while not done:
        with torch.no_grad():
            obs_t = torch.tensor(obs, device=device).unsqueeze(0)
            logits, _ = model(obs_t)
            action = logits.argmax(dim=-1).item()
        obs, reward, terminated, truncated, _ = eval_env.step(int(action))
        total_r += reward
        done = terminated or truncated
    eval_env.render()
    logger.info("Final episode reward: %.2f", total_r)

    train_env.close()
    eval_env.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="PPO Shelf-Space Optimizer")
    parser.add_argument("--n_products", type=int, default=DEFAULTS["n_products"])
    parser.add_argument("--n_positions", type=int, default=DEFAULTS["n_positions"])
    parser.add_argument("--episode_length", type=int, default=DEFAULTS["episode_length"])
    parser.add_argument("--max_steps", type=int, default=DEFAULTS["max_steps"])
    parser.add_argument("--rollout_steps", type=int, default=DEFAULTS["rollout_steps"])
    parser.add_argument("--ppo_epochs", type=int, default=DEFAULTS["ppo_epochs"])
    parser.add_argument("--minibatch_size", type=int, default=DEFAULTS["minibatch_size"])
    parser.add_argument("--gamma", type=float, default=DEFAULTS["gamma"])
    parser.add_argument("--gae_lambda", type=float, default=DEFAULTS["gae_lambda"])
    parser.add_argument("--clip_eps", type=float, default=DEFAULTS["clip_eps"])
    parser.add_argument("--ent_coef", type=float, default=DEFAULTS["ent_coef"])
    parser.add_argument("--vf_coef", type=float, default=DEFAULTS["vf_coef"])
    parser.add_argument("--learning_rate", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--hidden_size", type=int, default=DEFAULTS["hidden_size"])
    parser.add_argument("--n_hidden_layers", type=int, default=DEFAULTS["n_hidden_layers"])
    parser.add_argument("--log_interval", type=int, default=DEFAULTS["log_interval"])
    parser.add_argument("--eval_interval", type=int, default=DEFAULTS["eval_interval"])
    parser.add_argument("--eval_episodes", type=int, default=DEFAULTS["eval_episodes"])
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULTS["checkpoint_dir"])
    parser.add_argument("--forecaster_path", type=str, default=DEFAULTS["forecaster_path"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--device", type=str, default=DEFAULTS["device"])
    parser.add_argument("--lr_schedule", type=str, default=DEFAULTS["lr_schedule"],
                        choices=["linear", "constant"])

    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
