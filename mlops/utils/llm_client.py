"""
OpenRouter client with multi-model failover
===========================================
Wraps the OpenRouter chat-completions call with a cascade of candidate
models. If one model is down, rate-limited, or returns junk, the client
automatically tries the next. Only when *every* model in the cascade has
failed does the caller need to fall back to non-LLM logic (heuristic).

Configuration
-------------
The cascade is defined by the module-level ``DEFAULT_MODELS`` list and can
be overridden at runtime via the ``OPENROUTER_MODELS`` env var
(comma-separated slugs).

Public API
----------
``chat_with_failover(prompt, *, api_key, models=None, temperature=0.3,
                     max_tokens=3000, per_model_retries=2, timeout=30.0)``
    Returns the string response from the first model that succeeds, or
    ``None`` if all models failed.
"""

from __future__ import annotations

import os
import random
import time
from typing import Iterable, List, Optional


# Ordered cascade of free-tier OpenRouter models to try. When the first
# one fails (429, 5xx, timeout, bad JSON, empty string, etc.) the client
# falls through to the next. Adjust the order by prepending your
# preferred model or override completely via the OPENROUTER_MODELS env
# var. Keep all entries as valid OpenRouter slugs ("org/model[:tag]").
DEFAULT_MODELS: List[str] = [
    # Lista obtenida de https://openrouter.ai/api/v1/models (filtrando
    # pricing=0). Ordenados por robustez y ventana de contexto útil para
    # nuestros prompts (~1500-3000 tokens de entrada, 3000-5000 de salida).
    "meta-llama/llama-3.3-70b-instruct:free",       # 70B, 64K ctx — el más estable
    "qwen/qwen3-next-80b-a3b-instruct:free",        # 80B MoE, 262K ctx
    "openai/gpt-oss-120b:free",                      # 120B MoE, 131K ctx
    "nvidia/nemotron-3-super-120b-a12b:free",        # 120B MoE, 262K ctx
    "z-ai/glm-4.5-air:free",                         # GLM 4.5 Air, 131K ctx
    "google/gemma-3-27b-it:free",                    # 27B, 131K ctx
    "nousresearch/hermes-3-llama-3.1-405b:free",     # 405B Hermes, 131K ctx
    "arcee-ai/trinity-large-preview:free",           # tu modelo original, 131K ctx
]


def resolve_models(models: Optional[Iterable[str]] = None) -> List[str]:
    """Return the effective cascade — env override > argument > defaults."""
    env_override = os.getenv("OPENROUTER_MODELS", "").strip()
    if env_override:
        return [m.strip() for m in env_override.split(",") if m.strip()]
    if models:
        return list(models)
    return list(DEFAULT_MODELS)


def _looks_empty(text: str) -> bool:
    return not text or not text.strip()


def chat_with_failover(
    prompt: str,
    *,
    api_key: str,
    models: Optional[Iterable[str]] = None,
    temperature: float = 0.3,
    max_tokens: int = 3000,
    per_model_retries: int = 2,
    timeout: float = 30.0,
    logger=print,
) -> Optional[str]:
    """Call OpenRouter trying each candidate model until one succeeds.

    Returns the raw text response from the first model that answers, or
    ``None`` if every model failed. Never raises — the caller is expected
    to treat ``None`` as "fall back to heuristic".
    """
    from openai import OpenAI

    candidates = resolve_models(models)
    if not api_key:
        logger("   ⚠  No OPENROUTER_API_KEY — skipping LLM cascade.")
        return None

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=timeout,
    )

    for position, model in enumerate(candidates, start=1):
        short = model.split("/")[-1]
        for attempt in range(1, per_model_retries + 1):
            try:
                logger(f"   🤖 [{position}/{len(candidates)}] {short} "
                       f"(try {attempt}/{per_model_retries})")
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content or ""
                if _looks_empty(content):
                    logger("      ↳ empty response, retrying…")
                    continue
                logger(f"      ↳ ok ({len(content)} chars)")
                return content.strip()
            except Exception as exc:
                msg = str(exc)[:140]
                is_rate_limited = "429" in msg or "rate" in msg.lower()
                logger(f"      ↳ {'rate-limited' if is_rate_limited else 'error'}: {msg}")
                # Small backoff with jitter before the next retry / next model
                time.sleep(min(2 ** attempt + random.random(), 8.0))
        logger(f"   ⚠  Model {short} exhausted retries, falling through.")

    logger("   ⚠  All OpenRouter models in cascade failed.")
    return None
