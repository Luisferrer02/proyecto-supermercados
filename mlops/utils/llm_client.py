"""
LLM client — local (Hugging Face / Ollama) and remote (OpenRouter)
===================================================================
Three modes, auto-detected by environment variables:

  1. **Local HF** (default if ``HF_MODEL`` is set):
     Downloads and runs a model from Hugging Face with ``transformers``.
     No server needed. Set ``HF_MODEL=google/gemma-3-4b-it``.

  2. **Local Ollama** (if ``OLLAMA_MODEL`` is set):
     Connects to a running Ollama server at localhost:11434.
     Set ``OLLAMA_MODEL=gemma3:4b``.

  3. **Remote OpenRouter** (if ``OPENROUTER_API_KEY`` is set):
     Cascade of free-tier models with automatic failover.

Priority: HF_MODEL > OLLAMA_MODEL > OPENROUTER_API_KEY.

Public API
----------
``chat_with_failover(prompt, *, api_key, ...)``
    Returns the string response or ``None`` if everything failed.
"""

from __future__ import annotations

import os
import random
import threading
import time
from typing import Iterable, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Remote: OpenRouter model cascade
# ---------------------------------------------------------------------------

DEFAULT_MODELS: List[str] = [
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "z-ai/glm-4.5-air:free",
    "google/gemma-3-27b-it:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "arcee-ai/trinity-large-preview:free",
]

# ---------------------------------------------------------------------------
# Client / pipeline cache (thread-safe)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_cached_clients: dict[str, OpenAI] = {}
_hf_pipeline = None


def _get_client(base_url: str, api_key: str, timeout: float) -> OpenAI:
    with _lock:
        if base_url not in _cached_clients:
            _cached_clients[base_url] = OpenAI(
                base_url=base_url, api_key=api_key, timeout=timeout,
            )
        return _cached_clients[base_url]


def _get_hf_pipeline(model_id: str):
    global _hf_pipeline
    with _lock:
        if _hf_pipeline is None:
            import torch
            from transformers import AutoTokenizer, GenerationConfig, pipeline

            print(f"    Loading {model_id} from Hugging Face (first time may download)...")

            _hf_pipeline = pipeline(
                "text-generation",
                model=model_id,
                tokenizer=AutoTokenizer.from_pretrained(  # nosec B615
                    model_id, clean_up_tokenization_spaces=False
                ),
                dtype=torch.bfloat16,
                device_map="auto",
            )
            # Replace the model's generation_config entirely so our
            # per-call params (temperature, max_new_tokens) don't
            # conflict with defaults from generation_config.json.
            _hf_pipeline.model.generation_config = GenerationConfig()
            print("    Model loaded.")
        return _hf_pipeline
        return _hf_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_models(models: Optional[Iterable[str]] = None) -> List[str]:
    env_override = os.getenv("OPENROUTER_MODELS", "").strip()
    if env_override:
        return [m.strip() for m in env_override.split(",") if m.strip()]
    if models:
        return list(models)
    return list(DEFAULT_MODELS)


def _looks_empty(text: str) -> bool:
    return not text or not text.strip()


def _is_local_mode() -> bool:
    """True if a local model (HF or Ollama) is configured."""
    return bool(os.getenv("HF_MODEL", "").strip() or os.getenv("OLLAMA_MODEL", "").strip())


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

def _detect_mode() -> str:
    if os.getenv("HF_MODEL", "").strip():
        return "hf"
    if os.getenv("OLLAMA_MODEL", "").strip():
        return "ollama"
    if os.getenv("OPENROUTER_API_KEY", "").strip():
        return "openrouter"
    return "none"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def chat_with_failover(
    prompt: str,
    *,
    api_key: str = "",
    models: Optional[Iterable[str]] = None,
    temperature: float = 0.3,
    max_tokens: int = 3000,
    per_model_retries: int = 2,
    timeout: float = 30.0,
    logger=print,
) -> Optional[str]:
    """Send a prompt to the LLM. Auto-detects local vs remote mode."""
    mode = _detect_mode()

    if mode == "hf":
        return _chat_hf(prompt, temperature=temperature,
                        max_tokens=max_tokens, logger=logger)
    if mode == "ollama":
        return _chat_ollama(prompt, temperature=temperature,
                            max_tokens=max_tokens, timeout=timeout, logger=logger)
    if mode == "openrouter":
        api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        return _chat_remote(prompt, api_key=api_key, models=models,
                            temperature=temperature, max_tokens=max_tokens,
                            per_model_retries=per_model_retries,
                            timeout=timeout, logger=logger)

    logger("     No LLM configured. Set HF_MODEL, OLLAMA_MODEL, or OPENROUTER_API_KEY.")
    return None


# ---------------------------------------------------------------------------
# Local: Hugging Face transformers
# ---------------------------------------------------------------------------

def _chat_hf(
    prompt: str, *, temperature: float, max_tokens: int, logger,
) -> Optional[str]:
    model_id = os.getenv("HF_MODEL", "google/gemma-3-4b-it")
    pipe = _get_hf_pipeline(model_id)
    logger(f"    [local-hf] {model_id}")

    try:
        from transformers import GenerationConfig

        messages = [{"role": "user", "content": prompt}]
        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
        )
        result = pipe(
            messages,
            generation_config=gen_config,
        )
        # The pipeline returns a list of messages; the last one is the assistant's reply
        content = result[0]["generated_text"][-1]["content"]
        if _looks_empty(content):
            logger("      -> empty response")
            return None
        logger(f"      -> ok ({len(content)} chars)")
        return content.strip()
    except Exception as exc:
        logger(f"      -> error: {str(exc)[:140]}")
        return None


# ---------------------------------------------------------------------------
# Local: Ollama (OpenAI-compatible API)
# ---------------------------------------------------------------------------

def _chat_ollama(
    prompt: str, *, temperature: float, max_tokens: int, timeout: float, logger,
) -> Optional[str]:
    model = os.getenv("OLLAMA_MODEL", "gemma3:4b")
    base_url = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
    client = _get_client(base_url, api_key="ollama", timeout=timeout)
    logger(f"    [local-ollama] {model}")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        if _looks_empty(content):
            logger("      -> empty response")
            return None
        logger(f"      -> ok ({len(content)} chars)")
        return content.strip()
    except Exception as exc:
        logger(f"      -> error: {str(exc)[:140]}")
        return None


# ---------------------------------------------------------------------------
# Remote: OpenRouter cascade
# ---------------------------------------------------------------------------

def _chat_remote(
    prompt: str, *, api_key: str, models: Optional[Iterable[str]],
    temperature: float, max_tokens: int, per_model_retries: int,
    timeout: float, logger,
) -> Optional[str]:
    candidates = resolve_models(models)
    if not api_key:
        logger("     No OPENROUTER_API_KEY — skipping LLM cascade.")
        return None

    client = _get_client("https://openrouter.ai/api/v1", api_key, timeout)

    for position, model in enumerate(candidates, start=1):
        short = model.split("/")[-1]
        for attempt in range(1, per_model_retries + 1):
            try:
                logger(f"    [{position}/{len(candidates)}] {short} "
                       f"(try {attempt}/{per_model_retries})")
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content or ""
                if _looks_empty(content):
                    logger("      -> empty response, retrying...")
                    continue
                logger(f"      -> ok ({len(content)} chars)")
                return content.strip()
            except Exception as exc:
                msg = str(exc)[:140]
                is_rate_limited = "429" in msg or "rate" in msg.lower()
                logger(f"      -> {'rate-limited' if is_rate_limited else 'error'}: {msg}")
                time.sleep(min(2 ** attempt + random.random(), 8.0))
        logger(f"     Model {short} exhausted retries, falling through.")

    logger("     All OpenRouter models in cascade failed.")
    return None
