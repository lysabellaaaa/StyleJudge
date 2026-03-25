"""
Gemini 2.0 Flash client via Google AI Studio OpenAI-compatible endpoint.
Used for benchmark question quality verification (free tier: 15 RPM, 1M TPD).

Configured via GEMINI_API_KEY in config/api_keys.env.
"""
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.utils.rate_limiter import GEMINI_LIMITER
from src.utils import logger as log

load_dotenv(Path("config/api_keys.env"))

_client: OpenAI | None = None
DEFAULT_MODEL   = "gemini-2.0-flash"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY", ""),
            base_url=DEFAULT_BASE_URL,
        )
    return _client


class MaxRetriesExceeded(Exception):
    pass


def call(
    prompt: str,
    system_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    estimated_tokens: int = 300,
) -> dict:
    """Single API call. Returns {"content": str, "input_tokens": int, "output_tokens": int}."""
    GEMINI_LIMITER.acquire(estimated_tokens)
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    actual  = response.usage.completion_tokens if response.usage else estimated_tokens
    GEMINI_LIMITER.release_extra(actual, estimated_tokens)
    return {
        "content": content,
        "input_tokens":  response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": actual,
    }


def call_with_retry(
    prompt: str,
    system_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_retries: int = 3,
    base_delay: float = 4.0,  # Gemini free tier is slower to recover
) -> dict:
    last_exc = None
    for attempt in range(max_retries):
        try:
            return call(prompt, system_prompt, model, temperature, max_tokens)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                last_exc = e
                time.sleep(base_delay * (2 ** attempt))
            else:
                log.error("gemini_client", f"Unretriable error: {e}")
                raise
    raise MaxRetriesExceeded(f"Gemini API: failed after {max_retries} attempts") from last_exc
