"""
Custom research API client (OpenAI-compatible endpoint).
Used as exploratory third judge via aicohort.org.
Configured via ZHIPU_API_KEY and ZHIPU_BASE_URL in config/api_keys.env.
"""
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.utils.rate_limiter import ZHIPU_LIMITER
from src.utils import logger as log

load_dotenv(Path("config/api_keys.env"))

_client: OpenAI | None = None
DEFAULT_MODEL = "research-model"
DEFAULT_BASE_URL = "https://aicohort.org/v1"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ.get("ZHIPU_API_KEY", ""),
            base_url=os.environ.get("ZHIPU_BASE_URL", DEFAULT_BASE_URL),
        )
    return _client


class MaxRetriesExceeded(Exception):
    pass


def call(
    prompt: str,
    system_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    estimated_tokens: int = 500,
) -> dict:
    """Single API call. Returns {"content": str, "input_tokens": int, "output_tokens": int}."""
    ZHIPU_LIMITER.acquire(estimated_tokens)
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content
    actual = response.usage.completion_tokens if response.usage else estimated_tokens
    ZHIPU_LIMITER.release_extra(actual, estimated_tokens)
    return {
        "content": content,
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": actual,
    }


def call_with_retry(
    prompt: str,
    system_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    max_retries: int = 3,
    base_delay: float = 2.0,
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
                log.error("zhipu_client", f"Unretriable error: {e}")
                raise
    raise MaxRetriesExceeded(f"Research API: failed after {max_retries} attempts") from last_exc
