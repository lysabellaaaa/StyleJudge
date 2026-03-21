"""
Anthropic API client wrapper for StyleJudge.
Handles Claude Sonnet 4.6 (generation/rewriting) and Claude Haiku (if needed).
All keys loaded from config/api_keys.env via python-dotenv.
"""
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from src.utils.rate_limiter import ANTHROPIC_LIMITER
from src.utils import logger as log

load_dotenv(Path("config/api_keys.env"))

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


class ScoreExtractionError(Exception):
    pass


class MaxRetriesExceeded(Exception):
    pass


def call(
    prompt: str,
    system_prompt: str,
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    estimated_tokens: int = 500,
) -> dict:
    """
    Single API call. Returns {"content": str, "input_tokens": int, "output_tokens": int}.
    """
    ANTHROPIC_LIMITER.acquire(estimated_tokens)
    client = _get_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.content[0].text
    actual = response.usage.output_tokens
    ANTHROPIC_LIMITER.release_extra(actual, estimated_tokens)
    return {
        "content": content,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": actual,
    }


def call_with_retry(
    prompt: str,
    system_prompt: str,
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> dict:
    last_exc = None
    for attempt in range(max_retries):
        try:
            return call(prompt, system_prompt, model, temperature, max_tokens)
        except anthropic.RateLimitError as e:
            last_exc = e
            time.sleep(base_delay * (2 ** attempt))
        except anthropic.APITimeoutError as e:
            last_exc = e
            time.sleep(base_delay * (2 ** attempt))
        except Exception as e:
            log.error("anthropic_client", f"Unretriable error: {e}")
            raise
    raise MaxRetriesExceeded(f"Anthropic: failed after {max_retries} attempts") from last_exc
