"""
OpenAI API client wrapper for StyleJudge.
Used for: GPT-4o judge, QA verification, IRR classification, formality perception.
"""
import time
from pathlib import Path

import openai
from dotenv import load_dotenv

from src.utils.rate_limiter import OPENAI_LIMITER
from src.utils import logger as log

load_dotenv(Path("config/api_keys.env"))

_client: openai.OpenAI | None = None


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI()
    return _client


class MaxRetriesExceeded(Exception):
    pass


def call(
    prompt: str,
    system_prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    estimated_tokens: int = 500,
    response_format: dict | None = None,
) -> dict:
    """
    Single API call. Returns {"content": str, "input_tokens": int, "output_tokens": int}.
    Pass response_format={"type": "json_object"} for structured output.
    """
    OPENAI_LIMITER.acquire(estimated_tokens)
    client = _get_client()
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    if response_format:
        kwargs["response_format"] = response_format
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    actual = response.usage.completion_tokens
    OPENAI_LIMITER.release_extra(actual, estimated_tokens)
    return {
        "content": content,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": actual,
    }


def call_with_retry(
    prompt: str,
    system_prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    max_retries: int = 3,
    base_delay: float = 2.0,
    response_format: dict | None = None,
) -> dict:
    last_exc = None
    for attempt in range(max_retries):
        try:
            return call(prompt, system_prompt, model, temperature, max_tokens,
                       response_format=response_format)
        except openai.RateLimitError as e:
            last_exc = e
            time.sleep(base_delay * (2 ** attempt))
        except openai.APITimeoutError as e:
            last_exc = e
            time.sleep(base_delay * (2 ** attempt))
        except Exception as e:
            log.error("openai_client", f"Unretriable error: {e}")
            raise
    raise MaxRetriesExceeded(f"OpenAI: failed after {max_retries} attempts") from last_exc
