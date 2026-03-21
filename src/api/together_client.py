"""
Groq API client for Llama 3.3 70B (secondary judge).
Replaces Together.ai. Groq is free-tier; uses OpenAI-compatible SDK.
Supports logprobs for mechanistic analysis (Experiment M3).

File kept as together_client.py to avoid changing all imports.
"""
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from src.utils.rate_limiter import GROQ_LIMITER
from src.utils import logger as log

load_dotenv(Path("config/api_keys.env"))

_client: Groq | None = None
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
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
    logprobs: int = 0,
) -> dict:
    """
    Single API call. Returns {"content": str, "input_tokens": int, "output_tokens": int,
    "logprobs": list | None}.
    Set logprobs=5 for M3 mechanistic analysis (Groq uses top_logprobs=N with logprobs=True).
    """
    GROQ_LIMITER.acquire(estimated_tokens)
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
    if logprobs > 0:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = logprobs
    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    content = choice.message.content
    actual = response.usage.completion_tokens if response.usage else estimated_tokens
    GROQ_LIMITER.release_extra(actual, estimated_tokens)
    result = {
        "content": content,
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": actual,
        "logprobs": None,
    }
    if logprobs > 0 and hasattr(choice, "logprobs") and choice.logprobs:
        result["logprobs"] = choice.logprobs
    return result


def call_with_retry(
    prompt: str,
    system_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    max_retries: int = 3,
    base_delay: float = 2.0,
    logprobs: int = 0,
) -> dict:
    last_exc = None
    for attempt in range(max_retries):
        try:
            return call(prompt, system_prompt, model, temperature, max_tokens,
                       logprobs=logprobs)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                last_exc = e
                time.sleep(base_delay * (2 ** attempt))
            else:
                log.error("groq_client", f"Unretriable error: {e}")
                raise
    raise MaxRetriesExceeded(f"Groq: failed after {max_retries} attempts") from last_exc
