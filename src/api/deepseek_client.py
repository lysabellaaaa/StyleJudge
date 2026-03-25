"""
DeepSeek API client (OpenAI-compatible endpoint).
Supports both DeepSeek-V3 (deepseek-chat) and DeepSeek-R1 (deepseek-reasoner).

R1 calls automatically strip the <think>...</think> block from the response content,
returning only the final structured answer. The approximate word count of the stripped
block is returned as `think_block_words` for downstream analysis.

Configured via DEEPSEEK_API_KEY in config/api_keys.env.
"""
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.utils.rate_limiter import DEEPSEEK_LIMITER
from src.utils import logger as log

load_dotenv(Path("config/api_keys.env"))

_client: OpenAI | None = None
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
MODEL_V3 = "deepseek-chat"
MODEL_R1 = "deepseek-reasoner"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url=DEFAULT_BASE_URL,
        )
    return _client


def _strip_think_block(content: str) -> tuple[str, int]:
    """
    Remove <think>...</think> from R1 output.
    Returns (final_answer_text, approximate_think_word_count).
    """
    match = re.search(r"<think>(.*?)</think>\s*", content, re.DOTALL)
    if match:
        think_text = match.group(1)
        think_words = len(think_text.split())
        final = content[match.end():].strip()
        return final, think_words
    return content, 0


class MaxRetriesExceeded(Exception):
    pass


def call(
    prompt: str,
    system_prompt: str,
    model: str = MODEL_V3,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    estimated_tokens: int = 500,
) -> dict:
    """
    Single API call.

    Returns:
        {
            "content": str,            # final answer text (think block stripped for R1)
            "input_tokens": int,
            "output_tokens": int,
            "think_block_words": int,  # 0 for V3; approximate for R1
        }
    """
    DEEPSEEK_LIMITER.acquire(estimated_tokens)
    client = _get_client()

    # R1 (deepseek-reasoner) does not support temperature parameter
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
    )
    if model != MODEL_R1:
        kwargs["temperature"] = temperature

    response = client.chat.completions.create(**kwargs)
    raw_content = response.choices[0].message.content or ""
    actual = response.usage.completion_tokens if response.usage else estimated_tokens
    DEEPSEEK_LIMITER.release_extra(actual, estimated_tokens)

    if model == MODEL_R1:
        content, think_words = _strip_think_block(raw_content)
    else:
        content, think_words = raw_content, 0

    return {
        "content": content,
        "input_tokens":  response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": actual,
        "think_block_words": think_words,
    }


def call_with_retry(
    prompt: str,
    system_prompt: str,
    model: str = MODEL_V3,
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
                log.error("deepseek_client", f"Unretriable error: {e}")
                raise
    raise MaxRetriesExceeded(f"DeepSeek API: failed after {max_retries} attempts") from last_exc
