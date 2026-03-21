"""
Smoke test: validates all APIs respond correctly before any experiment runs.
Run this first: python scripts/smoke_test.py
Exits with code 1 if Anthropic or OpenAI fail (required). Groq and Zhipu warn only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api import anthropic_client, openai_client, together_client, zhipu_client

PING_PROMPT = "Reply with exactly: OK"
PING_SYSTEM = "You are a test assistant. Follow instructions precisely."


def test_anthropic() -> bool:
    print("Testing Anthropic (Claude Sonnet 4.6)...", end=" ")
    try:
        r = anthropic_client.call(PING_PROMPT, PING_SYSTEM, model="claude-sonnet-4-6",
                                   temperature=0.0, max_tokens=10)
        assert "OK" in r["content"].upper(), f"Unexpected response: {r['content']}"
        print(f"OK ({r['output_tokens']} tokens)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_openai() -> bool:
    print("Testing OpenAI (GPT-4o)...", end=" ")
    try:
        r = openai_client.call(PING_PROMPT, PING_SYSTEM, model="gpt-4o",
                                temperature=0.0, max_tokens=10)
        assert "OK" in r["content"].upper(), f"Unexpected response: {r['content']}"
        print(f"OK ({r['output_tokens']} tokens)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_groq() -> bool:
    print("Testing Groq (Llama 3.3 70B)...", end=" ")
    try:
        r = together_client.call(PING_PROMPT, PING_SYSTEM,
                                  model="llama-3.3-70b-versatile",
                                  temperature=0.0, max_tokens=20)
        assert r["content"] and len(r["content"]) > 0, "Empty response"
        print(f"OK ({r['output_tokens']} tokens) — '{r['content'][:40].strip()}'")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_zhipu() -> bool:
    print("Testing Zhipu AI (GLM-4)...", end=" ")
    try:
        r = zhipu_client.call(PING_PROMPT, PING_SYSTEM, model="research-model",
                               temperature=0.0, max_tokens=20)
        assert r["content"] and len(r["content"]) > 0, "Empty response"
        print(f"OK ({r['output_tokens']} tokens) — '{r['content'][:40].strip()}'")
        return True
    except Exception as e:
        print(f"FAILED: {str(e).encode('ascii', errors='replace').decode('ascii')}")
        return False


def test_groq_logprobs() -> bool:
    """Verify Groq returns logprobs for mechanistic analysis (M3)."""
    print("Testing Groq logprobs (for M3 mechanistic)...", end=" ")
    try:
        r = together_client.call(
            "Say 'Hello'.", PING_SYSTEM,
            model="llama-3.3-70b-versatile",
            temperature=0.0, max_tokens=10, logprobs=5,
        )
        if r["logprobs"] is not None:
            print("OK (logprobs returned)")
        else:
            print("WARN: logprobs not returned — M3 mechanistic experiment may fail")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def main():
    print("=" * 55)
    print("StyleJudge API Smoke Test")
    print("=" * 55)
    results = {
        "anthropic": test_anthropic(),
        "openai": test_openai(),
        "groq": test_groq(),
        "zhipu": test_zhipu(),
        "groq_logprobs": test_groq_logprobs(),
    }
    print("\n" + "=" * 55)
    passed = sum(results.values())
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    if not results["anthropic"] or not results["openai"]:
        print("CRITICAL: Anthropic and OpenAI are required. Cannot proceed.")
        sys.exit(1)
    if not results["groq"]:
        print("WARN: Groq unavailable — Llama 3.3 70B judge will be skipped.")
    if not results["zhipu"]:
        print("WARN: Zhipu AI unavailable — GLM-4 judge will be skipped.")
    print("Smoke test complete. Safe to run experiment.")


if __name__ == "__main__":
    main()
