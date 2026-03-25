"""
Token-bucket rate limiter per API.
Thread-safe; blocks until capacity is available.
"""
import threading
import time


class TokenBucketRateLimiter:
    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self._req_tokens = requests_per_minute
        self._tok_tokens = tokens_per_minute
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        self._req_tokens = min(self.rpm, self._req_tokens + elapsed * (self.rpm / 60))
        self._tok_tokens = min(self.tpm, self._tok_tokens + elapsed * (self.tpm / 60))

    def acquire(self, estimated_tokens: int = 500) -> None:
        """Block until one request slot and estimated_tokens are available."""
        while True:
            with self._lock:
                self._refill()
                if self._req_tokens >= 1 and self._tok_tokens >= estimated_tokens:
                    self._req_tokens -= 1
                    self._tok_tokens -= estimated_tokens
                    return
            time.sleep(0.5)

    def release_extra(self, actual_tokens: int, estimated_tokens: int) -> None:
        """Return unused token capacity if actual < estimated."""
        diff = estimated_tokens - actual_tokens
        if diff > 0:
            with self._lock:
                self._tok_tokens = min(self.tpm, self._tok_tokens + diff)


# Pre-configured limiters (conservative; adjust in experiment.yaml)
ANTHROPIC_LIMITER = TokenBucketRateLimiter(requests_per_minute=50,  tokens_per_minute=40_000)
OPENAI_LIMITER    = TokenBucketRateLimiter(requests_per_minute=60,  tokens_per_minute=60_000)
GROQ_LIMITER      = TokenBucketRateLimiter(requests_per_minute=30,  tokens_per_minute=6_000)
ZHIPU_LIMITER     = TokenBucketRateLimiter(requests_per_minute=30,  tokens_per_minute=20_000)
DEEPSEEK_LIMITER  = TokenBucketRateLimiter(requests_per_minute=60,  tokens_per_minute=60_000)
GEMINI_LIMITER    = TokenBucketRateLimiter(requests_per_minute=15,  tokens_per_minute=1_000_000)
