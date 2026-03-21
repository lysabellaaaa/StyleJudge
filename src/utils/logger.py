"""
Structured logger for all experiment agents.
Writes to state/experiment_log.jsonl (one JSON object per line).
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.Lock()
_LOG_PATH = Path("state/experiment_log.jsonl")


def _write(record: dict) -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _record(level: str, agent: str, message: str, **kwargs) -> None:
    _write({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "agent": agent,
        "message": message,
        **kwargs,
    })


def _safe_print(text: str) -> None:
    """Print to console with ASCII fallback for non-cp1252 characters."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def info(agent: str, message: str, **kwargs) -> None:
    _safe_print(f"[{agent}] {message}")
    _record("INFO", agent, message, **kwargs)


def warn(agent: str, message: str, **kwargs) -> None:
    _safe_print(f"[WARN][{agent}] {message}")
    _record("WARN", agent, message, **kwargs)


def error(agent: str, message: str, **kwargs) -> None:
    _safe_print(f"[ERROR][{agent}] {message}")
    _record("ERROR", agent, message, **kwargs)


def api_call(agent: str, model: str, tokens_used: int, phase: str, item_id: str) -> None:
    _record("API", agent, f"call to {model}", model=model, tokens_used=tokens_used,
            phase=phase, item_id=item_id)
