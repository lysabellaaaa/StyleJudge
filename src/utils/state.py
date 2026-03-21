"""
Experiment state manager with crash recovery.
Uses atomic file writes and threading.Lock() for thread-safe concurrent access.
"""
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path


class ExperimentState:
    def __init__(self, state_path: str):
        self.state_path = Path(state_path)
        self._lock = threading.Lock()
        self._data = self._default_state()

    def _default_state(self) -> dict:
        return {
            "current_phase": "dataset",
            "osf_preregistration_url": None,
            "completed": {
                "base_generation": [],
                "style_rewriting": [],
                "length_normalization": [],
                "irr_check": [],
                "formality_perception": [],
                "qa_verification": [],
                "human_spot_check": [],
                "adversarial_injection": [],
                "evaluation_gpt4o": [],
                "evaluation_llama70b": [],
                "evaluation_glm5": [],
                "mitigation_fixed_rubric": [],
                "mitigation_style_norm": [],
                "mitigation_style_agnostic": [],
                "mechanistic_position": [],
                "mechanistic_buffer": [],
                "mechanistic_logprob": [],
                "mechanistic_two_pass": [],
                "analysis": [],
            },
            "errors": [],
            "last_updated": self._now(),
        }

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def from_file(cls, path: str) -> "ExperimentState":
        instance = cls(path)
        if Path(path).exists():
            instance.load()
        else:
            instance.save()
        return instance

    def load(self) -> None:
        with self._lock:
            with open(self.state_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    def save(self) -> None:
        """Atomic save: write to .tmp then rename."""
        with self._lock:
            self._data["last_updated"] = self._now()
            tmp_path = self.state_path.with_suffix(".json.tmp")
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.state_path)

    def mark_complete(self, phase: str, item_id: str) -> None:
        with self._lock:
            if phase not in self._data["completed"]:
                self._data["completed"][phase] = []
            if item_id not in self._data["completed"][phase]:
                self._data["completed"][phase].append(item_id)
        self.save()

    def is_complete(self, phase: str, item_id: str) -> bool:
        with self._lock:
            return item_id in self._data["completed"].get(phase, [])

    def is_phase_complete(self, phase_key: str) -> bool:
        """Check if a sentinel 'done' marker exists for a whole phase."""
        return self.is_complete(phase_key, "done")

    def mark_phase_complete(self, phase_key: str) -> None:
        self.mark_complete(phase_key, "done")

    def get_phase(self) -> str:
        with self._lock:
            return self._data["current_phase"]

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._data["current_phase"] = phase
        self.save()

    def get_completed(self, phase: str) -> list:
        with self._lock:
            return list(self._data["completed"].get(phase, []))

    def log_error(self, phase: str, item_id: str, error: str) -> None:
        with self._lock:
            self._data["errors"].append({
                "phase": phase,
                "item_id": item_id,
                "error": error,
                "timestamp": self._now(),
            })
        self.save()

    @property
    def data(self) -> dict:
        with self._lock:
            return dict(self._data)

    def set_osf_url(self, url: str) -> None:
        with self._lock:
            self._data["osf_preregistration_url"] = url
        self.save()

    def get_osf_url(self) -> str | None:
        with self._lock:
            return self._data.get("osf_preregistration_url")
