"""
Behavioral fingerprint: a snapshot of model behavior at a point in time.

A fingerprint captures both aggregate metrics and per-response raw data,
enabling reproducibility analysis and detailed drift interpretation.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Fingerprint:
    """A behavioral fingerprint captured at a point in time."""

    model: str
    timestamp: str
    metrics: dict[str, float]
    raw_responses: list[str]

    # Multi-run data: per-prompt, per-run metrics for reproducibility
    # Shape: list of dicts, one per prompt, each with per-run metric lists
    per_prompt_metrics: list[dict] = field(default_factory=list)

    # Configuration used to collect this fingerprint
    config: dict = field(default_factory=dict)

    def save(self, path: str):
        """Save fingerprint to JSON."""
        data = {
            "model": self.model,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "raw_responses": self.raw_responses,
            "per_prompt_metrics": self.per_prompt_metrics,
            "config": self.config,
        }
        Path(path).write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load(cls, path: str) -> "Fingerprint":
        """Load fingerprint from JSON."""
        data = json.loads(Path(path).read_text())
        return cls(
            model=data["model"],
            timestamp=data["timestamp"],
            metrics=data["metrics"],
            raw_responses=data.get("raw_responses", []),
            per_prompt_metrics=data.get("per_prompt_metrics", []),
            config=data.get("config", {}),
        )

    @classmethod
    def create(cls, model: str, metrics: dict, raw_responses: list[str],
               per_prompt_metrics: list[dict] | None = None,
               config: dict | None = None) -> "Fingerprint":
        """Create a new fingerprint with current timestamp."""
        return cls(
            model=model,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            raw_responses=raw_responses,
            per_prompt_metrics=per_prompt_metrics or [],
            config=config or {},
        )
