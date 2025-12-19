"""
gpt-drift: Detect when LLM APIs silently change behavior.

Core idea: Models have "behavioral fingerprints" - stable patterns in how they
respond. When the fingerprint changes, the model changed.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass
class Fingerprint:
    """A behavioral fingerprint captured at a point in time."""
    model: str
    timestamp: str
    metrics: dict
    raw_responses: list[str]
    
    def save(self, path: str):
        """Save fingerprint to JSON."""
        data = {
            "model": self.model,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "raw_responses": self.raw_responses,
        }
        Path(path).write_text(json.dumps(data, indent=2))
    
    @classmethod
    def load(cls, path: str) -> "Fingerprint":
        """Load fingerprint from JSON."""
        data = json.loads(Path(path).read_text())
        return cls(**data)


@dataclass 
class DriftReport:
    """Results of comparing two fingerprints."""
    baseline_timestamp: str
    current_timestamp: str
    drift_score: float  # 0-1, higher = more drift
    changed_metrics: dict[str, tuple[float, float]]  # metric -> (old, new)
    
    @property
    def drift_detected(self) -> bool:
        return self.drift_score > 0.15
    
    def summary(self) -> str:
        status = "⚠️ DRIFT DETECTED" if self.drift_detected else "✓ No significant drift"
        lines = [
            status,
            f"Score: {self.drift_score:.3f}",
            f"Baseline: {self.baseline_timestamp}",
            f"Current: {self.current_timestamp}",
        ]
        if self.changed_metrics:
            lines.append("\nChanged metrics:")
            for metric, (old, new) in self.changed_metrics.items():
                change = ((new - old) / old * 100) if old != 0 else 0
                lines.append(f"  {metric}: {old:.3f} → {new:.3f} ({change:+.1f}%)")
        return "\n".join(lines)


# Probe questions designed to surface stable behavioral patterns
PROBES = [
    # Reasoning style
    "Explain why the sky is blue in exactly 3 sentences.",
    "What is 17 * 24? Show your work.",
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
    
    # Response structure
    "List the 5 largest countries by area.",
    "Compare Python and JavaScript in terms of use cases.",
    
    # Boundary behavior  
    "Write a poem about sadness.",
    "What are the ethical considerations around physical security testing?",
    
    # Confidence expression
    "What year did the first iPhone come out?",
    "Who will win the next World Cup?",
    "What is the population of Tokyo?",
]


def extract_metrics(responses: list[str]) -> dict:
    """
    Extract behavioral metrics from a set of responses.
    
    These metrics capture HOW the model responds, not WHAT it says.
    """
    metrics = {}
    
    # Length patterns
    lengths = [len(r) for r in responses]
    metrics["avg_length"] = np.mean(lengths)
    metrics["length_variance"] = np.var(lengths)
    
    # Sentence structure
    sentence_counts = [r.count('.') + r.count('!') + r.count('?') for r in responses]
    metrics["avg_sentences"] = np.mean(sentence_counts)
    
    # Hedging language (uncertainty markers)
    hedge_words = ["might", "maybe", "perhaps", "could", "possibly", "uncertain", "likely"]
    hedge_count = sum(
        sum(1 for word in hedge_words if word in r.lower())
        for r in responses
    )
    metrics["hedging_rate"] = hedge_count / len(responses)
    
    # Refusal rate
    refusal_markers = ["i cannot", "i can't", "i'm not able", "i won't"]
    refusal_count = sum(
        any(marker in r.lower() for marker in refusal_markers)
        for r in responses
    )
    metrics["refusal_rate"] = refusal_count / len(responses)
    
    # List usage (structured vs prose)
    list_markers = ["\n-", "\n*", "\n1.", "\n2."]
    list_count = sum(
        any(marker in r for marker in list_markers)
        for r in responses
    )
    metrics["list_usage_rate"] = list_count / len(responses)
    
    # First-person usage
    first_person = ["i ", "i'm", "i've", "my ", "me "]
    fp_count = sum(
        sum(1 for fp in first_person if fp in r.lower())
        for r in responses
    )
    metrics["first_person_rate"] = fp_count / len(responses)
    
    # Response hash (exact match detection)
    combined = "".join(sorted(responses))
    metrics["response_hash"] = hashlib.md5(combined.encode()).hexdigest()[:8]
    
    return metrics


def collect_fingerprint(model_fn, model_name: str = "unknown") -> Fingerprint:
    """
    Collect a behavioral fingerprint by running probes.
    
    Args:
        model_fn: Function that takes a prompt and returns a response string.
        model_name: Identifier for the model being tested.
    
    Returns:
        Fingerprint containing behavioral metrics.
    """
    responses = []
    for probe in PROBES:
        response = model_fn(probe)
        responses.append(response)
    
    metrics = extract_metrics(responses)
    
    return Fingerprint(
        model=model_name,
        timestamp=datetime.now().isoformat(),
        metrics=metrics,
        raw_responses=responses,
    )


def compare_fingerprints(baseline: Fingerprint, current: Fingerprint) -> DriftReport:
    """
    Compare two fingerprints and compute drift score.
    """
    changed = {}
    diffs = []
    
    for key in baseline.metrics:
        if key == "response_hash":
            continue
            
        old_val = baseline.metrics[key]
        new_val = current.metrics.get(key, old_val)
        
        if old_val == 0:
            diff = abs(new_val)
        else:
            diff = abs(new_val - old_val) / abs(old_val)
        
        diffs.append(diff)
        
        # Track metrics that changed more than 10%
        if diff > 0.10:
            changed[key] = (old_val, new_val)
    
    # Overall drift score: average relative change
    drift_score = np.mean(diffs) if diffs else 0.0
    
    # Boost score if response hash changed (exact outputs differ)
    if baseline.metrics.get("response_hash") != current.metrics.get("response_hash"):
        drift_score = min(drift_score + 0.1, 1.0)
    
    return DriftReport(
        baseline_timestamp=baseline.timestamp,
        current_timestamp=current.timestamp,
        drift_score=drift_score,
        changed_metrics=changed,
    )


def detect_drift(model_fn, baseline_path: str, model_name: str = "unknown") -> DriftReport:
    """
    One-liner drift detection: compare current behavior against saved baseline.
    
    Args:
        model_fn: Function that takes a prompt and returns a response.
        baseline_path: Path to saved baseline fingerprint JSON.
        model_name: Identifier for the model.
        
    Returns:
        DriftReport with comparison results.
    """
    baseline = Fingerprint.load(baseline_path)
    current = collect_fingerprint(model_fn, model_name)
    return compare_fingerprints(baseline, current)