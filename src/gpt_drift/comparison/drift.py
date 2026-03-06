"""
Drift comparison: statistical comparison of two behavioral fingerprints.

Computes per-construct drift using:
- Mean difference
- Effect size (Cohen's d)
- P-value (Welch's t-test or Mann-Whitney U)
- Coefficient of variation for stability classification
- Qualitative drift labels (negligible / small / medium / large)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats

from gpt_drift.fingerprint import Fingerprint
from gpt_drift.extractors.base import CORE_CONSTRUCTS


@dataclass
class ConstructResult:
    """Drift analysis result for a single behavioral construct."""

    name: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    drift_pct: float
    effect_size: float  # Cohen's d
    p_value: float
    cv_a: float
    cv_b: float
    drift_label: str  # negligible, small, medium, large
    stability_a: str  # stable, moderate, unstable
    stability_b: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mean_a": round(self.mean_a, 4),
            "mean_b": round(self.mean_b, 4),
            "std_a": round(self.std_a, 4),
            "std_b": round(self.std_b, 4),
            "drift_pct": round(self.drift_pct, 1),
            "effect_size": round(self.effect_size, 3),
            "p_value": round(self.p_value, 4),
            "cv_a": round(self.cv_a, 3),
            "cv_b": round(self.cv_b, 3),
            "drift_label": self.drift_label,
            "stability_a": self.stability_a,
            "stability_b": self.stability_b,
        }


@dataclass
class DriftReport:
    """Complete drift comparison report."""

    model_a: str
    model_b: str
    constructs: list[ConstructResult]
    drift_score: float  # Overall drift magnitude
    drift_detected: bool
    changed_metrics: dict[str, tuple[float, float]] = field(default_factory=dict)

    # For backward compatibility
    baseline_timestamp: str = ""
    current_timestamp: str = ""

    def summary(self) -> str:
        """Human-readable drift summary."""
        status = "⚠️  DRIFT DETECTED" if self.drift_detected else "✓ No significant drift"

        lines = [
            f"Behavior Drift Summary: {self.model_a} → {self.model_b}",
            "",
            status,
            f"Overall drift score: {self.drift_score:.3f}",
            "",
            f"{'Construct':<25} {'A (mean±sd)':<15} {'B (mean±sd)':<15} {'Drift':<10} {'Effect':<10} {'p-value':<10}",
            "-" * 85,
        ]

        for c in self.constructs:
            a_str = f"{c.mean_a:.2f}±{c.std_a:.2f}"
            b_str = f"{c.mean_b:.2f}±{c.std_b:.2f}"
            d_str = f"{c.drift_pct:+.0f}%"
            e_str = f"{c.effect_size:.2f}({c.drift_label[0].upper()})" if c.drift_label != "negligible" else f"{c.effect_size:.2f}"
            p_str = f"{c.p_value:.3f}" if c.p_value >= 0.001 else "<0.001"
            lines.append(f"{c.name:<25} {a_str:<15} {b_str:<15} {d_str:<10} {e_str:<10} {p_str:<10}")

        return "\n".join(lines)

    def to_json(self) -> dict:
        """Machine-readable JSON report."""
        return {
            "comparison": {
                "model_a": self.model_a,
                "model_b": self.model_b,
                "drift_score": round(self.drift_score, 4),
                "drift_detected": self.drift_detected,
            },
            "constructs": [c.to_dict() for c in self.constructs],
        }

    def save(self, path: str):
        """Save report to JSON file."""
        Path(path).write_text(json.dumps(self.to_json(), indent=2))


def compare_fingerprints(
    baseline: Fingerprint,
    current: Fingerprint,
    threshold: float = 0.5,
) -> DriftReport:
    """
    Compare two fingerprints and compute per-construct drift statistics.

    Uses effect sizes (Cohen's d) for drift magnitude and Welch's t-test
    for statistical significance. Falls back to relative change when
    per-run data is not available.

    Args:
        baseline: Reference fingerprint (Model A).
        current: Comparison fingerprint (Model B).
        threshold: Effect size threshold for drift detection (default: 0.5 = medium).

    Returns:
        DriftReport with per-construct analysis.
    """
    construct_results = []
    effect_sizes = []
    changed = {}

    for key in CORE_CONSTRUCTS:
        mean_a = baseline.metrics.get(key, 0.0)
        mean_b = current.metrics.get(key, 0.0)
        std_a = baseline.metrics.get(f"{key}_std", 0.0)
        std_b = current.metrics.get(f"{key}_std", 0.0)

        # Compute drift percentage
        if mean_a != 0:
            drift_pct = ((mean_b - mean_a) / abs(mean_a)) * 100
        else:
            drift_pct = 0.0 if mean_b == 0 else 100.0

        # Compute Cohen's d
        d = _cohens_d(mean_a, mean_b, std_a, std_b)

        # Compute p-value if we have per-run data
        p_value = _compute_p_value(baseline, current, key)

        # CV and stability classification
        cv_a = std_a / abs(mean_a) if mean_a != 0 else 0.0
        cv_b = std_b / abs(mean_b) if mean_b != 0 else 0.0

        result = ConstructResult(
            name=key,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            drift_pct=drift_pct,
            effect_size=abs(d),
            p_value=p_value,
            cv_a=cv_a,
            cv_b=cv_b,
            drift_label=_drift_label(abs(d)),
            stability_a=_stability_label(cv_a),
            stability_b=_stability_label(cv_b),
        )

        construct_results.append(result)
        effect_sizes.append(abs(d))

        # Track changed metrics (>10% change)
        if abs(drift_pct) > 10:
            changed[key] = (mean_a, mean_b)

    # Overall drift score: mean of absolute effect sizes
    overall = float(np.mean(effect_sizes)) if effect_sizes else 0.0
    detected = any(c.effect_size >= threshold for c in construct_results)

    return DriftReport(
        model_a=baseline.model,
        model_b=current.model,
        constructs=construct_results,
        drift_score=overall,
        drift_detected=detected,
        changed_metrics=changed,
        baseline_timestamp=baseline.timestamp,
        current_timestamp=current.timestamp,
    )


# ── Private Helpers ──────────────────────────────────────────────────────────

def _cohens_d(mean_a: float, mean_b: float, std_a: float, std_b: float) -> float:
    """Compute Cohen's d effect size."""
    if mean_a == mean_b:
        return 0.0

    # Both SDs effectively zero — no variance data available
    if std_a == 0 and std_b == 0:
        # Signal large effect but cap at reasonable maximum
        # (true d is undefined without variance)
        return min(abs(mean_b - mean_a) / max(abs(mean_a), abs(mean_b), 0.01), 10.0)

    pooled = np.sqrt((std_a ** 2 + std_b ** 2) / 2)

    # Guard against near-zero pooled SD (floating point artifacts
    # from deterministic or near-deterministic responses)
    if pooled < 1e-10:
        return min(abs(mean_b - mean_a) / max(abs(mean_a), abs(mean_b), 0.01), 10.0)

    d = (mean_b - mean_a) / pooled
    return min(abs(d), 10.0) * (1 if d >= 0 else -1)


def _compute_p_value(baseline: Fingerprint, current: Fingerprint, key: str) -> float:
    """
    Compute p-value for drift significance.

    Uses per-prompt metric distributions when available.
    Falls back to approximate test from summary statistics.
    """
    values_a = _extract_per_prompt_values(baseline, key)
    values_b = _extract_per_prompt_values(current, key)

    if values_a is not None and values_b is not None and len(values_a) > 1 and len(values_b) > 1:
        # Check if both arrays are constant (no variance to test)
        if np.std(values_a) < 1e-10 and np.std(values_b) < 1e-10:
            return 1.0 if np.mean(values_a) == np.mean(values_b) else 0.0
        # Welch's t-test (does not assume equal variance)
        _, p = stats.ttest_ind(values_a, values_b, equal_var=False)
        return float(p) if not np.isnan(p) else 1.0

    # Fallback: approximate from summary statistics
    mean_a = baseline.metrics.get(key, 0.0)
    mean_b = current.metrics.get(key, 0.0)
    std_a = baseline.metrics.get(f"{key}_std", 0.0)
    std_b = current.metrics.get(f"{key}_std", 0.0)
    n = max(baseline.config.get("n_probes", 10), 2)

    if std_a == 0 and std_b == 0:
        return 1.0 if mean_a == mean_b else 0.0

    se = np.sqrt(std_a ** 2 / n + std_b ** 2 / n)
    if se == 0:
        return 1.0

    t_stat = abs(mean_b - mean_a) / se
    # Approximate degrees of freedom (Welch-Satterthwaite)
    df = max((std_a ** 2 / n + std_b ** 2 / n) ** 2 /
             ((std_a ** 2 / n) ** 2 / (n - 1) + (std_b ** 2 / n) ** 2 / (n - 1))
             if (std_a > 0 or std_b > 0) else 1, 1)
    p = 2 * stats.t.sf(t_stat, df)
    return float(p)


def _extract_per_prompt_values(fp: Fingerprint, key: str) -> list[float] | None:
    """Extract per-prompt mean values for a construct from fingerprint data."""
    if not fp.per_prompt_metrics:
        return None

    values = []
    for prompt_data in fp.per_prompt_metrics:
        metrics = prompt_data.get("metrics", {})
        if key in metrics:
            values.append(metrics[key]["mean"])

    return values if values else None


def _drift_label(d: float) -> str:
    """Classify effect size into qualitative drift label."""
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def _stability_label(cv: float) -> str:
    """Classify coefficient of variation into stability label."""
    if cv < 0.15:
        return "stable"
    elif cv <= 0.30:
        return "moderate"
    else:
        return "unstable"
