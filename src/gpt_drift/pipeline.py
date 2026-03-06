"""
High-level pipeline functions for common gpt-drift workflows.
"""

from typing import Callable

from gpt_drift.fingerprint import Fingerprint
from gpt_drift.collector import collect_fingerprint
from gpt_drift.comparison import compare_fingerprints, DriftReport


def detect_drift(
    model_fn: Callable[[str], str],
    baseline_path: str,
    model_name: str = "unknown",
    n_runs: int = 1,
    threshold: float = 0.5,
) -> DriftReport:
    """
    One-liner drift detection: compare current behavior against saved baseline.

    Args:
        model_fn: Function that takes a prompt and returns a response.
        baseline_path: Path to saved baseline fingerprint JSON.
        model_name: Identifier for the model.
        n_runs: Number of runs per prompt for reproducibility.
        threshold: Effect size threshold for flagging drift.

    Returns:
        DriftReport with comparison results.
    """
    baseline = Fingerprint.load(baseline_path)
    current = collect_fingerprint(model_fn, model_name, n_runs=n_runs)
    return compare_fingerprints(baseline, current, threshold=threshold)


def compare_models(
    model_a_fn: Callable[[str], str],
    model_b_fn: Callable[[str], str],
    model_a_name: str = "model_a",
    model_b_name: str = "model_b",
    n_runs: int = 5,
    threshold: float = 0.5,
) -> DriftReport:
    """
    Compare two models directly without saving fingerprints.

    Args:
        model_a_fn: Function for model A (baseline).
        model_b_fn: Function for model B (comparison).
        model_a_name: Name for model A.
        model_b_name: Name for model B.
        n_runs: Number of runs per prompt.
        threshold: Effect size threshold for flagging drift.

    Returns:
        DriftReport with comparison results.
    """
    fp_a = collect_fingerprint(model_a_fn, model_a_name, n_runs=n_runs)
    fp_b = collect_fingerprint(model_b_fn, model_b_name, n_runs=n_runs)
    return compare_fingerprints(fp_a, fp_b, threshold=threshold)
