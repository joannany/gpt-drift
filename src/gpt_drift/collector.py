"""
Fingerprint collection with multi-run reproducibility.

For each probe prompt, collects N responses and extracts behavioral
signals from each independently. Reports measurement statistics
(mean, std, CI, CV) rather than single-run values.
"""

from typing import Callable

import numpy as np

from gpt_drift.fingerprint import Fingerprint
from gpt_drift.extractors.lexical import LexicalExtractor
from gpt_drift.extractors.base import CORE_CONSTRUCTS
from gpt_drift.probes import DEFAULT_PROBES


def collect_fingerprint(
    model_fn: Callable[[str], str],
    model_name: str = "unknown",
    probes: list[str] | None = None,
    n_runs: int = 1,
    extractor: LexicalExtractor | None = None,
) -> Fingerprint:
    """
    Collect a behavioral fingerprint by running probes.

    Args:
        model_fn: Function that takes a prompt and returns a response string.
        model_name: Identifier for the model being tested.
        probes: Prompt list. Defaults to built-in probe set.
        n_runs: Number of runs per prompt for reproducibility (default: 1).
                Use n_runs >= 5 for statistical analysis.
        extractor: Extractor instance. Defaults to LexicalExtractor.

    Returns:
        Fingerprint containing behavioral metrics and raw responses.
    """
    if probes is None:
        probes = DEFAULT_PROBES
    if extractor is None:
        extractor = LexicalExtractor()

    all_responses = []
    all_results = []
    per_prompt_metrics = []

    for probe in probes:
        prompt_runs = []
        prompt_results = []

        for run_idx in range(n_runs):
            response = model_fn(probe)
            result = extractor.extract(response)

            prompt_runs.append(response)
            prompt_results.append(result)

        all_responses.extend(prompt_runs)
        all_results.extend(prompt_results)

        # Store per-prompt, per-run metrics for reproducibility analysis
        if n_runs > 1:
            prompt_metrics = {}
            for key in CORE_CONSTRUCTS:
                values = [getattr(r, key) for r in prompt_results]
                prompt_metrics[key] = {
                    "values": values,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }
            per_prompt_metrics.append({
                "probe": probe,
                "metrics": prompt_metrics,
            })

    # Aggregate metrics across all responses
    aggregate = extractor.aggregate(all_results)

    # Add reproducibility statistics if multi-run
    if n_runs > 1:
        for key in CORE_CONSTRUCTS:
            all_values = [getattr(r, key) for r in all_results]
            mean = float(np.mean(all_values))
            std = float(np.std(all_values, ddof=1)) if len(all_values) > 1 else 0.0
            cv = std / mean if mean != 0 else 0.0
            aggregate[f"{key}_std"] = std
            aggregate[f"{key}_cv"] = cv

    config = {
        "n_probes": len(probes),
        "n_runs": n_runs,
        "extractor": "lexical",
    }

    return Fingerprint.create(
        model=model_name,
        metrics=aggregate,
        raw_responses=all_responses,
        per_prompt_metrics=per_prompt_metrics,
        config=config,
    )
