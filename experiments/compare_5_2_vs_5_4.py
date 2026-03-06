"""
gpt-drift Replication Example: GPT-5.2 vs GPT-5.4

OpenAI says GPT-5.4 reduces "unnecessary refusals and overly caveated
responses." This script measures how.

Usage:
    # Full comparison (requires OPENAI_API_KEY)
    python experiments/compare_5_2_vs_5_4.py

    # Single model fingerprint
    python experiments/compare_5_2_vs_5_4.py --model gpt-5.4

    # Fewer probes for quick test
    python experiments/compare_5_2_vs_5_4.py --quick

    # Custom runs per prompt
    python experiments/compare_5_2_vs_5_4.py --runs 10

Requirements:
    pip install gpt-drift[openai]
    export OPENAI_API_KEY=your_key
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("OpenAI package required: pip install openai")
    sys.exit(1)

from gpt_drift import collect_fingerprint, compare_fingerprints
from gpt_drift.probes import DEFAULT_PROBES, PROBE_CATEGORIES


# ── Configuration ────────────────────────────────────────────────────────────

MODEL_A = "gpt-5.2"
MODEL_B = "gpt-5.4"

# Per the design spec: temperature 0.7 approximates typical conversational
# deployment settings while preserving measurable output variability.
TEMPERATURE = 0.7
MAX_TOKENS = 1024
N_RUNS = 5

# Output directory
OUTPUT_DIR = Path("results")


def create_model_fn(client: OpenAI, model: str):
    """Create a function that queries an OpenAI model."""
    def model_fn(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content or ""
    return model_fn


def run_fingerprint(client: OpenAI, model: str, probes: list[str],
                    n_runs: int, output_dir: Path) -> Path:
    """Collect and save a fingerprint for a model."""
    print(f"\n{'='*60}")
    print(f"Collecting fingerprint: {model}")
    print(f"  Probes: {len(probes)}")
    print(f"  Runs per probe: {n_runs}")
    print(f"  Total API calls: {len(probes) * n_runs}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"{'='*60}\n")

    model_fn = create_model_fn(client, model)

    start = time.time()
    fp = collect_fingerprint(
        model_fn,
        model_name=model,
        probes=probes,
        n_runs=n_runs,
    )
    elapsed = time.time() - start

    # Save fingerprint
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model.replace('-', '_')}_{timestamp}.json"
    filepath = output_dir / filename
    fp.save(str(filepath))

    print(f"\nFingerprint collected in {elapsed:.1f}s")
    print(f"Saved to: {filepath}")
    print(f"\nMetrics:")
    for key, value in fp.metrics.items():
        if not key.endswith("_std") and not key.endswith("_cv"):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")

    # Print stability info if multi-run
    if n_runs > 1:
        print(f"\nStability (CV):")
        for key in ["hedging_rate", "refusal_rate", "confidence_rate",
                     "reasoning_verbosity", "sentiment_polarity", "safety_boundary"]:
            cv = fp.metrics.get(f"{key}_cv", 0)
            label = "stable" if cv < 0.15 else "moderate" if cv <= 0.30 else "unstable"
            print(f"  {key}: CV={cv:.3f} ({label})")

    return filepath


def run_comparison(path_a: str, path_b: str, output_dir: Path):
    """Compare two fingerprints and save the report."""
    from gpt_drift.fingerprint import Fingerprint

    fp_a = Fingerprint.load(path_a)
    fp_b = Fingerprint.load(path_b)

    report = compare_fingerprints(fp_a, fp_b)

    # Print summary
    print(f"\n{'='*60}")
    print(report.summary())
    print(f"{'='*60}")

    # Save JSON report
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"drift_report_{timestamp}.json"
    report.save(str(report_path))
    print(f"\nFull report saved to: {report_path}")

    # Print interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}\n")

    interpretations = {
        "hedging_rate": (
            "OpenAI described GPT-5.4 as reducing 'overly caveated responses.' "
            "A decrease in hedging rate would confirm this claim."
        ),
        "refusal_rate": (
            "The release notes mention reducing 'unnecessary refusals.' "
            "A decrease in refusal rate would confirm this claim."
        ),
        "confidence_rate": (
            "If hedging decreases, we would expect confidence to increase "
            "as the model becomes more direct in its assertions."
        ),
        "reasoning_verbosity": (
            "GPT-5.4 is described as more 'token efficient.' "
            "A decrease in reasoning verbosity would align with this."
        ),
        "sentiment_polarity": (
            "Users complained about artificially warm 'cringe' tone in prior versions. "
            "A shift in sentiment polarity would indicate tone adjustment."
        ),
        "safety_boundary": (
            "The release notes mention 'reducing unnecessary refusals' while "
            "'preserving strong protections against misuse.' A decrease would "
            "suggest relaxed safety framing on non-sensitive topics."
        ),
    }

    for c in report.constructs:
        direction = "increased" if c.drift_pct > 0 else "decreased"
        print(f"{c.name}: {direction} {abs(c.drift_pct):.0f}% (d={c.effect_size:.2f}, {c.drift_label})")
        if c.name in interpretations:
            print(f"  Context: {interpretations[c.name]}")
        print()

    return report


def main():
    parser = argparse.ArgumentParser(
        description="gpt-drift: GPT-5.2 vs GPT-5.4 behavioral comparison"
    )
    parser.add_argument("--model", default=None,
                        help="Run single model only (e.g., gpt-5.4)")
    parser.add_argument("--model-a", default=MODEL_A,
                        help=f"Baseline model (default: {MODEL_A})")
    parser.add_argument("--model-b", default=MODEL_B,
                        help=f"Comparison model (default: {MODEL_B})")
    parser.add_argument("--runs", type=int, default=N_RUNS,
                        help=f"Runs per prompt (default: {N_RUNS})")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 10 probes, 3 runs")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help="Output directory")
    parser.add_argument("--compare", nargs=2, metavar=("FILE_A", "FILE_B"),
                        help="Compare two existing fingerprint files")
    args = parser.parse_args()

    output_dir = Path(args.output)

    # Compare existing files
    if args.compare:
        run_comparison(args.compare[0], args.compare[1], output_dir)
        return

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Export your API key: export OPENAI_API_KEY=your_key")
        sys.exit(1)

    client = OpenAI()

    # Select probes
    if args.quick:
        # 2 probes per category = 10 total
        probes = []
        for category_probes in PROBE_CATEGORIES.values():
            probes.extend(category_probes[:2])
        n_runs = 3
        print("Quick mode: 10 probes, 3 runs per prompt")
    else:
        probes = DEFAULT_PROBES
        n_runs = args.runs

    # Estimate cost
    est_calls = len(probes) * n_runs * (1 if args.model else 2)
    print(f"\nEstimated API calls: {est_calls}")
    print(f"Estimated cost: ~${est_calls * 0.01:.2f} (rough estimate)")

    # Single model mode
    if args.model:
        run_fingerprint(client, args.model, probes, n_runs, output_dir)
        return

    # Full comparison
    print(f"\nRunning full comparison: {args.model_a} vs {args.model_b}")

    path_a = run_fingerprint(client, args.model_a, probes, n_runs, output_dir)
    path_b = run_fingerprint(client, args.model_b, probes, n_runs, output_dir)

    run_comparison(str(path_a), str(path_b), output_dir)


if __name__ == "__main__":
    main()
