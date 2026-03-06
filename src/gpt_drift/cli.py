"""
gpt-drift command-line interface.

Usage:
    gpt-drift run <model> [--output FILE] [--runs N]
    gpt-drift compare <file_a> <file_b> [--threshold T] [--output FILE]
    gpt-drift regression <baseline> <candidate> [--threshold T]
"""

import argparse
import json
import sys
from pathlib import Path

from gpt_drift.fingerprint import Fingerprint
from gpt_drift.collector import collect_fingerprint
from gpt_drift.comparison import compare_fingerprints


def main():
    parser = argparse.ArgumentParser(
        prog="gpt-drift",
        description="Behavioral drift measurement for language models.",
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # ── run: collect fingerprint ──
    run_parser = sub.add_parser("run", help="Collect a behavioral fingerprint")
    run_parser.add_argument("model", help="Model name (e.g., gpt-5.2, gpt-5.4)")
    run_parser.add_argument("--output", "-o", default=None, help="Output JSON path")
    run_parser.add_argument("--runs", "-n", type=int, default=1, help="Runs per prompt (default: 1)")
    run_parser.add_argument("--mock", action="store_true", help="Use mock model (no API)")
    run_parser.add_argument("--mock-version", default="v1", help="Mock version (v1 or v2)")

    # ── compare: compare two fingerprints ──
    cmp_parser = sub.add_parser("compare", help="Compare two fingerprints")
    cmp_parser.add_argument("file_a", help="Baseline fingerprint JSON")
    cmp_parser.add_argument("file_b", help="Comparison fingerprint JSON")
    cmp_parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Effect size threshold")
    cmp_parser.add_argument("--output", "-o", default=None, help="Save report to JSON")
    cmp_parser.add_argument("--json", action="store_true", help="Output JSON instead of table")

    # ── regression: CI/CD behavioral regression test ──
    reg_parser = sub.add_parser("regression", help="Behavioral regression test")
    reg_parser.add_argument("baseline", help="Baseline fingerprint JSON")
    reg_parser.add_argument("candidate", help="Candidate fingerprint JSON")
    reg_parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Effect size threshold")

    args = parser.parse_args()

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "compare":
        _cmd_compare(args)
    elif args.command == "regression":
        _cmd_regression(args)
    else:
        parser.print_help()


def _cmd_run(args):
    """Collect a behavioral fingerprint."""
    model_fn = _get_model_fn(args)
    model_name = args.model if not args.mock else f"mock-{args.mock_version}"

    print(f"Collecting fingerprint for {model_name} (n_runs={args.runs})...")
    fp = collect_fingerprint(model_fn, model_name, n_runs=args.runs)

    output = args.output or f"{model_name.replace('/', '_')}_fingerprint.json"
    fp.save(output)

    print(f"Fingerprint saved to {output}")
    print(f"\nMetrics:")
    for key, value in fp.metrics.items():
        if not key.endswith("_std") and not key.endswith("_cv"):
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")


def _cmd_compare(args):
    """Compare two fingerprints."""
    fp_a = Fingerprint.load(args.file_a)
    fp_b = Fingerprint.load(args.file_b)

    report = compare_fingerprints(fp_a, fp_b, threshold=args.threshold)

    if args.json:
        print(json.dumps(report.to_json(), indent=2))
    else:
        print(report.summary())

    if args.output:
        report.save(args.output)
        print(f"\nReport saved to {args.output}")


def _cmd_regression(args):
    """Behavioral regression test for CI/CD pipelines."""
    fp_baseline = Fingerprint.load(args.baseline)
    fp_candidate = Fingerprint.load(args.candidate)

    report = compare_fingerprints(fp_baseline, fp_candidate, threshold=args.threshold)

    print("Behavioral Regression Report")
    print("=" * 60)
    print()

    failures = []
    for c in report.constructs:
        if c.effect_size >= args.threshold:
            status = "⚠ EXCEEDS THRESHOLD"
            failures.append(c.name)
        else:
            status = "✓ within threshold"
        print(f"  {c.name:<25} {c.drift_pct:+.0f}% (d={c.effect_size:.2f})  {status}")

    print()
    if failures:
        print(f"STATUS: REVIEW REQUIRED ({len(failures)} construct(s) exceed threshold)")
        sys.exit(1)
    else:
        print("STATUS: PASS (all constructs within threshold)")
        sys.exit(0)


def _get_model_fn(args):
    """Get model function based on CLI args."""
    if args.mock:
        return _mock_model_fn(args.mock_version)

    # Try OpenAI
    try:
        from openai import OpenAI
        import os
        if os.getenv("OPENAI_API_KEY"):
            client = OpenAI()
            model = args.model

            def openai_fn(prompt: str) -> str:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1024,
                )
                return response.choices[0].message.content
            return openai_fn
    except ImportError:
        pass

    print("No OpenAI API key found. Using mock model.")
    print("Install openai and set OPENAI_API_KEY for real models.\n")
    return _mock_model_fn(args.mock_version if hasattr(args, 'mock_version') else "v1")


def _mock_model_fn(version: str = "v1"):
    """Mock model for testing without API access."""
    def model_fn(prompt: str) -> str:
        base = f"This is a response to: {prompt[:50]}..."
        if version == "v1":
            return f"{base}\n\n1. First point\n2. Second point\n3. Third point"
        else:
            return (
                f"I think {base.lower()} Perhaps this helps. "
                f"Maybe I should also mention that this could vary. "
                f"It's hard to say for certain, but likely the answer "
                f"involves several considerations."
            )
    return model_fn
