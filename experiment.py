"""
Experiment: Compare GPT behavior across API calls.

This script demonstrates drift detection in practice. Run it periodically
to track if your LLM provider silently updates their model.

Usage:
    # First run: create baseline
    python experiment.py --create-baseline
    
    # Later runs: check for drift
    python experiment.py --check
"""

import argparse
import os
from pathlib import Path

# Use OpenAI if available, otherwise simulate
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from drift import collect_fingerprint, detect_drift, Fingerprint


BASELINE_PATH = "baseline.json"


def create_openai_model_fn(model: str = "gpt-4o-mini"):
    """Create a function that queries OpenAI."""
    client = OpenAI()
    
    def model_fn(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # Deterministic for fingerprinting
            max_tokens=500,
        )
        return response.choices[0].message.content
    
    return model_fn


def create_mock_model_fn(version: str = "v1"):
    """
    Mock model for testing without API access.
    Simulates two different "versions" to demonstrate drift detection.
    """
    def model_fn(prompt: str) -> str:
        # Simulate different behavior based on version
        base_response = f"This is a response to: {prompt[:50]}..."
        
        if version == "v1":
            # Original behavior: concise, uses lists
            return f"{base_response}\n\n1. First point\n2. Second point"
        else:
            # "Updated" behavior: verbose, more hedging
            return f"I think {base_response.lower()} Perhaps this helps. Maybe I should also mention that this could vary."
    
    return model_fn


def main():
    parser = argparse.ArgumentParser(description="GPT Drift Detection Experiment")
    parser.add_argument("--create-baseline", action="store_true", help="Create new baseline fingerprint")
    parser.add_argument("--check", action="store_true", help="Check current behavior against baseline")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to test")
    parser.add_argument("--mock", action="store_true", help="Use mock model (no API needed)")
    parser.add_argument("--mock-version", default="v1", help="Mock model version (v1 or v2)")
    args = parser.parse_args()
    
    # Select model function
    if args.mock:
        print(f"Using mock model (version: {args.mock_version})")
        model_fn = create_mock_model_fn(args.mock_version)
        model_name = f"mock-{args.mock_version}"
    elif HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        print(f"Using OpenAI model: {args.model}")
        model_fn = create_openai_model_fn(args.model)
        model_name = args.model
    else:
        print("No OpenAI API key found. Using mock model.")
        print("Set OPENAI_API_KEY to test with real models.\n")
        model_fn = create_mock_model_fn(args.mock_version)
        model_name = f"mock-{args.mock_version}"
    
    if args.create_baseline:
        print(f"\nCollecting baseline fingerprint for {model_name}...")
        fingerprint = collect_fingerprint(model_fn, model_name)
        fingerprint.save(BASELINE_PATH)
        print(f"Baseline saved to {BASELINE_PATH}")
        print(f"\nMetrics captured:")
        for key, value in fingerprint.metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
                
    elif args.check:
        if not Path(BASELINE_PATH).exists():
            print(f"No baseline found at {BASELINE_PATH}")
            print("Run with --create-baseline first.")
            return
        
        print(f"\nChecking {model_name} against baseline...")
        report = detect_drift(model_fn, BASELINE_PATH, model_name)
        print(f"\n{report.summary()}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
