# gpt-drift

Detect when LLM APIs silently change behavior.

## The Problem

LLM APIs evolve continuously, and behavioral changes can occur as models are updated or re-tuned in production. In some cases, these shifts have been observed publicly without corresponding version changes.

For downstream applications that depend on stable behavioral patterns, such silent changes can introduce failures that are hard to detect without dedicated monitoring.

## How It Works

Instead of checking if answers are "correct," we measure behavioral patterns:

| Metric | What It Captures |
|--------|------------------|
| `avg_length` | Response verbosity |
| `hedging_rate` | Uncertainty language ("maybe", "perhaps") |
| `refusal_rate` | Safety boundary strictness |
| `list_usage_rate` | Structured vs prose responses |
| `first_person_rate` | Self-reference frequency |

When these metrics shift significantly, the model has changed.

## Quick Start

```bash
pip install openai numpy
```

```python
from drift import collect_fingerprint, compare_fingerprints

# Define how to call your model
def my_model(prompt: str) -> str:
    # Your API call here
    return response

# Collect baseline
baseline = collect_fingerprint(my_model, "gpt-4")
baseline.save("baseline.json")

# Later: check for drift
from drift import detect_drift
report = detect_drift(my_model, "baseline.json", "gpt-4")
print(report.summary())
```

## Example Output

```
⚠️ DRIFT DETECTED
Score: 0.234
Baseline: 2024-01-15T10:30:00
Current: 2024-02-01T14:22:00

Changed metrics:
  hedging_rate: 0.180 → 0.340 (+88.9%)
  avg_length: 245.000 → 312.000 (+27.3%)
  refusal_rate: 0.100 → 0.200 (+100.0%)
```

## CLI Usage

```bash
# Create baseline
python experiment.py --create-baseline --model gpt-4o-mini

# Check for drift
python experiment.py --check --model gpt-4o-mini

# Test without API (uses mock model)
python experiment.py --create-baseline --mock --mock-version v1
python experiment.py --check --mock --mock-version v2
```

## Run Tests

```bash
python test_drift.py
```

## Design Decisions

**Why behavioral metrics instead of output comparison?**

Exact output matching is too brittle—models are stochastic. Behavioral metrics capture stable patterns that persist even when specific outputs vary.

**Why these specific probes?**

The probe set tests different capabilities: reasoning, factual recall, structured output, boundary behavior. Changes in any area will surface in the metrics.

**What drift score threshold should I use?**

The default is 0.15 (15% average change). Access via `report.drift_detected`. Tune based on your tolerance for false positives.

## Limitations

- Requires deterministic settings (temperature=0) for reliable fingerprinting
- Probe set may not cover all relevant behaviors for your use case
- Detection ≠ diagnosis: tells you *something* changed, not *what* or *why*

## Disclaimer

This tool measures behavioral drift, not correctness. It is intended for model evaluation and monitoring purposes. Do not include sensitive prompts or store sensitive data in raw responses.
> Note: This project is not affiliated with or endorsed by any LLM provider.

## License

MIT

## Citation

```bibtex
@software{jo_2026_gptdrift,
  author = {Jo, Anna},
  title  = {GPT Drift},
  year   = {2026},
  url    = {https://github.com/joannany/gpt-drift}
}
```
