"""Tests for gpt-drift."""

import tempfile
from drift import (
    extract_metrics,
    collect_fingerprint,
    compare_fingerprints,
    Fingerprint,
)


def test_extract_metrics_captures_length():
    """Metrics should capture response length patterns."""
    responses = ["Short.", "This is a longer response with more words."]
    metrics = extract_metrics(responses)
    
    assert "avg_length" in metrics
    assert metrics["avg_length"] > 0


def test_extract_metrics_detects_hedging():
    """Metrics should detect uncertainty language."""
    confident = ["The answer is 42.", "Paris is the capital."]
    hedging = ["Maybe the answer is 42.", "Perhaps Paris could be the capital."]
    
    confident_metrics = extract_metrics(confident)
    hedging_metrics = extract_metrics(hedging)
    
    assert hedging_metrics["hedging_rate"] > confident_metrics["hedging_rate"]


def test_extract_metrics_detects_refusals():
    """Metrics should detect refusal patterns."""
    helpful = ["Here's how to do it.", "The answer is..."]
    refusing = ["I cannot help with that.", "I'm not able to assist."]
    
    helpful_metrics = extract_metrics(helpful)
    refusing_metrics = extract_metrics(refusing)
    
    assert refusing_metrics["refusal_rate"] > helpful_metrics["refusal_rate"]


def test_extract_metrics_rates_within_bounds():
    """Rates should be normalized to 0-1 per response."""
    responses = [
        "I think maybe this could work.",
        "Here is a direct answer without hedging.",
    ]
    metrics = extract_metrics(responses)
    
    assert 0 <= metrics["hedging_rate"] <= 1
    assert 0 <= metrics["first_person_rate"] <= 1
    assert 0 <= metrics["refusal_rate"] <= 1
    assert 0 <= metrics["list_usage_rate"] <= 1


def test_fingerprint_save_load():
    """Fingerprints should roundtrip through JSON."""
    fp = Fingerprint(
        model="test-model",
        timestamp="2024-01-01T00:00:00",
        metrics={"avg_length": 100.0, "hedging_rate": 0.2},
        raw_responses=["response 1", "response 2"],
    )
    
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        fp.save(f.name)
        loaded = Fingerprint.load(f.name)
    
    assert loaded.model == fp.model
    assert loaded.metrics == fp.metrics


def test_compare_fingerprints_no_drift():
    """Identical fingerprints should show no drift."""
    fp1 = Fingerprint(
        model="test",
        timestamp="2024-01-01",
        metrics={"avg_length": 100.0, "hedging_rate": 0.2},
        raw_responses=["test"],
    )
    fp2 = Fingerprint(
        model="test",
        timestamp="2024-01-02",
        metrics={"avg_length": 100.0, "hedging_rate": 0.2},
        raw_responses=["test"],
    )
    
    report = compare_fingerprints(fp1, fp2)
    
    assert report.drift_score < 0.1
    assert not report.drift_detected


def test_compare_fingerprints_detects_drift():
    """Different fingerprints should show drift."""
    fp1 = Fingerprint(
        model="test",
        timestamp="2024-01-01",
        metrics={"avg_length": 100.0, "hedging_rate": 0.1},
        raw_responses=["short"],
    )
    fp2 = Fingerprint(
        model="test",
        timestamp="2024-01-02",
        metrics={"avg_length": 200.0, "hedging_rate": 0.5},  # 2x length, 5x hedging
        raw_responses=["much longer response"],
    )
    
    report = compare_fingerprints(fp1, fp2)
    
    assert report.drift_score > 0.15
    assert report.drift_detected
    assert "avg_length" in report.changed_metrics


def test_collect_fingerprint():
    """Should collect fingerprint from model function."""
    def mock_model(prompt: str) -> str:
        return f"Response to: {prompt}"
    
    fp = collect_fingerprint(mock_model, "mock")
    
    assert fp.model == "mock"
    assert len(fp.raw_responses) > 0
    assert "avg_length" in fp.metrics


if __name__ == "__main__":
    # Simple test runner
    import sys
    
    tests = [
        test_extract_metrics_captures_length,
        test_extract_metrics_detects_hedging,
        test_extract_metrics_detects_refusals,
        test_extract_metrics_rates_within_bounds,        
        test_fingerprint_save_load,
        test_compare_fingerprints_no_drift,
        test_compare_fingerprints_detects_drift,
        test_collect_fingerprint,
    ]
    
    passed = 0
    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
    
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
