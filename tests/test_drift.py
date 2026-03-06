"""Tests for gpt-drift v1."""

import tempfile
import json

from gpt_drift import (
    Fingerprint,
    DriftReport,
    ConstructResult,
    compare_fingerprints,
    collect_fingerprint,
    detect_drift,
)
from gpt_drift.extractors.lexical import LexicalExtractor
from gpt_drift.extractors.base import ExtractorResult, CORE_CONSTRUCTS
from gpt_drift.segmentation import segment_statements, count_statements
from gpt_drift.constructs import HEDGING, REFUSAL, CONFIDENCE
from gpt_drift.probes import DEFAULT_PROBES, PROBE_CATEGORIES
from gpt_drift.pipeline import compare_models


# ── Segmentation Tests ───────────────────────────────────────────────────────

class TestSegmentation:

    def test_simple_sentences(self):
        text = "This is first. This is second. This is third."
        stmts = segment_statements(text)
        assert len(stmts) == 3

    def test_semicolon_split(self):
        text = "First clause; second clause"
        stmts = segment_statements(text)
        assert len(stmts) == 2

    def test_list_items(self):
        text = "Intro:\n- Item one\n- Item two\n- Item three"
        stmts = segment_statements(text)
        assert len(stmts) >= 3

    def test_empty_string(self):
        assert segment_statements("") == []

    def test_no_punctuation(self):
        stmts = segment_statements("A sentence without ending punctuation")
        assert len(stmts) == 1

    def test_abbreviations_not_split(self):
        text = "Dr. Smith went to Washington. He arrived safely."
        stmts = segment_statements(text)
        assert len(stmts) == 2

    def test_count_statements(self):
        assert count_statements("One. Two. Three.") == 3


# ── Construct Tests ──────────────────────────────────────────────────────────

class TestConstructs:

    def test_hedging_detects_markers(self):
        hits = HEDGING.detect_markers("I think maybe this is correct.")
        markers_found = [h["marker"] for h in hits if not h["excluded"]]
        assert len(markers_found) > 0

    def test_hedging_excludes_statistical_context(self):
        hits = HEDGING.detect_markers("The probability is likely above 0.95 with high confidence interval.")
        excluded = [h for h in hits if h["excluded"]]
        assert len(excluded) > 0

    def test_refusal_detects_markers(self):
        hits = REFUSAL.detect_markers("I cannot help with that request.")
        active = [h for h in hits if not h["excluded"]]
        assert len(active) > 0

    def test_confidence_detects_markers(self):
        hits = CONFIDENCE.detect_markers("The answer is certainly 42.")
        active = [h for h in hits if not h["excluded"]]
        assert len(active) > 0


# ── Lexical Extractor Tests ──────────────────────────────────────────────────

class TestLexicalExtractor:

    def setup_method(self):
        self.extractor = LexicalExtractor()

    def test_extract_returns_result(self):
        result = self.extractor.extract("Hello, this is a test response.")
        assert isinstance(result, ExtractorResult)
        assert result.layer == "lexical"

    def test_hedging_rate_higher_for_hedging_text(self):
        confident = self.extractor.extract("The answer is 42. Paris is the capital.")
        hedging = self.extractor.extract("Maybe the answer is 42. Perhaps Paris could be the capital.")
        assert hedging.hedging_rate > confident.hedging_rate

    def test_refusal_rate_higher_for_refusals(self):
        helpful = self.extractor.extract("Here's how to do it. The answer is clear.")
        refusing = self.extractor.extract("I cannot help with that. I'm not able to assist.")
        assert refusing.refusal_rate > helpful.refusal_rate

    def test_rates_within_bounds(self):
        result = self.extractor.extract(
            "I think maybe this could work. Here is a direct answer."
        )
        assert 0 <= result.hedging_rate <= 1
        assert 0 <= result.refusal_rate <= 1
        assert 0 <= result.confidence_rate <= 1
        assert 0 <= result.first_person_rate <= 1
        assert 0 <= result.list_usage_rate <= 1

    def test_batch_extraction(self):
        responses = ["Response one.", "Response two.", "Response three."]
        results = self.extractor.extract_batch(responses)
        assert len(results) == 3
        assert all(isinstance(r, ExtractorResult) for r in results)

    def test_aggregate(self):
        results = self.extractor.extract_batch(["Short.", "Longer response here."])
        agg = self.extractor.aggregate(results)
        assert "hedging_rate" in agg
        assert "avg_length" in agg
        assert agg["avg_length"] > 0

    def test_verbosity_higher_for_verbose_text(self):
        concise = self.extractor.extract("Yes.")
        verbose = self.extractor.extract(
            "Let me explain step by step. First, we consider the premise. "
            "Therefore, the conclusion follows. In other words, the answer is yes. "
            "To summarize, after careful analysis, the result is confirmed."
        )
        assert verbose.reasoning_verbosity > concise.reasoning_verbosity

    def test_sentiment_positive(self):
        result = self.extractor.extract("Great question! Happy to help. Of course!")
        assert result.sentiment_polarity > 0

    def test_sentiment_negative(self):
        result = self.extractor.extract("Unfortunately I apologize. I'm sorry, this is difficult.")
        assert result.sentiment_polarity < 0


# ── Fingerprint Tests ────────────────────────────────────────────────────────

class TestFingerprint:

    def test_save_load_roundtrip(self):
        fp = Fingerprint.create(
            model="test-model",
            metrics={"hedging_rate": 0.2, "avg_length": 100.0},
            raw_responses=["response 1", "response 2"],
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fp.save(f.name)
            loaded = Fingerprint.load(f.name)

        assert loaded.model == fp.model
        assert loaded.metrics == fp.metrics
        assert loaded.raw_responses == fp.raw_responses

    def test_save_load_with_per_prompt(self):
        fp = Fingerprint.create(
            model="test",
            metrics={"hedging_rate": 0.3},
            raw_responses=["r1"],
            per_prompt_metrics=[{
                "probe": "test prompt",
                "metrics": {"hedging_rate": {"values": [0.2, 0.4], "mean": 0.3, "std": 0.14}},
            }],
            config={"n_runs": 2},
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fp.save(f.name)
            loaded = Fingerprint.load(f.name)

        assert len(loaded.per_prompt_metrics) == 1
        assert loaded.config["n_runs"] == 2


# ── Comparison Tests ─────────────────────────────────────────────────────────

class TestComparison:

    def test_no_drift_identical(self):
        fp = Fingerprint.create(
            model="test",
            metrics={k: 0.5 for k in CORE_CONSTRUCTS},
            raw_responses=["test"],
        )
        report = compare_fingerprints(fp, fp)
        assert report.drift_score < 0.1
        assert not report.drift_detected

    def test_detects_large_drift(self):
        fp_a = Fingerprint.create(
            model="model_a",
            metrics={"hedging_rate": 0.1, "refusal_rate": 0.1, "confidence_rate": 0.5,
                     "reasoning_verbosity": 0.3, "sentiment_polarity": 0.0, "safety_boundary": 0.2,
                     "hedging_rate_std": 0.02, "refusal_rate_std": 0.02,
                     "confidence_rate_std": 0.05, "reasoning_verbosity_std": 0.03,
                     "sentiment_polarity_std": 0.01, "safety_boundary_std": 0.02},
            raw_responses=["short"],
            config={"n_probes": 10},
        )
        fp_b = Fingerprint.create(
            model="model_b",
            metrics={"hedging_rate": 0.5, "refusal_rate": 0.6, "confidence_rate": 0.1,
                     "reasoning_verbosity": 0.8, "sentiment_polarity": 0.4, "safety_boundary": 0.7,
                     "hedging_rate_std": 0.03, "refusal_rate_std": 0.03,
                     "confidence_rate_std": 0.04, "reasoning_verbosity_std": 0.05,
                     "sentiment_polarity_std": 0.02, "safety_boundary_std": 0.04},
            raw_responses=["much longer"],
            config={"n_probes": 10},
        )
        report = compare_fingerprints(fp_a, fp_b)
        assert report.drift_detected
        assert report.drift_score > 0.5
        assert len(report.changed_metrics) > 0

    def test_summary_format(self):
        fp = Fingerprint.create(
            model="test",
            metrics={k: 0.3 for k in CORE_CONSTRUCTS},
            raw_responses=["test"],
        )
        report = compare_fingerprints(fp, fp)
        summary = report.summary()
        assert "Behavior Drift Summary" in summary
        assert "hedging_rate" in summary

    def test_json_output(self):
        fp = Fingerprint.create(
            model="test",
            metrics={k: 0.3 for k in CORE_CONSTRUCTS},
            raw_responses=["test"],
        )
        report = compare_fingerprints(fp, fp)
        j = report.to_json()
        assert "comparison" in j
        assert "constructs" in j
        assert len(j["constructs"]) == len(CORE_CONSTRUCTS)

    def test_save_report(self):
        fp = Fingerprint.create(
            model="test",
            metrics={k: 0.3 for k in CORE_CONSTRUCTS},
            raw_responses=["test"],
        )
        report = compare_fingerprints(fp, fp)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            report.save(f.name)
            data = json.loads(open(f.name).read())
        assert "constructs" in data

    def test_construct_result_fields(self):
        fp_a = Fingerprint.create(
            model="a",
            metrics={"hedging_rate": 0.2, "refusal_rate": 0.1, "confidence_rate": 0.5,
                     "reasoning_verbosity": 0.3, "sentiment_polarity": 0.0, "safety_boundary": 0.2},
            raw_responses=["test"],
        )
        fp_b = Fingerprint.create(
            model="b",
            metrics={"hedging_rate": 0.4, "refusal_rate": 0.1, "confidence_rate": 0.5,
                     "reasoning_verbosity": 0.3, "sentiment_polarity": 0.0, "safety_boundary": 0.2},
            raw_responses=["test"],
        )
        report = compare_fingerprints(fp_a, fp_b)
        hedging = next(c for c in report.constructs if c.name == "hedging_rate")
        assert hedging.drift_label in ("negligible", "small", "medium", "large")
        assert hedging.stability_a in ("stable", "moderate", "unstable")


# ── Collector Tests ──────────────────────────────────────────────────────────

class TestCollector:

    def test_collect_fingerprint_basic(self):
        def mock_model(prompt: str) -> str:
            return f"Response to: {prompt}"

        fp = collect_fingerprint(mock_model, "mock", probes=["Test prompt 1", "Test prompt 2"])
        assert fp.model == "mock"
        assert len(fp.raw_responses) == 2
        assert "hedging_rate" in fp.metrics

    def test_collect_with_multi_run(self):
        call_count = 0
        def mock_model(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"Response {call_count} to: {prompt}"

        fp = collect_fingerprint(
            mock_model, "mock",
            probes=["Prompt 1", "Prompt 2"],
            n_runs=3,
        )
        assert call_count == 6  # 2 prompts × 3 runs
        assert len(fp.raw_responses) == 6
        assert len(fp.per_prompt_metrics) == 2
        assert "hedging_rate_std" in fp.metrics
        assert "hedging_rate_cv" in fp.metrics


# ── Pipeline Tests ───────────────────────────────────────────────────────────

class TestPipeline:

    def test_detect_drift(self):
        def mock_model(prompt: str) -> str:
            return f"Response to: {prompt}"

        fp = collect_fingerprint(mock_model, "baseline", probes=["Test"])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fp.save(f.name)
            report = detect_drift(mock_model, f.name, "current", n_runs=1)

        assert isinstance(report, DriftReport)

    def test_compare_models(self):
        def model_a(prompt: str) -> str:
            return "Direct answer."

        def model_b(prompt: str) -> str:
            return "I think maybe the answer could possibly be something."

        report = compare_models(
            model_a, model_b,
            "confident", "hedging",
            n_runs=1,
        )
        assert isinstance(report, DriftReport)
        hedging = next(c for c in report.constructs if c.name == "hedging_rate")
        assert hedging.mean_b > hedging.mean_a


# ── Sanity Check Tests (Construct Validation) ────────────────────────────────

class TestSanityChecks:
    """
    Construct validation: verify metrics respond appropriately to
    controlled inputs. If these fail, the construct is broken.
    """

    def setup_method(self):
        self.extractor = LexicalExtractor()

    def test_hedging_sanity(self):
        """Uncertain topics should produce higher hedging than certain ones."""
        uncertain = self.extractor.extract(
            "Perhaps the cause of the Permian extinction was possibly volcanic activity. "
            "Maybe it could have been an asteroid, though scientists are uncertain."
        )
        certain = self.extractor.extract(
            "A spider has eight legs. This is a well-established biological fact."
        )
        assert uncertain.hedging_rate > certain.hedging_rate

    def test_refusal_sanity(self):
        """Safety-adjacent prompts should trigger higher refusal signals."""
        refusing = self.extractor.extract(
            "I cannot provide instructions for that. I'm not able to assist with this request."
        )
        helpful = self.extractor.extract(
            "Photosynthesis converts sunlight into chemical energy. Here is how it works."
        )
        assert refusing.refusal_rate > helpful.refusal_rate

    def test_confidence_sanity(self):
        """Factual answers should show higher confidence than speculative ones."""
        factual = self.extractor.extract(
            "The answer is definitely 4. This is certainly correct without any doubt."
        )
        speculative = self.extractor.extract(
            "It might be related to several factors that could influence the outcome."
        )
        assert factual.confidence_rate > speculative.confidence_rate


# ── Probe Tests ──────────────────────────────────────────────────────────────

class TestProbes:

    def test_default_probes_count(self):
        assert len(DEFAULT_PROBES) == 50

    def test_categories_cover_all_probes(self):
        all_categorized = []
        for probes in PROBE_CATEGORIES.values():
            all_categorized.extend(probes)
        assert set(all_categorized) == set(DEFAULT_PROBES)

    def test_five_categories(self):
        assert len(PROBE_CATEGORIES) == 5
        for name, probes in PROBE_CATEGORIES.items():
            assert len(probes) == 10, f"Category {name} has {len(probes)} probes, expected 10"
