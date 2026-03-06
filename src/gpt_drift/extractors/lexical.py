"""
Layer 1: Lexical heuristic extraction.

Fast, deterministic, transparent. Each behavioral construct is measured
by pattern-matching against curated marker lists with context-window
filtering to reduce false positives.

This layer is the reproducibility backbone — results are fully
deterministic given the same input and marker configuration.
"""

import re
from gpt_drift.constructs import (
    ALL_CONSTRUCTS, HEDGING, REFUSAL, CONFIDENCE,
    VERBOSITY, SENTIMENT, SAFETY_BOUNDARY,
)
from gpt_drift.segmentation import segment_statements
from gpt_drift.extractors.base import ExtractorResult


# Negative sentiment markers (tracked separately from the Construct)
_NEGATIVE_MARKERS = [
    "unfortunately", "I'm sorry", "regrettably", "sadly",
    "I apologize", "I regret", "difficult", "challenging",
]

# Positive sentiment markers (from the SENTIMENT construct)
_POSITIVE_MARKERS = SENTIMENT.markers


class LexicalExtractor:
    """
    Layer 1 behavioral signal extraction using lexical heuristics.

    For each response, extracts per-statement rates for each construct
    by checking marker presence with context-window exclusion filtering.
    """

    def extract(self, response: str) -> ExtractorResult:
        """
        Extract behavioral metrics from a single response.

        Args:
            response: The model's response text.

        Returns:
            ExtractorResult with all construct measurements.
        """
        statements = segment_statements(response)
        n_statements = len(statements) if statements else 1

        return ExtractorResult(
            hedging_rate=self._rate(HEDGING, statements, n_statements),
            refusal_rate=self._rate(REFUSAL, statements, n_statements),
            confidence_rate=self._rate(CONFIDENCE, statements, n_statements),
            reasoning_verbosity=self._verbosity(response, statements, n_statements),
            sentiment_polarity=self._sentiment(response, statements),
            safety_boundary=self._rate(SAFETY_BOUNDARY, statements, n_statements),
            avg_length=len(response),
            statement_count=n_statements,
            list_usage_rate=self._list_rate(response),
            first_person_rate=self._first_person_rate(statements, n_statements),
            layer="lexical",
        )

    def extract_batch(self, responses: list[str]) -> list[ExtractorResult]:
        """Extract metrics from multiple responses."""
        return [self.extract(r) for r in responses]

    def aggregate(self, results: list[ExtractorResult]) -> dict[str, float]:
        """
        Aggregate multiple ExtractorResults into mean metrics.

        Used for computing fingerprint-level metrics from per-response results.
        """
        if not results:
            return {}

        keys = [
            "hedging_rate", "refusal_rate", "confidence_rate",
            "reasoning_verbosity", "sentiment_polarity", "safety_boundary",
            "avg_length", "statement_count", "list_usage_rate", "first_person_rate",
        ]

        aggregated = {}
        for key in keys:
            values = [getattr(r, key) for r in results]
            aggregated[key] = sum(values) / len(values)

        return aggregated

    # ── Private Methods ──────────────────────────────────────────────────

    def _rate(self, construct, statements: list[str], n: int) -> float:
        """Compute per-statement rate for a construct with exclusion filtering."""
        if n == 0:
            return 0.0

        flagged = 0
        for stmt in statements:
            hits = construct.detect_markers(stmt)
            # Count statement as flagged if it has at least one non-excluded hit
            if any(not h["excluded"] for h in hits):
                flagged += 1

        return flagged / n

    def _verbosity(self, full_text: str, statements: list[str], n: int) -> float:
        """
        Compute reasoning verbosity as a composite signal.

        Combines statement count, reasoning marker density, and raw length
        into a normalized score. Higher = more verbose reasoning.
        """
        # Reasoning marker density
        marker_hits = VERBOSITY.detect_markers(full_text)
        active_hits = sum(1 for h in marker_hits if not h["excluded"])
        marker_density = active_hits / max(n, 1)

        # Normalized length (characters per statement)
        chars_per_stmt = len(full_text) / max(n, 1)

        # Composite: weight marker density and normalized length
        # Scale chars_per_stmt to roughly 0-1 range (200 chars/stmt = 1.0)
        length_score = min(chars_per_stmt / 200.0, 2.0)

        return (marker_density * 0.4 + length_score * 0.6)

    def _sentiment(self, full_text: str, statements: list[str]) -> float:
        """
        Compute sentiment polarity as positive - negative signal density.

        Returns value centered around 0:
        - Positive values = net positive sentiment
        - Negative values = net negative sentiment
        - Near zero = neutral
        """
        text_lower = full_text.lower()
        n = len(statements) if statements else 1

        pos_count = sum(1 for m in _POSITIVE_MARKERS if m.lower() in text_lower)
        neg_count = sum(1 for m in _NEGATIVE_MARKERS if m.lower() in text_lower)

        # Normalize by statement count, bound to [-1, 1]
        raw = (pos_count - neg_count) / max(n, 1)
        return max(-1.0, min(1.0, raw))

    def _list_rate(self, text: str) -> float:
        """Detect structured list usage (bullets, numbered lists)."""
        list_patterns = [r'\n\s*[-*•]\s+', r'\n\s*\d+[.)]\s+']
        total_matches = sum(
            len(re.findall(p, text)) for p in list_patterns
        )
        # Normalize: 5+ list items = 1.0
        return min(total_matches / 5.0, 1.0)

    def _first_person_rate(self, statements: list[str], n: int) -> float:
        """Detect first-person pronoun usage rate."""
        if n == 0:
            return 0.0

        fp_markers = ["i ", "i'm", "i've", "i'll", "my ", "me ", "mine"]
        flagged = 0
        for stmt in statements:
            lower = f" {stmt.lower()} "
            if any(f" {m}" in lower or lower.startswith(m) for m in fp_markers):
                flagged += 1

        return flagged / n
