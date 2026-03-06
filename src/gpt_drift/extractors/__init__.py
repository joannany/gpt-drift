"""
Signal extraction layers for behavioral measurement.

Layer 1: Lexical heuristics (core, deterministic)
Layer 2: Classifier-based extraction (core, requires models)
Layer 3: LLM-as-judge interpretation (experimental, optional)
"""

from gpt_drift.extractors.lexical import LexicalExtractor
from gpt_drift.extractors.base import ExtractorResult

__all__ = ["LexicalExtractor", "ExtractorResult"]
