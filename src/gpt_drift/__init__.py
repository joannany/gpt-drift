"""
gpt-drift: Behavioral drift measurement for language models.

Detect and interpret behavioral changes between model versions.
"""

from gpt_drift.fingerprint import Fingerprint
from gpt_drift.comparison import compare_fingerprints, DriftReport, ConstructResult
from gpt_drift.collector import collect_fingerprint
from gpt_drift.pipeline import detect_drift

__version__ = "1.0.0"

__all__ = [
    "Fingerprint",
    "DriftReport",
    "ConstructResult",
    "compare_fingerprints",
    "collect_fingerprint",
    "detect_drift",
]
