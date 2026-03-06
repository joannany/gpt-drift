"""Base types for the extraction layer."""

from dataclasses import dataclass, field


@dataclass
class ExtractorResult:
    """Result from extracting behavioral signals from a single response."""
    hedging_rate: float = 0.0
    refusal_rate: float = 0.0
    confidence_rate: float = 0.0
    reasoning_verbosity: float = 0.0
    sentiment_polarity: float = 0.0
    safety_boundary: float = 0.0

    # Additional structural metrics
    avg_length: float = 0.0
    statement_count: int = 0
    list_usage_rate: float = 0.0
    first_person_rate: float = 0.0

    # Layer metadata
    layer: str = "unknown"

    def to_dict(self) -> dict:
        """Convert to dict, excluding metadata fields."""
        return {
            "hedging_rate": self.hedging_rate,
            "refusal_rate": self.refusal_rate,
            "confidence_rate": self.confidence_rate,
            "reasoning_verbosity": self.reasoning_verbosity,
            "sentiment_polarity": self.sentiment_polarity,
            "safety_boundary": self.safety_boundary,
            "avg_length": self.avg_length,
            "statement_count": self.statement_count,
            "list_usage_rate": self.list_usage_rate,
            "first_person_rate": self.first_person_rate,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractorResult":
        """Create from dict, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


# The six core behavioral constructs
CORE_CONSTRUCTS = [
    "hedging_rate",
    "refusal_rate",
    "confidence_rate",
    "reasoning_verbosity",
    "sentiment_polarity",
    "safety_boundary",
]
