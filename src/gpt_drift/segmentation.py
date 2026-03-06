"""
Statement segmentation for behavioral metric computation.

A statement is the base unit for computing behavioral rates (e.g., hedging
rate = hedging statements / total statements). Implemented as a hybrid
rule-based sentence tokenizer with fallback boundary detection.
"""

import re


# Abbreviations that shouldn't trigger sentence splits
_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
    "vs.", "etc.", "inc.", "ltd.", "corp.",
    "e.g.", "i.e.", "fig.", "eq.", "approx.",
    "st.", "ave.", "blvd.",
}

# Pattern for sentence-ending punctuation
_SENT_END = re.compile(r'([.!?])\s+(?=[A-Z"\'])|([.!?])$')

# Pattern for list items (numbered or bulleted)
_LIST_ITEM = re.compile(r'^\s*(?:\d+[.)]|\*|-|•)\s+', re.MULTILINE)

# Pattern for semicolons splitting independent clauses
_SEMICOLON = re.compile(r';\s+')


def segment_statements(text: str) -> list[str]:
    """
    Split text into statements.

    Rules:
    - Terminal punctuation (. ! ?) followed by whitespace + capital letter
    - Semicolons split into separate statements
    - List items are individual statements
    - Compound sentences with coordinating conjunctions stay as one statement
    - Code blocks and special formatting are treated as single statements

    Returns:
        List of statement strings, stripped of leading/trailing whitespace.
    """
    if not text or not text.strip():
        return []

    # Handle code blocks as single units
    text = _collapse_code_blocks(text)

    # Split on list items first
    segments = _split_on_lists(text)

    # For each segment, split on sentence boundaries
    statements = []
    for segment in segments:
        statements.extend(_split_sentences(segment))

    # Clean up
    result = [s.strip() for s in statements if s.strip()]
    return result if result else [text.strip()]


def _collapse_code_blocks(text: str) -> str:
    """Replace code blocks with a single placeholder token."""
    # Triple backtick blocks
    text = re.sub(r'```[\s\S]*?```', '[CODE_BLOCK]', text)
    # Indented code (4+ spaces at line start, consecutive lines)
    text = re.sub(r'(?:^    .+\n?)+', '[CODE_BLOCK]\n', text, flags=re.MULTILINE)
    return text


def _split_on_lists(text: str) -> list[str]:
    """Split text on list item boundaries."""
    parts = _LIST_ITEM.split(text)
    return [p for p in parts if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """Split a text block into sentences, respecting abbreviations."""
    # First split on semicolons
    clauses = _SEMICOLON.split(text)

    sentences = []
    for clause in clauses:
        sentences.extend(_split_on_periods(clause))

    return sentences


def _split_on_periods(text: str) -> list[str]:
    """Split on sentence-ending punctuation, skipping abbreviations."""
    results = []
    current = []
    tokens = text.split()

    for token in tokens:
        current.append(token)
        token_lower = token.lower().rstrip('"\')')

        # Check if this token ends a sentence
        if token_lower.endswith(('.', '!', '?')):
            # Skip if it's a known abbreviation
            if token_lower in _ABBREVIATIONS:
                continue
            # Skip if it looks like a decimal number
            if re.match(r'\d+\.\d*$', token_lower):
                continue

            sentence = ' '.join(current)
            results.append(sentence)
            current = []

    # Don't lose trailing text without terminal punctuation
    if current:
        trailing = ' '.join(current)
        results.append(trailing)

    return results


def count_statements(text: str) -> int:
    """Count statements in text. Convenience wrapper."""
    return len(segment_statements(text))
