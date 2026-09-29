#!/usr/bin/env python3
"""Normalization and value-matching functions for deterministic scoring.

The P2 scorer and answer-bearing-turn builder import this module directly so
they apply the same phrase-boundary rules.
"""
from __future__ import annotations

import re
import unicodedata


_PUNCT_RE = re.compile(r"[^\w\s:/-]+")
_WS_RE = re.compile(r"\s+")

_TIME_AMPM_RE = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)\b", flags=re.IGNORECASE
)
_TIME_HHMM_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_TIME_BARE_HHMM_RE = re.compile(r"\b(\d{3,4})\b")

_MW_ALIAS = {
    "centre": "center",
    "color": "colour",
    "1 person": "1",
    "one person": "1",
    "2 people": "2",
    "two people": "2",
    "3 people": "3",
    "three people": "3",
    "4 people": "4",
    "five people": "5",
    "5 people": "5",
    "moderate price": "moderate",
    "moderately priced": "moderate",
    "expensive priced": "expensive",
    "cheap priced": "cheap",
    "guesthouse": "guest house",
    "asian oriental": "asian",
    "north american": "american",
    "panasian": "asian",
}


def _strip_punct(s: str) -> str:
    return _PUNCT_RE.sub(" ", s)


def _to_24h(h: int, m: int, ampm: str) -> str:
    ampm = ampm.lower().replace(".", "")
    if ampm == "pm" and h != 12:
        h += 12
    if ampm == "am" and h == 12:
        h = 0
    return f"{h:02d}:{m:02d}"


def _normalize_times(s: str) -> str:
    def _ampm_sub(match):
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        return _to_24h(hour, minute, match.group(3))

    s = _TIME_AMPM_RE.sub(_ampm_sub, s)

    def _hhmm_sub(match):
        return f"{int(match.group(1)):02d}:{int(match.group(2)):02d}"

    s = _TIME_HHMM_RE.sub(_hhmm_sub, s)

    def _bare_sub(match):
        value = match.group(1)
        if len(value) == 3:
            hour, minute = int(value[0]), int(value[1:])
        else:
            hour, minute = int(value[:2]), int(value[2:])
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"
        return value

    return _TIME_BARE_HHMM_RE.sub(_bare_sub, s)


def _apply_aliases(s: str) -> str:
    for source, target in _MW_ALIAS.items():
        s = re.sub(rf"\b{re.escape(source)}\b", target, s)
    return s


def normalize_value(s: str | None, dataset: str = "sgd") -> str:
    """Normalize a slot value or candidate answer for comparison."""
    if s is None:
        return ""
    value = unicodedata.normalize("NFKC", str(s)).lower().strip()
    if dataset == "multiwoz":
        value = _apply_aliases(value)
    value = _normalize_times(value)
    value = _strip_punct(value)
    return _WS_RE.sub(" ", value).strip()


def normalize_turn(s: str, dataset: str = "sgd") -> str:
    """Apply the scorer's value normalization to dialogue text."""
    return normalize_value(s, dataset)


def normalized_phrase_occurs(needle: str, haystack: str) -> bool:
    """Return whether a normalized phrase occurs at word-character bounds.

    The boundary check rules out short-value accidents such as ``sf`` inside
    ``successfully`` and ``2`` inside ``20`` while retaining normalized times,
    entities, slashes, colons, and hyphens.
    """
    if not needle:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack))


def value_occurs_in_text(
    value: str | None, text: str | None, dataset: str = "sgd"
) -> bool:
    """Apply the shared scorer normalization and bounded occurrence rule."""
    normalized_value = normalize_value(value, dataset)
    normalized_text = normalize_turn(text or "", dataset)
    return normalized_phrase_occurs(normalized_value, normalized_text)


def values_match(a: str | None, b: str | None, dataset: str = "sgd") -> bool:
    """Return whether two non-empty values normalize identically."""
    normalized_a = normalize_value(a, dataset)
    normalized_b = normalize_value(b, dataset)
    return bool(normalized_a) and normalized_a == normalized_b


def values_match_loose(
    prediction: str | None, reference: str | None, dataset: str = "sgd"
) -> bool:
    """Return whether the complete normalized reference occurs in a prediction.

    P2 answer-bearing turns and loose P2 answers use the same directional
    predicate: the complete reference value must occur at phrase boundaries in
    the candidate text. A prediction may add context around the reference, but
    a shortened fragment of a multiword reference receives no credit.
    """
    return value_occurs_in_text(reference, prediction, dataset)


def value_in_turns(value: str | None, turns: list[str], dataset: str = "sgd") -> bool:
    """Normalized-substring eligibility filter for probe construction.

    Answer-bearing-turn detection uses the phrase-bounded
    ``value_occurs_in_text`` predicate.
    """
    normalized_value = normalize_value(value, dataset)
    if not normalized_value:
        return False
    return any(
        normalized_value in normalize_turn(turn, dataset) for turn in turns
    )


def humanize_slot(canonical: str) -> str:
    """Convert a machine-readable slot identifier to a readable label."""
    return re.sub(r"[_\-]+", " ", canonical).strip().lower()


if __name__ == "__main__":
    assert normalize_value("12 pm", "sgd") == "12:00"
    assert normalize_value("guesthouse", "multiwoz") == "guest house"
    assert value_occurs_in_text("Atlanta, GA", "Leaving from Atlanta GA.", "sgd")
    assert not value_occurs_in_text("sf", "successfully", "sgd")
    assert values_match_loose("Atlanta GA airport", "Atlanta, GA", "sgd")
    assert not values_match_loose("Atlanta", "Atlanta, GA", "sgd")
    assert not values_match_loose("2", "20", "sgd")
    print("normalize self-test: pass")
