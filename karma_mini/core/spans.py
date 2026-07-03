"""
Verbatim-span utilities.

Gold phrases are exact character spans of the Stanza-tokenized sentence, so a
predicted phrase only scores if it is byte-identical to a slice of its source
line. These helpers locate a phrase in its line and repair the common LLM
drifts (casing, re-joined punctuation such as "fixed-length" for the Stanza
"fixed - length") by snapping the phrase back to the exact sentence slice.
"""

import re
from typing import Optional, Tuple


def find_span(text: str, phrase: str) -> Optional[Tuple[int, int]]:
    """Locate ``phrase`` in ``text``, tolerating casing and spacing drift.

    Tries, in order: word-boundary exact match, word-boundary case-insensitive
    match, and space-insensitive match. Matches are anchored at word boundaries
    because gold phrases are token-aligned — a plain substring search would
    match short phrases inside words (e.g. "of" inside "softmax").
    Returns (start, end) character offsets into ``text``, or None.
    """
    phrase = phrase.strip()
    if not phrase or not text:
        return None

    escaped = re.escape(phrase)
    m = re.search(r"(?<!\w)" + escaped + r"(?!\w)", text)
    if m:
        return m.start(), m.end()

    m = re.search(r"(?<!\w)" + escaped + r"(?!\w)", text, flags=re.IGNORECASE)
    if m:
        return m.start(), m.end()

    # Space-insensitive: compare with all spaces removed, then map the match
    # back to the original offsets. Repairs "fixed-length" -> "fixed - length"
    # and tokenizer-mangled URLs like "https://github. com/...".
    compact_phrase = phrase.replace(" ", "").lower()
    if len(compact_phrase) < 4:  # too short; despaced matching would misfire
        return None
    chars, index_map = [], []
    for idx, ch in enumerate(text):
        if ch != " ":
            chars.append(ch.lower())
            index_map.append(idx)
    k = "".join(chars).find(compact_phrase)
    if k < 0:
        return None
    start = index_map[k]
    end = index_map[k + len(compact_phrase) - 1] + 1
    return start, end


def snap_phrase(text: str, phrase: str) -> Optional[str]:
    """Return the exact slice of ``text`` that matches ``phrase``, or None."""
    span = find_span(text, phrase)
    if span is None:
        return None
    return text[span[0]:span[1]]
