"""
Chunking: Stanza sentences -> overlapping windows.

Each chunk is a sliding window of consecutive Stanza sentences with 50%
overlap and carrying its paper id, line range, and nearest section header so
answers can cite exact locations.
"""

import logging
from typing import Dict, List

from karma_mini.loader import iter_papers

logger = logging.getLogger(__name__)

# sentences per chunk
WINDOW = 4
# sentences between window starts (50% overlap)
STRIDE = 2
# drop header-only lines and almost empty windows
MIN_TOKENS = 10


def build_chunks(trial_root: str) -> List[Dict]:
    """Chunk every paper under ``trial_root`` into overlapping windows.

    Returns a list of
        {"id", "paper_id", "start_line", "end_line", "section", "text"}
    """
    chunks: List[Dict] = []
    papers = 0

    for paper in iter_papers(trial_root):
        papers += 1
        sents = paper["sentences"]
        hints = paper["section_hints"]
        start = 0

        while start < len(sents):
            window = sents[start:start + WINDOW]
            text = " ".join(t for _, t in window).strip()

            if len(text.split()) >= MIN_TOKENS:
                chunks.append({
                    "id": f"{paper['paper_id']}:{window[0][0]}-{window[-1][0]}",
                    "paper_id": paper["paper_id"],
                    "start_line": window[0][0],
                    "end_line": window[-1][0],
                    "section": hints.get(window[0][0], ""),
                    "text": text,
                })

            if start + WINDOW >= len(sents):
                break

            start += STRIDE

    logger.info(f"Chunked {papers} paper(s) into {len(chunks)} chunks "
                f"(window={WINDOW}, stride={STRIDE}).")
    return chunks
