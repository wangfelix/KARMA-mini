"""
Trial-data loader for the NCG task.

Reads the Stanza-tokenized plaintext (``*-Stanza-out.txt``) for each paper.
"""

import os
import glob
import logging
from typing import Dict, List, Tuple, Optional, Iterator

logger = logging.getLogger(__name__)

# Lines in the Stanza file that contain section headers rather than prose.
_KNOWN_HEADERS = {
    "title", "abstract", "introduction", "related work", "background",
    "model", "models", "approach", "method", "methods", "methodology",
    "architecture", "experiments", "experimental setup", "experiment",
    "setup", "results", "evaluation", "analysis", "ablation",
    "ablation study", "discussion", "conclusion", "conclusions",
    "dataset", "datasets", "data", "training", "training details",
    "baselines", "implementation", "implementation details", "tasks",
}


def _stanza_path(folder: str) -> Optional[str]:
    """Return the *-Stanza-out.txt path inside a paper folder, if present."""
    matches = glob.glob(os.path.join(folder, "*-Stanza-out.txt"))
    return matches[0] if matches else None


def _is_header_line(text: str) -> bool:
    """Heuristic: short, capitalized-ish line that is not a full sentence."""
    stripped = text.strip()
    if not stripped:
        return False
    low = stripped.lower().rstrip(" .")
    if low in _KNOWN_HEADERS:
        return True
    tokens = stripped.split()

    if len(tokens) <= 6 and not stripped.endswith((".", "?", "!", ":", ",")):
        # avoid treating fragments with lowercase-only first word as headers
        if stripped[0].isupper() or low in _KNOWN_HEADERS:
            return True
    return False


def _build_section_hints(sentences: List[Tuple[int, str]]) -> Dict[int, str]:
    """Attach the nearest preceding header line to each sentence line number.

    Returns {line_no: header}.
    Sentences before any header get an empty hint.
    """
    hints: Dict[int, str] = {}
    current = ""
    for line_no, text in sentences:
        if _is_header_line(text):
            current = text.strip()
        hints[line_no] = current
    return hints


def load_paper(folder: str) -> Dict:
    """Load a single paper folder.

    Returns:
        {
          "paper_id": <relative folder path, e.g. "machine-translation/0">,
          "sentences": [(line_no:int, text:str), ...],   # from *-Stanza-out.txt
          "section_hints": {line_no: nearest_header_str},
        }
    """
    stanza = _stanza_path(folder)
    if stanza is None:
        raise FileNotFoundError(f"No *-Stanza-out.txt found in {folder}")

    sentences: List[Tuple[int, str]] = []
    with open(stanza, "r", encoding="utf-8") as f:
        for idx, raw in enumerate(f, start=1):
            text = raw.rstrip("\n")
            sentences.append((idx, text))

    # paper_id = the two trailing path components (<task>/<n>) when available,
    # else the basename. This is what the writer/scorer mirror.
    norm = os.path.normpath(folder)
    parts = norm.split(os.sep)
    paper_id = os.path.join(*parts[-2:]) if len(parts) >= 2 else parts[-1]

    return {
        "paper_id": paper_id,
        "folder": folder,
        "sentences": sentences,
        "section_hints": _build_section_hints(sentences),
    }


def iter_papers(root: str) -> Iterator[Dict]:
    """load_paper(...) for every paper folder under ``root``.

    Accepts either a root (``<task>/<n>/`` folders) or a single paper
    folder (one that directly contains a ``*-Stanza-out.txt`` file).
    """
    # Single paper folder?
    if _stanza_path(root) is not None:
        yield load_paper(root)
        return

    found = False
    for stanza in sorted(glob.glob(os.path.join(root, "*", "*", "*-Stanza-out.txt"))):
        found = True
        yield load_paper(os.path.dirname(stanza))

    if not found:
        # Fall back to a one-level layout (<n>/ directly under root).
        for stanza in sorted(glob.glob(os.path.join(root, "*", "*-Stanza-out.txt"))):
            yield load_paper(os.path.dirname(stanza))
