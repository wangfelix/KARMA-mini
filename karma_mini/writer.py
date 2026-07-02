"""
Prediction writer for the NCG task.

Serializes a per-paper :class:`KnowledgeGraph` into the directory layout
and file format the scorer consumes:

    <out_root>/<task>/<n>/
        triples/<iu>.txt   # one "(subject||predicate||object)" per line
        sentences.txt      # one contribution line number per line
        entities.txt       # "<line>\t<start>\t<end>\t<phrase>" per line
"""

import os
import logging
from typing import Dict, Optional, Tuple, List

from karma_mini.core.data_structures import (
    KnowledgeGraph,
    IU_SPEC,
    is_structural_node,
)

logger = logging.getLogger(__name__)


def _format_triple(subject: str, predicate: str, obj: str) -> str:
    return f"({subject}||{predicate}||{obj})"


def write_predictions(paper_id: str, graph: KnowledgeGraph, out_root: str,
                      sentences: Optional[Dict[int, str]] = None) -> Dict[str, int]:
    """Write all prediction files for one paper.

    Args:
        paper_id:  relative paper path, e.g. "machine-translation/0".
        graph:     the assembled per-paper KnowledgeGraph.
        out_root:  predictions root.
        sentences: {line_no: text} from the Stanza file, used to compute the
                   character offsets for entities.txt. If omitted, entities.txt
                   is written empty.

    Returns:
        {"triples": n, "info_units": n, "sentences": n, "entities": n}
    """
    out_dir = os.path.join(out_root, paper_id)
    triples_dir = os.path.join(out_dir, "triples")
    os.makedirs(triples_dir, exist_ok=True)

    grouped = graph.group_by_info_unit()

    # --- triples/<iu>.txt ---
    triple_count = 0
    for unit, triples in grouped.items():
        filename = IU_SPEC[unit].filename + ".txt"
        seen = set()
        lines: List[str] = []
        for t in triples:
            line = _format_triple(t.subject, t.predicate, t.object)
            if line not in seen:
                seen.add(line)
                lines.append(line)
        with open(os.path.join(triples_dir, filename), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        triple_count += len(lines)

    # --- sentences.txt ---
    contrib_lines = graph.contribution_lines()
    with open(os.path.join(out_dir, "sentences.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(str(n) for n in contrib_lines) + ("\n" if contrib_lines else ""))

    # --- entities.txt ---
    entities = _collect_entities(graph, sentences or {})
    with open(os.path.join(out_dir, "entities.txt"), "w", encoding="utf-8") as f:
        for line_no, start, end, phrase in entities:
            f.write(f"{line_no}\t{start}\t{end}\t{phrase}\n")

    return {
        "triples": triple_count,
        "info_units": len(grouped),
        "sentences": len(contrib_lines),
        "entities": len(entities),
    }


def _collect_entities(graph: KnowledgeGraph,
                      sentences: Dict[int, str]) -> List[Tuple[int, int, int, str]]:
    """Build deduplicated (line, start, end, phrase) tuples from phrase nodes.

    A phrase's char offset is found by locating it verbatim within its source
    Stanza line (predicted phrases are drawn from that text, so they string
    match). Structural nodes ("Contribution" and info-unit labels) are skipped,
    as are phrases that cannot be located in their line.
    """
    seen = set()
    out: List[Tuple[int, int, int, str]] = []

    for t in graph.phrase_nodes():
        line_no = t.source_line
        text = sentences.get(line_no)
        if not text:
            continue
        for phrase in (t.subject, t.object):
            if not phrase or is_structural_node(phrase):
                continue
            start = text.find(phrase)
            if start < 0:
                continue
            end = start + len(phrase)
            key = (line_no, start, end, phrase)
            if key not in seen:
                seen.add(key)
                out.append(key)

    out.sort(key=lambda e: (e[0], e[1], e[2]))
    return out
