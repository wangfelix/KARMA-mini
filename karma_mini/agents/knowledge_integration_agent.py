"""
Knowledge Integration Agent (KIA) for KARMA Mini — per-paper graph assembly.
"""

import logging
from typing import List, Dict

from karma_mini.core.base_agent import BaseAgent
from karma_mini.core.data_structures import (
    KnowledgeTriple,
    KnowledgeGraph,
    IU_SPEC,
    INFO_UNITS,
    MANDATORY_UNITS,
    MODEL_OR_APPROACH,
    ROOT,
)

logger = logging.getLogger(__name__)


class KnowledgeIntegrationAgent(BaseAgent):
    """
    Agent 4: Knowledge Integration Agent (KIA)

    Role: assemble the per-paper rooted contribution graph. It adds the
    structural ``(Contribution||has||<InfoUnit>)`` edges for every info
    unit present, keeps the ``(term||predicate||term)`` edges, merges duplicate
    phrase nodes (identical strings collapse to one node, creating a DAG),
    de-duplicates identical triples, and groups the result by info unit.
    """

    def __init__(self, client=None, model_name: str = ""):
        super().__init__(client, model_name, system_prompt="")

    def process(self, aligned_triples: List[Dict], paper_id: str = "") -> KnowledgeGraph:
        kg = KnowledgeGraph(paper_id=paper_id)

        # Bucket the canonicalized term-edges per info unit, preserving order
        per_unit: Dict[str, List[KnowledgeTriple]] = {u: [] for u in INFO_UNITS}

        for t in aligned_triples:
            unit = t.get("info_unit")
            if unit not in IU_SPEC:
                continue
            spec = IU_SPEC[unit]
            edge = self._canonicalize(t, unit, spec)
            if edge is not None:
                per_unit[unit].append(edge)

        # Assemble in canonical INFO_UNITS order: structural edge first,
        # then the term edges of that unit.
        for unit in INFO_UNITS:
            edges = per_unit[unit]
            if not edges:
                continue
            spec = IU_SPEC[unit]
            if not spec.direct:
                kg.add_triple(KnowledgeTriple(
                    subject=ROOT,
                    predicate="has",
                    object=spec.node_label,
                    info_unit=unit,
                    source_line=-1,
                    source_paper=paper_id,
                    evidence="",
                ))
            for edge in edges:
                kg.add_triple(edge)

        self._report_mandatory(kg, paper_id)
        return kg

    def _canonicalize(self, t: Dict, unit: str, spec) -> KnowledgeTriple:
        """Turn a raw aligned triple into a canonical KnowledgeTriple.

        Snaps the info-unit node label to its canonical casing, and rewrites the
        root edge of the two "direct" units (RESEARCHPROBLEM, CODE) so that the
        term attaches to ``Contribution`` with the correct predicate
        """
        subject = str(t.get("subject", "")).strip()
        predicate = str(t.get("predicate", "")).strip()
        obj = str(t.get("object", "")).strip()
        line = t.get("from_line", -1)
        try:
            line = int(line)
        except (TypeError, ValueError):
            line = -1
        evidence = str(t.get("evidence", ""))

        if spec.direct:
            # RESEARCHPROBLEM / CODE: (Contribution || <root_pred> || term)
            term = self._pick_term(subject, obj, spec)
            return KnowledgeTriple(
                subject=ROOT,
                predicate=spec.root_pred,
                object=term,
                info_unit=unit,
                source_line=line,
                source_paper=t.get("source_paper", ""),
                evidence=evidence,
            )

        # Normal unit: keep text verbatim, only snap node-label casing.
        subject = self._snap_label(subject, spec.node_label)
        obj = self._snap_label(obj, spec.node_label)
        return KnowledgeTriple(
            subject=subject,
            predicate=predicate,
            object=obj,
            info_unit=unit,
            source_line=line,
            source_paper=t.get("source_paper", ""),
            evidence=evidence,
        )

    @staticmethod
    def _pick_term(subject: str, obj: str, spec) -> str:
        """Choose the phrase that is the actual term for a direct-unit edge."""
        structural = {ROOT.lower(), spec.node_label.lower(), "research problem"}
        # Prefer the object; fall back to subject if the object is structural.
        if obj and obj.lower() not in structural:
            return obj
        if subject and subject.lower() not in structural:
            return subject
        return obj or subject

    @staticmethod
    def _snap_label(text: str, node_label: str) -> str:
        """Normalize casing of an info-unit node label so it merges with the backbone."""
        if text.strip().lower() == node_label.lower():
            return node_label
        return text

    def _report_mandatory(self, kg: KnowledgeGraph, paper_id: str) -> None:
        present = set(kg.group_by_info_unit().keys())
        missing = [u for u in MANDATORY_UNITS if u not in present]
        if not (present & set(MODEL_OR_APPROACH)):
            missing.append("MODEL/APPROACH")
        if missing:
            logger.warning(f"[{paper_id}] missing mandatory info unit(s): {', '.join(missing)}")
