"""
Triple Extraction Agent (TEA) for KARMA Mini — per-sentence phrase/triple extraction.
"""

import logging
from typing import List, Dict, Optional

from karma_mini.core.base_agent import BaseAgent
from karma_mini.core.spans import snap_phrase
from karma_mini.core.data_structures import (
    IU_SPEC,
    ROOT,
    STRUCTURAL_PREDICATES,
)

logger = logging.getLogger(__name__)

# Lowercased structural node names (root + info-unit labels) that are allowed
# as triple endpoints without being spans of the current sentence.
_STRUCTURAL_LOWER = {ROOT.lower()} | {s.node_label.lower() for s in IU_SPEC.values()}

# Cap on how many existing graph nodes are offered for chaining.
MAX_KNOWN_NODES = 40


class TripleExtractionAgent(BaseAgent):
    """
    Agent 3: Triple Extraction Agent (TEA)

    Role: given ONE contribution sentence and its (already aligned) info unit,
    extract the scientific-term and predicate phrases and wire them into
    (subject, predicate, object) triples. Working one sentence at a time keeps
    every phrase a verbatim span of that sentence; a deterministic
    snap-to-span pass repairs small casing/spacing drift, and triples whose
    terms cannot be located (and are not known graph nodes) are dropped.
    """

    def __init__(self, client, model_name: str):
        system_prompt = """You extract knowledge-graph triples from ONE contribution sentence of a scholarly NLP paper (SemEval-2021 NLPContributionGraph task).

The sentence is given exactly as produced by the Stanza tokenizer: punctuation is spaced out (e.g. "RNN Encoder - Decoder", "two recurrent neural networks ( RNN )", "phrase - based SMT").

RULES:
  - subject, predicate and object MUST be copied VERBATIM as contiguous spans of the sentence (same spacing as shown). Never paraphrase, re-tokenize, or invent words.
  - predicates are FREE TEXT from the sentence wording (e.g. "consists of", "act as", "improves the performance", "built using"). Do not map them to any vocabulary.
  - extract only what the sentence actually states; 1-6 triples is typical.
  - chain triples: the object of one triple can be the subject of a deeper one.

STRUCTURE by info unit (the info unit is given to you):
  - RESEARCHPROBLEM: emit ("Contribution", "has research problem", <problem phrase>) — one per distinct problem/task name in the sentence.
  - CODE: emit ("Contribution", "Code", <the URL exactly as written>).
  - Any other unit with node label L (e.g. MODEL -> "Model", RESULTS -> "Results"): start the chain at the label — (L, "has", <top term>) or (L, <sentence verb>, <term>) — then chain deeper (<term>, <predicate>, <term>).
  - If EXISTING GRAPH NODES are provided, you may use one as a subject to attach this sentence's information to the graph built so far.

WORKED EXAMPLES:
Sentence (MODEL): "The proposed neural network architecture , which we will refer to as an RNN Encoder - Decoder , consists of two recurrent neural networks ( RNN ) that act as an encoder and a decoder pair ."
[
  {"subject":"Model","predicate":"has","object":"neural network architecture"},
  {"subject":"neural network architecture","predicate":"refer to as","object":"RNN Encoder - Decoder"},
  {"subject":"neural network architecture","predicate":"consists of","object":"two recurrent neural networks ( RNN )"},
  {"subject":"two recurrent neural networks ( RNN )","predicate":"act as","object":"encoder"},
  {"subject":"two recurrent neural networks ( RNN )","predicate":"act as","object":"decoder"}
]
Sentence (RESEARCHPROBLEM): "Learning Phrase Representations using RNN Encoder - Decoder for Statistical Machine Translation"
[
  {"subject":"Contribution","predicate":"has research problem","object":"Statistical Machine Translation"}
]
Sentence (RESULTS): "As expected , adding features computed by neural networks consistently improves the performance over the baseline performance ."
[
  {"subject":"Results","predicate":"improves the performance","object":"adding features"},
  {"subject":"adding features","predicate":"computed by","object":"neural networks"},
  {"subject":"Results","predicate":"improves the performance","object":"over the baseline performance"}
]

OUTPUT FORMAT:
Return ONLY a valid JSON array of {"subject": ..., "predicate": ..., "object": ...} objects (no markdown, no prose)."""
        super().__init__(client, model_name, system_prompt)

    def process(self, line_no: int, text: str, info_unit: str,
                known_nodes: Optional[List[str]] = None) -> List[Dict]:
        """Extract triples from one contribution sentence.

        Args:
            line_no:     1-indexed Stanza line number of the sentence.
            text:        the exact Stanza sentence.
            info_unit:   canonical INFO_UNITS token (already aligned by the SAA).
            known_nodes: phrase nodes already in this paper's graph, offered to
                         the model as chaining targets for cross-sentence links.

        Returns:
            list of validated triple dicts with info_unit / from_line / evidence
            injected from the known inputs (never trusted from the LLM).
        """
        spec = IU_SPEC.get(info_unit)
        if spec is None or not text.strip():
            return []

        known = [n for n in (known_nodes or []) if n][:MAX_KNOWN_NODES]

        parts = [f'Info unit: {info_unit} (node label: "{spec.node_label}")',
                 f"Line {line_no}: {text}"]
        if known:
            parts.append("EXISTING GRAPH NODES (usable as subjects for chaining):")
            parts.extend(f"  - {n}" for n in known)
        parts.append("Return ONLY the JSON array of triples.")
        prompt = "\n".join(parts)

        response_text = self._make_llm_call(prompt, temperature=0.1)
        raw = self._parse_json_response(response_text)

        triples: List[Dict] = []
        for t in raw:
            if not isinstance(t, dict):
                continue
            subject = str(t.get("subject", "")).strip()
            predicate = str(t.get("predicate", "")).strip()
            obj = str(t.get("object", "")).strip()
            if not (subject and predicate and obj):
                continue

            subject = self._resolve_endpoint(subject, text, known)
            obj = self._resolve_endpoint(obj, text, known)
            if subject is None or obj is None:
                logger.debug(f"TEA dropped non-verbatim triple on line {line_no}: {t}")
                continue

            if predicate not in STRUCTURAL_PREDICATES:
                predicate = snap_phrase(text, predicate) or predicate

            triples.append({
                "info_unit": info_unit,
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "from_line": line_no,
                "evidence": text,
            })

        logger.info(f"TEA line {line_no} [{info_unit}]: {len(triples)} triple(s).")
        return triples

    @staticmethod
    def _resolve_endpoint(phrase: str, text: str,
                          known_nodes: List[str]) -> Optional[str]:
        """Snap an endpoint to a verbatim sentence span, a structural node, or a
        known graph node. Returns None if it is none of these (triple dropped)."""
        if phrase.lower() in _STRUCTURAL_LOWER:
            return phrase  # canonical casing is applied later by the KIA
        snapped = snap_phrase(text, phrase)
        if snapped is not None:
            return snapped
        for node in known_nodes:
            if node == phrase or node.lower() == phrase.lower():
                return node
        return None
