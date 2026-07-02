"""
Schema Alignment Agent (SAA) for KARMA Mini — NCG info-unit alignment.
"""

import json
import logging
from typing import List, Dict

from karma_mini.core.base_agent import BaseAgent
from karma_mini.core.data_structures import INFO_UNITS, normalize_unit

logger = logging.getLogger(__name__)


class SchemaAlignmentAgent(BaseAgent):
    """
    Agent 2: Schema Alignment Agent (SAA)

    Role: align each triple's ``info_unit`` to the fixed 12-unit NCG inventory,
    applying the normalization rules. The subject / predicate / object text is
    left untouched, only the info_unit label is standardized.

    The process is mostly deterministic (a rule table in ``normalize_unit``). A LLM
    is used only as a fallback for the borderline labels the rules can't map.
    """

    def __init__(self, client, model_name: str):
        units = ", ".join(INFO_UNITS)
        system_prompt = f"""You align messy information-unit labels to a fixed inventory for the NCG task.

The ONLY valid information units are: {units}.

Normalization rules:
  - "method" or "application" -> APPROACH
  - "system" or "architecture" -> MODEL
  - EXPERIMENTALSETUP only when hardware is mentioned, otherwise HYPERPARAMETERS

You will receive a JSON array of objects, each with a "guess" label and the
"subject"/"predicate"/"object" of the triple for context. For each item, return
the single best-matching info unit token from the inventory. Do NOT change any
text. Return ONLY a JSON array of objects: [{{"info_unit": "<TOKEN>"}}], one per
input item, in the same order."""
        super().__init__(client, model_name, system_prompt)

    def process(self, triples: List[Dict]) -> List[Dict]:
        """Align the ``info_unit`` of every triple to the 12-unit inventory.

        Returns a new list of triples (text fields unchanged) whose ``info_unit``
        is a canonical INFO_UNITS token. Triples that cannot be mapped at all are
        dropped.
        """
        if not triples:
            return []

        aligned: List[Dict] = []
        unresolved: List[int] = []  # indices needing the LLM fallback

        for i, t in enumerate(triples):
            canonical = normalize_unit(t.get("info_unit", ""))
            if canonical is None:
                unresolved.append(i)
            new_t = dict(t)
            new_t["info_unit"] = canonical  # may be None; filled by fallback
            aligned.append(new_t)

        if unresolved:
            self._resolve_with_llm(triples, aligned, unresolved)

        # Drop anything still unmapped.
        result = [t for t in aligned if t.get("info_unit") in INFO_UNITS]
        dropped = len(aligned) - len(result)
        logger.info(
            f"SAA aligned {len(result)} triples to the NCG inventory"
            + (f" (dropped {dropped} unmappable)" if dropped else "")
        )
        return result

    def _resolve_with_llm(self, triples: List[Dict], aligned: List[Dict],
                          unresolved: List[int]) -> None:
        payload = [
            {
                "guess": triples[i].get("info_unit", ""),
                "subject": triples[i].get("subject", ""),
                "predicate": triples[i].get("predicate", ""),
                "object": triples[i].get("object", ""),
            }
            for i in unresolved
        ]

        prompt = f"""Align each of these to one info unit token from the inventory.

{json.dumps(payload, indent=2, ensure_ascii=False)}

Return ONLY a JSON array of {{"info_unit": "<TOKEN>"}} in the same order."""

        response_text = self._make_llm_call(prompt, temperature=0.0)
        parsed = self._parse_json_response(response_text)

        for slot, item in zip(unresolved, parsed):
            if isinstance(item, dict):
                canonical = normalize_unit(item.get("info_unit", ""))
                aligned[slot]["info_unit"] = canonical
