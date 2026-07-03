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

    Role: align the info-unit label of each selected contribution sentence to
    the fixed 12-unit NCG inventory, applying the normalization rules. Because
    the unit is fixed per SENTENCE (before triple extraction), every triple
    later drawn from that sentence inherits the same unit. The sentence text is
    never touched.

    The process is mostly deterministic (the rule table in ``normalize_unit``);
    the LLM is used only as a fallback for borderline labels the rules can't map.
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
"sentence" it was assigned to, for context. For each item, return the single
best-matching info unit token from the inventory. Return ONLY a JSON array of
objects: [{{"info_unit": "<TOKEN>"}}], one per input item, in the same order."""
        super().__init__(client, model_name, system_prompt)

    def process(self, selections: List[Dict]) -> List[Dict]:
        """Align the ``info_unit`` of every selected sentence to the inventory.

        Args:
            selections: [{"line": int, "info_unit": <raw label>, "text": str}, ...]

        Returns:
            the same selections with ``info_unit`` replaced by a canonical
            INFO_UNITS token; selections that cannot be mapped are dropped.
        """
        if not selections:
            return []

        aligned: List[Dict] = []
        unresolved: List[int] = []  # indices needing the LLM fallback

        for i, sel in enumerate(selections):
            canonical = normalize_unit(sel.get("info_unit", ""))
            if canonical is None:
                unresolved.append(i)
            new_sel = dict(sel)
            new_sel["info_unit"] = canonical  # may be None; filled by fallback
            aligned.append(new_sel)

        if unresolved:
            self._resolve_with_llm(selections, aligned, unresolved)

        result = [s for s in aligned if s.get("info_unit") in INFO_UNITS]
        dropped = len(aligned) - len(result)
        logger.info(
            f"SAA aligned {len(result)} contribution sentence(s)"
            + (f" (dropped {dropped} unmappable)" if dropped else "")
        )
        return self._enforce_solution_exclusivity(result)

    @staticmethod
    def _enforce_solution_exclusivity(selections: List[Dict]) -> List[Dict]:
        """Gold annotates a paper's solution as EITHER a MODEL or an APPROACH,
        almost never both (33 model vs 18 approach files across the 50 trial
        papers — at most one overlap). Predicting both guarantees one wholly
        spurious info-unit file and splits the real solution's triples across
        two files. If both appear, relabel the minority family into the
        dominant one (ties go to MODEL, the corpus prior)."""
        model_n = sum(1 for s in selections if s.get("info_unit") == "MODEL")
        approach_n = sum(1 for s in selections if s.get("info_unit") == "APPROACH")
        if not (model_n and approach_n):
            return selections
        winner = "MODEL" if model_n >= approach_n else "APPROACH"
        loser = "APPROACH" if winner == "MODEL" else "MODEL"
        logger.info(f"SAA: both MODEL ({model_n}) and APPROACH ({approach_n}) present; "
                    f"relabeling {loser} -> {winner}.")
        for s in selections:
            if s.get("info_unit") == loser:
                s["info_unit"] = winner
        return selections

    def _resolve_with_llm(self, selections: List[Dict], aligned: List[Dict],
                          unresolved: List[int]) -> None:
        payload = [
            {
                "guess": selections[i].get("info_unit", ""),
                "sentence": str(selections[i].get("text", ""))[:300],
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
