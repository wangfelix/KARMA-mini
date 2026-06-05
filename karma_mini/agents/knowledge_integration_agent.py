"""
Knowledge Integration Agent (KIA) for KARMA Mini.
"""

import logging
from typing import List, Dict
import json
from collections import defaultdict

from karma_mini.core.base_agent import BaseAgent
from karma_mini.core.data_structures import KnowledgeTriple, KGEntity, KnowledgeGraph

logger = logging.getLogger(__name__)

class KnowledgeIntegrationAgent(BaseAgent):
    """
    Agent 3: Knowledge Integration Agent (KIA)

    Role: Takes all aligned triples globally, merges duplicates, groups by
    entity pairs, detects conflicts, and uses LLM-based reasoning to resolve
    those conflicts before adding them to the final Knowledge Graph.
    """

    def __init__(self, client, model_name: str):
        system_prompt = """You are a Knowledge Integration Agent (KIA) for a biomedical knowledge graph.
Your task is to resolve conflicting relationships between two entities based on evidence from different sources.

You will be given:
1. Entity A (Head)
2. Entity B (Tail)
3. A list of conflicting relations and the evidence sentences that produced them.

GUIDELINES for Resolution:
1. CONTEXTUAL SPLIT: If both are true but in different contexts (e.g., high dose vs low dose, or mouse vs human), keep BOTH relationships but append a brief context note to the relation type (e.g., "ACTIVATES (in vitro)").
2. GENERALIZATION: If the specific interactions contradict but both indicate interaction, generalize them to a safer relation like "REGULATES" or "INTERACTS_WITH".
3. OVERRULE: If one evidence is clearly a strong finding and the other is a weak hypothesis or negated, pick the strong one.

CONFIDENCE SCORING (0.0 to 1.0):
- 0.9 - 1.0: Very strong, explicit finding supported by clear, uncontradicted evidence.
- 0.7 - 0.8: Strong finding but generalized from conflicting specifics, or highly contextual.
- 0.5 - 0.6: Weak, implied relationship, or significant ambiguity remaining after resolution.

OUTPUT FORMAT:
Return ONLY a valid JSON array of the resolved relationships. Do not include markdown or explanations.
[
  {
    "head": "Entity A",
    "relation": "RESOLVED_RELATION",
    "tail": "Entity B",
    "confidence": 0.9,
    "resolution_note": "Brief explanation of why this was chosen."
  }
]"""
        super().__init__(client, model_name, system_prompt)

    def process(self, aligned_triples: List[Dict]) -> KnowledgeGraph:
        """
        Processes aligned triples to build the final Knowledge Graph.

        Args:
            aligned_triples: List of dictionaries representing aligned triples.

        Returns:
            A KnowledgeGraph object ready for export.
        """
        kg = KnowledgeGraph()

        # Group triples by head and tail (case-insensitive for basic merging)
        grouped_triples = defaultdict(list)

        for t in aligned_triples:
            # We use lowercase keys for grouping to merge "Aspirin" and "aspirin"
            head_key = t.get("head", "").strip().lower()
            tail_key = t.get("tail", "").strip().lower()
            if head_key and tail_key:
                pair_key = (head_key, tail_key)
                grouped_triples[pair_key].append(t)

        logger.info(f"KIA grouped {len(aligned_triples)} triples into {len(grouped_triples)} unique entity pairs.")

        for (head_key, tail_key), triples in grouped_triples.items():
            # Check if there are multiple DIFFERENT relations for this pair
            relations = list(set([t.get("relation") for t in triples]))

            # Create standard entity objects (using the first occurrence's casing and type)
            head_ent = KGEntity(
                entity_id=head_key,
                name=triples[0].get("head"),
                entity_type=triples[0].get("head_type", "OTHER")
            )
            tail_ent = KGEntity(
                entity_id=tail_key,
                name=triples[0].get("tail"),
                entity_type=triples[0].get("tail_type", "OTHER")
            )

            kg.add_entity(head_ent)
            kg.add_entity(tail_ent)

            if len(relations) == 1:
                # No conflict! Merge them.
                # Confidence could be based on the number of sources (e.g., 1 source = 0.5, 2 = 0.7, 3+ = 0.9)
                confidence = min(0.5 + (len(triples) - 1) * 0.2, 0.95)
                sources = ", ".join(list(set([t.get("source_id") for t in triples])))

                triple_obj = KnowledgeTriple(
                    head=head_ent.entity_id,
                    relation=relations[0],
                    tail=tail_ent.entity_id,
                    confidence=confidence,
                    source=sources
                )
                kg.add_triple(triple_obj)
            else:
                # Conflict detected! Ask the LLM to resolve it.
                logger.info(f"KIA resolving conflict for {head_ent.name} -> {tail_ent.name} ({relations})")
                resolved_triples = self._resolve_conflict(head_ent.name, tail_ent.name, triples)

                for rt in resolved_triples:
                    sources = ", ".join(list(set([t.get("source_id") for t in triples])))
                    triple_obj = KnowledgeTriple(
                        head=head_ent.entity_id,
                        relation=rt.get("relation", "INTERACTS_WITH"),
                        tail=tail_ent.entity_id,
                        confidence=float(rt.get("confidence", 0.5)),
                        source=f"Resolved from: {sources}"
                    )
                    kg.add_triple(triple_obj)

        return kg

    def _resolve_conflict(self, head_name: str, tail_name: str, conflicting_triples: List[Dict]) -> List[Dict]:
        """Ask LLM to resolve a conflict between multiple triples for the same entity pair."""

        evidence_list = []
        for t in conflicting_triples:
            evidence_list.append({
                "relation": t.get("relation"),
                "evidence": t.get("evidence"),
                "source": t.get("source_id")
            })

        prompt = f"""Resolve the relationship conflict between '{head_name}' and '{tail_name}'.

CONFLICTING EVIDENCE:
{json.dumps(evidence_list, indent=2)}

Determine the final relationship(s) and return ONLY the JSON array."""

        response_text = self._make_llm_call(prompt, temperature=0.1)
        resolved = self._parse_json_response(response_text)

        # Ensure it returns something valid even if it fails
        if not resolved:
            return [{"head": head_name, "relation": "INTERACTS_WITH", "tail": tail_name, "confidence": 0.4}]
        return resolved
