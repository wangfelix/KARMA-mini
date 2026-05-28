"""
Schema Alignment Agent (SAA) for KARMA Mini.
"""

import logging
from typing import List, Dict
from karma_mini.core.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class SchemaAlignmentAgent(BaseAgent):
    """
    Agent 2: Schema Alignment Agent (SAA)
    
    Role: Takes raw extracted triples (with messy, natural language relations 
    and entity types) and standardizes them into a strict predefined ontology.
    """

    def __init__(self, client, model_name: str):
        system_prompt = """You are a Schema Alignment Agent (SAA) for a biomedical knowledge graph.
Your task is to take raw, unstandardized biomedical relationships and map them to a strict, predefined vocabulary.

ALLOWED ENTITY CATEGORIES (Node Labels):
- DRUG: Pharmaceuticals, therapeutic compounds (e.g., aspirin, chemotherapy)
- DISEASE: Medical conditions, disorders, symptoms (e.g., cancer, headache)
- GENE: Genetic elements
- PROTEIN: Enzymes, receptors, antibodies (e.g., COX-2, HER2)
- CHEMICAL: Small molecules, metabolites, ions (e.g., PGE2)
- PATHWAY: Biological pathways
- ANATOMY: Organs, tissues, cells
- OTHER: Use only if absolutely none of the above fit

ALLOWED RELATION TYPES (Edge Types):
- TREATS: Drug/intervention cures or alleviates disease/symptom
- INHIBITS: Blocks, reduces activity, or suppresses
- ACTIVATES: Stimulates, enhances, or upregulates
- CAUSES: Triggers or leads to a condition
- ASSOCIATED_WITH: Statistical or observational link
- REGULATES: Controls expression or activity
- INCREASES: Raises levels or quantity
- DECREASES: Lowers levels or quantity
- INTERACTS_WITH: Direct binding, targeting, or physical connection

GUIDELINES:
1. You will be provided with a JSON array of raw triples.
2. For each triple, map the `head_type_guess` and `tail_type_guess` to the closest ALLOWED ENTITY CATEGORY.
3. Map the `relation` to the closest ALLOWED RELATION TYPE. 
   - E.g., "reduces the production of" -> DECREASES
   - E.g., "targets" -> INTERACTS_WITH
   - E.g., "alleviates" -> TREATS
4. Retain the exact `head`, `tail`, `evidence`, and `source_id` from the input. Do not change the actual entity names.
5. If a relation cannot be mapped sensibly, default to INTERACTS_WITH or ASSOCIATED_WITH.

OUTPUT FORMAT:
Return ONLY a valid JSON array of objects. Do not include markdown formatting or explanations.
[
  {
    "head": "exact input head",
    "head_type": "MAPPED_CATEGORY",
    "relation": "MAPPED_RELATION",
    "tail": "exact input tail",
    "tail_type": "MAPPED_CATEGORY",
    "evidence": "exact input evidence",
    "source_id": "exact input source_id"
  }
]"""
        super().__init__(client, model_name, system_prompt)

    def process(self, raw_triples: List[Dict]) -> List[Dict]:
        """
        Process a list of raw triples to align their schema.

        Args:
            raw_triples: List of dictionaries representing raw triples.

        Returns:
            A list of dictionaries representing the aligned triples.
        """
        if not raw_triples:
            return []
            
        import json
        
        prompt = f"""Align the following raw triples to the predefined schema.

RAW TRIPLES:
{json.dumps(raw_triples, indent=2)}

Return ONLY a JSON array of the aligned triples."""

        logger.info(f"SAA aligning {len(raw_triples)} raw triples...")
        
        # We use a very low temperature (0.0) for alignment to ensure strict 
        # adherence to the allowed schema categories.
        response_text = self._make_llm_call(prompt, temperature=0.0)
        
        aligned_triples = self._parse_json_response(response_text)
        
        valid_aligned = []
        for triple in aligned_triples:
            if isinstance(triple, dict) and "head" in triple and "tail" in triple and "relation" in triple:
                # Ensure fields exist
                triple["source_id"] = triple.get("source_id", "unknown")
                valid_aligned.append(triple)
                
        logger.info(f"SAA successfully aligned {len(valid_aligned)} triples.")
        return valid_aligned
