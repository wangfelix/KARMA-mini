"""
Information Extraction Agent (IEA) for KARMA Mini.
"""

import logging
from typing import List, Dict
from karma_mini.core.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class InformationExtractionAgent(BaseAgent):
    """
    Agent 1: Information Extraction Agent (IEA)
    
    Role: Reads raw text (e.g., an abstract) and extracts explicitly stated 
    biomedical entities and their relationships. It captures the "raw" 
    terminology used in the text without trying to standardize it.
    """

    def __init__(self, client, model_name: str):
        system_prompt = """You are an Information Extraction Agent (IEA) specializing in biomedical literature.
Your task is to read a scientific abstract and extract all explicit biomedical relationships into raw triples.

OBJECTIVE: 
Extract entities and the relationships between them exactly as they are described in the text. 
Do not attempt to standardize or categorize them into strict ontologies yet—capture the raw terms and phrases.

GUIDELINES:
1. Only extract relationships that are explicitly stated or clearly implied in the text.
2. The "head" and "tail" should be the specific entities mentioned (e.g., proteins, drugs, diseases, cells).
3. Provide a raw guess for the entity types based on context (e.g., "protein", "chemical", "disease", "symptom").
4. The "relation" should reflect the verbs or phrases used in the text (e.g., "reduces", "was found to inhibit", "correlates with").
5. Always provide the exact sentence from the text as "evidence".

OUTPUT FORMAT:
You must return ONLY a valid JSON array of objects. Do not include markdown formatting like ```json or any explanations.
[
  {
    "head": "raw entity name 1",
    "head_type_guess": "raw type guess",
    "relation": "raw relation phrase",
    "tail": "raw entity name 2",
    "tail_type_guess": "raw type guess",
    "evidence": "Exact sentence from the text proving this relationship."
  }
]

EXAMPLE:
Text: "We observed that aspirin significantly lowers the frequency of severe headaches in our patient cohort."
Output:
[
  {
    "head": "aspirin",
    "head_type_guess": "drug",
    "relation": "significantly lowers",
    "tail": "severe headaches",
    "tail_type_guess": "symptom",
    "evidence": "We observed that aspirin significantly lowers the frequency of severe headaches in our patient cohort."
  }
]"""
        super().__init__(client, model_name, system_prompt)

    def process(self, abstract_text: str, abstract_id: str = "unknown") -> List[Dict]:
        """
        Process a raw abstract to extract raw triples.

        Args:
            abstract_text: The plain text of the scientific abstract.
            abstract_id: An optional identifier for the source abstract.

        Returns:
            A list of dictionaries representing the raw extracted triples.
        """
        prompt = f"""Extract raw biomedical relationships from the following abstract.

Abstract ID: {abstract_id}
Text:
{abstract_text}

Return ONLY a JSON array of the raw relationships."""

        logger.info(f"IEA processing abstract {abstract_id}...")
        
        # We use a slightly higher temperature (e.g., 0.2) for extraction to allow 
        # the model to flexibly capture the exact phrases used in the text.
        response_text = self._make_llm_call(prompt, temperature=0.2)
        
        raw_triples = self._parse_json_response(response_text)
        
        # Inject the source ID into the extracted records for traceability
        valid_triples = []
        for triple in raw_triples:
            if isinstance(triple, dict) and "head" in triple and "tail" in triple:
                triple["source_id"] = abstract_id
                valid_triples.append(triple)
                
        logger.info(f"IEA extracted {len(valid_triples)} raw triples from abstract {abstract_id}.")
        return valid_triples
