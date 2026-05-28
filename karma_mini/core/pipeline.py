import json
import logging
from typing import List, Dict

from karma_mini.agents import InformationExtractionAgent, SchemaAlignmentAgent

logger = logging.getLogger(__name__)

class KARMAPipeline:
    """
    Orchestrates the KARMA Mini 3-Agent workflow.
    """
    
    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name
        
        # Initialize agents
        self.iea = InformationExtractionAgent(client, model_name)
        self.saa = SchemaAlignmentAgent(client, model_name)
        # self.kia = KnowledgeIntegrationAgent(client, model_name)
        
    def process_abstracts(self, file_path: str):
        """
        Runs the pipeline on a JSON file containing abstracts.
        Currently only runs Agent 1 (IEA) and prints the results.
        """
        # 1. Load data
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                abstracts = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load abstracts from {file_path}: {e}")
            return

        logger.info(f"Loaded {len(abstracts)} abstracts for processing.")
        
        all_raw_triples = []
        
        # 2. Agent 1: Information Extraction (Run per abstract)
        print("\n" + "="*60)
        print(" AGENT 1: INFORMATION EXTRACTION ".center(60, "="))
        print("="*60)
        
        for abstract in abstracts:
            abs_id = abstract.get("id", "unknown_id")
            text = abstract.get("text", "")
            
            print(f"\nProcessing: {abs_id} - '{abstract.get('title', 'No Title')}'")
            
            raw_triples = self.iea.process(text, abstract_id=abs_id)
            all_raw_triples.extend(raw_triples)
            
            # Print the results to terminal
            for i, triple in enumerate(raw_triples, 1):
                print(f"  [{i}] {triple.get('head')} ({triple.get('head_type_guess')})")
                print(f"      -[ {triple.get('relation')} ]->")
                print(f"      {triple.get('tail')} ({triple.get('tail_type_guess')})")
                print(f"      Evidence: \"{triple.get('evidence')}\"\n")
                
        print(f"\nTotal raw triples extracted: {len(all_raw_triples)}")

        # 3. Agent 2: Schema Alignment (Run on all raw triples)
        print("\n" + "="*60)
        print(" AGENT 2: SCHEMA ALIGNMENT ".center(60, "="))
        print("="*60)
        
        # To avoid exceeding context windows, we can chunk the raw triples if there are too many.
        # But for 30 abstracts, chunking by ~10 triples should be safe and fast.
        all_aligned_triples = []
        chunk_size = 10
        for i in range(0, len(all_raw_triples), chunk_size):
            chunk = all_raw_triples[i:i + chunk_size]
            aligned_chunk = self.saa.process(chunk)
            all_aligned_triples.extend(aligned_chunk)
            
        print(f"\nTotal aligned triples: {len(all_aligned_triples)}")
        for i, triple in enumerate(all_aligned_triples, 1):
            print(f"  [{i}] {triple.get('head')} ({triple.get('head_type')})")
            print(f"      -[ {triple.get('relation')} ]->")
            print(f"      {triple.get('tail')} ({triple.get('tail_type')})")
            print(f"      Source: {triple.get('source_id')}\n")
