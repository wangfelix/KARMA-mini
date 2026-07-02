"""
KARMA Mini pipeline for the SemEval-2021 NLPContributionGraph (NCG) task.

Each paper is processed independently and yields its own contribution graph
rooted at a single "Contribution" node. There is no cross-paper merging and no
global knowledge graph.
"""

import logging

from karma_mini.agents import (
    InformationExtractionAgent,
    SchemaAlignmentAgent,
    KnowledgeIntegrationAgent,
)
from karma_mini.loader import iter_papers
from karma_mini.writer import write_predictions

logger = logging.getLogger(__name__)


class KARMAPipeline:

    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name

        # Agent 1: contribution extraction over a whole paper.
        self.iea = InformationExtractionAgent(client, model_name)
        # Agent 2: align each triple's info_unit to the 12-unit inventory.
        self.saa = SchemaAlignmentAgent(client, model_name)
        # Agent 3: assemble the per-paper rooted contribution graph.
        self.kia = KnowledgeIntegrationAgent(client, model_name)

    def process_papers(self, trial_root: str, out_root: str):
        """Run the pipeline over every paper.

        For each paper: loader -> IEA -> SAA -> KIA -> writer. Prints per-paper
        triple counts. Writes predictions mirroring the gold folder layout.
        """
        papers = list(iter_papers(trial_root))
        if not papers:
            logger.error(f"No papers found under {trial_root}")
            return

        print(f"\nFound {len(papers)} paper(s) under {trial_root}\n")

        totals = {"papers": 0, "triples": 0, "info_units": 0}

        for paper in papers:
            paper_id = paper["paper_id"]
            sentences = paper["sentences"]
            hints = paper["section_hints"]
            sent_map = {ln: txt for ln, txt in sentences}

            print("=" * 60)
            print(f" {paper_id} ".center(60, "="))
            print("=" * 60)

            # 1. Information Extraction (contribution triples over the paper).
            raw = self.iea.process(sentences, section_hints=hints)
            for t in raw:
                t["source_paper"] = paper_id

            # 2. Schema Alignment (info-unit normalization).
            aligned = self.saa.process(raw)

            # 3. Knowledge Integration (per-paper rooted graph assembly).
            graph = self.kia.process(aligned, paper_id=paper_id)

            # 4. Write predictions in the gold-matching layout.
            stats = write_predictions(paper_id, graph, out_root, sentences=sent_map)

            units = ", ".join(graph.group_by_info_unit().keys()) or "(none)"
            print(f"  triples    : {stats['triples']}")
            print(f"  info units : {stats['info_units']}  [{units}]")
            print(f"  sentences  : {stats['sentences']}")
            print(f"  entities   : {stats['entities']}")
            print(f"  -> {out_root}/{paper_id}\n")

            totals["papers"] += 1
            totals["triples"] += stats["triples"]
            totals["info_units"] += stats["info_units"]

        print("=" * 60)
        print(" SUMMARY ".center(60, "="))
        print("=" * 60)
        print(f"Papers processed : {totals['papers']}")
        print(f"Total triples    : {totals['triples']}")
        print(f"Predictions in   : {out_root}\n")
