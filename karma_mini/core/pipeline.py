"""
KARMA Mini pipeline for the SemEval-2021 NLPContributionGraph (NCG) task.

Each paper is processed independently and yields its own contribution graph
rooted at a single "Contribution" node. There is no cross-paper merging and no
global knowledge graph.
"""

import logging

from karma_mini.agents import (
    ContributionSentenceAgent,
    SchemaAlignmentAgent,
    TripleExtractionAgent,
    KnowledgeIntegrationAgent,
)
from karma_mini.core.data_structures import is_structural_node
from karma_mini.loader import iter_papers
from karma_mini.writer import write_predictions

logger = logging.getLogger(__name__)


class KARMAPipeline:

    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name

        # Agent 1: select contribution sentences + one info unit per sentence.
        self.csa = ContributionSentenceAgent(client, model_name)
        # Agent 2: align the sentence-level info units to the 12-unit inventory.
        self.saa = SchemaAlignmentAgent(client, model_name)
        # Agent 3: extract phrases + triples, one sentence at a time.
        self.tea = TripleExtractionAgent(client, model_name)
        # Agent 4: assemble the per-paper rooted contribution graph.
        self.kia = KnowledgeIntegrationAgent(client, model_name)

    def process_papers(self, trial_root: str, out_root: str):
        """Run the pipeline over every paper.

        For each paper: loader -> CSA -> SAA -> TEA (per sentence) -> KIA ->
        writer. Prints per-paper counts. Writes predictions mirroring the gold
        folder layout.
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

            # 1. Contribution sentence selection (+ per-sentence info unit).
            selections = self.csa.process(sentences, section_hints=hints)

            # 2. Schema alignment of the sentence-level info units.
            aligned = self.saa.process(selections)

            # 3. Triple extraction, one sentence at a time; nodes extracted so
            #    far are offered back to the agent for cross-sentence chaining.
            triples = []
            known_nodes = []
            for sel in aligned:
                ts = self.tea.process(sel["line"], sel["text"], sel["info_unit"],
                                      known_nodes=known_nodes)
                for t in ts:
                    t["source_paper"] = paper_id
                    for phrase in (t["subject"], t["object"]):
                        if not is_structural_node(phrase) and phrase not in known_nodes:
                            known_nodes.append(phrase)
                triples.extend(ts)

            # 4. Knowledge integration (per-paper rooted graph assembly).
            graph = self.kia.process(triples, paper_id=paper_id)

            # 5. Write predictions in the gold-matching layout. sentences.txt is
            #    the CSA's selection (even lines that yielded no triples).
            pred_lines = sorted({s["line"] for s in aligned})
            stats = write_predictions(paper_id, graph, out_root,
                                      sentences=sent_map,
                                      contribution_lines=pred_lines)

            units = ", ".join(graph.group_by_info_unit().keys()) or "(none)"
            print(f"  sentences  : {stats['sentences']} selected")
            print(f"  triples    : {stats['triples']}")
            print(f"  info units : {stats['info_units']}  [{units}]")
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
