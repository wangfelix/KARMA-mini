"""
Load KARMA Mini triple predictions into a Neo4j graph database.

Reads:  data/ncg/predictions/<task>/<n>/triples/<info_unit>.txt
        (each line: "(subject||predicate||object)")

Usage:
    pip install neo4j

    # via CLI args
    python load_triples_to_neo4j.py \
        --predictions data/ncg/predictions \
        --uri bolt://localhost:7687 \
        --user neo4j --password yourpassword

    # or via .env (NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD)
    python load_triples_to_neo4j.py --predictions data/ncg/predictions

    # wipe everything first
    python load_triples_to_neo4j.py --predictions data/ncg/predictions --clear
"""

import argparse
import os
import re
from pathlib import Path

from neo4j import GraphDatabase

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def sanitize_rel_type(predicate: str) -> str:
    """Turn free-text predicate into a legal Cypher relationship type."""
    rel = predicate.strip().upper()
    rel = re.sub(r"[^A-Z0-9]+", "_", rel).strip("_")
    if not rel:
        rel = "RELATED_TO"
    if rel[0].isdigit():
        rel = "R_" + rel
    return rel


def parse_triple_line(line: str):
    """Parse '(subject||predicate||object)' -> (subject, predicate, object)."""
    line = line.strip()
    if not line:
        return None
    if line.startswith("(") and line.endswith(")"):
        line = line[1:-1]
    parts = [p.strip() for p in line.split("||")]
    if len(parts) != 3 or not all(parts):
        return None
    return tuple(parts)


# These are KARMA Mini's structural backbone node names -- the intermediate
# nodes that stand in for an info-unit category (e.g. "Model", "Results")
# plus the root "Contribution" node. If merged globally by name they would
# become meaningless super-hubs connecting nearly every paper. Keep these
# scoped per-paper (paper_id stays a NODE property, as before). Every other
# entity name is real extracted content and gets merged globally across
# papers so that the same real-world concept (e.g. "LSTM") becomes one
# shared node reachable from every paper that mentions it.
STRUCTURAL_NODE_NAMES = {
    "contribution", "model", "approach", "dataset", "results",
    "baselines", "hyperparameters", "experimental setup", "tasks",
    "experiments", "ablation analysis", "code",
}


def is_structural(name: str) -> bool:
    return name.strip().lower() in STRUCTURAL_NODE_NAMES


class KarmaNeo4jLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_all(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def ensure_constraints(self):
        # NODE KEY constraints require Neo4j Enterprise Edition. Community
        # Edition only supports single-property uniqueness constraints, so we
        # use plain indexes instead -- they won't reject duplicates, but they
        # make the MERGE lookups fast, which is all we actually need here.
        # Two separate indexes: one for paper-scoped structural nodes
        # (paper_id + name), one for globally-shared content entities (name
        # only, since those nodes have no paper_id property at all).
        with self.driver.session() as session:
            session.run(
                "CREATE INDEX entity_paper_name IF NOT EXISTS "
                "FOR (e:Entity) ON (e.paper_id, e.name)"
            )
            session.run(
                "CREATE INDEX entity_name IF NOT EXISTS "
                "FOR (e:Entity) ON (e.name)"
            )

    def load_triple(self, paper_id, info_unit, subj, pred, obj):
        rel_type = sanitize_rel_type(pred)
        subj_pattern = (
            "{paper_id: $paper_id, name: $subj}" if is_structural(subj)
            else "{name: $subj}"
        )
        obj_pattern = (
            "{paper_id: $paper_id, name: $obj}" if is_structural(obj)
            else "{name: $obj}"
        )
        # paper_id lives on the RELATIONSHIP (always, regardless of whether
        # either endpoint is a shared or paper-scoped node) -- this is the
        # one reliable place to find "which paper does this edge belong to",
        # since a shared entity node itself no longer carries paper_id.
        query = f"""
        MERGE (p:Paper {{paper_id: $paper_id}})
        MERGE (s:Entity {subj_pattern})
        MERGE (o:Entity {obj_pattern})
        MERGE (s)-[r:{rel_type} {{paper_id: $paper_id}}]->(o)
          ON CREATE SET r.predicate_text = $pred, r.info_unit = $info_unit
        WITH p, s
        FOREACH (_ IN CASE WHEN s.name = 'Contribution' THEN [1] ELSE [] END |
            MERGE (p)-[:HAS_ROOT]->(s))
        """
        with self.driver.session() as session:
            session.run(
                query,
                paper_id=paper_id,
                subj=subj,
                obj=obj,
                pred=pred,
                info_unit=info_unit,
            )


def load_all_predictions(loader: KarmaNeo4jLoader, predictions_root: str):
    root = Path(predictions_root)
    n_papers, n_triples, n_skipped = 0, 0, 0

    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        for paper_dir in sorted(task_dir.iterdir()):
            if not paper_dir.is_dir():
                continue

            # Two known layouts:
            #   predictions/<task>/<n>/triples/*.txt          (our pipeline output)
            #   trial-data/<task>/<n>/info-units/triples/*.txt (gold annotations)
            triples_dir = paper_dir / "triples"
            if not triples_dir.is_dir():
                triples_dir = paper_dir / "info-units" / "triples"
            if not triples_dir.is_dir():
                continue

            paper_id = f"{task_dir.name}/{paper_dir.name}"
            n_papers += 1

            for iu_file in sorted(triples_dir.glob("*.txt")):
                info_unit = iu_file.stem
                for line in iu_file.read_text(encoding="utf-8").splitlines():
                    parsed = parse_triple_line(line)
                    if parsed is None:
                        if line.strip():
                            n_skipped += 1
                        continue
                    subj, pred, obj = parsed
                    loader.load_triple(paper_id, info_unit, subj, pred, obj)
                    n_triples += 1

    print(f"Loaded {n_triples} triples from {n_papers} papers "
          f"({n_skipped} malformed lines skipped).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True,
                     help="Path to data/ncg/predictions")
    ap.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    ap.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.getenv("NEO4J_PASSWORD"))
    ap.add_argument("--clear", action="store_true",
                     help="Wipe the whole database before loading")
    args = ap.parse_args()

    if not args.password:
        raise SystemExit(
            "Neo4j password not set. Pass --password or set NEO4J_PASSWORD in .env"
        )

    loader = KarmaNeo4jLoader(args.uri, args.user, args.password)
    try:
        if args.clear:
            print("Clearing existing graph...")
            loader.clear_all()
        loader.ensure_constraints()
        load_all_predictions(loader, args.predictions)
    finally:
        loader.close()


if __name__ == "__main__":
    main()