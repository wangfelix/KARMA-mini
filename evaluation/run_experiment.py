"""Run Plain RAG and gold-GraphRAG over a frozen question split.

The JSONL output contains answers, retrieved evidence, traces, latency, and a
full experiment configuration. Existing matching output is resumed safely;
an incompatible output file is never silently mixed with a new experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

from graph_rag.qa_neo4j import get_schema_text, retrieve_question
from plain_rag import Embedder, load_index, retrieve

from .config import (
    FINAL_ANSWER_MODEL,
    FINAL_DENSE_WEIGHT,
    FINAL_RETRIEVAL_MODE,
    FINAL_TOP_K,
)
from .plain_retrieval import assert_open_corpus, format_retrieval_hits


SYSTEMS = ("plain_rag", "graph_rag_gold")
THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)

COMMON_ANSWER_SYSTEM_PROMPT = """You answer questions about a collection of NLP research papers using only the supplied evidence.

Rules:
- Do not use outside knowledge or fill gaps with assumptions.
- Cite factual claims with the supporting paper id in square brackets, for example [machine-translation/0].
- Treat each Graph evidence row as a projected fact: keys such as a.name, r.predicate_text, and b.name denote the subject, relation, and object. The r.paper_id or query_scope_paper_ids field supplies its paper provenance.
- Graph rows carry explicit direction: fact_subject and fact_object (or fact_subject_1/fact_object_1, fact_subject_2/fact_object_2 for a two-step path) give the true subject and object of each relation. Trust these over any other node column, and never state a relation in the reverse direction.
- A Graph row may project a path rather than a single fact. Keys numbered by hop, such as a.name, r1.predicate_text, mid.name, r2.predicate_text and b.name, describe a chain: subject, first relation, intermediate entity, second relation, endpoint. Read the whole chain. When a question states one relation and asks about a second, the answer is the endpoint of the matching chain, not the intermediate.
- A question about the problems a paper addresses can be answered from its title, abstract, motivation, introduction, or explicit research-problem facts; the literal words "research problem" need not occur.
- For list questions, include every item supported by the evidence.
- For count questions, give the count and enough detail to make the counted set auditable.
- If the evidence does not contain the answer, say exactly: The supplied evidence does not contain the answer.
- Be concise and answer the question directly."""

ABSTENTION_ANSWER = "The supplied evidence does not contain the answer."


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _experiment_config(args: argparse.Namespace, question_hash: str) -> dict[str, Any]:
    config = {
        "question_file_sha256": question_hash,
        "split": args.split,
        "systems": sorted(args.systems),
        "answer_model": args.model,
        "answer_prompt_version": "common-grounded-answer-v2",
        "answer_postprocessing": "strip-think-blocks-v1",
        "answer_temperature": 0.0,
        "graph_cypher_temperature": 0.0,
        "plain_rag": {
            "index": str(args.index.resolve()),
            "retrieval_mode": args.retrieval_mode,
            "dense_weight": args.dense_weight,
            "top_k": args.top_k,
            "scope_policy": "filter-explicit-paper-ids-v1",
        },
        "graph_rag": {
            "source_label": args.graph_source_label,
            "neo4j_uri": args.neo4j_uri,
            "neo4j_user": args.neo4j_user,
            "expected_paper_count": args.expected_graph_papers,
            "provenance_policy": "retain-cypher-paper-scope-v1",
            "cypher_postprocessing": "strip-think-fences-v1",
        },
    }
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config["experiment_id"] = hashlib.sha256(serialized.encode()).hexdigest()[:16]
    return config


def _select_questions(records: list[dict[str, Any]], split: str,
                      question_ids: list[str] | None, limit: int | None,
                      allow_draft: bool) -> list[dict[str, Any]]:
    selected = [record for record in records if split == "all" or record["split"] == split]
    if question_ids:
        requested = set(question_ids)
        selected = [record for record in selected if record["id"] in requested]
        missing = requested - {record["id"] for record in selected}
        if missing:
            raise KeyError(f"Question IDs not found in selected split: {sorted(missing)}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        selected = selected[:limit]
    if not selected:
        raise RuntimeError("No questions matched the requested selection")
    unverified = [
        record["id"] for record in selected
        if record.get("review_status") != "human_verified"
    ]
    if unverified and not allow_draft:
        raise RuntimeError(
            f"Refusing to run {len(unverified)} unverified questions. "
            "Review and mark them human_verified, or explicitly acknowledge "
            "the unverified benchmark with --allow-unverified-questions."
        )
    return selected


def _make_client(timeout: float) -> OpenAI:
    api_key = os.getenv("KIT_API_KEY")
    base_url = os.getenv("KIT_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("KIT_API_KEY and KIT_BASE_URL must be set")
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _generate_common_answer(client: OpenAI, model: str, question: str,
                            evidence: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    prompt = (
        f"QUESTION:\n{question}\n\n"
        "EVIDENCE:\n"
        + json.dumps(evidence, ensure_ascii=False, indent=2)
        + "\n\nANSWER:"
    )
    answer = ""
    raw_outputs = []
    reasoning_removed = False
    for attempt in range(1, 3):
        attempt_prompt = prompt
        if attempt == 2:
            attempt_prompt += (
                "\n\nSECOND-PASS CHECK:\n"
                "Your first response abstained even though retrieved evidence was "
                "available. Re-read every evidence item, including graph "
                "subject/relation/object fields and paper provenance. Answer when "
                "the requested information is supported; otherwise repeat the "
                "required abstention sentence."
            )
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": COMMON_ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": attempt_prompt},
            ],
        )
        raw_answer = (response.choices[0].message.content or "").strip()
        raw_outputs.append(raw_answer)
        answer = THINK_BLOCK_PATTERN.sub("", raw_answer).strip()
        reasoning_removed = reasoning_removed or answer != raw_answer
        if answer != ABSTENTION_ANSWER or not evidence:
            return answer, {
                "answer_attempts": attempt,
                "answer_reasoning_removed": reasoning_removed,
                "raw_answer_outputs": raw_outputs,
            }
    return answer, {
        "answer_attempts": 2,
        "answer_reasoning_removed": reasoning_removed,
        "raw_answer_outputs": raw_outputs,
    }


class PlainBackend:
    def __init__(self, client: OpenAI, model: str, index_path: Path,
                 retrieval_mode: str, dense_weight: float, top_k: int):
        self.client = client
        self.model = model
        self.index = load_index(str(index_path))
        self.retrieval_mode = retrieval_mode
        self.dense_weight = dense_weight
        self.top_k = top_k
        self.embedder = None
        if retrieval_mode in {"dense", "hybrid"}:
            self.embedder = Embedder(
                client,
                model=self.index.meta["embed_model"],
            )

    def run(self, question: str) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        # Open-corpus retrieval: every question is ranked against all chunks.
        # Locating the relevant papers is part of the task being measured, so
        # no question-derived paper filter is applied.
        hits = retrieve(
            self.index,
            self.embedder,
            question,
            k=self.top_k,
            mode=self.retrieval_mode,
            dense_weight=self.dense_weight,
            paper_ids=None,
        )
        evidence = format_retrieval_hits(hits)
        generated, answer_trace = _generate_common_answer(
            self.client,
            self.model,
            question,
            evidence,
        )
        return generated, evidence, {
            "retrieved_count": len(hits),
            "paper_scope_policy": "open_corpus_no_filter",
            "paper_scope_filter": [],
            "candidate_chunk_count": len(self.index.chunks),
            "retrieved_paper_count": len({
                item.get("paper_id", "") for item in evidence
            }),
            **answer_trace,
        }


class GoldGraphBackend:
    def __init__(self, client: OpenAI, model: str, uri: str, user: str,
                 password: str, expected_papers: int,
                 allow_count_mismatch: bool):
        self.client = client
        self.model = model
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        with self.driver.session() as session:
            stats = session.run(
                "MATCH (p:Paper) "
                "WITH count(p) AS paper_count "
                "MATCH ()-[r]->() WHERE r.info_unit IS NOT NULL "
                "RETURN paper_count, count(r) AS fact_count"
            ).single()
        self.graph_stats = dict(stats) if stats is not None else {
            "paper_count": 0,
            "fact_count": 0,
        }
        if (
            self.graph_stats["paper_count"] != expected_papers
            and not allow_count_mismatch
        ):
            self.close()
            raise RuntimeError(
                "Neo4j contains "
                f"{self.graph_stats['paper_count']} papers; expected {expected_papers}. "
                "Load the intended gold graph or pass --allow-graph-count-mismatch "
                "only for diagnostics."
            )
        self.schema_text = get_schema_text(self.driver)

    def close(self) -> None:
        self.driver.close()

    def run(self, question: str) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        result = retrieve_question(
            self.driver,
            self.client,
            self.model,
            self.schema_text,
            question,
        )
        evidence = _json_safe(result.get("records", []))
        if result.get("fallback_answer") is not None:
            generated = result["fallback_answer"]
            answer_trace = {
                "answer_attempts": 0,
                "answer_reasoning_removed": False,
                "raw_answer_outputs": [],
            }
        else:
            generated, answer_trace = _generate_common_answer(
                self.client,
                self.model,
                question,
                evidence,
            )
        trace = {
            "cypher": result.get("cypher"),
            "subgraph": _json_safe(result.get("subgraph")),
            "graph_stats": self.graph_stats,
            **answer_trace,
        }
        return generated, evidence, trace


def _existing_records(
    output_path: Path,
    experiment_id: str,
    overwrite: bool,
) -> dict[tuple[str, str], dict[str, Any]]:
    if not output_path.exists() or overwrite:
        return {}
    records = load_jsonl(output_path)
    incompatible = [
        record for record in records
        if record.get("experiment", {}).get("experiment_id") != experiment_id
    ]
    if incompatible:
        raise RuntimeError(
            f"{output_path} contains a different experiment. Choose a new output "
            "path or pass --overwrite explicitly."
        )
    indexed = {}
    for record in records:
        key = (record["question_id"], record["system"])
        if key in indexed:
            raise RuntimeError(
                f"{output_path} contains duplicate records for {key}. "
                "Choose a clean output path or repair the file before resuming."
            )
        indexed[key] = record
    return indexed


def _write_checkpoint(
    output_path: Path,
    records: dict[tuple[str, str], dict[str, Any]],
) -> None:
    """Atomically replace a JSONL checkpoint while preserving record order."""
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        for record in records.values():
            stream.write(
                json.dumps(record, ensure_ascii=False, default=str) + "\n"
            )
    temporary_path.replace(output_path)


def _retry_metadata(
    previous: dict[str, Any] | None,
) -> tuple[int, list[dict[str, Any]]]:
    if previous is None:
        return 1, []
    attempt = int(previous.get("run_attempt", 1)) + 1
    previous_errors = list(previous.get("previous_errors", []))
    if previous.get("status") == "error":
        previous_errors.append({
            "attempt": attempt - 1,
            "error": previous.get("error"),
            "latency_seconds": previous.get("latency_seconds"),
        })
    return attempt, previous_errors


def run_experiment(args: argparse.Namespace) -> None:
    load_dotenv()
    questions = load_jsonl(args.questions)
    selected = _select_questions(
        questions,
        args.split,
        args.question_id,
        args.limit,
        args.allow_draft,
    )
    assert_open_corpus(selected)
    config = _experiment_config(args, _file_sha256(args.questions))
    records = _existing_records(
        args.out,
        config["experiment_id"],
        args.overwrite,
    )
    completed = {
        key for key, record in records.items()
        if record.get("status") == "ok"
    }
    selected_keys = [
        (question["id"], system)
        for question in selected
        for system in args.systems
    ]
    pending_systems = {
        system for question_id, system in selected_keys
        if (question_id, system) not in completed
    }
    if not pending_systems:
        print(
            f"Experiment {config['experiment_id']}: wrote 0, "
            f"resumed {len(selected_keys)} -> {args.out}"
        )
        return
    client = _make_client(args.timeout)

    backends: dict[str, PlainBackend | GoldGraphBackend] = {}
    graph_backend = None
    try:
        if "plain_rag" in pending_systems:
            backends["plain_rag"] = PlainBackend(
                client,
                args.model,
                args.index,
                args.retrieval_mode,
                args.dense_weight,
                args.top_k,
            )
        if "graph_rag_gold" in pending_systems:
            password = os.getenv("NEO4J_PASSWORD")
            if not password:
                raise RuntimeError("NEO4J_PASSWORD must be set for graph_rag_gold")
            graph_backend = GoldGraphBackend(
                client,
                args.model,
                args.neo4j_uri,
                args.neo4j_user,
                password,
                args.expected_graph_papers,
                args.allow_graph_count_mismatch,
            )
            backends["graph_rag_gold"] = graph_backend

        args.out.parent.mkdir(parents=True, exist_ok=True)
        total = len(selected) * len(args.systems)
        written = 0
        skipped = 0
        retried = 0
        for question in selected:
            for system in args.systems:
                key = (question["id"], system)
                if key in completed:
                    skipped += 1
                    continue
                previous = records.get(key)
                run_attempt, previous_errors = _retry_metadata(previous)
                retried += int(previous is not None)
                started = time.perf_counter()
                try:
                    generated, evidence, trace = backends[system].run(
                        question["question"]
                    )
                    status = "ok"
                    error = None
                except Exception as exc:
                    generated = ""
                    evidence = []
                    trace = {}
                    status = "error"
                    error = f"{type(exc).__name__}: {exc}"
                record = {
                    "question_id": question["id"],
                    "split": question["split"],
                    "category": question["category"],
                    "review_status": question.get("review_status"),
                    "system": system,
                    "status": status,
                    "answer": generated,
                    "evidence": evidence,
                    "trace": trace,
                    "error": error,
                    "latency_seconds": time.perf_counter() - started,
                    "run_attempt": run_attempt,
                    "previous_errors": previous_errors,
                    "experiment": config,
                }
                records[key] = record
                _write_checkpoint(args.out, records)
                written += 1
                retry_note = f" (attempt {run_attempt})" if run_attempt > 1 else ""
                print(
                    f"[{written + skipped}/{total}] "
                    f"{question['id']} {system}: {status}{retry_note}"
                )
        print(
            f"Experiment {config['experiment_id']}: wrote {written}, "
            f"resumed {skipped}, retried {retried} -> {args.out}"
        )
    finally:
        if graph_backend is not None:
            graph_backend.close()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/evaluation/questions.jsonl"),
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=SYSTEMS,
        default=list(SYSTEMS),
    )
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--question-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--model",
        default=os.getenv("ANSWER_MODEL", FINAL_ANSWER_MODEL),
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--index", type=Path, default=Path("data/rag"))
    parser.add_argument(
        "--retrieval-mode",
        choices=("bm25", "dense", "hybrid"),
        default=FINAL_RETRIEVAL_MODE,
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=None,
    )
    parser.add_argument("--top-k", type=int, default=FINAL_TOP_K)
    parser.add_argument(
        "--neo4j-uri",
        default=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
    )
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--graph-source-label", default="semeval_gold")
    parser.add_argument("--expected-graph-papers", type=int, default=50)
    parser.add_argument("--allow-graph-count-mismatch", action="store_true")
    parser.add_argument(
        "--allow-draft",
        "--allow-unverified-questions",
        dest="allow_draft",
        action="store_true",
        help=(
            "Allow automatically curated questions without human verification. "
            "Their review status remains recorded in every response."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.retrieval_mode != "hybrid" and args.dense_weight is not None:
        parser.error("--dense-weight only changes hybrid retrieval")
    if args.dense_weight is None:
        args.dense_weight = (
            FINAL_DENSE_WEIGHT
            if args.retrieval_mode == "hybrid"
            else 0.0
        )
    if not 0.0 <= args.dense_weight <= 1.0:
        parser.error("--dense-weight must lie in [0, 1]")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    run_experiment(args)


if __name__ == "__main__":
    main()
