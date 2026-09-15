import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from karma_mini.rag.bm25 import BM25Index, tokenize
from karma_mini.rag.index import RAGIndex
from karma_mini.rag.retriever import retrieve
from graph_rag.qa_neo4j import (
    add_direction_projection,
    attach_cypher_scope_provenance,
    clean_cypher_response,
    extract_cypher_scope_paper_ids,
)
from rag_eval.calibration import (
    apply_temperature,
    calibration_metrics,
    fit_temperature,
    inverse_softmax_logits,
)
from rag_eval.compare_ablation import (
    _paired_randomization_p,
    _validate_matched_configs,
)
from rag_eval.compare_systems import compare_systems, validate_response_file
from rag_eval.judge import _validate_result
from rag_eval.paired_statistics import paired_randomization_p
from rag_eval.plain_retrieval import assert_open_corpus, explicit_paper_scope
from rag_eval.questions import build_questions
from rag_eval.retrieval_sensitivity import (
    build_configurations,
    question_stratum,
    summarize,
)
from rag_eval.run_experiment import (
    ABSTENTION_ANSWER,
    _generate_common_answer,
    run_experiment,
)
from rag_eval.run_comparison import ComparisonPaths, run_pipeline
from rag_eval.score import deterministic_metrics, retrieval_metrics, token_f1


ROOT = Path(__file__).resolve().parent.parent


class DirectionProjectionTests(unittest.TestCase):
    """An undirected match must not be reported as if it were directed."""

    def test_adds_direction_columns_to_a_plain_projection(self):
        rewritten = add_direction_projection(
            "MATCH (e)-[r]-(n) RETURN DISTINCT r.paper_id, e.name")
        self.assertIn("startNode(r).name AS fact_subject", rewritten)
        self.assertIn("endNode(r).name AS fact_object", rewritten)

    def test_annotates_each_hop_of_a_two_step_path(self):
        rewritten = add_direction_projection(
            "MATCH (a)-[r1]-(m)-[r2]-(b) RETURN r1.paper_id")
        self.assertIn("fact_subject_1", rewritten)
        self.assertIn("fact_object_2", rewritten)

    def test_leaves_aggregates_and_paths_untouched(self):
        for query in ("MATCH (e)-[r]-(n) RETURN count(*) AS c",
                      "MATCH (a)-[r*1..2]-(b) RETURN a.name"):
            self.assertEqual(add_direction_projection(query), query)


class QuestionBankTests(unittest.TestCase):
    def test_builds_expected_stratified_size(self):
        questions = build_questions(ROOT / "data" / "ncg" / "trial-data")
        self.assertEqual(len(questions), 80)
        self.assertEqual(sum(q["split"] == "dev" for q in questions), 20)
        self.assertEqual(sum(q["split"] == "test" for q in questions), 60)
        self.assertTrue(all(
            q["review_status"] == "team_accepted_no_expert_review"
            for q in questions
        ))
        self.assertEqual(len({q["id"] for q in questions}), 80)
        categories = {q["category"] for q in questions}
        for category in categories:
            self.assertTrue(any(
                q["category"] == category and q["split"] == "dev"
                for q in questions
            ))

    def test_questions_never_name_a_paper(self):
        """Open-corpus evaluation breaks if a question leaks a paper id.

        A leaked identifier reinstates the Plain-RAG scope filter, which makes
        paper-level retrieval correct by string matching instead of by ranking.
        """
        questions = build_questions(ROOT / "data" / "ncg" / "trial-data")
        assert_open_corpus(questions)
        for question in questions:
            self.assertEqual(explicit_paper_scope(question["question"]), set())

    def test_assert_open_corpus_rejects_a_named_paper(self):
        with self.assertRaises(RuntimeError):
            assert_open_corpus([
                {"id": "x-1", "question": "In paper machine-translation/0, what is X?"},
            ])


class CalibrationTests(unittest.TestCase):
    def test_inverse_softmax_round_trip_at_unit_temperature(self):
        probabilities = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
        np.testing.assert_allclose(
            apply_temperature(probabilities, 1.0),
            probabilities,
            atol=1e-10,
        )
        self.assertEqual(inverse_softmax_logits(probabilities).shape, (2, 3))

    def test_temperature_and_metrics_are_finite(self):
        probabilities = np.asarray([
            [0.95, 0.04, 0.01],
            [0.90, 0.08, 0.02],
            [0.05, 0.15, 0.80],
            [0.10, 0.80, 0.10],
        ])
        labels = [0, 2, 2, 1]
        temperature = fit_temperature(probabilities, labels)
        metrics = calibration_metrics(apply_temperature(probabilities, temperature), labels)
        self.assertGreater(temperature, 0)
        for key in ("accuracy", "nll", "brier", "ece"):
            self.assertTrue(np.isfinite(metrics[key]))

    def test_paired_randomization_handles_ties_and_direction(self):
        self.assertEqual(_paired_randomization_p(np.asarray([0.0, 0.0])), 1.0)
        probability = _paired_randomization_p(np.asarray([1.0, 1.0, 1.0]))
        self.assertEqual(probability, 0.25)

    def test_top_k_variation_requires_explicit_sensitivity_mode(self):
        common = {
            "question_file_sha256": "questions",
            "split": "dev",
            "answer_model": "answer-model",
            "answer_prompt_version": "prompt-v1",
            "answer_temperature": 0.0,
        }
        configs = {
            "k8": {
                **common,
                "plain_rag": {"top_k": 8, "index": "index"},
            },
            "k10": {
                **common,
                "plain_rag": {"top_k": 10, "index": "index"},
            },
        }
        with self.assertRaisesRegex(ValueError, "same top_k"):
            _validate_matched_configs(configs)
        _validate_matched_configs(configs, allow_top_k_variation=True)

    def test_large_randomization_is_seeded_and_finite(self):
        differences = np.ones(21)
        first = paired_randomization_p(differences, seed=7, samples=1_000)
        second = paired_randomization_p(differences, seed=7, samples=1_000)
        self.assertEqual(first, second)
        self.assertGreater(first, 0.0)
        self.assertLessEqual(first, 1.0)


class RetrievalModeTests(unittest.TestCase):
    class FakeEmbedder:
        def embed_query(self, query):
            return np.asarray([1.0, 0.0], dtype=np.float32)

    def setUp(self):
        chunks = [
            {"id": "p/0:1-1", "paper_id": "p/0", "text": "apple"},
            {"id": "p/1:1-1", "paper_id": "p/1", "text": "banana"},
        ]
        self.index = RAGIndex(
            chunks=chunks,
            embeddings=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
            bm25=BM25Index([tokenize(chunk["text"]) for chunk in chunks]),
            meta={},
        )

    def test_modes_select_expected_chunk(self):
        bm25 = retrieve(self.index, None, "apple", mode="bm25", k=1)
        dense = retrieve(
            self.index,
            self.FakeEmbedder(),
            "apple",
            mode="dense",
            k=1,
        )
        self.assertEqual(bm25[0]["chunk"]["id"], "p/0:1-1")
        self.assertEqual(dense[0]["chunk"]["id"], "p/1:1-1")
        self.assertIsNone(bm25[0]["cosine"])

    def test_explicit_paper_scope_filters_candidates_before_ranking(self):
        scoped = retrieve(
            self.index,
            None,
            "apple",
            mode="bm25",
            k=5,
            paper_ids={"p/1"},
        )
        self.assertEqual([hit["chunk"]["paper_id"] for hit in scoped], ["p/1"])

    def test_unknown_paper_scope_fails_instead_of_falling_back_globally(self):
        with self.assertRaisesRegex(ValueError, "No indexed chunks matched"):
            retrieve(
                self.index,
                None,
                "apple",
                mode="bm25",
                paper_ids={"p/99"},
            )

    def test_explicit_paper_scope_is_normalized(self):
        self.assertEqual(
            explicit_paper_scope("Compare Machine-Translation/0 with qa/1."),
            {"machine-translation/0", "qa/1"},
        )


class RetrievalSensitivityTests(unittest.TestCase):
    def test_builds_stable_grid(self):
        configurations = build_configurations([10, 3, 3], [0.75, 0.25])
        self.assertEqual(
            [config.name for config in configurations],
            [
                "bm25-k3",
                "bm25-k10",
                "hybrid-a025-k3",
                "hybrid-a025-k10",
                "hybrid-a075-k3",
                "hybrid-a075-k10",
            ],
        )

    def test_strata_keep_graph_native_questions_separate(self):
        self.assertEqual(
            question_stratum({"category": "cross_paper_entity_count"}),
            "graph_native",
        )
        self.assertEqual(
            question_stratum({"category": "single_paper_fact"}),
            "shared_answerability",
        )

    def test_summary_does_not_impute_missing_metrics(self):
        report = summarize([
            {
                "metrics": {"reference_item_recall": 1.0},
                "retrieved_chunk_count": 3,
                "unique_paper_count": 1,
                "latency_ms": 2.0,
            },
            {
                "metrics": {},
                "retrieved_chunk_count": 3,
                "unique_paper_count": 2,
                "latency_ms": 4.0,
            },
        ])
        self.assertEqual(
            report["metric_means"]["reference_item_recall"],
            {"mean": 1.0, "n": 1},
        )
        self.assertEqual(report["mean_unique_papers"], 1.5)


class GraphProvenanceTests(unittest.TestCase):
    def test_cleans_reasoning_and_markdown_from_cypher(self):
        raw = (
            "<think>construct the query</think>\n"
            "```cypher\nMATCH (a)-[r]->(b) RETURN a\n```"
        )
        self.assertEqual(
            clean_cypher_response(raw),
            "MATCH (a)-[r]->(b) RETURN a",
        )

    def test_retains_paper_id_from_equality_filter(self):
        cypher = (
            'MATCH (a:Entity)-[r]->(b:Entity) '
            'WHERE r.paper_id = "machine-translation/0" '
            'RETURN a.name, b.name'
        )
        self.assertEqual(
            extract_cypher_scope_paper_ids(cypher),
            ["machine-translation/0"],
        )
        records = attach_cypher_scope_provenance(
            [{"a.name": "Contribution", "b.name": "SMT"}],
            cypher,
        )
        self.assertEqual(
            records[0]["query_scope_paper_ids"],
            ["machine-translation/0"],
        )

    def test_retains_multiple_paper_ids_from_in_filter(self):
        cypher = (
            "MATCH (a)-[r]->(b) "
            "WHERE r.paper_id IN ['question-answering/2', 'text-classification/1'] "
            "RETURN count(*)"
        )
        self.assertEqual(
            extract_cypher_scope_paper_ids(cypher),
            ["question-answering/2", "text-classification/1"],
        )


class AnswerGenerationTests(unittest.TestCase):
    class FakeCompletions:
        def __init__(self, answers):
            self.answers = iter(answers)
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            answer = next(self.answers)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=answer),
            )])

    def test_rechecks_nonempty_evidence_after_abstention(self):
        completions = self.FakeCompletions([
            ABSTENTION_ANSWER,
            "SMT [machine-translation/0]",
        ])
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        )
        answer, trace = _generate_common_answer(
            client,
            "test-model",
            "What problem is addressed?",
            [{"paper_id": "machine-translation/0", "text": "SMT"}],
        )
        self.assertEqual(answer, "SMT [machine-translation/0]")
        self.assertEqual(trace["answer_attempts"], 2)
        self.assertEqual(completions.calls, 2)

    def test_does_not_retry_empty_evidence(self):
        completions = self.FakeCompletions([ABSTENTION_ANSWER])
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        )
        answer, trace = _generate_common_answer(
            client,
            "test-model",
            "What problem is addressed?",
            [],
        )
        self.assertEqual(answer, ABSTENTION_ANSWER)
        self.assertEqual(trace["answer_attempts"], 1)
        self.assertEqual(completions.calls, 1)

    def test_strips_reasoning_block_but_retains_raw_output(self):
        raw = "<think>private reasoning</think>\nFinal answer [p/0]"
        completions = self.FakeCompletions([raw])
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        )
        answer, trace = _generate_common_answer(
            client,
            "test-model",
            "What problem is addressed?",
            [{"paper_id": "p/0", "text": "Final answer"}],
        )
        self.assertEqual(answer, "Final answer [p/0]")
        self.assertTrue(trace["answer_reasoning_removed"])
        self.assertEqual(trace["raw_answer_outputs"], [raw])


class ExperimentResumeTests(unittest.TestCase):
    class FakePlainBackend:
        def __init__(self):
            self.calls = 0

        def run(self, question):
            self.calls += 1
            return "plain answer", [{"paper_id": "paper/1"}], {}

    class FakeGraphBackend:
        def __init__(self):
            self.calls = 0

        def run(self, question):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient service failure")
            return "graph answer", [{"paper_id": "paper/1"}], {}

        def close(self):
            pass

    def test_resume_retries_only_failed_responses(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            question_path = directory / "questions.jsonl"
            output_path = directory / "responses.jsonl"
            question_path.write_text(json.dumps({
                "id": "q1",
                "split": "test",
                "category": "fact",
                "question": "What is reported?",
                "review_status": "human_verified",
            }) + "\n", encoding="utf-8")
            arguments = SimpleNamespace(
                questions=question_path,
                systems=["plain_rag", "graph_rag_gold"],
                split="test",
                question_id=None,
                limit=None,
                out=output_path,
                model="answer-model",
                timeout=120.0,
                index=Path("data/rag"),
                retrieval_mode="hybrid",
                dense_weight=0.25,
                top_k=8,
                neo4j_uri="bolt://127.0.0.1:7687",
                neo4j_user="neo4j",
                graph_source_label="semeval_gold",
                expected_graph_papers=50,
                allow_graph_count_mismatch=False,
                allow_draft=False,
                overwrite=False,
            )
            plain_backend = self.FakePlainBackend()
            graph_backend = self.FakeGraphBackend()
            with patch(
                "rag_eval.run_experiment._make_client",
                return_value=object(),
            ), patch(
                "rag_eval.run_experiment.PlainBackend",
                return_value=plain_backend,
            ) as plain_constructor, patch(
                "rag_eval.run_experiment.GoldGraphBackend",
                return_value=graph_backend,
            ), patch.dict(os.environ, {"NEO4J_PASSWORD": "test-password"}):
                with contextlib.redirect_stdout(io.StringIO()):
                    run_experiment(arguments)
                    run_experiment(arguments)

            records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(len(records), 2)
        self.assertEqual(plain_backend.calls, 1)
        self.assertEqual(plain_constructor.call_count, 1)
        self.assertEqual(graph_backend.calls, 2)
        graph_record = next(
            record for record in records
            if record["system"] == "graph_rag_gold"
        )
        self.assertEqual(graph_record["status"], "ok")
        self.assertEqual(graph_record["run_attempt"], 2)
        self.assertEqual(len(graph_record["previous_errors"]), 1)


class DeterministicScoreTests(unittest.TestCase):
    def test_paper_set_and_integer_metrics(self):
        set_question = {
            "answer_type": "paper_id_set",
            "reference_items": ["machine-translation/0", "question-answering/2"],
        }
        set_response = {
            "status": "ok",
            "answer": "[machine-translation/0] and [question-answering/2]",
        }
        self.assertEqual(
            deterministic_metrics(set_question, set_response)["set_exact"],
            1.0,
        )
        integer_question = {"answer_type": "integer", "reference_items": [8]}
        integer_response = {"status": "ok", "answer": "There are eight papers."}
        self.assertEqual(
            deterministic_metrics(integer_question, integer_response)["integer_accuracy"],
            1.0,
        )

    def test_token_f1(self):
        self.assertEqual(token_f1("BERT model", "BERT model"), 1.0)

    def test_abstention_metric_accepts_supported_explanation(self):
        question = {"answer_type": "unanswerable", "reference_items": []}
        response = {
            "status": "ok",
            "answer": (
                "The supplied evidence does not contain any mention of BERT. "
                "The retrieved paper discusses convolutional models instead."
            ),
        }
        self.assertEqual(
            deterministic_metrics(question, response)["abstention_accuracy"],
            1.0,
        )

    def test_paper_level_retrieval_metrics_use_evidence_not_answer(self):
        question = {
            "evidence": [
                {"paper_id": "machine-translation/0"},
                {"paper_id": "question-answering/2"},
            ],
        }
        response = {
            "answer": "The answer cites [question-answering/2].",
            "evidence": [
                {"paper_id": "machine-translation/0"},
                {"paper_id": "named-entity-recognition/1"},
            ],
        }
        metrics = retrieval_metrics(question, response)
        self.assertEqual(metrics["paper_precision"], 0.5)
        self.assertEqual(metrics["paper_recall"], 0.5)
        self.assertEqual(metrics["paper_f1"], 0.5)

    def test_retrieval_metrics_skip_questions_without_positive_evidence(self):
        self.assertEqual(
            retrieval_metrics({"evidence": []}, {"evidence": []}),
            {},
        )

    def test_reference_item_recall_checks_retrieved_content(self):
        question = {
            "answer_type": "list",
            "reference_items": ["Statistical Machine Translation", "phrase-based SMT"],
            "evidence": [{"paper_id": "machine-translation/0"}],
        }
        response = {
            "evidence": [{
                "paper_id": "machine-translation/0",
                "text": "We study statistical machine translation.",
            }],
        }
        metrics = retrieval_metrics(question, response)
        self.assertEqual(metrics["reference_item_recall"], 0.5)
        self.assertEqual(metrics["reference_item_all"], 0.0)


class JudgeValidationTests(unittest.TestCase):
    def test_validates_and_normalizes_probabilities(self):
        result = _validate_result({
            "label": "fully_correct",
            "probabilities": {
                "fully_correct": 0.8,
                "partially_correct": 0.15,
                "incorrect": 0.04,
            },
            "scores": {
                "correctness": 4,
                "faithfulness": 4,
                "completeness": 3,
                "relevance": 4,
            },
            "unsupported_claims": [],
            "missing_information": ["one minor detail"],
            "rationale": "The answer is supported by the supplied evidence.",
        }, raw_response="raw")
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0)
        self.assertEqual(result.label, "fully_correct")


class SystemComparisonTests(unittest.TestCase):
    systems = ("plain_rag", "graph_rag_gold")

    @staticmethod
    def _write_jsonl(path, records):
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def _fixture(self, directory):
        question_path = directory / "questions.jsonl"
        response_path = directory / "responses.jsonl"
        judgment_path = directory / "judgments.jsonl"
        questions = [
            {
                "id": "q1",
                "split": "test",
                "category": "fact",
                "answer_type": "short_answer",
                "reference_answer": "alpha",
                "reference_items": ["alpha"],
                "evidence": [{"paper_id": "paper/1"}],
                "review_status": "draft_needs_human_review",
            },
            {
                "id": "q2",
                "split": "test",
                "category": "unanswerable",
                "answer_type": "unanswerable",
                "reference_answer": "No answer",
                "reference_items": [],
                "evidence": [],
                "review_status": "draft_needs_human_review",
            },
        ]
        experiment = {
            "experiment_id": "experiment-1",
            "split": "test",
            "systems": ["graph_rag_gold", "plain_rag"],
            "answer_model": "answer-model",
            "answer_prompt_version": "answer-prompt",
            "answer_temperature": 0.0,
            "plain_rag": {
                "retrieval_mode": "hybrid",
                "dense_weight": 0.25,
                "top_k": 8,
            },
            "graph_rag": {"source_label": "semeval_gold"},
        }
        labels = {
            ("q1", "plain_rag"): "partially_correct",
            ("q1", "graph_rag_gold"): "fully_correct",
            ("q2", "plain_rag"): "fully_correct",
            ("q2", "graph_rag_gold"): "fully_correct",
        }
        responses = []
        judgments = []
        for question in questions:
            for system in self.systems:
                answer = (
                    "alpha"
                    if question["id"] == "q1"
                    else "The supplied evidence does not contain the answer."
                )
                evidence = (
                    [{"paper_id": "paper/1", "text": "alpha"}]
                    if question["id"] == "q1"
                    else []
                )
                responses.append({
                    "question_id": question["id"],
                    "split": "test",
                    "category": question["category"],
                    "review_status": question["review_status"],
                    "system": system,
                    "status": "ok",
                    "answer": answer,
                    "evidence": evidence,
                    "latency_seconds": 1.0,
                    "error": None,
                    "experiment": experiment,
                })
                label = labels[(question["id"], system)]
                judgments.append({
                    "question_id": question["id"],
                    "split": "test",
                    "category": question["category"],
                    "system": system,
                    "judge_model": "judge-model",
                    "prompt_version": "judge-prompt",
                    "label": label,
                    "probabilities": {
                        "fully_correct": 0.8 if label == "fully_correct" else 0.1,
                        "partially_correct": 0.8 if label == "partially_correct" else 0.1,
                        "incorrect": 0.1,
                    },
                    "scores": {
                        "correctness": 4,
                        "faithfulness": 4,
                        "completeness": 4,
                        "relevance": 4,
                    },
                })
        self._write_jsonl(question_path, questions)
        self._write_jsonl(response_path, responses)
        self._write_jsonl(judgment_path, judgments)
        return question_path, response_path, judgment_path

    def test_builds_paired_system_report(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self._fixture(Path(temporary_directory))
            report = compare_systems(
                *paths,
                bootstrap_samples=100,
                randomization_samples=100,
            )
        self.assertEqual(report["question_count"], 2)
        self.assertEqual(report["status"], "complete_unverified_questions")
        primary = report["primary_paired_comparison"]
        self.assertEqual(primary["comparison_wins"], 1)
        self.assertEqual(primary["ties"], 1)
        self.assertEqual(
            primary["fully_correct_rate"]["difference"],
            0.5,
        )

    def test_rejects_missing_judgment_pair(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self._fixture(Path(temporary_directory))
            judgments = paths[2].read_text(encoding="utf-8").splitlines()
            paths[2].write_text("\n".join(judgments[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing judgments"):
                compare_systems(
                    *paths,
                    bootstrap_samples=10,
                    randomization_samples=10,
                )

    def test_validates_complete_response_file_before_judging(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            question_path, response_path, _ = self._fixture(
                Path(temporary_directory)
            )
            validation = validate_response_file(question_path, response_path)
        self.assertEqual(validation["question_count"], 2)
        self.assertEqual(validation["response_count"], 4)

    def test_pipeline_paths_are_stable(self):
        paths = ComparisonPaths.under(Path("results/final_comparison"), "test")
        self.assertEqual(
            paths.comparison,
            Path("results/final_comparison/test-comparison.json"),
        )

    def test_pipeline_orchestrates_all_local_stages(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            question_path, source_responses, source_judgments = self._fixture(
                directory
            )
            output_dir = directory / "output"

            def fake_generate(arguments):
                arguments.out.parent.mkdir(parents=True, exist_ok=True)
                arguments.out.write_text(
                    source_responses.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )

            def fake_judge(
                questions,
                responses,
                output,
                model,
                timeout,
                overwrite=False,
                allow_draft=False,
            ):
                output.write_text(
                    source_judgments.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )

            arguments = SimpleNamespace(
                questions=question_path,
                split="test",
                output_dir=output_dir,
                index=Path("data/rag"),
                answer_model="answer-model",
                judge_model="judge-model",
                timeout=120.0,
                neo4j_uri="bolt://127.0.0.1:7687",
                neo4j_user="neo4j",
                graph_source_label="semeval_gold",
                expected_graph_papers=50,
                allow_graph_count_mismatch=False,
                allow_unverified_questions=True,
                seed=7,
                bootstrap_samples=100,
                randomization_samples=100,
                overwrite=False,
            )
            with patch(
                "rag_eval.run_comparison.run_experiment",
                side_effect=fake_generate,
            ), patch(
                "rag_eval.run_comparison.run_judge",
                side_effect=fake_judge,
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    manifest = run_pipeline(arguments)

            paths = ComparisonPaths.under(output_dir, "test")
            self.assertEqual(manifest["response_count"], 4)
            self.assertTrue(paths.scores.exists())
            self.assertTrue(paths.comparison.exists())
            self.assertTrue(paths.manifest.exists())


if __name__ == "__main__":
    unittest.main()
