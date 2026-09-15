# Shared RAG evaluation

This package compares the text retriever in `plain_rag/` with the Neo4j
retriever in `graph_rag/`. Both systems use common answer generation, judging,
and scoring stages. Contribution-graph extraction is evaluated separately by
`eval_ncg.py`. See the [project README](../README.md) for the full package layout.

## Setup

Run commands from the repository root with Python 3.10 or newer:

```bash
pip install -r requirements.txt
python -m evaluation --help
```

The help command lists options without starting an experiment. For model calls
and graph retrieval, configure `.env` in the repository root:

```env
KIT_API_KEY=your_api_key
KIT_BASE_URL=https://ki-toolbox.scc.kit.edu/api/v1
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

A running Neo4j instance and access to the configured KIT models are required.
Reuse the existing Plain-RAG index in `data/rag/`. If it is missing, build it:

```bash
python -m plain_rag index
```

Index creation makes embedding API calls, including when the later evaluation
uses BM25. Load the gold contribution graphs into a dedicated evaluation
Neo4j database if they are not already loaded:

```bash
python -m graph_rag.load_neo4j --predictions data/ncg/trial-data --clear
```

`--clear` deletes the configured database's contents. The evaluation runner
expects 50 `Paper` nodes. This gold-graph setup is an oracle structured-retrieval
condition: it uses human annotations rather than KARMA-Mini extraction outputs.

## Benchmark

[questions.py](questions.py) derives 80 questions from the 50-paper SemEval-2021
Task 11 trial corpus: 20 development questions and 60 held-out test questions.
It uses gold triples and paper text to construct questions and references.
Questions cover paper retrieval, facts, multi-hop relations, cross-paper
profiles, lists, counts, and unanswerable queries. Paper selection is part of
the task; the question text does not supply paper identifiers.

The benchmark is project-created and has no expert human verification.
Generated records retain their review status, and the runner requires
`--allow-unverified-questions` to evaluate them. Judge probabilities are also
uncalibrated because no expert correctness labels are available.

Generated benchmark files under `data/evaluation/` and experiment outputs under
`results/` are not included in the code submission. If
`data/evaluation/questions.jsonl` is missing, generate it locally:

```bash
python -m evaluation.questions
```

Reuse an existing frozen question file when reproducing a saved experiment.
The generator writes its output directly, so do not run it over the benchmark
used by a completed experiment. Preserve the question file, response records,
judgments, and manifest together. Do not retune settings on the held-out split.

## Current defaults

[config.py](config.py) defines the current comparison settings:

| Setting | Default |
| --- | --- |
| Plain-RAG retrieval | BM25 |
| Retrieved passages | 15 |
| Dense weight | 0.0 |
| Answer model | `kit.mistral-small-4-119b-a8b` |
| Judge model | `kit.gpt-oss-120b` |

These describe the current code. For a completed experiment, use its saved
manifest and response configuration as the record of what actually ran.
The answer and judge models can be overridden with command-line options;
`JUDGE_MODEL` also overrides the default judge model.

## Run a development comparison

Use a separate output directory for a local development run:

```bash
python -m evaluation \
  --split dev \
  --allow-unverified-questions \
  --output-dir results/local-dev
```

This runs both systems on the 20 development questions, judges the 40 answers,
and calculates their metrics and paired comparison. It makes model API calls.
The command defaults to the held-out `test` split when `--split` is omitted,
so keep `--split dev` explicit during development.

Compatible runs resume existing responses and judgments. Keep completed-run
artifacts intact and use a new output directory for a different configuration.
`--overwrite` replaces existing response and judgment files.

## Outputs and interpretation

Each run writes the following files, prefixed with `dev-` or `test-`:

| File | Contents |
| --- | --- |
| `responses.jsonl` | Answers, retrieved evidence, model traces, timings, and configuration |
| `judgments.jsonl` | Raw and parsed pointwise judgments |
| `scores.json` | Aggregated retrieval and answer metrics |
| `comparison.json` | Paired system differences and category summaries |
| `run-manifest.json` | Run status, models, settings, counts, and output paths |

Retrieval metrics measure paper provenance and reference-item coverage in the
retrieved evidence. Deterministic answer metrics include token overlap,
reference-term recall, set matching, count accuracy, and abstention accuracy
where appropriate. These are separate from the judge's `fully_correct`,
`partially_correct`, and `incorrect` labels. High term coverage does not imply
that an answer is fully correct or supported by the evidence.

The judge evaluates each answer against references and evidence without being
told which system produced it. Interpret its labels as automated assessments,
with the benchmark's lack of expert verification kept explicit.

Individual stages are available through `evaluation.run_experiment`,
`evaluation.judge`, `evaluation.score`, and `evaluation.compare_systems`.
Use each module's `--help` for its input and output options.
