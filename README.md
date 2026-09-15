# KARMA Mini: NLPContributionGraph Extraction (SemEval-2021 Task 11)

KARMA Mini is a streamlined, 4-agent LLM pipeline that extracts a paper's
**contribution knowledge graph** for the SemEval-2021 Task 11
[NLPContributionGraph (NCG)](https://ncg-task.github.io/) shared task.

The repo also contains a **plain-RAG baseline** over the same corpus
(`plain_rag/`) for a GraphRAG-vs-RAG comparison — see
[RAG baseline](#rag-baseline-graphrag-vs-rag) below.

Each scholarly NLP paper is processed **independently** and yields its own graph
rooted at a single node literally named `Contribution`. Graphs are never merged
across papers. This is a simplified, NCG-focused reproduction of the
[KARMA architecture](https://github.com/YuxingLu613/KARMA).

## Repository layout

| Package | Responsibility | Entry point |
| --- | --- | --- |
| `karma_mini/` | Contribution-graph extraction agents and shared corpus loading | `python -m karma_mini` |
| `plain_rag/` | Text chunking, embeddings, BM25/hybrid retrieval, and grounded answers | `python -m plain_rag` |
| `graph_rag/` | Neo4j graph retrieval and answers; the existing comparison UI | `python -m graph_rag`; `streamlit run graph_rag/app.py` |
| `evaluation/` | Shared benchmark, experiment runner, judging, scoring, and paired comparison | `python -m evaluation` |

The two retrieval implementations are sibling packages. `evaluation/` evaluates
both through shared stages and stays separate from either implementation.
Extraction quality is evaluated separately by `eval_ncg.py`.

## What it produces

For every paper, a rooted multi-way tree / DAG of triples:

```
(Contribution || has research problem || Statistical Machine Translation)
(Contribution || has || Model)
(Model || has || neural network architecture)
(neural network architecture || refer to as || RNN Encoder - Decoder)
(Contribution || has || Results)
(Results || improves the performance || adding features)
```

- **Predicates are free text** taken verbatim from the sentence wording (plus the
  structural `has`). They are never mapped to a fixed vocabulary.
- **Phrases are exact Stanza tokens**, copied verbatim from the input so they can
  string-match the gold annotations (e.g. `phrase - based SMT`,
  `two recurrent neural networks ( RNN )`).
- Two info units attach **directly** to the root: `research-problem`
  (`Contribution || has research problem || <term>`) and `code`
  (`Contribution || Code || <url>`). All others get an intermediate node
  (`Model`, `Results`, `Experimental setup`, …).

### The fixed 12 information units

`RESEARCHPROBLEM, APPROACH, MODEL, CODE, DATASET, EXPERIMENTALSETUP,
HYPERPARAMETERS, BASELINES, RESULTS, TASKS, EXPERIMENTS, ABLATIONANALYSIS`

Mandatory per paper: `RESEARCHPROBLEM`, `RESULTS`, and at least one of
`MODEL` / `APPROACH`. Normalization: `method`/`application` → `APPROACH`;
`system`/`architecture` → `MODEL`; `EXPERIMENTALSETUP` only when hardware is
mentioned, otherwise `HYPERPARAMETERS`.

## The 4-Agent Architecture

The pipeline (`karma_mini/core/pipeline.py`) runs four agents **per paper**,
mirroring the task's own granularities (sentences → phrases → triples):

### 1. Contribution Sentence Agent (CSA) — *sentence selection + IU tagging*
`karma_mini/agents/contribution_sentence_agent.py`

Reads the **whole paper** as numbered Stanza sentences (one per line, 1-indexed)
and selects the handful of **contribution sentences** (what *this* paper
contributes — usually in the title, abstract, intro, and the opening of the
model/results sections), tagging each with **exactly one** information unit.
Deciding the unit once, at the sentence level, means every triple later drawn
from a sentence lands in the same `triples/<iu>.txt` file — related edges can
never scatter across files.

### 2. Schema Alignment Agent (SAA) — *info-unit alignment*
`karma_mini/agents/schema_alignment_agent.py`

Aligns each selected sentence's `info_unit` to the fixed 12-unit inventory,
applying the official normalization rules. The sentence text is left
**untouched**. Mostly deterministic (a rule table); the LLM is a
temperature-0.0 fallback only for borderline labels.

### 3. Triple Extraction Agent (TEA) — *per-sentence phrase + triple extraction*
`karma_mini/agents/triple_extraction_agent.py`

Given **one** contribution sentence and its aligned info unit, extracts the
scientific-term and predicate phrases and wires them into
`(subject, predicate, object)` triples. Working one sentence at a time keeps
every phrase a **verbatim span** of that sentence; a deterministic
snap-to-span pass (`karma_mini/core/spans.py`) repairs casing/spacing drift
(e.g. `fixed-length` → `fixed - length`), and triples whose terms cannot be
located are dropped. Nodes extracted from earlier sentences are offered back to
the agent so later sentences can chain onto them (cross-sentence links).

### 4. Knowledge Integration Agent (KIA) — *per-paper graph assembly*
`karma_mini/agents/knowledge_integration_agent.py`

Assembles the rooted graph: adds the `(Contribution || has || <InfoUnit>)`
backbone edges, special-cases the two direct units, keeps the term→term edges,
**merges duplicate phrase nodes** (identical strings collapse into one node,
creating the DAG), de-duplicates identical triples, and groups by info unit.
Fully deterministic Python.

## Input data

NCG trial data lives under `data/ncg/trial-data/<task>/<n>/`. The canonical input
is `<id>-Stanza-out.txt` (tokenized, one sentence per line, 1-indexed). The
loader (`karma_mini/loader.py`) reads it and attaches simple section hints from
the standalone header lines Stanza preserves (`title`, `abstract`,
`Introduction`, …). **No OCR is performed** — the dataset ships plaintext.

## Setup & Usage

Use Python 3.10 or newer.

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment variables** (`.env` in the project root):
   ```env
   KIT_API_KEY=your_actual_api_key_here
   KIT_BASE_URL=https://ki-toolbox.scc.kit.edu/api/v1
   ```

3. **Run the pipeline** over the whole trial set (writes predictions mirroring
   the gold folder layout):
   ```bash
   python -m karma_mini --data data/ncg/trial-data --out data/ncg/predictions
   ```
   Run on a single paper folder (handy for inspection):
   ```bash
   python -m karma_mini --data data/ncg/trial-data/machine-translation/0
   ```
   Pick a model (`--model`) and request timeout (`--timeout`) as needed:
   ```bash
   python -m karma_mini --model kit.gpt-oss-120b --timeout 120
   ```

   Predictions are written to `data/ncg/predictions/<task>/<n>/`:
   ```
   triples/<iu>.txt   # "(subject||predicate||object)" per line
   sentences.txt      # contribution sentence line numbers
   entities.txt       # "<line>\t<start>\t<end>\t<phrase>"
   ```

## Evaluation

Scoring uses the **official** SemEval-2021 Task 11 scorer, pinned as a Git
submodule in `scoring/`. Initialize it after cloning this repository:

```bash
git submodule update --init scoring
```

Then score the predictions against the gold trial data:

```bash
python eval_ncg.py --gold data/ncg/trial-data --pred data/ncg/predictions
```

`eval_ncg.py` reuses the official `evaluate()` matching logic and prints
precision / recall / F1 for **sentences**, **phrases**, **info units**, and
**triples**, plus a per-info-unit triple breakdown. (The official scorer imports
`scipy`/`numpy` it never uses; `eval_ncg.py` stubs them so no heavy deps are
required.)

## RAG baseline (GraphRAG vs RAG)

`plain_rag/` implements a standard retrieval-augmented generation pipeline over the
**raw Stanza text** of the trial papers — the plain-RAG side of a
GraphRAG-vs-RAG comparison (the GraphRAG side retrieves over the gold
contribution triples of the same papers).

The command-line implementation lives in `plain_rag/cli.py` and is launched
with `python -m plain_rag`. The original `rag.py` remains a compatibility
launcher with the same arguments.

Pipeline (per the classic RAG architecture):

1. **Chunking** (`plain_rag/corpus.py`): sliding windows of 4 Stanza
   sentences, stride 2 (50% overlap), each carrying paper id, line range, and
   nearest section header.
2. **Embedding** (`plain_rag/embedder.py`): `kit.qwen3-embedding-8b`
   (4096-dim), L2-normalized, batched.
3. **Hybrid retrieval** (`plain_rag/retriever.py`): every chunk is scored
   with **BM25** (pure-Python Okapi, `bm25.py`) and **embedding cosine
   similarity**; both are min-max normalized over the collection and combined
   as their **average** — the final ranking score.
4. **Generation** (`plain_rag/generator.py`): an LLM answers from the
   retrieved excerpts only, citing sources as `[<task>/<n>:<lines>]`.

Usage:

```bash
python -m plain_rag index                          # one-time: chunk + embed the corpus
python -m plain_rag search "multi-head attention"  # retrieval only, shows BM25/cosine/combined
python -m plain_rag ask "What is the RNN Encoder - Decoder used for?"
python -m plain_rag ask "..." --model azure.gpt-4.1-mini -k 8   # any chat model on the endpoint
```

The index lives in `data/rag/` (gitignored; rebuild anytime with
`python -m plain_rag index`). Existing indexes require no migration.

## GraphRAG and comparison UI

The Neo4j-backed GraphRAG implementation and its Streamlit interface live in
`graph_rag/`:

- `graph_rag/load_neo4j.py` loads extracted or gold contribution triples.
- `graph_rag/qa_neo4j.py` translates questions to Cypher and summarizes the
  graph results.
- `graph_rag/prompts.py` contains the Cypher-generation and answer prompts.
- `graph_rag/app.py` runs GraphRAG and plain RAG for the same question and
  displays both answers side by side, including a draggable, force-directed
  Neo4j evidence graph, generated Cypher, and retrieved text passages.

Configure Neo4j in `.env` in addition to the KIT API variables:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

Launch interactive graph question answering with:

```bash
python -m graph_rag
```

Prepare both retrieval backends once, then launch the comparison UI:

```bash
python -m graph_rag.load_neo4j --predictions data/ncg/trial-data --clear
python -m plain_rag index
streamlit run graph_rag/app.py
```

By default, the plain-RAG index is read from `data/rag/` and its top five
passages are used. Override these with `RAG_INDEX_PATH` and `RAG_TOP_K`.

## Shared RAG evaluation

`evaluation/` evaluates both `plain_rag/` and `graph_rag/` using the same benchmark,
answer generation, pointwise judging, deterministic metrics, and paired
comparison. Its Plain-RAG adapter, `evaluation/plain_retrieval.py`, formats
retrieval hits for evaluation and validates the benchmark's open-corpus
constraint; the retriever itself lives in `plain_rag/retriever.py`.

`python -m evaluation` launches the complete comparison pipeline and accepts the
same arguments as `python -m evaluation.run_comparison`. Individual stages remain
available as modules such as `evaluation.run_experiment`, `evaluation.judge`,
`evaluation.score`, and `evaluation.compare_systems`.

See [the evaluation guide](evaluation/README.md) for preparation, frozen
configuration, and commands. Inspect the entry point without starting a run:

```bash
python -m evaluation --help
```
