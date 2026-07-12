# KARMA Mini: NLPContributionGraph Extraction (SemEval-2021 Task 11)

KARMA Mini is a streamlined, 4-agent LLM pipeline that extracts a paper's
**contribution knowledge graph** for the SemEval-2021 Task 11
[NLPContributionGraph (NCG)](https://ncg-task.github.io/) shared task.

The repo also contains a **plain-RAG baseline** over the same corpus
(`rag.py`, `karma_mini/rag/`) for a GraphRAG-vs-RAG comparison — see
[RAG baseline](#rag-baseline-graphrag-vs-rag) below.

Each scholarly NLP paper is processed **independently** and yields its own graph
rooted at a single node literally named `Contribution`. Graphs are never merged
across papers. This is a simplified, NCG-focused reproduction of the
[KARMA architecture](https://github.com/YuxingLu613/KARMA).

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
   python main.py --data data/ncg/trial-data --out data/ncg/predictions
   ```
   Run on a single paper folder (handy for inspection):
   ```bash
   python main.py --data data/ncg/trial-data/machine-translation/0
   ```
   Pick a model (`--model`) and request timeout (`--timeout`) as needed:
   ```bash
   python main.py --model kit.gpt-oss-120b --timeout 120
   ```

   Predictions are written to `data/ncg/predictions/<task>/<n>/`:
   ```
   triples/<iu>.txt   # "(subject||predicate||object)" per line
   sentences.txt      # contribution sentence line numbers
   entities.txt       # "<line>\t<start>\t<end>\t<phrase>"
   ```

## Evaluation

Scoring uses the **official** SemEval-2021 Task 11 scorer. Clone it once into
`scoring/` (gitignored):

```bash
git clone https://github.com/ncg-task/scoring-program.git scoring
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

`rag.py` implements a standard retrieval-augmented generation pipeline over the
**raw Stanza text** of the trial papers — the plain-RAG side of a
GraphRAG-vs-RAG comparison (the GraphRAG side retrieves over the gold
contribution triples of the same papers).

Pipeline (per the classic RAG architecture):

1. **Chunking** (`karma_mini/rag/corpus.py`): sliding windows of 4 Stanza
   sentences, stride 2 (50% overlap), each carrying paper id, line range, and
   nearest section header.
2. **Embedding** (`karma_mini/rag/embedder.py`): `kit.qwen3-embedding-8b`
   (4096-dim), L2-normalized, batched.
3. **Hybrid retrieval** (`karma_mini/rag/retriever.py`): every chunk is scored
   with **BM25** (pure-Python Okapi, `bm25.py`) and **embedding cosine
   similarity**; both are min-max normalized over the collection and combined
   as their **average** — the final ranking score.
4. **Generation** (`karma_mini/rag/generator.py`): an LLM answers from the
   retrieved excerpts only, citing sources as `[<task>/<n>:<lines>]`.

Usage:

```bash
python rag.py index                          # one-time: chunk + embed the corpus
python rag.py search "multi-head attention"  # retrieval only, shows BM25/cosine/combined
python rag.py ask "What is the RNN Encoder - Decoder used for?"
python rag.py ask "..." --model azure.gpt-4.1-mini -k 8   # any chat model on the endpoint
```

The index lives in `data/rag/` (gitignored; rebuild anytime with
`python rag.py index`).

## GraphRAG and comparison UI

The Neo4j-backed GraphRAG implementation and its Streamlit interface live in
`graph_rag/`:

- `graph_rag/load_neo4j.py` loads extracted or gold contribution triples.
- `graph_rag/qa_neo4j.py` translates questions to Cypher and summarizes the
  graph results.
- `graph_rag/app.py` runs GraphRAG and plain RAG for the same question and
  displays both answers side by side, including the relevant Neo4j evidence
  subgraph, generated Cypher, and retrieved text passages.

Configure Neo4j in `.env` in addition to the KIT API variables:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

Prepare both retrieval backends once, then launch the comparison UI:

```bash
python -m graph_rag.load_neo4j --predictions data/ncg/trial-data --clear
python rag.py index
streamlit run graph_rag/app.py
```

By default, the plain-RAG index is read from `data/rag/` and its top five
passages are used. Override these with `RAG_INDEX_PATH` and `RAG_TOP_K`.
