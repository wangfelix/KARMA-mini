# GraphRAG — Cypher Generation Skill

This document teaches an LLM how to translate a natural-language question
about the KARMA Mini contribution knowledge graph into a correct Cypher
query. It is loaded verbatim into RAGthe system prompt before every query.

## Graph structure

- `(:Paper {paper_id: string})`
- `(:Entity {name: string})` — `name` is a verbatim phrase copied from a
  paper's text. Two kinds: STRUCTURAL nodes (Contribution, Model, Results,
  Approach, ...) also carry `paper_id` and are scoped to one paper each.
  CONTENT nodes (real extracted terms) have NO `paper_id` and are shared
  globally across papers whenever the exact phrase matches.
- `(:Paper)-[:HAS_ROOT]->(:Entity {name: "Contribution"})` — every paper
  has exactly one root entity literally named `"Contribution"`.
- `(:Entity)-[<relationship type>]->(:Entity)` with three properties on
  every relationship: `paper_id` (which paper this specific fact comes
  from — always present, this is the reliable source of truth),
  `predicate_text` (the original free-text predicate), and `info_unit`
  (the category this triple belongs to, e.g. `"research-problem"`,
  `"results"`, `"model"`, `"approach"`, `"experiments"`, `"baselines"`,
  `"dataset"`). The exact relationship type name (`HAS`,
  `HAS_RESEARCH_PROBLEM`, ...) is derived from the predicate text and
  varies per triple — do not assume a fixed small set.

Content entity nodes are shared globally: the same real-world concept
mentioned with the exact same phrase in multiple papers is ONE node,
reachable via relationships from all of those papers — this is what lets
you traverse from one paper to related papers through a shared concept.
Slightly different verbatim text (e.g. "LSTM" vs "an LSTM network") still
creates separate nodes, so use fuzzy matching to catch variants.

## Critical rules

1. **Entity nodes are of two kinds** (see the live schema block above for
   the exact structural name list): STRUCTURAL nodes (Contribution, Model,
   Results, Approach, ...) carry a `paper_id` property and are scoped to
   one paper. CONTENT nodes (actual extracted terms, e.g. "LSTM") are
   GLOBAL/SHARED across papers by exact name and have NO `paper_id`
   property at all. **Every relationship, however, always has a
   `paper_id` property** — this is the one reliable place to find which
   paper a fact belongs to, regardless of node type.

   To scope a question to one paper, prefer filtering on the
   relationship's paper_id:
   ```cypher
   MATCH (a:Entity)-[r]->(b:Entity)
   WHERE r.paper_id = "machine-translation/0"
   ```
   Filtering `a.paper_id = "..."` also works IF `a` happens to be a
   structural node (e.g. the paper's own "Contribution" or "Model" node),
   but do NOT rely on `.paper_id` existing on content entities — it won't.

   Never search for the paper id inside `Entity.name`.

2. **To find a specific category of information** (research problem, model,
   dataset, results, baselines, ...), filter on the relationship property
   `r.info_unit` — never on entity name text (the category name never
   appears inside `Entity.name`). **Always use the exact spelling/casing
   listed under "info_unit values in use" in the live schema block that
   precedes this document** — do not assume a naming convention (they are
   lowercase and hyphenated, e.g. `"research-problem"`, not
   `"RESEARCHPROBLEM"`).

   Standard pattern for "what is the `<category>` of `<paper_id>`":
   ```cypher
   MATCH (a:Entity)-[r]->(b:Entity)
   WHERE r.paper_id = "machine-translation/0"
     AND r.info_unit = "research-problem"
   RETURN a.name, r.predicate_text, b.name
   LIMIT 25
   ```
   This direct property-filter pattern is more reliable than walking the
   exact `HAS_ROOT` backbone structure — prefer it.

3. **Never AND multiple keywords onto one entity's name.** The graph is
   fine-grained: a fact is usually spread across a short chain of small
   entities, not packed into one entity. Match on ONE distinctive
   keyword, then explore its 1-2 hop neighborhood instead of filtering
   everything in one compound `WHERE e.name CONTAINS X AND e.name
   CONTAINS Y` clause.

4. **Never invent a relationship type name.** `REL_TYPE` is not a real
   type — it does not exist in the database. If you don't know the exact
   type, use an untyped pattern instead: `-[r]-` or, for multi-hop,
   `-[r*1..3]-`. Always give a bound variable name to any relationship
   you reference later in `RETURN`/`WHERE` — an unbound variable is a
   syntax error.

5. **Use fuzzy matching for free text, exact matching for known
   properties.** `toLower(x) CONTAINS toLower(y)` for phrase text;
   exact `=` for `paper_id` and `info_unit` once you know the value.

6. **Always return `paper_id` and `predicate_text`** so results are
   traceable back to a specific paper and interpretable by a human.

7. **If the question is unrelated to this graph** (general world knowledge,
   e.g. "what is the capital of France") or genuinely cannot be answered by
   any Cypher query against this schema, respond with a short plain-English
   sentence saying so — do NOT attempt to write a Cypher query for it, and
   do NOT wrap it in Cypher syntax. A short natural-language refusal is
   correct in this one case only; every other answer must be pure Cypher.

8. **`model` and `approach` are closely related, overlapping categories.**
   A paper's method may be tagged under either one (not necessarily both).
   If a question asks "what model/method/approach does paper X use" and
   filtering on `r.info_unit = "model"` returns nothing, also try
   `r.info_unit = "approach"`, or match both at once:
   ```cypher
   WHERE r.info_unit IN ["model", "approach"]
   ```

9. **`paper_id` values are prefixed with their task category**
   (e.g. `machine-translation/0`, `named-entity-recognition/3`,
   `question-answering/5`, `relation-classification/8`,
   `text-classification/2`). For questions like "which papers are about
   named entity recognition" or "papers on question answering", prefer
   filtering directly on `paper_id` rather than searching for the task
   name inside `Entity.name` (papers rarely describe their own task by
   name in the extracted phrases):
   ```cypher
   MATCH (p:Paper)
   WHERE toLower(p.paper_id) CONTAINS toLower("named-entity-recognition")
   RETURN p.paper_id
   ```

10. **`LIMIT 25` caps ROWS, not distinct papers.** A single paper can
    produce several matching rows (multiple entity variants, multiple
    relationships), so a small LIMIT can silently truncate the paper list
    and make a "which papers..." answer incomplete/inconsistent with a
    separate `count()` query on the same topic. There are only 50 papers
    total in this graph, so for any question that lists or counts
    DISTINCT papers, use `LIMIT 100` (safely above the total) or omit
    LIMIT entirely — never use a small LIMIT like 25 for this kind of
    query. Small LIMITs are only appropriate when returning individual
    fact rows (e.g. "what is the research problem of paper X").

11. **"Papers with BOTH X and Y" (conjunction) must be checked within the
    SAME paper_id — do NOT use a "shared neighbor entity" pattern for
    this.** Two entities merely sharing a graph neighbor does NOT mean
    they come from the same paper (content entities are shared globally,
    so a neighbor can easily be reached from two different papers). The
    "shared neighbor" broadening trick is fine for pure relatedness
    questions, but for explicit AND/"both" questions it silently changes
    the meaning and can produce a WRONG answer that looks plausible.
    Correct pattern — resolve paper_id for each condition separately,
    then require equality:
    ```cypher
    MATCH (e1:Entity)-[r1]-(x)
    WHERE toLower(e1.name) CONTAINS toLower("attention")
    WITH DISTINCT r1.paper_id AS pid
    MATCH (e2:Entity)-[r2]-(y)
    WHERE toLower(e2.name) CONTAINS toLower("LSTM") AND r2.paper_id = pid
    RETURN DISTINCT pid
    ```

12. **"Papers that use the same `<category>` as paper P" is a two-step
    lookup — do NOT navigate through `HAS_ROOT` and assume a fixed hop
    count.** Backbone depth (how many hops from Contribution to the
    actual content) varies by info_unit and is not something to guess.
    Instead, use the direct `r.paper_id` + `r.info_unit` filter from rule
    2 for BOTH steps:
    ```cypher
    -- Step 1: what value does paper P have for this category?
    MATCH (a:Entity)-[r]->(v:Entity)
    WHERE r.paper_id = "machine-translation/3" AND r.info_unit = "dataset"
    WITH DISTINCT v.name AS value
    -- Step 2: which OTHER papers share that same value?
    MATCH (v2:Entity {name: value})<-[r2]-()
    WHERE r2.info_unit = "dataset" AND r2.paper_id <> "machine-translation/3"
    RETURN DISTINCT r2.paper_id
    ```
    If step 1 returns nothing, that paper simply may not have a triple in
    that exact category (e.g. check "approach" too per rule 8) — report
    that plainly rather than forcing a broadened but meaningless match.

13. **Aggregate answers must also return their supporting evidence.** The
    Streamlit UI visualizes the relevant subgraph using two stable aliases:
    `paper_ids` and `entity_names`. For count questions, return the numeric
    aggregate AND collect the matching paper ids and entity names. Do not
    return only a bare count. Example:
    ```cypher
    MATCH (e:Entity)-[r]-(neighbor:Entity)
    WHERE toLower(e.name) CONTAINS toLower("LSTM")
    RETURN count(DISTINCT r.paper_id) AS paper_count,
           collect(DISTINCT r.paper_id) AS paper_ids,
           collect(DISTINCT e.name) AS entity_names
    ```
    For a global paper count where no entity was matched, still return
    `collect(DISTINCT p.paper_id) AS paper_ids` alongside `count(p)`.

## Examples

**Q: "what is the research problem of machine-translation/0"**
```cypher
MATCH (a:Entity)-[r]->(b:Entity)
WHERE r.paper_id = "machine-translation/0"
  AND r.info_unit = "research-problem"
RETURN a.name, r.predicate_text, b.name
LIMIT 25
```

**Q: "what is a decoder modified to be composed of"**
```cypher
MATCH (e:Entity)-[r]-(neighbor:Entity)
WHERE toLower(e.name) CONTAINS toLower("decoder")
RETURN r.paper_id, e.name, type(r), r.predicate_text, neighbor.name
LIMIT 25
```

**Q: "which papers have contributed to LSTM"**
```cypher
MATCH (e:Entity)-[r]-(neighbor:Entity)
WHERE toLower(e.name) CONTAINS toLower("LSTM")
RETURN DISTINCT r.paper_id, e.name
LIMIT 100
```

## Anti-examples (do NOT do these)

```cypher
-- WRONG: compound AND on one entity's name
MATCH (e:Entity)
WHERE toLower(e.name) CONTAINS toLower("decoder")
  AND toLower(e.name) CONTAINS toLower("N=6")
  AND toLower(e.name) CONTAINS toLower("stacks")
RETURN e

-- WRONG: paper id searched inside entity name
MATCH (e:Entity)
WHERE toLower(e.name) CONTAINS toLower("machine-translation/0")
RETURN e

-- WRONG: category name searched inside entity name
MATCH (e:Entity)
WHERE toLower(e.name) CONTAINS toLower("research problem")
RETURN e

-- WRONG: invented relationship type
MATCH (e:Entity)-[:REL_TYPE]->(x)
RETURN e, x

-- WRONG: type(r) on a variable-length path -- r is a LIST of relationships
-- there (0/1/2+ hops), not a single relationship, so type(r) fails
MATCH (a)-[r*1..2]-(b)
RETURN type(r)
-- if you need the type, don't use a variable-length path, or don't call
-- type()/predicate_text on the list variable at all
```
