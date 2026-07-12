"""
Simple NLQ -> Cypher -> Neo4j -> natural-language-answer QA system 



Usage:
    python qa_system.py --uri bolt://localhost:7687 --user neo4j --password kdseminar26ss

    You are dropped into an interactive prompt:
      > which papers have contributed to LSTM
      > what is the research problem of machine-translation/0
      > exit
"""

import argparse
import os
import re

from neo4j import GraphDatabase

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Same client setup as main.py: OpenAI-compatible client against the KIT
# endpoint, same model roster, so this script and the extraction pipeline
# share one config.
# ---------------------------------------------------------------------------
from openai import OpenAI

AVAILABLE_MODELS = [
    "kit.gemma4-31b-it",
    "kit.gpt-oss-120b",
    "kit.minimax-m2.5-229b",
    "kit.minimax-m2.7-229b",
    "kit.mistral-small-4-119b-a8b",
]


def make_llm_client(timeout: float):
    api_key = os.getenv("KIT_API_KEY")
    base_url = os.getenv("KIT_BASE_URL")
    if not api_key or not base_url:
        raise SystemExit(
            "[ERROR] KIT_API_KEY or KIT_BASE_URL environment variable is not set!"
        )
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def call_llm(client, model_name, system_prompt, user_prompt, _retry=True):
    resp = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = resp.choices[0].message.content
    finish_reason = resp.choices[0].finish_reason

    if content is None:
        print(f"[warning] LLM returned empty content (finish_reason={finish_reason})")
        if _retry:
            print("[retrying LLM call once]")
            return call_llm(client, model_name, system_prompt, user_prompt, _retry=False)
        raise RuntimeError(
            f"LLM returned empty content twice in a row (finish_reason={finish_reason}). "
            "This is likely a transient API issue -- try again in a moment."
        )

    return content.strip()


# ---------------------------------------------------------------------------
# Schema introspection: give the LLM a live, accurate picture of the graph
# instead of a hand-written (and possibly stale) description.
# ---------------------------------------------------------------------------

def get_schema_text(driver):
    with driver.session() as session:
        rel_types = [r["relationshipType"] for r in session.run(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        )]
        info_units = [r["iu"] for r in session.run(
            "MATCH ()-[r]->() WHERE r.info_unit IS NOT NULL "
            "RETURN DISTINCT r.info_unit AS iu"
        )]
        n_papers = session.run("MATCH (p:Paper) RETURN count(p) AS c").single()["c"]
        sample_names = [r["name"] for r in session.run(
            "MATCH (e:Entity) RETURN DISTINCT e.name AS name LIMIT 15"
        )]

    return f"""
Graph schema:
- (:Paper {{paper_id: string}})
- (:Entity {{name: string}}) -- name is a verbatim phrase from a paper.
  TWO KINDS of Entity nodes:
  * STRUCTURAL nodes (Contribution, Model, Approach, Dataset, Results,
    Baselines, Hyperparameters, "Experimental Setup", Tasks, Experiments,
    "Ablation Analysis", Code) -- these ALSO have a paper_id property and
    are scoped to one paper each (not shared across papers).
  * CONTENT nodes (everything else, e.g. "LSTM", "RNN Encoder-Decoder") --
    these have NO paper_id property. They are GLOBAL/SHARED: if the exact
    same phrase appears in multiple papers, it is the SAME node, reachable
    from all of those papers' relationships. This is what lets you answer
    "which papers mention X" by looking at who points to that one node.
- (:Paper)-[:HAS_ROOT]->(:Entity {{name: "Contribution"}})
- (:Entity)-[<one of the concrete relationship types listed below>
             {{paper_id: string, predicate_text: string, info_unit: string}}]->(:Entity)
  Every relationship ALWAYS has a paper_id property -- this is the reliable
  way to know which paper a given edge belongs to, regardless of whether
  its endpoint nodes are structural (paper-scoped) or content (shared).

The ACTUAL relationship type names in this database are (pick one of these
literal strings when you need a specific type -- "REL_TYPE" itself is NOT a
real type, it is just a placeholder name in this description):
  {", ".join(sorted(rel_types)) or "(none yet)"}

info_unit values in use: {", ".join(sorted(info_units)) or "(none yet)"}

There are {n_papers} papers loaded. Entity.name examples: {sample_names}

TIP: if you are not confident which specific relationship type applies,
prefer an untyped pattern -[r]- (matches any type, either direction) over
guessing a type name -- an untyped pattern is much safer than a wrong or
placeholder type name, which silently matches nothing.

IMPORTANT: content entities are now GLOBAL/SHARED across papers by exact
name (see "TWO KINDS of Entity nodes" above) -- but the same real-world
concept can still appear as slightly DIFFERENT verbatim text across papers
(e.g. "LSTM" vs "an LSTM network" are two different nodes because the exact
text differs). For cross-paper questions ("which papers mention/use/
contribute X"), match on Entity.name with case-insensitive substring
search, then read off paper_id from the connected RELATIONSHIPS (not from
the entity node itself, since a shared content node has no paper_id of its
own):
  MATCH (e:Entity)-[r]-(neighbor:Entity)
  WHERE toLower(e.name) CONTAINS toLower("X")
  RETURN DISTINCT r.paper_id, e.name
Do NOT use exact equality unless the user quotes an exact phrase.
Always return r.paper_id (and r.predicate_text where relevant) so results
can be traced back to a specific paper -- prefer r.paper_id over
e.paper_id, since e.paper_id only exists on structural nodes.
""".strip()


# ---------------------------------------------------------------------------
# NLQ -> Cypher -> results -> natural language answer
# ---------------------------------------------------------------------------

def load_skill(path=None):
    """Load the Cypher-generation instructions from an external markdown
    file so they can be edited without touching this script. Resolves the
    default path relative to THIS script's own location (not the current
    working directory), so it works no matter where the program is
    launched from -- important once teammates run this on their own
    machines. Falls back to a minimal built-in prompt if the file is
    missing."""
    if path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, "skill.md")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"[warning] {path} not found, using minimal built-in prompt")
        return (
            "You translate a natural-language question about a scholarly-"
            "paper contribution knowledge graph into a single Cypher query. "
            "Use toLower(...) CONTAINS toLower(...) for fuzzy text matching, "
            "filter paper_id and info_unit as exact properties (never search "
            "for them inside Entity.name), and never invent relationship "
            "type names."
        )


CYPHER_SYSTEM_PROMPT = (
    load_skill() + "\n\n"
    "Return ONLY the Cypher query itself, no explanation, no markdown fences."
)

ANSWER_SYSTEM_PROMPT = """You answer the user's question using ONLY the \
provided Cypher query results, in natural, conversational English -- like \
a knowledgeable research assistant explaining findings, not a database \
dump. Write full sentences that weave in the actual details from the \
results (the original predicate_text wording, the info_unit category, how \
entities connect) rather than just listing bare names or IDs. Give enough \
context that someone who hasn't seen the raw data would understand WHY \
the answer is true, not just WHAT it is. For example, instead of "Paper X: \
LSTM", say something like "Paper X uses an LSTM-based architecture, \
specifically describing it as '<predicate_text wording>'." If multiple \
papers or facts are involved, briefly note what's similar or different \
between them instead of just concatenating them.

Ground rules that still apply:
- Never state anything not directly supported by the provided results --
  no outside knowledge, no filling in gaps with assumptions.
- If results are empty, say so plainly instead of guessing.
- Always mention which paper(s) (paper_id) a specific claim comes from.
- For a pure aggregate answer (a single count/total with no per-paper
  breakdown), just state the number naturally -- don't force a paper_id
  citation onto it, but a sentence of context is still welcome (e.g. "27
  papers mention LSTM, spanning all five task categories.").
- If the results contain a list of items, you MUST include every single
  one -- never silently truncate, sample, or drop items, no matter how
  many there are. A natural-sounding answer does not mean a shorter one;
  weave every item in, grouped or summarized narratively if that helps
  readability, but nothing gets left out."""


def get_known_info_units(driver):
    with driver.session() as session:
        return [r["iu"] for r in session.run(
            "MATCH ()-[r]->() WHERE r.info_unit IS NOT NULL "
            "RETURN DISTINCT r.info_unit AS iu"
        )]


def _normalize_key(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def fix_info_unit_literals(cypher, known_info_units):
    """LLMs sometimes write info_unit filters using a plausible-looking but
    wrong format (e.g. "RESEARCHPROBLEM" instead of the real
    "research-problem"). Auto-correct any quoted string used in an
    `.info_unit = "..."` comparison to the real stored value whenever it
    normalizes (lowercase, letters/digits only) to the same thing."""
    lookup = {_normalize_key(v): v for v in known_info_units}

    pattern = re.compile(r'(\.info_unit\s*=\s*)(["\'])([^"\']*)\2')

    def repl(match):
        prefix, quote, value = match.group(1), match.group(2), match.group(3)
        key = _normalize_key(value)
        if key in lookup and lookup[key] != value:
            print(f"[auto-corrected info_unit: \"{value}\" -> \"{lookup[key]}\"]")
            return f"{prefix}{quote}{lookup[key]}{quote}"
        return match.group(0)

    return pattern.sub(repl, cypher)


def generate_cypher(client, model_name, schema_text, question, prior_error=None):
    prompt = f"{schema_text}\n\nQuestion: {question}\n\nCypher query:"
    if prior_error:
        prompt += (
            f"\n\n(Your previous query failed with this error, fix it:\n"
            f"{prior_error})"
        )
    cypher = call_llm(client, model_name, CYPHER_SYSTEM_PROMPT, prompt)
    return cypher.strip().strip("`").replace("cypher\n", "", 1)


def run_cypher(driver, cypher):
    with driver.session() as session:
        result = session.run(cypher)
        return [record.data() for record in result]


CYPHER_START_KEYWORDS = (
    "MATCH", "OPTIONAL", "WITH", "CALL", "UNWIND", "RETURN", "MERGE", "CREATE",
)


def looks_like_cypher(text):
    """Heuristic: does this response actually start like a Cypher query, or
    did the LLM ignore instructions and reply in plain English instead
    (e.g. explaining that the question is unrelated to the graph)?"""
    first_word = text.strip().split(None, 1)[0].upper() if text.strip() else ""
    return first_word in CYPHER_START_KEYWORDS


def safe_run_cypher(driver, cypher):
    """run_cypher that never raises -- returns (records, error_or_None)."""
    try:
        return run_cypher(driver, cypher), None
    except Exception as e:
        return None, e


def answer_question(driver, client, model_name, schema_text, question):
    """Returns a dict: {"answer": str, "cypher": str or None}.
    "cypher" is the actual query that was executed (or None if the LLM
    determined the question was out of scope and never ran a query at all)
    -- this is what a UI should display in a traceability/verification
    panel, instead of trying to scrape it back out of console print()
    output (which is never part of the return value)."""
    known_info_units = get_known_info_units(driver)

    cypher = generate_cypher(client, model_name, schema_text, question)
    cypher = fix_info_unit_literals(cypher, known_info_units)
    print(f"[cypher] {cypher}")

    # The LLM sometimes ignores "return only Cypher" and instead explains in
    # plain English that the question is out of scope (e.g. general trivia
    # unrelated to the paper graph). Treat that explanation as the answer
    # directly instead of trying to execute it as a query.
    if not looks_like_cypher(cypher):
        return {"answer": cypher, "cypher": None}

    records, err = safe_run_cypher(driver, cypher)

    if err is not None:
        print(f"[retry after error: {err}]")
        cypher = generate_cypher(client, model_name, schema_text, question, str(err))
        cypher = fix_info_unit_literals(cypher, known_info_units)
        print(f"[cypher retry] {cypher}")
        if not looks_like_cypher(cypher):
            return {"answer": cypher, "cypher": None}
        records, err = safe_run_cypher(driver, cypher)
        if err is not None:
            print(f"[retry also failed: {err}]")
            return {"answer": "No matching results found in the graph.", "cypher": cypher}

    if not records:
        # Query was valid but returned nothing -- likely too many/too strict
        # AND conditions on a single node. Retry once with an explicit nudge
        # to broaden the search instead of giving up immediately.
        print("[empty result, retrying with a broader query]")
        broaden_hint = (
            "Your previous query returned zero results:\n"
            f"{cypher}\n"
            "This is probably because it required too many conditions to "
            "match on a single entity, or filtered on the wrong property. "
            "Rewrite it: pick only the single most distinctive keyword from "
            "the question, match entities containing that one keyword, then "
            "return their 1-2 hop neighborhood (relationships and connected "
            "entities) instead of filtering everything at once. Make sure "
            "every relationship variable you reference in RETURN/WHERE is "
            "actually bound with a name in the MATCH pattern."
        )
        cypher = generate_cypher(client, model_name, schema_text, question, broaden_hint)
        cypher = fix_info_unit_literals(cypher, known_info_units)
        print(f"[cypher broadened] {cypher}")
        if not looks_like_cypher(cypher):
            return {"answer": cypher, "cypher": None}
        records, err = safe_run_cypher(driver, cypher)
        if err is not None:
            print(f"[broadened query also failed: {err}]")
            records = []
    if not records:
        return {"answer": "No matching results found in the graph.", "cypher": cypher}

    prompt = f"Question: {question}\n\nCypher results:\n{records}\n\nAnswer:"
    answer = call_llm(client, model_name, ANSWER_SYSTEM_PROMPT, prompt)
    return {"answer": answer, "cypher": cypher}


def main():
    ap = argparse.ArgumentParser(description="KARMA Mini Graph QA")
    ap.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    ap.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.getenv("NEO4J_PASSWORD"))
    ap.add_argument(
        "--model",
        choices=AVAILABLE_MODELS,
        default="kit.mistral-small-4-119b-a8b",
        help="Select the LLM model to query",
    )
    ap.add_argument(
        "--timeout", type=float, default=60.0,
        help="API request timeout in seconds (default: 60.0)",
    )
    args = ap.parse_args()

    if not args.password:
        raise SystemExit(
            "Neo4j password not set. Pass --password or set NEO4J_PASSWORD in .env"
        )

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    client = make_llm_client(args.timeout)
    schema_text = get_schema_text(driver)
    print(schema_text, "\n")
    print("Ask a question (or 'exit'):")

    try:
        while True:
            question = input("> ").strip()
            if question.lower() in ("exit", "quit"):
                break
            if not question:
                continue
            result = answer_question(driver, client, args.model, schema_text, question)
            print(result["answer"])
            print()
    finally:
        driver.close()


if __name__ == "__main__":
    main()