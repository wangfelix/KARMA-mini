"""Build a reproducible QA benchmark from the NCG gold corpus.

Generator version 3, open corpus. Every question is answerable only by
searching the whole 50-paper collection. No question names a paper, and no
paper identifier is supplied to the retrieval systems as metadata, so paper
selection is part of the task being measured rather than given away.

This follows the two-stage protocol used for template-based knowledge-graph QA
corpora: a deterministic generator produces an auditable draft, and human
annotators paraphrase and verify it afterwards. Records therefore stay marked
as awaiting that human pass.

Design constraints that matter for validity:

1. Fact, multi-hop, and paper-retrieval questions are drawn only from gold
   triples that can be verbalized. Roughly 40 percent of the gold annotation is
   structural scaffolding such as (Contribution || has || Model), which is
   identical in every paper and cannot discriminate between them. Most of the
   remainder has bare prepositions or sentence fragments as predicates.
2. The entity a question asks about must occur in exactly one paper of the
   collection. Without that constraint an open-corpus question has no single
   correct answer.
3. Cross-paper questions use a text-grounded reference. Gold graph membership
   is a strict subset of the papers that actually mention an entity, so a
   graph-derived reference marks a correct text answer wrong.

Known coverage gap: the corpus supports no relational multi-hop across papers.
NCG annotates each paper independently and never merges graphs, so the only
inter-paper links are incidental string collisions on generic terms.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


GENERATOR_VERSION = "v3-open-corpus"

STRUCTURAL_NAMES = {
    "contribution",
    "model",
    "approach",
    "dataset",
    "results",
    "baselines",
    "hyperparameters",
    "experimental setup",
    "experimental settings",
    "tasks",
    "experiments",
    "ablation analysis",
    "code",
}

STRUCTURAL_PREDICATES = {"has", "has research problem", "code"}

# Entities used for the cross-paper questions. Each occurs in at least two gold
# graphs and in between three and eight paper texts. The upper bound matters:
# Plain RAG retrieves eight chunks, so an entity appearing in thirty papers
# would make exhaustive enumeration impossible for a top-k retriever regardless
# of retrieval quality. Fixing this list keeps regeneration stable.
SHARED_ENTITIES = (
    "BERT",
    "BioBERT",
    "multi-head attention",
    "character embeddings",
    "GloVe embeddings",
    "ReLU activation",
    "MRR",
    "mini-batch size",
    "dropout rate",
    "natural language understanding",
)

# Plausible modern NLP terms verified absent from all fifty papers. They post-
# date the collection, so a system that answers rather than abstaining is
# drawing on parametric knowledge instead of the retrieved evidence.
ABSENT_ENTITIES = (
    "retrieval augmented generation",
    "chain of thought prompting",
    "LoRA",
    "reinforcement learning from human feedback",
    "FlashAttention",
    "rotary position embedding",
    "speculative decoding",
    "direct preference optimization",
    "instruction tuning",
    "vision transformer",
)

# Predicates that read as a relation rather than as a fragment of a sentence.
VERB_PREDICATE = re.compile(
    r"^(is|are|was|were|has|have|uses?|used|using|trains?|trained|"
    r"outperforms?|proposes?|proposed|presents?|achieves?|achieved|improves?|"
    r"improved|introduces?|introduced|leads? to|based on|consists? of|"
    r"evaluated on|applied to|performs?|provides?|requires?|contains?|"
    r"includes?|employs?|adopts?|extends?|reports?|obtains?|yields?|"
    r"produces?|combines?|learns?|predicts?|generates?|considers?|considered|"
    r"treats?|models?|encodes?|represents?|compares?|reduces?|increases?)\b",
    re.IGNORECASE,
)

# Openings that signal the span is a clause fragment, not a noun phrase.
FRAGMENT_START = re.compile(
    r"^(when|why|how|what|which|and|or|but|that|this|these|those|it|we|our)\b",
    re.IGNORECASE,
)

COPULA_PREDICATES = {"is", "are", "was", "were"}

# Edges that introduce an alias for the subject rather than a distinct entity.
ALIAS_PREDICATES = {"has acronym", "have acronym", "is", "are",
                    "refer to as", "referred to as", "name"}

# Entities opening with these read as generic references ("all previous
# models", "both datasets") rather than as a thing a reader could ask about.
GENERIC_ENTITY_HEAD = {
    "all", "both", "each", "every", "higher", "lower", "baseline", "other",
    "others", "previous", "same", "such", "these", "those", "many", "most",
    "several", "various", "different", "single", "new", "our", "their",
}

# Predicate heads that read as past participles and therefore take "is" rather
# than "does" when the triple is turned into a question.
PARTICIPLE_HEADS = {
    "based", "trained", "applied", "evaluated", "used", "proposed",
    "introduced", "considered", "improved", "achieved", "obtained",
    "produced", "combined", "learned", "generated", "encoded",
    "represented", "compared", "reduced", "increased", "pretrained",
}

PREFERRED_UNITS = {"model", "approach", "results", "tasks", "experiments"}


@dataclass(frozen=True)
class Triple:
    paper_id: str
    info_unit: str
    subject: str
    predicate: str
    object: str
    source_file: str

    @property
    def text(self) -> str:
        return f"({self.subject}||{self.predicate}||{self.object})"

    def evidence_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "info_unit": self.info_unit,
            "triple": self.text,
            "source_file": self.source_file,
        }


def parse_triple(line: str) -> tuple[str, str, str] | None:
    line = line.strip()
    if not line:
        return None
    if line.startswith("(") and line.endswith(")"):
        line = line[1:-1]
    parts = tuple(part.strip() for part in line.split("||"))
    if len(parts) != 3 or not all(parts):
        return None
    return parts


def load_triples(corpus_root: Path) -> list[Triple]:
    triples: list[Triple] = []
    pattern = "*/*/triples/*.txt"
    for path in sorted(corpus_root.glob(pattern)):
        task = path.parents[2].name
        paper_number = path.parents[1].name
        paper_id = f"{task}/{paper_number}"
        for line in path.read_text(encoding="utf-8").splitlines():
            parsed = parse_triple(line)
            if parsed is None:
                continue
            triples.append(
                Triple(
                    paper_id=paper_id,
                    info_unit=path.stem,
                    subject=parsed[0],
                    predicate=parsed[1],
                    object=parsed[2],
                    source_file=str(path),
                )
            )
    if not triples:
        raise RuntimeError(f"No triples found below {corpus_root}")
    return triples


def tokens(text: str) -> list[str]:
    """Tokenize exactly as evaluation.score does, so matching stays consistent."""
    return re.findall(r"[a-z0-9]+", text.lower())


def contains_phrase(haystack_tokens: list[str], phrase: str) -> bool:
    """True when the phrase occurs as a contiguous token run.

    Token matching absorbs Stanza spacing (``multi - head`` versus
    ``multi-head``) while still respecting word boundaries, so ``BERT`` does
    not match inside ``RoBERTa``.
    """
    needle = tokens(phrase)
    if not needle:
        return False
    span = len(needle)
    for start in range(len(haystack_tokens) - span + 1):
        if haystack_tokens[start:start + span] == needle:
            return True
    return False


def is_noun_phrase(span: str, max_length: int) -> bool:
    span = span.strip()
    if not 3 <= len(span) <= max_length:
        return False
    if FRAGMENT_START.match(span):
        return False
    if span.count(",") > 1:
        return False
    if span.strip().lower() in STRUCTURAL_NAMES:
        return False
    return bool(re.search(r"[A-Za-z]{3}", span))


def is_verbalizable(triple: Triple) -> bool:
    """Whether this triple can become a question with a coherent reading."""
    predicate = triple.predicate.strip()
    if predicate.lower() in STRUCTURAL_PREDICATES:
        return False
    if not VERB_PREDICATE.match(predicate):
        return False
    return is_noun_phrase(triple.subject, 80) and is_noun_phrase(triple.object, 90)


def _is_plural(subject: str) -> bool:
    """Rough plural test on the head noun, used only for verb agreement."""
    words = re.findall(r"[A-Za-z0-9]+", subject)
    if not words:
        return False
    last = words[-1].lower()
    if last.endswith(("ss", "us", "is", "as")):
        return False
    return last.endswith("s") and len(last) > 3


def _third_person(verb: str) -> str:
    """Inflect a bare verb for a singular subject, inverting _base_form."""
    lowered = verb.lower()
    if lowered in {"have"}:
        return "has"
    # Participles are already inflected; adding "s" produces "useds".
    if lowered.endswith(("ed", "ing")):
        return verb
    if lowered.endswith(("s", "ch", "sh", "x", "z")):
        return verb if lowered.endswith("s") else verb + "es"
    return verb + "s"


def _base_form(verb: str) -> str:
    """Strip third-person singular inflection so it can follow \"does\"."""
    lowered = verb.lower()
    if lowered == "has":
        return "have"
    if lowered.endswith("sses") or lowered.endswith("ss"):
        return verb
    if lowered.endswith("es") and lowered[:-2].endswith(("ch", "sh", "x", "z")):
        return verb[:-2]
    if lowered.endswith("s"):
        return verb[:-1]
    return verb


def _base_question(category: str, question: str, answer_type: str,
                   reference_answer: str, reference_items: list,
                   evidence: Iterable[Triple], scope_paper_ids: list[str],
                   **extra) -> dict:
    """Assemble one benchmark record.

    ``scope_paper_ids`` is gold provenance for scoring only. It is deliberately
    not passed to any retrieval backend: which papers answer the question is
    part of what the experiment measures.
    """
    record = {
        "category": category,
        "question": question,
        "answer_type": answer_type,
        "reference_answer": reference_answer,
        "reference_items": reference_items,
        "evidence": [triple.evidence_dict() for triple in evidence],
        "scope_paper_ids": scope_paper_ids,
        "source": "SemEval-2021 Task 11 NLPContributionGraph gold triples and Stanza text",
        "generator_version": GENERATOR_VERSION,
        "review_status": "team_accepted_no_expert_review",
        "paraphrase_status": "not_paraphrased",
    }
    record.update(extra)
    return record


def ask_relation(subject: str, predicate: str) -> str:
    """Render a subject-predicate pair as a grammatical open-corpus question.

    NCG predicates are verbatim sentence wording, so they arrive as finite
    verbs, past participles, or copulas. Each needs a different frame.
    """
    predicate = predicate.strip()
    normalized = predicate.lower()
    if normalized in {"has acronym", "have acronym"}:
        return f"What is the acronym for {subject}?"

    head, _, rest = predicate.partition(" ")
    rest = f" {rest}" if rest else ""
    if head.lower() in COPULA_PREDICATES:
        copula = "are" if _is_plural(subject) and head.lower() == "is" else predicate
        return f"What {copula} {subject}?"
    if (head.lower() in PARTICIPLE_HEADS
            or head.lower().endswith(("ed", "ing"))):
        verb = "are" if _is_plural(subject) else "is"
        return f"What {verb} {subject} {predicate}?"
    auxiliary = "do" if _is_plural(subject) else "does"
    return f"What {auxiliary} {subject} {_base_form(head)}{rest}?"


class Corpus:
    """Paper texts plus the corpus-uniqueness lookup the questions depend on."""

    def __init__(self, corpus_root: Path, paper_ids: Iterable[str]):
        self.paper_ids = sorted(paper_ids)
        self.tokens: dict[str, list[str]] = {}
        for paper_id in self.paper_ids:
            task, number = paper_id.split("/", 1)
            files = sorted((corpus_root / task / number).glob("*-Stanza-out.txt"))
            if len(files) != 1:
                raise RuntimeError(f"Expected one Stanza file for {paper_id}")
            self.tokens[paper_id] = tokens(files[0].read_text(encoding="utf-8"))
        self._cache: dict[str, list[str]] = {}

    def mentioning(self, phrase: str) -> list[str]:
        """Papers whose text contains the phrase, as a contiguous token run."""
        if phrase not in self._cache:
            self._cache[phrase] = [
                paper_id for paper_id in self.paper_ids
                if contains_phrase(self.tokens[paper_id], phrase)
            ]
        return self._cache[phrase]

    def unique_to(self, phrase: str, paper_id: str) -> bool:
        """True when exactly one paper mentions the phrase, and it is this one."""
        return self.mentioning(phrase) == [paper_id]


def _paper_retrieval_questions(triples: list[Triple], corpus: Corpus,
                               wanted: int) -> list[dict]:
    """"Which paper addresses X?" where X identifies exactly one paper.

    Built from the research-problem edge, but the question uses only its object.
    The structural subject and predicate never appear.
    """
    problems: dict[str, list[Triple]] = defaultdict(list)
    for triple in triples:
        if (triple.subject == "Contribution"
                and triple.predicate.lower() == "has research problem"):
            problems[triple.object].append(triple)

    candidates = []
    for problem, evidence in sorted(problems.items()):
        owners = {triple.paper_id for triple in evidence}
        if len(owners) != 1 or not 3 <= len(problem) <= 70:
            continue
        owner = next(iter(owners))
        if not corpus.unique_to(problem, owner):
            continue
        candidates.append((owner, problem, evidence))

    questions, used = [], defaultdict(int)
    # One question per paper first, so the category spans as many papers as
    # possible before any paper contributes a second.
    for limit in (1, 2, 3):
        for owner, problem, evidence in candidates:
            if len(questions) == wanted:
                break
            if used[owner] >= limit:
                continue
            used[owner] += 1
            questions.append(_base_question(
                category="paper_retrieval",
                question=f"Which paper addresses {problem}?",
                answer_type="paper_id_set",
                reference_answer=owner,
                reference_items=[owner],
                evidence=evidence,
                scope_paper_ids=[owner],
            ))
        if len(questions) == wanted:
            break
    return questions


def _fact_rank(triple: Triple) -> tuple:
    return (
        -int(triple.info_unit in PREFERRED_UNITS),
        len(triple.subject) + len(triple.object),
        triple.text,
    )


def _fact_questions(gated: list[Triple], corpus: Corpus, wanted: int) -> list[dict]:
    """One gold triple per question; the object is the reference answer."""
    candidates = [
        triple for triple in sorted(gated, key=_fact_rank)
        if corpus.unique_to(triple.subject, triple.paper_id)
    ]
    questions, used, seen = [], defaultdict(int), set()
    for limit in (1, 2, 3, 4):
        for triple in candidates:
            if len(questions) == wanted:
                break
            key = (triple.subject.lower(), triple.predicate.lower())
            if key in seen or used[triple.paper_id] >= limit:
                continue
            seen.add(key)
            used[triple.paper_id] += 1
            questions.append(_base_question(
                category="single_fact",
                question=ask_relation(triple.subject, triple.predicate),
                answer_type="short_answer",
                reference_answer=triple.object,
                reference_items=[triple.object],
                evidence=[triple],
                scope_paper_ids=[triple.paper_id],
            ))
        if len(questions) == wanted:
            break
    return questions


def _multi_hop_questions(gated: list[Triple], corpus: Corpus,
                         wanted: int) -> list[dict]:
    """Two chained gold triples where the linking entity is withheld.

    The question states only the entry point and both relations. A system must
    resolve the first hop to discover where the second one lives.
    """
    by_paper: dict[str, list[Triple]] = defaultdict(list)
    for triple in gated:
        by_paper[triple.paper_id].append(triple)

    candidates = []
    for paper_id in sorted(by_paper):
        outgoing: dict[str, list[Triple]] = defaultdict(list)
        for triple in by_paper[paper_id]:
            outgoing[triple.subject].append(triple)
        for first in by_paper[paper_id]:
            if not corpus.unique_to(first.subject, paper_id):
                continue
            if ":" in first.subject:
                continue
            for second in outgoing.get(first.object, []):
                # A copula second hop produces "what is it is?"; negations
                # produce "what does it were not?". Neither is a question.
                if second.predicate.split()[0].lower() in COPULA_PREDICATES:
                    continue
                if second.object in {first.subject, first.object}:
                    continue
                candidates.append((first, second))

    candidates.sort(key=lambda pair: (
        -int(pair[0].info_unit in PREFERRED_UNITS),
        len(pair[0].subject) + len(pair[1].object),
        pair[0].text + pair[1].text,
    ))

    questions, used, seen = [], defaultdict(int), set()
    for first, second in candidates:
        if len(questions) == wanted:
            break
        key = (first.subject.lower(), second.object.lower())
        if key in seen or used[first.paper_id] >= 2:
            continue
        seen.add(key)
        used[first.paper_id] += 1
        # An acronym edge is an alias rather than a step to a different
        # entity, so naming the intermediate would be redundant and the
        # two-clause form reads as broken English ("X has acronym
        # something"). Collapsing onto the subject keeps the question
        # natural while still requiring the alias to be resolved.
        if first.predicate.strip().lower() in ALIAS_PREDICATES:
            question = ask_relation(first.subject, second.predicate)
        else:
            head, _, rest = first.predicate.strip().partition(" ")
            rest = f" {rest}" if rest else ""
            verb = head if _is_plural(first.subject) else _third_person(head)
            question = (
                f"{first.subject} {verb}{rest} something. "
                f"{ask_relation('it', second.predicate)}"
            )
        questions.append(_base_question(
            category="multi_hop",
            question=question,
            answer_type="short_answer",
            reference_answer=second.object,
            reference_items=[second.object],
            evidence=[first, second],
            scope_paper_ids=[first.paper_id],
        ))
    return questions


def _entity_occurrences(triples: list[Triple]) -> dict[str, list[Triple]]:
    occurrences: dict[str, list[Triple]] = defaultdict(list)
    for triple in triples:
        for entity in {triple.subject, triple.object}:
            if entity.strip().lower() not in STRUCTURAL_NAMES:
                occurrences[entity].append(triple)
    return occurrences


def _cross_paper_questions(triples: list[Triple], corpus: Corpus,
                           list_wanted: int, count_wanted: int) -> list[dict]:
    """Aggregation over the collection, with a text-grounded reference.

    The reference counts every paper whose text mentions the entity, not every
    paper whose gold graph records it. Graph membership is a strict subset, so
    a graph-derived reference would mark a correct text answer wrong.
    """
    listed, counted = [], []
    occurrences = _entity_occurrences(triples)
    for entity in SHARED_ENTITIES:
        graph_evidence = occurrences.get(entity, [])
        graph_papers = sorted({triple.paper_id for triple in graph_evidence})
        if len(graph_papers) < 2:
            raise RuntimeError(f"Expected {entity!r} in at least two graphs")
        text_papers = corpus.mentioning(entity)
        if not 2 <= len(text_papers) <= 8:
            raise RuntimeError(
                f"{entity!r} spans {len(text_papers)} papers; outside the "
                "range a top-k retriever can enumerate"
            )
        extra = {
            "entity": entity,
            "entity_reference_basis": "text",
            "graph_paper_count": len(graph_papers),
            "text_paper_count": len(text_papers),
        }
        listed.append(_base_question(
            category="cross_paper_list",
            question=f"Which papers in this collection mention {entity}?",
            answer_type="paper_id_set",
            reference_answer="; ".join(text_papers),
            reference_items=text_papers,
            evidence=graph_evidence,
            scope_paper_ids=text_papers,
            **extra,
        ))
        counted.append(_base_question(
            category="cross_paper_count",
            question=f"How many papers in this collection mention {entity}?",
            answer_type="integer",
            reference_answer=str(len(text_papers)),
            reference_items=[len(text_papers)],
            evidence=graph_evidence,
            scope_paper_ids=text_papers,
            **extra,
        ))
    return listed[:list_wanted] + counted[:count_wanted]


# Terms that describe an outcome or a generic property rather than naming a
# thing. They make poor answer-key items because no fluent answer lists them.
UNINFORMATIVE_TERMS = {
    "weights", "training", "performance", "results", "each task", "our models",
    "higher scores", "better", "softmax", "vectors", "attention", "jointly",
    "state - of - the - art performance", "new state - of - the - art performance",
    "input sentence", "entity recognition", "sentences", "further analysis",
}


def _is_named_entity(entity: str) -> bool:
    """Whether the entity names a specific model, method, metric, or dataset.

    Profile questions ask what a collection reports about a thing. That only
    reads as a question when the thing has a name, so bare descriptions such
    as "larger datasets" or "comparable performance" are rejected even though
    they occur in several graphs.
    """
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", entity)
    if not words:
        return False
    if words[0].lower() in GENERIC_ENTITY_HEAD:
        return False
    if entity.strip().lower() in UNINFORMATIVE_TERMS:
        return False
    # An acronym, or an internal capital, marks a named artefact.
    return any(
        word.isupper() and len(word) >= 2 for word in words
    ) or any(
        word[0].isupper() for word in words
    )


def _is_salient_item(term: str) -> bool:
    """Whether a reference term is specific enough to expect in an answer."""
    stripped = term.strip()
    if len(stripped) < 4 or stripped.lower() in UNINFORMATIVE_TERMS:
        return False
    words = stripped.split()
    if words[0].lower() in GENERIC_ENTITY_HEAD:
        return False
    named = bool(re.search(r"[A-Z][A-Za-z0-9-]*", stripped))
    return named or len(words) >= 3


def _is_readable_term(term: str) -> bool:
    """Reject spans whose Unicode was lost in the distributed Stanza text.

    Greek letters and subscripts arrive as literal question marks, so items
    like "? 1 = 0.9 , ? 2 = 0.98" carry no recoverable meaning and cannot be
    matched against a generated answer.
    """
    if "?" in term:
        return False
    letters = sum(character.isalpha() for character in term)
    return letters >= 3 and letters >= len(term) * 0.5


def _entity_profile_questions(triples: list[Triple], corpus: Corpus,
                              wanted: int) -> list[dict]:
    """"What is X?" where X is described across several papers.

    This is the category the graph should suit best: one entity node carries
    edges contributed by different papers, so a single lookup gathers facts
    that plain retrieval has to assemble from separate documents.
    """
    related: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    evidence: dict[str, list[Triple]] = defaultdict(list)
    for triple in triples:
        subject, obj = triple.subject.strip(), triple.object.strip()
        if subject.lower() in STRUCTURAL_NAMES or obj.lower() in STRUCTURAL_NAMES:
            continue
        if triple.predicate.strip().lower() in STRUCTURAL_PREDICATES:
            continue
        if 2 <= len(subject) <= 45 and 2 <= len(obj) <= 60:
            related[subject][triple.paper_id].add(obj)
            evidence[subject].append(triple)
        if 2 <= len(obj) <= 45 and 2 <= len(subject) <= 60:
            related[obj][triple.paper_id].add(subject)
            evidence[obj].append(triple)

    candidates = []
    for entity, by_paper in related.items():
        if not _is_named_entity(entity):
            continue
        by_paper = {
            paper_id: {
                term for term in terms
                if _is_readable_term(term) and _is_salient_item(term)
            }
            for paper_id, terms in by_paper.items()
        }
        by_paper = {k: v for k, v in by_paper.items() if v}
        if len(by_paper) < 2:
            continue
        if len(by_paper) < 2:
            continue
        head = re.findall(r"[A-Za-z0-9]+", entity)
        if not head or head[0].lower() in GENERIC_ENTITY_HEAD:
            continue
        mentions = corpus.mentioning(entity)
        if not 2 <= len(mentions) <= 12:
            continue
        items = sorted({term for terms in by_paper.values() for term in terms})
        if not 3 <= len(items) <= 8:
            continue
        candidates.append((len(by_paper), len(items), entity, items))

    candidates.sort(key=lambda c: (-c[0], -c[1], c[2].lower()))
    questions = []
    for paper_count, _, entity, items in candidates[:wanted]:
        questions.append(_base_question(
            category="cross_paper_profile",
            question=(
                f"What do the papers in this collection report about {entity}?"
            ),
            answer_type="list",
            reference_answer="; ".join(items),
            reference_items=items,
            evidence=evidence[entity],
            scope_paper_ids=sorted(related[entity]),
            entity=entity,
            contributing_paper_count=paper_count,
        ))
    return questions


def _unanswerable_questions(corpus: Corpus, wanted: int) -> list[dict]:
    """Terms absent from every paper. The only correct response is abstention."""
    questions = []
    for entity in ABSENT_ENTITIES[:wanted]:
        present = corpus.mentioning(entity)
        if present:
            raise RuntimeError(f"{entity!r} is present in {present}")
        questions.append(_base_question(
            category="unanswerable",
            question=f"What do the papers in this collection report about {entity}?",
            answer_type="unanswerable",
            reference_answer="The available evidence does not contain the answer.",
            reference_items=[],
            evidence=[],
            scope_paper_ids=[],
        ))
    return questions


TARGET_COUNTS = {
    "paper_retrieval": 15,
    "single_fact": 15,
    "multi_hop": 10,
    "cross_paper_profile": 15,
    "cross_paper_list": 10,
    "cross_paper_count": 5,
    "unanswerable": 10,
}

DEV_COUNTS = {
    "paper_retrieval": 4,
    "single_fact": 4,
    "multi_hop": 2,
    "cross_paper_profile": 4,
    "cross_paper_list": 2,
    "cross_paper_count": 1,
    "unanswerable": 3,
}

TOTAL_QUESTIONS = 80


def build_questions(corpus_root: Path) -> list[dict]:
    triples = load_triples(corpus_root)
    corpus = Corpus(corpus_root, {triple.paper_id for triple in triples})
    gated = [triple for triple in triples if is_verbalizable(triple)]

    cross = _cross_paper_questions(
        triples, corpus,
        TARGET_COUNTS["cross_paper_list"], TARGET_COUNTS["cross_paper_count"])
    profiles = _entity_profile_questions(
        triples, corpus, TARGET_COUNTS["cross_paper_profile"])
    unanswerable = _unanswerable_questions(corpus, TARGET_COUNTS["unanswerable"])
    retrieval = _paper_retrieval_questions(
        triples, corpus, TARGET_COUNTS["paper_retrieval"])
    multi_hop = _multi_hop_questions(gated, corpus, TARGET_COUNTS["multi_hop"])

    # Single facts have by far the largest candidate pool, so they absorb any
    # shortfall in the scarcer categories rather than leaving the bank short.
    shortfall = (
        (TARGET_COUNTS["paper_retrieval"] - len(retrieval))
        + (TARGET_COUNTS["multi_hop"] - len(multi_hop))
        + (TARGET_COUNTS["cross_paper_profile"] - len(profiles))
    )
    facts = _fact_questions(
        gated, corpus, TARGET_COUNTS["single_fact"] + shortfall)

    questions = retrieval + facts + multi_hop + profiles + cross + unanswerable
    if len(questions) != TOTAL_QUESTIONS:
        raise AssertionError(
            f"Expected {TOTAL_QUESTIONS} questions, built {len(questions)}"
        )

    for index, question in enumerate(questions, start=1):
        question["id"] = f"ncg-{index:03d}"
        question["split"] = "test"

    # Keep 25 percent for development, spread evenly inside each category.
    for category, dev_count in DEV_COUNTS.items():
        in_category = [q for q in questions if q["category"] == category]
        if not in_category or dev_count == 0:
            continue
        dev_count = min(dev_count, len(in_category))
        if dev_count == 1:
            selected = {0}
        else:
            selected = {
                round(i * (len(in_category) - 1) / (dev_count - 1))
                for i in range(dev_count)
            }
        for index in selected:
            in_category[index]["split"] = "dev"
    return questions


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/ncg/trial-data"))
    parser.add_argument("--out", type=Path, default=Path("data/evaluation/questions.jsonl"))
    args = parser.parse_args()
    questions = build_questions(args.corpus)
    write_jsonl(args.out, questions)
    dev = sum(question["split"] == "dev" for question in questions)
    counts: dict[str, int] = defaultdict(int)
    for question in questions:
        counts[question["category"]] += 1
    print(f"Wrote {len(questions)} questions ({dev} dev, {len(questions)-dev} test) to {args.out}")
    for category in TARGET_COUNTS:
        print(f"  {category:24s} {counts[category]:3d}")


if __name__ == "__main__":
    main()
