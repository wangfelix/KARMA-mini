"""
Information Extraction Agent (IEA) for KARMA Mini — NCG contribution extraction.
"""

import logging
from typing import List, Dict, Tuple, Optional

from karma_mini.core.base_agent import BaseAgent
from karma_mini.core.data_structures import INFO_UNITS

logger = logging.getLogger(__name__)

# Sentences are sent to the model in windows to stay within context limits
MAX_SENTENCES_PER_CALL = 120


class InformationExtractionAgent(BaseAgent):
    """
    Agent 1: Information Extraction Agent (IEA)

    Role: contribution extraction over a whole paper. Given the paper as
    numbered Stanza sentences (one per line, 1-indexed), it first identifies the
    few contribution sentences （what this paper contributes - research problem,
    model/approach, results, datasets, code, baselines - usually in the title,
    abstract, introduction, and the first lines of the model/results sections and
    then emits (subject, predicate, object) triples drawn only from those
    sentences, using the exact tokenized phrases.
    """

    def __init__(self, client, model_name: str):
        units = ", ".join(INFO_UNITS)
        system_prompt = f"""You extract the CONTRIBUTION KNOWLEDGE GRAPH of a single scholarly NLP paper for the SemEval-2021 NLPContributionGraph (NCG) task.

You are given the paper as numbered sentences (one per line, 1-indexed), exactly as produced by the Stanza tokenizer. Note the tokenization: punctuation is spaced out (e.g. "RNN Encoder - Decoder", "two recurrent neural networks ( RNN )", "phrase - based SMT").

STEP 1 — Find the CONTRIBUTION sentences.
Select only the handful of sentences that state what THIS paper contributes:
  - the research problem / task it addresses,
  - the model or approach it proposes,
  - its main results,
  - datasets it introduces or uses, released code, and baselines compared against.
These are almost always in the title, the abstract, the introduction, and the opening sentences of the model/approach and results sections. Do NOT pull background, related-work, or generic statements.

STEP 2 — From ONLY those sentences, emit (subject, predicate, object) triples.
Rules:
  - subject, predicate, and object MUST be copied VERBATIM as contiguous spans of tokens from the sentence (same spacing as shown). Never paraphrase, re-tokenize, or invent words.
  - predicates are FREE TEXT taken from the wording of the sentence (e.g. "consists of", "act as", "improves the performance", "trained", "built using"). Do not map them to any fixed vocabulary.
  - tag each triple with a best-guess info_unit from this fixed inventory: {units}.
  - record from_line = the 1-indexed line number the triple was drawn from, and evidence = that exact sentence.

STRUCTURE conventions (match these so the graph is rooted correctly):
  - RESEARCHPROBLEM: emit subject "Contribution", predicate "has research problem", object = the research-problem phrase. Emit one per distinct problem/task name (often the title and an intro sentence).
  - CODE: if a code/repo URL is stated, emit subject "Contribution", predicate "Code", object = the exact URL.
  - For MODEL, APPROACH, RESULTS, DATASET, BASELINES, EXPERIMENTALSETUP, HYPERPARAMETERS, TASKS, EXPERIMENTS, ABLATIONANALYSIS: connect the first/top term to the info-unit node using that node's name as the subject. The node names are: Model, Approach, Results, Dataset, Baselines, Experimental setup, Hyperparameters, Tasks, Experiments, Ablation analysis. Then chain deeper triples phrase->phrase.

info_unit normalization: a "method" or "application" -> APPROACH; a "system" or "architecture" -> MODEL. Use EXPERIMENTALSETUP only when hardware is mentioned, otherwise HYPERPARAMETERS.

ALWAYS include the mandatory units when present in the paper: RESEARCHPROBLEM, RESULTS, and at least one of MODEL or APPROACH.

OUTPUT FORMAT:
Return ONLY a valid JSON array (no markdown, no prose). Each element:
{{
  "info_unit": "<one token from the inventory>",
  "subject": "<verbatim span, or 'Contribution', or an info-unit node name>",
  "predicate": "<verbatim wording, or 'has'/'has research problem'/'Code'>",
  "object": "<verbatim span from the sentence>",
  "from_line": <int line number>,
  "evidence": "<the exact sentence>"
}}

WORKED EXAMPLE (input lines):
2  Learning Phrase Representations using RNN Encoder - Decoder for Statistical Machine Translation
16  The proposed neural network architecture , which we will refer to as an RNN Encoder - Decoder , consists of two recurrent neural networks ( RNN ) that act as an encoder and a decoder pair .
159  As expected , adding features computed by neural networks consistently improves the performance over the baseline performance .

Valid output:
[
  {{"info_unit":"RESEARCHPROBLEM","subject":"Contribution","predicate":"has research problem","object":"Statistical Machine Translation","from_line":2,"evidence":"Learning Phrase Representations using RNN Encoder - Decoder for Statistical Machine Translation"}},
  {{"info_unit":"MODEL","subject":"Model","predicate":"has","object":"neural network architecture","from_line":16,"evidence":"The proposed neural network architecture , which we will refer to as an RNN Encoder - Decoder , consists of two recurrent neural networks ( RNN ) that act as an encoder and a decoder pair ."}},
  {{"info_unit":"MODEL","subject":"neural network architecture","predicate":"refer to as","object":"RNN Encoder - Decoder","from_line":16,"evidence":"The proposed neural network architecture , which we will refer to as an RNN Encoder - Decoder , consists of two recurrent neural networks ( RNN ) that act as an encoder and a decoder pair ."}},
  {{"info_unit":"MODEL","subject":"neural network architecture","predicate":"consists of","object":"two recurrent neural networks ( RNN )","from_line":16,"evidence":"The proposed neural network architecture , which we will refer to as an RNN Encoder - Decoder , consists of two recurrent neural networks ( RNN ) that act as an encoder and a decoder pair ."}},
  {{"info_unit":"RESULTS","subject":"Results","predicate":"improves the performance","object":"adding features","from_line":159,"evidence":"As expected , adding features computed by neural networks consistently improves the performance over the baseline performance ."}}
]"""
        super().__init__(client, model_name, system_prompt)

    def process(self, sentences: List[Tuple[int, str]],
                section_hints: Optional[Dict[int, str]] = None) -> List[Dict]:
        """Extract contribution triples from a paper.

        Args:
            sentences: list of (line_no, text) from the Stanza output
            section_hints: {line_no: nearest_header} for context.

        Returns:
            list of dicts: {info_unit, subject, predicate, object, from_line, evidence}
        """
        all_triples: List[Dict] = []
        for start in range(0, len(sentences), MAX_SENTENCES_PER_CALL):
            window = sentences[start:start + MAX_SENTENCES_PER_CALL]
            all_triples.extend(self._process_window(window, section_hints))

        # Keep only well-formed triples.
        valid: List[Dict] = []
        for t in all_triples:
            if not isinstance(t, dict):
                continue
            if not all(k in t for k in ("subject", "predicate", "object")):
                continue
            if not (str(t.get("subject", "")).strip() and str(t.get("object", "")).strip()):
                continue
            try:
                t["from_line"] = int(t.get("from_line", -1))
            except (TypeError, ValueError):
                t["from_line"] = -1
            t["info_unit"] = str(t.get("info_unit", "")).strip()
            valid.append(t)

        logger.info(f"IEA extracted {len(valid)} contribution triples.")
        return valid

    def _process_window(self, window: List[Tuple[int, str]],
                        section_hints: Optional[Dict[int, str]]) -> List[Dict]:
        numbered = []
        for line_no, text in window:
            hint = (section_hints or {}).get(line_no, "")
            prefix = f"[{hint}] " if hint else ""
            numbered.append(f"{line_no}\t{prefix}{text}")
        body = "\n".join(numbered)

        prompt = f"""Here is the paper as numbered Stanza sentences (a leading [section] tag may indicate the current section). Identify the contribution sentences, then emit the contribution triples.

{body}

Return ONLY the JSON array of triples."""

        # A low-but-nonzero temperature is suficient for verbatim copying
        response_text = self._make_llm_call(prompt, temperature=0.1)
        return self._parse_json_response(response_text)
