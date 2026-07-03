"""
Contribution Sentence Agent (CSA) for KARMA Mini — sentence selection + IU tagging.
"""

import logging
from typing import List, Dict, Tuple, Optional

from karma_mini.core.base_agent import BaseAgent
from karma_mini.core.data_structures import INFO_UNITS, normalize_unit

logger = logging.getLogger(__name__)

# Sentences are sent to the model in windows to stay within context limits.
MAX_SENTENCES_PER_CALL = 120

# Hard cap on selections per paper. Gold papers annotate only a handful of
# contribution sentences (~10% of lines; typically 8-15), and every extra
# selection floods the downstream triple/phrase predictions with false
# positives — the scorer is exact-match, so precision dies fast.
MAX_SELECTIONS = 15


class ContributionSentenceAgent(BaseAgent):
    """
    Agent 1: Contribution Sentence Agent (CSA)

    Role: read the whole paper (numbered Stanza sentences, 1-indexed) and select
    the few CONTRIBUTION sentences, tagging each with exactly ONE information
    unit. Deciding the info unit once, at the sentence level, means every triple
    later extracted from that sentence inherits the same unit — related edges
    can never scatter across different triples/<iu>.txt files.
    """

    def __init__(self, client, model_name: str):
        units = ", ".join(INFO_UNITS)
        system_prompt = f"""You select the CONTRIBUTION sentences of a single scholarly NLP paper for the SemEval-2021 NLPContributionGraph (NCG) task.

You are given the paper as numbered sentences (one per line, 1-indexed), exactly as produced by the Stanza tokenizer. A leading [section] tag may indicate the current section.

A CONTRIBUTION sentence states what THIS paper contributes:
  - the research problem / task it addresses,
  - the model or approach it proposes,
  - its main results,
  - datasets it introduces or uses, released code, baselines compared against,
  - the hyperparameters / experimental setup of ITS solution, its experiments and ablations.
Be HIGHLY selective: only about 10% of a paper's sentences qualify — typically 5-15 sentences for the WHOLE paper, found in the title, the abstract, the introduction, and the opening sentences of the model/approach, experiments and results sections. NEVER select more than 15. Do NOT select background, related-work, generic statements, or routine experimental detail. Order your output by importance: the most central contribution statements first.

Tag every selected sentence with EXACTLY ONE information unit from this fixed inventory: {units}.
  - ALWAYS select the paper title (the sentence right after the "title" header line) and tag it RESEARCHPROBLEM — in this corpus the title virtually always names the research problem.
  - A "method" or "application" -> APPROACH; a "system" or "architecture" -> MODEL.
  - A paper proposes EITHER a MODEL or an APPROACH, never both: decide which one fits the paper's solution overall (concrete implementation -> MODEL; abstract method -> APPROACH) and use that single unit for ALL its solution sentences.
  - EXPERIMENTALSETUP only when hardware is mentioned, otherwise HYPERPARAMETERS.
  - Papers must have RESEARCHPROBLEM, RESULTS, and one of MODEL/APPROACH — always select sentences for these when the paper states them.

OUTPUT FORMAT:
Return ONLY a valid JSON array (no markdown, no prose), one object per selected sentence:
[
  {{"line": 2, "info_unit": "RESEARCHPROBLEM"}},
  {{"line": 16, "info_unit": "MODEL"}},
  {{"line": 159, "info_unit": "RESULTS"}}
]"""
        super().__init__(client, model_name, system_prompt)

    def process(self, sentences: List[Tuple[int, str]],
                section_hints: Optional[Dict[int, str]] = None) -> List[Dict]:
        """Select contribution sentences over a whole paper.

        Args:
            sentences: list of (line_no, text) from the Stanza output.
            section_hints: optional {line_no: nearest_header} for context.

        Returns:
            list of {"line": int, "info_unit": <raw label>, "text": <sentence>},
            sorted by line, at most one entry per line.
        """
        by_line = {ln: txt for ln, txt in sentences}

        raw: List[Dict] = []
        for start in range(0, len(sentences), MAX_SENTENCES_PER_CALL):
            window = sentences[start:start + MAX_SENTENCES_PER_CALL]
            raw.extend(self._process_window(window, section_hints))

        seen = set()
        selections: List[Dict] = []
        for item in raw:  # emission order = the model's importance ranking
            if not isinstance(item, dict):
                continue
            try:
                line = int(item.get("line"))
            except (TypeError, ValueError):
                continue
            unit = str(item.get("info_unit", "")).strip()
            if line not in by_line or line in seen or not unit:
                continue
            seen.add(line)
            selections.append({"line": line, "info_unit": unit, "text": by_line[line]})

        if len(selections) > MAX_SELECTIONS:
            logger.warning(f"CSA over-selected ({len(selections)}); capping to {MAX_SELECTIONS}.")
            selections = self._cap(selections)

        selections = self._ensure_research_problem(selections, by_line, section_hints)

        selections.sort(key=lambda s: s["line"])
        logger.info(f"CSA selected {len(selections)} contribution sentence(s).")
        return selections

    @staticmethod
    def _ensure_research_problem(selections: List[Dict], by_line: Dict[int, str],
                                 section_hints: Optional[Dict[int, str]]) -> List[Dict]:
        """Deterministic backstop for the mandatory RESEARCHPROBLEM unit: if no
        selection maps to it, select (or retag) the paper title. Gold annotates
        the title as a research-problem sentence in nearly every paper, and a
        missing RESEARCHPROBLEM zeroes an entire mandatory info unit."""
        if any(normalize_unit(s.get("info_unit", "")) == "RESEARCHPROBLEM" for s in selections):
            return selections

        title_line = None
        for line_no in sorted(by_line):
            hint = (section_hints or {}).get(line_no, "")
            if hint.strip().lower() == "title" and by_line[line_no].strip().lower() != "title":
                title_line = line_no
                break
        if title_line is None and by_line.get(1, "").strip().lower() == "title" and 2 in by_line:
            title_line = 2
        if title_line is None:
            return selections

        logger.warning("CSA selected no RESEARCHPROBLEM sentence; using the title as backstop.")
        existing = next((s for s in selections if s["line"] == title_line), None)
        if existing:
            existing["info_unit"] = "RESEARCHPROBLEM"
            return selections
        entry = {"line": title_line, "info_unit": "RESEARCHPROBLEM", "text": by_line[title_line]}
        if len(selections) >= MAX_SELECTIONS:
            selections = selections[:-1]  # drop the lowest-ranked kept item
        selections.append(entry)
        return selections

    @staticmethod
    def _cap(selections: List[Dict]) -> List[Dict]:
        """Keep the top MAX_SELECTIONS by rank, but never drop the last sentence
        of a mandatory unit family (RESEARCHPROBLEM / MODEL-or-APPROACH / RESULTS)
        that was selected somewhere in the full list."""
        def family(sel: Dict) -> str:
            unit = normalize_unit(sel.get("info_unit", "")) or ""
            if unit == "RESEARCHPROBLEM":
                return "RP"
            if unit == "RESULTS":
                return "RES"
            if unit in ("MODEL", "APPROACH"):
                return "MA"
            return ""

        kept = selections[:MAX_SELECTIONS]
        for fam in ("RP", "MA", "RES"):
            if any(family(s) == fam for s in selections) and \
                    not any(family(s) == fam for s in kept):
                best = next(s for s in selections if family(s) == fam)
                for i in range(len(kept) - 1, -1, -1):
                    if family(kept[i]) == "":
                        kept[i] = best
                        break
                else:
                    kept[-1] = best
        return kept

    def _process_window(self, window: List[Tuple[int, str]],
                        section_hints: Optional[Dict[int, str]]) -> List[Dict]:
        numbered = []
        for line_no, text in window:
            hint = (section_hints or {}).get(line_no, "")
            prefix = f"[{hint}] " if hint else ""
            numbered.append(f"{line_no}\t{prefix}{text}")
        body = "\n".join(numbered)

        first, last = window[0][0], window[-1][0]
        prompt = f"""Here are sentences {first}-{last} of the paper as numbered Stanza sentences. Select the contribution sentences and tag each with its information unit. The budget of at most 15 selections applies to the WHOLE paper, not to this excerpt — pick only sentences that would make that paper-wide cut.

{body}

Return ONLY the JSON array."""

        response_text = self._make_llm_call(prompt, temperature=0.1)
        return self._parse_json_response(response_text)
