"""
Core data structures and constants for the KARMA Mini NCG pipeline.

This module defines the fixed NLPContributionGraph (NCG) information-unit
inventory and the per-paper contribution graph that the pipeline assembles.
Each paper yields its OWN graph rooted at a single node literally named
"Contribution"; graphs are never merged across papers.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from collections import OrderedDict, defaultdict
import json

# ---------------------------------------------------------------------------
# NCG information-unit inventory (fixed, closed set of 12)
# ---------------------------------------------------------------------------

# Canonical info-unit tokens, in the order they are reported by the scorer.
INFO_UNITS = [
    "RESEARCHPROBLEM",
    "APPROACH",
    "MODEL",
    "CODE",
    "DATASET",
    "EXPERIMENTALSETUP",
    "HYPERPARAMETERS",
    "BASELINES",
    "RESULTS",
    "TASKS",
    "EXPERIMENTS",
    "ABLATIONANALYSIS",
]

# A valid contribution graph must contain the research problem, the results,
# and at least one of MODEL / APPROACH (the paper's core contribution).
MANDATORY_UNITS = ["RESEARCHPROBLEM", "RESULTS"]
MODEL_OR_APPROACH = ["MODEL", "APPROACH"]


@dataclass(frozen=True)
class IUSpec:
    """Serialization spec for one information unit.

    filename     : gold triples file name (lowercase, hyphenated)
    node_label   : capitalized node name used in the graph (e.g. "Model")
    root_pred    : predicate on the edge leaving the "Contribution" root
    direct       : if True, terms attach DIRECTLY to "Contribution" via
                   root_pred (no intermediate info-unit node). This is the
                   special structure used by RESEARCHPROBLEM and CODE.
    """

    filename: str
    node_label: str
    root_pred: str
    direct: bool


# token -> IUSpec.  The structural conventions here were read directly off the
# gold trial data triples/*.txt files.
IU_SPEC: "OrderedDict[str, IUSpec]" = OrderedDict([
    ("RESEARCHPROBLEM", IUSpec("research-problem", "Research problem", "has research problem", True)),
    ("APPROACH",        IUSpec("approach", "Approach", "has", False)),
    ("MODEL",           IUSpec("model", "Model", "has", False)),
    ("CODE",            IUSpec("code", "Code", "Code", True)),
    ("DATASET",         IUSpec("dataset", "Dataset", "has", False)),
    ("EXPERIMENTALSETUP", IUSpec("experimental-setup", "Experimental setup", "has", False)),
    ("HYPERPARAMETERS", IUSpec("hyperparameters", "Hyperparameters", "has", False)),
    ("BASELINES",       IUSpec("baselines", "Baselines", "has", False)),
    ("RESULTS",         IUSpec("results", "Results", "has", False)),
    ("TASKS",           IUSpec("tasks", "Tasks", "has", False)),
    ("EXPERIMENTS",     IUSpec("experiments", "Experiments", "has", False)),
    ("ABLATIONANALYSIS", IUSpec("ablation-analysis", "Ablation analysis", "has", False)),
])

# The literal root node name shared by every contribution graph.
ROOT = "Contribution"

# Node strings that are structural (not phrases drawn from the paper text) and
# must therefore be excluded from the entities/phrase predictions.
_STRUCTURAL_NODES = {ROOT} | {spec.node_label for spec in IU_SPEC.values()}

# Predicates that are graph scaffolding rather than sentence wording; they are
# never emitted as phrase (entity) predictions.
STRUCTURAL_PREDICATES = {"has", "has research problem", "Code"}


def normalize_unit(raw: str) -> Optional[str]:
    """Map a free-text / messy info-unit label to a canonical INFO_UNITS token.

    Applies the NCG normalization rules:
      - method / application      -> APPROACH
      - system / architecture     -> MODEL
    Returns None if the label cannot be mapped to one of the 12 units.
    """
    if not raw:
        return None
    key = "".join(ch for ch in raw.upper() if ch.isalnum())

    aliases = {
        "RESEARCHPROBLEM": "RESEARCHPROBLEM",
        "PROBLEM": "RESEARCHPROBLEM",
        "METHOD": "APPROACH",
        "APPLICATION": "APPROACH",
        "APPROACH": "APPROACH",
        "SYSTEM": "MODEL",
        "ARCHITECTURE": "MODEL",
        "MODEL": "MODEL",
        "CODE": "CODE",
        "DATASET": "DATASET",
        "DATA": "DATASET",
        "EXPERIMENTALSETUP": "EXPERIMENTALSETUP",
        "EXPSETUP": "EXPERIMENTALSETUP",
        "SETUP": "EXPERIMENTALSETUP",
        "HYPERPARAMETERS": "HYPERPARAMETERS",
        "HYPERPARAMETER": "HYPERPARAMETERS",
        "BASELINES": "BASELINES",
        "BASELINE": "BASELINES",
        "RESULTS": "RESULTS",
        "RESULT": "RESULTS",
        "TASKS": "TASKS",
        "TASK": "TASKS",
        "EXPERIMENTS": "EXPERIMENTS",
        "EXPERIMENT": "EXPERIMENTS",
        "ABLATIONANALYSIS": "ABLATIONANALYSIS",
        "ABLATION": "ABLATIONANALYSIS",
    }
    return aliases.get(key)


# ---------------------------------------------------------------------------
# Triple + per-paper graph
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeTriple:
    """A single edge in a paper's contribution graph.

    subject / predicate / object are kept verbatim (predicates are free text
    taken from the sentence wording; phrases are exact Stanza tokens).
    """

    subject: str
    predicate: str
    object: str
    info_unit: str = ""            # canonical INFO_UNITS token
    source_line: int = -1          # 1-indexed Stanza line, or -1 for structural edges
    source_paper: str = ""         # relative paper folder, e.g. "machine-translation/0"
    evidence: str = ""             # exact Stanza sentence the triple was drawn from

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "KnowledgeTriple":
        return cls(**{k: data[k] for k in data if k in cls.__dataclass_fields__})

    def signature(self):
        """Identity used for de-duplication / node merging (exact strings)."""
        return (self.subject, self.predicate, self.object)


@dataclass
class KnowledgeGraph:
    """A per-paper contribution graph.

    Stores triples, de-duplicates identical edges (which also merges
    string-identical phrase nodes into one node, yielding a DAG), and groups
    triples by information unit for serialization.
    """

    paper_id: str = ""
    triples: List[KnowledgeTriple] = field(default_factory=list)
    _seen: set = field(default_factory=set, repr=False)

    def add_triple(self, triple: KnowledgeTriple) -> bool:
        """Add a triple, skipping exact duplicates. Returns True if added."""
        sig = triple.signature()
        if sig in self._seen:
            return False
        self._seen.add(sig)
        self.triples.append(triple)
        return True

    @property
    def nodes(self) -> set:
        ns = set()
        for t in self.triples:
            ns.add(t.subject)
            ns.add(t.object)
        return ns

    def group_by_info_unit(self) -> "OrderedDict[str, List[KnowledgeTriple]]":
        """Return triples grouped by canonical info-unit token, in INFO_UNITS order."""
        groups = defaultdict(list)
        for t in self.triples:
            groups[t.info_unit].append(t)
        ordered = OrderedDict()
        for unit in INFO_UNITS:
            if unit in groups:
                ordered[unit] = groups[unit]
        # any stray units not in the inventory (shouldn't happen post-SAA)
        for unit, ts in groups.items():
            if unit not in ordered:
                ordered[unit] = ts
        return ordered

    def contribution_lines(self) -> List[int]:
        """Sorted unique 1-indexed lines that contributed a term (for sentences.txt)."""
        lines = {t.source_line for t in self.triples if t.source_line and t.source_line > 0}
        return sorted(lines)

    def phrase_nodes(self) -> List[KnowledgeTriple]:
        """Triples whose endpoints reference text phrases (for entities.txt)."""
        return [t for t in self.triples if t.source_line and t.source_line > 0]

    def get_statistics(self) -> Dict:
        groups = self.group_by_info_unit()
        return {
            "paper_id": self.paper_id,
            "triple_count": len(self.triples),
            "info_units": list(groups.keys()),
            "info_unit_count": len(groups),
        }

    def to_dict(self) -> Dict:
        return {
            "paper_id": self.paper_id,
            "triples": [t.to_dict() for t in self.triples],
            "statistics": self.get_statistics(),
        }

    def save_to_file(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def is_structural_node(name: str) -> bool:
    """True for the root and info-unit node labels (not phrases from text)."""
    return name in _STRUCTURAL_NODES
