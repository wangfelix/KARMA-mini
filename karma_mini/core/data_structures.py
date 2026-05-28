"""
Core data structures for the KARMA mini framework.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Union
import json

@dataclass
class KnowledgeTriple:
    head: str
    relation: str
    tail: str
    confidence: float = 0.0
    source: str = "unknown"
    relevance: float = 0.0
    clarity: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'KnowledgeTriple':
        return cls(**data)

@dataclass
class KGEntity:
    entity_id: str
    entity_type: str = "Unknown"
    name: str = ""
    normalized_id: str = "N/A"
    aliases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'KGEntity':
        return cls(**data)

@dataclass
class KnowledgeGraph:
    entities: set = field(default_factory=set)
    triples: List[KnowledgeTriple] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def add_entity(self, entity: Union[str, KGEntity]):
        if isinstance(entity, KGEntity):
            self.entities.add(entity.name)
        else:
            self.entities.add(entity)

    def add_triple(self, triple: KnowledgeTriple):
        self.triples.append(triple)
        self.entities.add(triple.head)
        self.entities.add(triple.tail)

    def get_statistics(self) -> Dict:
        relation_counts = {}
        for triple in self.triples:
            relation_counts[triple.relation] = relation_counts.get(triple.relation, 0) + 1

        return {
            'entity_count': len(self.entities),
            'triple_count': len(self.triples),
            'unique_relations': len(relation_counts),
            'relation_distribution': relation_counts,
            'avg_confidence': sum(t.confidence for t in self.triples) / len(self.triples) if self.triples else 0
        }

    def to_dict(self) -> Dict:
        return {
            'entities': list(self.entities),
            'triples': [triple.to_dict() for triple in self.triples],
            'metadata': self.metadata,
            'statistics': self.get_statistics()
        }

    def save_to_file(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
