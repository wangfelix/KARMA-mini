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
        
    def __hash__(self):
        return hash(self.entity_id)

    def __eq__(self, other):
        if isinstance(other, KGEntity):
            return self.entity_id == other.entity_id
        return False

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
            'entities': [e.to_dict() if isinstance(e, KGEntity) else {"name": e} for e in self.entities],
            'triples': [triple.to_dict() for triple in self.triples],
            'metadata': self.metadata,
            'statistics': self.get_statistics()
        }

    def save_to_file(self, filepath: str):
        # We need to make sure self.entities is serialized properly if it contains KGEntity objects.
        data_to_save = self.to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)

    def export_to_neo4j_csv(self, nodes_path: str, relationships_path: str):
        import csv
        
        # Write nodes
        with open(nodes_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['entity_id:ID', 'name', 'LABEL'])
            for entity in self.entities:
                if isinstance(entity, KGEntity):
                    writer.writerow([entity.entity_id, entity.name, entity.entity_type])
                else:
                    # Fallback for plain strings
                    writer.writerow([entity, entity, 'OTHER'])

        # Write relationships
        with open(relationships_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([':START_ID', ':END_ID', ':TYPE', 'confidence', 'source'])
            for triple in self.triples:
                writer.writerow([triple.head, triple.tail, triple.relation, triple.confidence, triple.source])
