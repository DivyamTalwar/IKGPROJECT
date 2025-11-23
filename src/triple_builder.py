"""
Triple Formation Module

Converts entities and relations into structured RDF-style triples
with entity normalization, validation, and deduplication.
"""

from typing import List, Dict, Any, Set, Tuple, Optional
import hashlib
from datetime import datetime
import logging
import json
from collections import Counter
from .utils import save_json, normalize_entity_text, create_entity_id, get_timestamp


class TripleBuilder:
    """
    Builds RDF-style triples from entities and relations.

    Handles entity normalization, ID generation, triple validation,
    and deduplication.

    Attributes:
        entity_mapping: Dictionary mapping entity texts to normalized IDs
        triples: List of constructed triples

    Examples:
        >>> builder = TripleBuilder()
        >>> entities = [{"text": "Tesla", "label": "ORG"}]
        >>> relations = [{
        ...     "subject": "Elon Musk", "subject_type": "PERSON",
        ...     "predicate": "CEO_OF",
        ...     "object": "Tesla", "object_type": "ORG",
        ...     "confidence": 0.9
        ... }]
        >>> result = builder.build_triples(entities, relations)
        >>> len(result['triples']) > 0
        True
    """

    def __init__(self, enable_normalization: bool = True):
        """
        Initialize triple builder.

        Args:
            enable_normalization: Whether to normalize entity names
        """
        self.entity_mapping = {}
        self.triples = []
        self.enable_normalization = enable_normalization
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"TripleBuilder initialized (normalization={enable_normalization})")

    def build_triples(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        source_id: str = None
    ) -> Dict[str, Any]:
        """
        Build triples from entities and relations.

        Args:
            entities: List of extracted entities
            relations: List of extracted relations
            source_id: Identifier for the source document

        Returns:
            Dictionary containing:
                - triples: List of triple dictionaries
                - entity_mapping: Mapping of entity texts to IDs
                - statistics: Triple statistics

        Examples:
            >>> builder = TripleBuilder()
            >>> entities = [
            ...     {"text": "Apple", "label": "ORG"},
            ...     {"text": "California", "label": "GPE"}
            ... ]
            >>> relations = [{
            ...     "subject": "Apple", "subject_type": "ORG",
            ...     "predicate": "LOCATED_IN",
            ...     "object": "California", "object_type": "GPE",
            ...     "confidence": 0.85, "context": "based in"
            ... }]
            >>> result = builder.build_triples(entities, relations)
            >>> result['statistics']['total_triples'] > 0
            True
        """
        if not relations:
            self.logger.warning("No relations provided for triple building")
            return {
                "triples": [],
                "entity_mapping": {},
                "statistics": self._empty_statistics()
            }

        triples = []
        timestamp = get_timestamp()

        # Build entity mapping
        for entity in entities:
            entity_text = entity.get('text', '')
            entity_type = entity.get('label', '')
            if entity_text and entity_text not in self.entity_mapping:
                entity_id = self.normalize_entity(entity_text, entity_type)
                self.entity_mapping[entity_text] = entity_id

        # Create triples from relations
        for relation in relations:
            # Get or create entity IDs
            subject_text = relation.get('subject', '')
            object_text = relation.get('object', '')
            subject_type = relation.get('subject_type', '')
            object_type = relation.get('object_type', '')

            if not subject_text or not object_text:
                continue

            # Normalize and get IDs
            if subject_text not in self.entity_mapping:
                self.entity_mapping[subject_text] = self.normalize_entity(
                    subject_text, subject_type
                )
            if object_text not in self.entity_mapping:
                self.entity_mapping[object_text] = self.normalize_entity(
                    object_text, object_type
                )

            subject_id = self.entity_mapping[subject_text]
            object_id = self.entity_mapping[object_text]

            # Create triple
            triple = self.create_triple(
                subject=subject_text,
                subject_type=subject_type,
                predicate=relation.get('predicate', ''),
                object_=object_text,
                object_type=object_type,
                confidence=relation.get('confidence', 0.5),
                metadata={
                    "context": relation.get('context', ''),
                    "sentence": relation.get('sentence', ''),
                    "method": relation.get('method', ''),
                    "source_id": source_id,
                    "date_extracted": timestamp
                }
            )

            # Validate triple
            if self.validate_triple(triple):
                triples.append(triple)

        # Deduplicate triples
        unique_triples = self.deduplicate_triples(triples)

        # Calculate statistics
        statistics = self.get_statistics(unique_triples)

        result = {
            "triples": unique_triples,
            "entity_mapping": self.entity_mapping,
            "statistics": statistics
        }

        self.logger.info(
            f"Built {len(unique_triples)} triples from {len(relations)} relations"
        )

        return result

    def normalize_entity(self, entity_text: str, entity_type: str) -> str:
        if self.enable_normalization:
            entity_id = create_entity_id(entity_text, entity_type)
        else:
            simple_text = entity_text.lower().replace(' ', '_')
            prefix = entity_type.lower()[:4]
            entity_id = f"{prefix}_{simple_text}"

        return entity_id

    def create_triple(
        self,
        subject: str,
        subject_type: str,
        predicate: str,
        object_: str,
        object_type: str,
        confidence: float,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        subject_id = self.entity_mapping.get(
            subject,
            self.normalize_entity(subject, subject_type)
        )
        object_id = self.entity_mapping.get(
            object_,
            self.normalize_entity(object_, object_type)
        )

        triple = {
            "subject": subject,
            "subject_type": subject_type,
            "subject_id": subject_id,
            "predicate": predicate,
            "object": object_,
            "object_type": object_type,
            "object_id": object_id,
            "confidence": round(confidence, 2),
            "metadata": metadata
        }

        return triple

    def validate_triple(self, triple: Dict[str, Any]) -> bool:
        required_fields = [
            'subject', 'subject_type', 'subject_id',
            'predicate',
            'object', 'object_type', 'object_id',
            'confidence'
        ]

        for field in required_fields:
            if field not in triple:
                self.logger.warning(f"Triple missing required field: {field}")
                return False

            if isinstance(triple[field], str) and not triple[field]:
                self.logger.warning(f"Triple has empty field: {field}")
                return False

        if triple['subject'] == triple['object']:
            self.logger.debug(f"Skipping self-referential triple: {triple['subject']}")
            return False

        confidence = triple.get('confidence', 0)
        if not (0 <= confidence <= 1):
            self.logger.warning(f"Invalid confidence value: {confidence}")
            return False

        return True

    def deduplicate_triples(
        self,
        triples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate triples, keeping highest confidence.

        Triples are considered duplicates if they have the same
        (subject, predicate, object) tuple.

        Args:
            triples: List of triples

        Returns:
            Deduplicated triple list

        Examples:
            >>> builder = TripleBuilder()
            >>> t1 = {
            ...     "subject": "A", "predicate": "WORKS_AT", "object": "B",
            ...     "subject_type": "PERSON", "object_type": "ORG",
            ...     "subject_id": "pers_a", "object_id": "orga_b",
            ...     "confidence": 0.8, "metadata": {}
            ... }
            >>> t2 = {
            ...     "subject": "A", "predicate": "WORKS_AT", "object": "B",
            ...     "subject_type": "PERSON", "object_type": "ORG",
            ...     "subject_id": "pers_a", "object_id": "orga_b",
            ...     "confidence": 0.9, "metadata": {}
            ... }
            >>> result = builder.deduplicate_triples([t1, t2])
            >>> len(result)
            1
            >>> result[0]['confidence']
            0.9
        """
        if not triples:
            return []

        # Group by (subject, predicate, object)
        triple_groups = {}

        for triple in triples:
            key = (
                triple.get('subject', ''),
                triple.get('predicate', ''),
                triple.get('object', '')
            )

            if key not in triple_groups:
                triple_groups[key] = []
            triple_groups[key].append(triple)

        # Keep highest confidence from each group
        deduplicated = []
        for group in triple_groups.values():
            # Sort by confidence (descending)
            best_triple = max(group, key=lambda t: t.get('confidence', 0))

            # Optionally merge metadata from all triples in group
            if len(group) > 1:
                methods = set()
                for t in group:
                    method = t.get('metadata', {}).get('method', '')
                    if method:
                        methods.add(method)
                best_triple['metadata']['methods'] = list(methods)

            deduplicated.append(best_triple)

        original_count = len(triples)
        dedup_count = len(deduplicated)
        removed = original_count - dedup_count

        if removed > 0:
            self.logger.info(f"Removed {removed} duplicate triples")

        return deduplicated

    def get_statistics(self, triples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics for triple collection.

        Args:
            triples: List of triples

        Returns:
            Statistics dictionary with counts and distributions

        Examples:
            >>> builder = TripleBuilder()
            >>> triples = [
            ...     {"subject": "A", "predicate": "WORKS_AT", "object": "B",
            ...      "subject_type": "PERSON", "object_type": "ORG",
            ...      "subject_id": "pers_a", "object_id": "orga_b",
            ...      "confidence": 0.9, "metadata": {}}
            ... ]
            >>> stats = builder.get_statistics(triples)
            >>> stats['total_triples']
            1
        """
        if not triples:
            return self._empty_statistics()

        # Count unique subjects and objects
        unique_subjects = set(t['subject'] for t in triples)
        unique_objects = set(t['object'] for t in triples)

        # Count by relation type
        relation_distribution = Counter(t['predicate'] for t in triples)

        # Count by entity types
        subject_type_dist = Counter(t['subject_type'] for t in triples)
        object_type_dist = Counter(t['object_type'] for t in triples)

        # Calculate average confidence
        avg_confidence = sum(t.get('confidence', 0) for t in triples) / len(triples)

        # Find confidence distribution
        high_conf = sum(1 for t in triples if t.get('confidence', 0) >= 0.8)
        med_conf = sum(1 for t in triples if 0.5 <= t.get('confidence', 0) < 0.8)
        low_conf = sum(1 for t in triples if t.get('confidence', 0) < 0.5)

        statistics = {
            "total_triples": len(triples),
            "unique_subjects": len(unique_subjects),
            "unique_objects": len(unique_objects),
            "unique_entities": len(unique_subjects | unique_objects),
            "relation_distribution": dict(relation_distribution),
            "subject_type_distribution": dict(subject_type_dist),
            "object_type_distribution": dict(object_type_dist),
            "average_confidence": round(avg_confidence, 2),
            "confidence_distribution": {
                "high (>= 0.8)": high_conf,
                "medium (0.5-0.8)": med_conf,
                "low (< 0.5)": low_conf
            }
        }

        return statistics

    def _empty_statistics(self) -> Dict[str, Any]:
        """Return empty statistics dictionary."""
        return {
            "total_triples": 0,
            "unique_subjects": 0,
            "unique_objects": 0,
            "unique_entities": 0,
            "relation_distribution": {},
            "subject_type_distribution": {},
            "object_type_distribution": {},
            "average_confidence": 0.0,
            "confidence_distribution": {
                "high (>= 0.8)": 0,
                "medium (0.5-0.8)": 0,
                "low (< 0.5)": 0
            }
        }

    def export_triples(
        self,
        triples: List[Dict[str, Any]],
        output_path: str,
        format: str = "json"
    ) -> None:
        """
        Export triples to file.

        Args:
            triples: List of triples
            output_path: Output file path
            format: Output format (json, csv, ttl)

        Examples:
            >>> builder = TripleBuilder()
            >>> triples = [{"subject": "A", "predicate": "rel", "object": "B",
            ...              "subject_type": "X", "object_type": "Y",
            ...              "subject_id": "x_a", "object_id": "y_b",
            ...              "confidence": 0.9, "metadata": {}}]
            >>> builder.export_triples(triples, "output.json", "json")
        """
        if format == "json":
            save_json(triples, output_path, indent=2)
            self.logger.info(f"Exported {len(triples)} triples to {output_path} (JSON)")

        elif format == "csv":
            # Flatten triples for CSV
            import csv
            from pathlib import Path

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                if triples:
                    fieldnames = [
                        'subject', 'subject_type', 'subject_id',
                        'predicate',
                        'object', 'object_type', 'object_id',
                        'confidence'
                    ]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    for triple in triples:
                        row = {k: triple.get(k, '') for k in fieldnames}
                        writer.writerow(row)

            self.logger.info(f"Exported {len(triples)} triples to {output_path} (CSV)")

        elif format == "ttl":
            # Export as Turtle/RDF
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("@prefix kg: <http://example.org/kg#> .\n\n")

                for triple in triples:
                    subj = triple['subject_id'].replace(' ', '_')
                    pred = triple['predicate']
                    obj = triple['object_id'].replace(' ', '_')

                    f.write(f"kg:{subj} kg:{pred} kg:{obj} .\n")

            self.logger.info(f"Exported {len(triples)} triples to {output_path} (TTL)")

        else:
            raise ValueError(f"Unsupported format: {format}")

    def clear(self) -> None:
        """Clear all stored triples and entity mappings."""
        self.triples = []
        self.entity_mapping = {}
        self.logger.info("Cleared all triples and entity mappings")


# Utility Functions

def generate_entity_id(entity_text: str, entity_type: str) -> str:
    """
    Generate unique entity ID.

    Args:
        entity_text: Entity text
        entity_type: Entity type

    Returns:
        Unique ID string

    Examples:
        >>> generate_entity_id("Tesla Inc.", "ORGANIZATION")
        'orga_tesla'
    """
    # Use the utility function
    return create_entity_id(entity_text, entity_type)


def normalize_text(text: str) -> str:
    """
    Normalize text for entity matching.

    Args:
        text: Input text

    Returns:
        Normalized text

    Examples:
        >>> normalize_text("  Tesla Inc.  ")
        'tesla'
        >>> normalize_text("New York")
        'new york'
    """
    if not text:
        return ""

    # Convert to lowercase
    normalized = text.lower()

    # Remove extra whitespace
    normalized = ' '.join(normalized.split())

    # Remove common punctuation
    normalized = normalized.replace('.', '').replace(',', '')

    # Remove common suffixes
    suffixes = ['inc', 'llc', 'corp', 'ltd', 'co']
    for suffix in suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)].strip()

    return normalized


def calculate_triple_confidence(
    entity_confidence: float,
    relation_confidence: float
) -> float:
    """
    Calculate overall triple confidence.

    Uses harmonic mean to ensure low confidence in either component
    results in low overall confidence.

    Args:
        entity_confidence: Confidence of entity extraction
        relation_confidence: Confidence of relation extraction

    Returns:
        Combined confidence score

    Examples:
        >>> calculate_triple_confidence(0.9, 0.8)
        0.85
        >>> calculate_triple_confidence(0.9, 0.3)
        0.45
    """
    # Harmonic mean
    if entity_confidence + relation_confidence == 0:
        return 0.0

    harmonic_mean = 2 * (entity_confidence * relation_confidence) / (
        entity_confidence + relation_confidence
    )

    return round(harmonic_mean, 2)


# Module logger
logger = logging.getLogger(__name__)
