"""
Relation Extraction Module

This module extracts semantic relationships between entities using
dependency parsing and pattern-based matching techniques.
"""

import spacy
from typing import List, Dict, Any, Tuple, Optional, Set
import re
import logging
from itertools import combinations
from collections import defaultdict
from .utils import time_function


RELATION_PATTERNS = {
    "CEO_OF": [
        r"{subj}.*(is|was|became).*(CEO|chief executive officer).*{obj}",
        r"{subj}.*(leads|headed|heads).*{obj}",
        r"{obj}.*CEO.*(is|was).*{subj}",
    ],
    "WORKS_AT": [
        r"{subj}.*(works|worked|employed).*{obj}",
        r"{subj}.*(joined|join).*{obj}",
        r"{subj}.*(employee|staff|team member).*{obj}",
    ],
    "LOCATED_IN": [
        r"{subj}.*(in|located in|based in|headquartered in).*{obj}",
        r"{subj}.*(headquarters|office).*{obj}",
        r"{obj}.*based.*{subj}",
    ],
    "HEADQUARTERED_IN": [
        r"{subj}.*(headquartered|headquarters).*{obj}",
        r"{subj}.*based.*{obj}",
    ],
    "ACQUIRED": [
        r"{subj}.*(acquired|bought|purchased).*{obj}",
        r"{obj}.*(acquisition|purchase).*{subj}",
        r"{subj}.*acquiring.*{obj}",
    ],
    "FOUNDED": [
        r"{subj}.*(founded|established|created|started).*{obj}",
        r"{obj}.*(founded|established).*(by).*{subj}",
        r"{subj}.*founder.*{obj}",
    ],
    "LAUNCHED_ON": [
        r"{subj}.*(launched|released|introduced).*{obj}",
        r"{obj}.*launch.*{subj}",
    ],
    "PARTNERED_WITH": [
        r"{subj}.*(partnered|partner|partnership).*{obj}",
        r"{subj}.*(collaborated|collaboration).*{obj}",
    ],
    "INVESTED_IN": [
        r"{subj}.*(invested|investment|investing).*{obj}",
        r"{subj}.*funding.*{obj}",
    ],
    "SUBSIDIARY_OF": [
        r"{subj}.*(subsidiary|owned by|division of).*{obj}",
        r"{obj}.*(owns|acquired).*{subj}",
    ],
}

# Dependency patterns for relation extraction
DEPENDENCY_PATTERNS = {
    "CEO_OF": ["nsubj", "ROOT", "attr"],
    "WORKS_AT": ["nsubj", "ROOT", "prep", "pobj"],
    "LOCATED_IN": ["nsubjpass", "ROOT", "prep", "pobj"],
}


class RelationExtractor:
    """
    Extracts relationships between named entities.

    Uses dependency parsing and pattern matching to identify
    semantic relationships in text.

    Attributes:
        nlp: SpaCy NLP model
        use_dependency: Whether to use dependency-based extraction
        use_patterns: Whether to use pattern-based extraction
        max_distance: Maximum word distance between entities

    Examples:
        >>> extractor = RelationExtractor()
        >>> text = "Elon Musk is the CEO of Tesla."
        >>> entities = [
        ...     {"text": "Elon Musk", "label": "PERSON", "start_char": 0, "end_char": 9},
        ...     {"text": "Tesla", "label": "ORG", "start_char": 24, "end_char": 29}
        ... ]
        >>> result = extractor.extract_relations(text, entities)
        >>> len(result['relations']) > 0
        True
    """

    def __init__(
        self,
        nlp_model=None,
        use_dependency_parsing: bool = True,
        use_pattern_matching: bool = True,
        max_entity_distance: int = 10
    ):
        """
        Initialize relation extractor.

        Args:
            nlp_model: Pre-loaded SpaCy model (optional, will load if None)
            use_dependency_parsing: Enable dependency-based extraction
            use_pattern_matching: Enable pattern-based extraction
            max_entity_distance: Maximum words between entities to consider
        """
        self.logger = logging.getLogger(__name__)

        # Load or use provided SpaCy model
        if nlp_model is None:
            try:
                self.logger.info("Loading SpaCy model for relation extraction")
                self.nlp = spacy.load("en_core_web_lg")
            except OSError:
                self.logger.error("SpaCy model not found")
                raise
        else:
            self.nlp = nlp_model

        self.use_dependency = use_dependency_parsing
        self.use_patterns = use_pattern_matching
        self.max_distance = max_entity_distance

        self.logger.info(
            f"RelationExtractor initialized "
            f"(dependency={use_dependency_parsing}, "
            f"patterns={use_pattern_matching})"
        )

    @time_function
    def extract_relations(
        self,
        text: str,
        entities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract relations from text given known entities.

        Args:
            text: Input text
            entities: List of extracted entities

        Returns:
            Dictionary with:
                - text: Original text
                - relations: List of extracted relations
                - relation_count: Count by relation type

        Examples:
            >>> extractor = RelationExtractor()
            >>> text = "Apple is headquartered in Cupertino, California."
            >>> entities = [
            ...     {"text": "Apple", "label": "ORG", "start_char": 0, "end_char": 5},
            ...     {"text": "Cupertino", "label": "GPE", "start_char": 26, "end_char": 35}
            ... ]
            >>> result = extractor.extract_relations(text, entities)
            >>> 'relations' in result
            True
        """
        if not text or not entities:
            return {
                "text": text,
                "relations": [],
                "relation_count": {}
            }

        # Process text with SpaCy
        doc = self.nlp(text)

        # Get entity pairs
        entity_pairs = self.get_entity_pairs(entities)

        all_relations = []

        # Dependency-based extraction
        if self.use_dependency:
            dep_relations = self._dependency_based_extraction(doc, entity_pairs, entities)
            all_relations.extend(dep_relations)

        # Pattern-based extraction
        if self.use_patterns:
            pattern_relations = self._pattern_based_extraction(text, entities, entity_pairs)
            all_relations.extend(pattern_relations)

        # Deduplicate relations
        unique_relations = self._deduplicate_relations(all_relations)

        # Count relations by type
        relation_count = {}
        for rel in unique_relations:
            rel_type = rel['predicate']
            relation_count[rel_type] = relation_count.get(rel_type, 0) + 1

        result = {
            "text": text,
            "relations": unique_relations,
            "relation_count": relation_count
        }

        self.logger.debug(
            f"Extracted {len(unique_relations)} relations from text"
        )

        return result

    def _dependency_based_extraction(
        self,
        doc,
        entity_pairs: List[Tuple],
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        relations = []

        # Create entity lookup by character positions
        entity_lookup = {}
        for ent in entities:
            for i in range(ent['start_char'], ent['end_char']):
                entity_lookup[i] = ent

        # Find entity spans in doc
        entity_spans = {}
        for ent in doc.ents:
            for entity in entities:
                if (entity['text'] == ent.text and
                    entity['start_char'] == ent.start_char):
                    entity_spans[entity['text']] = ent
                    break

        # Check each entity pair
        for subj, obj in entity_pairs:
            # Find connecting verb/root
            subj_text = subj['text']
            obj_text = obj['text']

            if subj_text not in entity_spans or obj_text not in entity_spans:
                continue

            subj_span = entity_spans[subj_text]
            obj_span = entity_spans[obj_text]

            # Find path between entities
            dep_path = self._find_dependency_path(subj_span.root, obj_span.root)

            if dep_path:
                # Extract relation from path
                relation_type = self._classify_from_dependency(
                    dep_path,
                    subj['label'],
                    obj['label']
                )

                if relation_type:
                    # Extract context
                    sent = list(subj_span.root.sent)
                    context = self._extract_context(sent, subj_span, obj_span)

                    relation = {
                        "subject": subj['text'],
                        "subject_type": subj['label'],
                        "predicate": relation_type,
                        "object": obj['text'],
                        "object_type": obj['label'],
                        "confidence": 0.85,
                        "context": context,
                        "sentence": subj_span.sent.text,
                        "method": "dependency"
                    }
                    relations.append(relation)

        return relations

    def _pattern_based_extraction(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        entity_pairs: List[Tuple]
    ) -> List[Dict[str, Any]]:
        """
        Extract relations using predefined patterns.

        Args:
            text: Input text
            entities: List of entities
            entity_pairs: Pairs to check

        Returns:
            List of extracted relations
        """
        relations = []

        for subj, obj in entity_pairs:
            # Get text between entities
            start = min(subj['start_char'], obj['start_char'])
            end = max(subj['end_char'], obj['end_char'])
            context_text = text[start:end].lower()

            # Try all relation patterns
            for relation_type, patterns in RELATION_PATTERNS.items():
                for pattern_template in patterns:
                    # Replace placeholders with entity text
                    pattern = pattern_template.replace('{subj}', re.escape(subj['text'].lower()))
                    pattern = pattern.replace('{obj}', re.escape(obj['text'].lower()))

                    # Check if pattern matches
                    if re.search(pattern, text.lower(), re.IGNORECASE):
                        # Extract context
                        context = self._extract_pattern_context(
                            text,
                            subj,
                            obj
                        )

                        # Calculate confidence based on pattern specificity
                        confidence = self._calculate_pattern_confidence(
                            pattern_template,
                            subj,
                            obj
                        )

                        relation = {
                            "subject": subj['text'],
                            "subject_type": subj['label'],
                            "predicate": relation_type,
                            "object": obj['text'],
                            "object_type": obj['label'],
                            "confidence": confidence,
                            "context": context,
                            "sentence": self._extract_sentence(text, subj, obj),
                            "method": "pattern"
                        }
                        relations.append(relation)
                        break  # Found a match, no need to try other patterns for this type

        return relations

    def _classify_relation(
        self,
        subject_type: str,
        object_type: str,
        context: str
    ) -> Optional[str]:
        """
        Classify the relationship type based on entity types and context.

        Args:
            subject_type: Type of subject entity
            object_type: Type of object entity
            context: Textual context between entities

        Returns:
            Relation type or None
        """
        context_lower = context.lower()

        # CEO relationships
        if ("ceo" in context_lower or "chief executive" in context_lower):
            return "CEO_OF"

        # Work relationships
        if ("works" in context_lower or "employed" in context_lower or
            "joined" in context_lower):
            return "WORKS_AT"

        # Location relationships
        if ("headquartered" in context_lower or "based" in context_lower):
            return "HEADQUARTERED_IN"
        if ("in" in context_lower or "located" in context_lower):
            return "LOCATED_IN"

        # Acquisition
        if ("acquired" in context_lower or "bought" in context_lower):
            return "ACQUIRED"

        # Founded
        if ("founded" in context_lower or "established" in context_lower):
            return "FOUNDED"

        # Partnership
        if ("partner" in context_lower or "collaborated" in context_lower):
            return "PARTNERED_WITH"

        return None

    def _classify_from_dependency(
        self,
        dep_path: List,
        subject_type: str,
        object_type: str
    ) -> Optional[str]:
        """
        Classify relation from dependency path.

        Args:
            dep_path: Dependency path tokens
            subject_type: Subject entity type
            object_type: Object entity type

        Returns:
            Relation type or None
        """
        if not dep_path:
            return None

        # Extract verb from path (usually the ROOT)
        verb = None
        for token in dep_path:
            if token.pos_ == "VERB":
                verb = token.lemma_
                break

        if not verb:
            return None

        # Classify based on verb and entity types
        if verb in ["be", "become"] and subject_type == "PERSON" and object_type in ["ORG", "ORGANIZATION"]:
            return "CEO_OF"

        if verb in ["work", "employ", "join"] and object_type in ["ORG", "ORGANIZATION"]:
            return "WORKS_AT"

        if verb in ["locate", "base", "headquarter"]:
            return "LOCATED_IN"

        if verb in ["acquire", "buy", "purchase"]:
            return "ACQUIRED"

        if verb in ["found", "establish", "create", "start"]:
            return "FOUNDED"

        return None

    def get_entity_pairs(
        self,
        entities: List[Dict[str, Any]],
        max_distance: Optional[int] = None
    ) -> List[Tuple]:
        """
        Generate all possible entity pairs for relation extraction.

        Args:
            entities: List of entities
            max_distance: Maximum character distance between entities (optional)

        Returns:
            List of (subject, object) entity pairs

        Examples:
            >>> extractor = RelationExtractor()
            >>> entities = [
            ...     {"text": "A", "start_char": 0, "end_char": 1},
            ...     {"text": "B", "start_char": 10, "end_char": 11}
            ... ]
            >>> pairs = extractor.get_entity_pairs(entities)
            >>> len(pairs) >= 1
            True
        """
        if max_distance is None:
            max_distance = self.max_distance * 10  # Approximate word distance to char distance

        pairs = []

        for subj, obj in combinations(entities, 2):
            # Calculate distance
            distance = abs(subj['start_char'] - obj['end_char'])

            if distance <= max_distance:
                # Add both orderings
                pairs.append((subj, obj))
                # Also try reverse order
                pairs.append((obj, subj))

        self.logger.debug(f"Generated {len(pairs)} entity pairs from {len(entities)} entities")

        return pairs

    def _find_dependency_path(self, token1, token2) -> List:
        """
        Find dependency path between two tokens.

        Args:
            token1: First token
            token2: Second token

        Returns:
            List of tokens in the path
        """
        # Find common ancestor
        ancestors1 = set([token1] + list(token1.ancestors))
        ancestors2 = set([token2] + list(token2.ancestors))

        common_ancestors = ancestors1 & ancestors2

        if not common_ancestors:
            return []

        # Find lowest common ancestor
        lca = min(common_ancestors, key=lambda t: t.i)

        # Build path
        path1 = []
        current = token1
        while current != lca:
            path1.append(current)
            current = current.head

        path2 = []
        current = token2
        while current != lca:
            path2.append(current)
            current = current.head

        path = path1 + [lca] + list(reversed(path2))

        return path

    def _extract_context(self, sentence, subj_span, obj_span) -> str:
        """
        Extract context between two entity spans.

        Args:
            sentence: Sentence tokens
            subj_span: Subject entity span
            obj_span: Object entity span

        Returns:
            Context string
        """
        # Get indices
        start_idx = max(subj_span.end, 0)
        end_idx = min(obj_span.start, len(sentence))

        if start_idx >= end_idx:
            start_idx = max(obj_span.end, 0)
            end_idx = min(subj_span.start, len(sentence))

        context_tokens = sentence[start_idx:end_idx]
        context = " ".join([t.text for t in context_tokens])

        return context.strip()

    def _extract_pattern_context(
        self,
        text: str,
        subj: Dict[str, Any],
        obj: Dict[str, Any]
    ) -> str:
        """
        Extract context between entities for pattern matching.

        Args:
            text: Full text
            subj: Subject entity
            obj: Object entity

        Returns:
            Context string
        """
        start = min(subj['end_char'], obj['end_char'])
        end = max(subj['start_char'], obj['start_char'])

        context = text[start:end].strip()

        return context

    def _extract_sentence(
        self,
        text: str,
        subj: Dict[str, Any],
        obj: Dict[str, Any]
    ) -> str:
        """
        Extract the sentence containing both entities.

        Args:
            text: Full text
            subj: Subject entity
            obj: Object entity

        Returns:
            Sentence text
        """
        # Simple sentence extraction (could be improved with NLP)
        start = min(subj['start_char'], obj['start_char'])
        end = max(subj['end_char'], obj['end_char'])

        # Find sentence boundaries
        sent_start = text.rfind('.', 0, start) + 1
        if sent_start == 0:
            sent_start = 0

        sent_end = text.find('.', end)
        if sent_end == -1:
            sent_end = len(text)
        else:
            sent_end += 1

        sentence = text[sent_start:sent_end].strip()

        return sentence

    def _calculate_pattern_confidence(
        self,
        pattern: str,
        subj: Dict[str, Any],
        obj: Dict[str, Any]
    ) -> float:
        """
        Calculate confidence for pattern-based extraction.

        Args:
            pattern: Matched pattern
            subj: Subject entity
            obj: Object entity

        Returns:
            Confidence score (0-1)
        """
        base_confidence = 0.75

        # Longer patterns are more specific
        pattern_length = len(pattern)
        if pattern_length > 50:
            length_factor = 1.0
        elif pattern_length > 30:
            length_factor = 0.95
        else:
            length_factor = 0.9

        # Entity confidence
        ent_confidence = (
            subj.get('confidence', 0.8) + obj.get('confidence', 0.8)
        ) / 2

        confidence = base_confidence * length_factor * ent_confidence

        return round(min(confidence, 1.0), 2)

    def _deduplicate_relations(
        self,
        relations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        relation_groups = {}

        for relation in relations:
            key = (
                relation['subject'],
                relation['predicate'],
                relation['object']
            )

            if key not in relation_groups:
                relation_groups[key] = []
            relation_groups[key].append(relation)

        # Keep highest confidence from each group
        unique_relations = []
        for group in relation_groups.values():
            best_relation = max(group, key=lambda r: r.get('confidence', 0))
            unique_relations.append(best_relation)

        return unique_relations


def extract_dependency_path(doc, entity1_span, entity2_span) -> str:
    """
    Extract dependency path between two entities as string.

    Args:
        doc: SpaCy Doc
        entity1_span: Span of first entity
        entity2_span: Span of second entity

    Returns:
        Dependency path as string

    Examples:
        >>> # Would require actual SpaCy doc and spans
        >>> # path = extract_dependency_path(doc, span1, span2)
        pass
    """
    # Get root tokens
    root1 = entity1_span.root
    root2 = entity2_span.root

    # Find path
    ancestors1 = list(root1.ancestors)
    ancestors2 = list(root2.ancestors)

    # Find common ancestor
    common = set(ancestors1) & set(ancestors2)

    if not common:
        return ""

    lca = min(common, key=lambda t: t.i)

    # Build path string
    path_tokens = []

    current = root1
    while current != lca:
        path_tokens.append(f"{current.text}({current.dep_})")
        current = current.head

    path_tokens.append(f"{lca.text}(ROOT)")

    current = root2
    reverse_path = []
    while current != lca:
        reverse_path.append(f"{current.text}({current.dep_})")
        current = current.head

    path_tokens.extend(reversed(reverse_path))

    return " -> ".join(path_tokens)


# Module logger
logger = logging.getLogger(__name__)
