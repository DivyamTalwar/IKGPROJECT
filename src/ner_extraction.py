"""
Named Entity Recognition Module

This module provides functionality for extracting named entities from text
using SpaCy's large language models with support for batch processing,
confidence scoring, and custom entity types.
"""

import spacy
from typing import List, Dict, Any, Optional, Union
import json
import logging
from pathlib import Path
from collections import Counter, defaultdict
from .utils import save_json, time_function
 

class NERExtractor:
    """
    Named Entity Recognition extractor using SpaCy.

    This class provides comprehensive NER capabilities including single and batch
    text processing, entity statistics, and confidence scoring.

    Attributes:
        model_name (str): SpaCy model name
        nlp: Loaded SpaCy model
        entity_types (List[str]): Entity types to extract

    Examples:
        >>> ner = NERExtractor(model_name="en_core_web_lg")
        >>> result = ner.extract_entities("Elon Musk founded Tesla.")
        >>> len(result['entities']) > 0
        True
    """

    def __init__(
        self,
        model_name: str = "en_core_web_lg",
        custom_entity_types: Optional[List[str]] = None,
        disable_components: Optional[List[str]] = None
    ):
        """
        Initialize NER extractor.

        Args:
            model_name: SpaCy model to use (default: en_core_web_lg)
            custom_entity_types: Additional entity types to recognize
            disable_components: Pipeline components to disable for speed

        Raises:
            OSError: If SpaCy model is not installed
        """
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

        # Load SpaCy model
        try:
            self.logger.info(f"Loading SpaCy model: {model_name}")
            self.nlp = spacy.load(model_name)
            self.logger.info(f"Successfully loaded {model_name}")
        except OSError:
            self.logger.error(
                f"SpaCy model '{model_name}' not found. "
                f"Install it with: python -m spacy download {model_name}"
            )
            raise

        # Disable unnecessary components for speed
        if disable_components is None:
            # Keep only essential components for NER
            all_pipes = self.nlp.pipe_names
            disable_components = [
                pipe for pipe in all_pipes
                if pipe not in ["tok2vec", "tagger", "parser", "ner", "transformer"]
            ]

        if disable_components:
            self.nlp.disable_pipes(*disable_components)
            self.logger.info(f"Disabled pipeline components: {disable_components}")

        # Set up entity types
        self.entity_types = list(self.nlp.get_pipe("ner").labels)
        if custom_entity_types:
            self.entity_types.extend(custom_entity_types)

        self.logger.info(f"Initialized NER with {len(self.entity_types)} entity types")

    def extract_entities(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            self.logger.warning("Empty text provided for entity extraction")
            return {
                "text": text,
                "entities": [],
                "entity_count": {}
            }

        try:
            # Process text with SpaCy
            doc = self.nlp(text)

            # Extract entities
            entities = []
            for ent in doc.ents:
                entity_dict = {
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "confidence": self._calculate_confidence(ent)
                }
                entities.append(entity_dict)

            # Count entities by type
            entity_count = Counter(ent['label'] for ent in entities)

            result = {
                "text": text,
                "entities": entities,
                "entity_count": dict(entity_count)
            }

            self.logger.debug(
                f"Extracted {len(entities)} entities from text "
                f"({len(text)} characters)"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error processing text: {e}")
            return {
                "text": text,
                "entities": [],
                "entity_count": {},
                "error": str(e)
            }

    @time_function
    def extract_entities_batch(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        if not documents:
            self.logger.warning("No documents provided for batch extraction")
            return []

        results = []
        texts = [doc.get('text', '') for doc in documents]

        self.logger.info(
            f"Processing {len(documents)} documents in batches of {batch_size}"
        )

        try:
            # Process documents in batches
            for i, (doc_obj, doc_metadata) in enumerate(
                zip(self.nlp.pipe(texts, batch_size=batch_size), documents)
            ):
                # Extract entities from processed document
                entities = []
                for ent in doc_obj.ents:
                    entity_dict = {
                        "text": ent.text,
                        "label": ent.label_,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                        "confidence": self._calculate_confidence(ent)
                    }
                    entities.append(entity_dict)

                # Count entities by type
                entity_count = Counter(ent['label'] for ent in entities)

                # Create result dictionary
                result = {
                    **doc_metadata,  # Include original metadata (id, source, etc.)
                    "entities": entities,
                    "entity_count": dict(entity_count)
                }

                results.append(result)

                # Log progress
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(documents)} documents")

            self.logger.info(
                f"Batch processing complete. Processed {len(results)} documents"
            )

            return results

        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")
            raise

    def _calculate_confidence(self, entity) -> float:
        """
        Calculate confidence score for an entity.

        Confidence is based on:
        - Entity length (longer entities are more reliable)
        - Entity type frequency
        - Capitalization patterns

        Args:
            entity: SpaCy entity span

        Returns:
            Confidence score between 0 and 1

        Note:
            This is a heuristic approach. SpaCy doesn't provide direct
            confidence scores for NER in all models.
        """
        base_confidence = 0.8

        # Length factor: longer entities are more reliable
        length = len(entity.text)
        if length > 10:
            length_factor = 1.0
        elif length > 5:
            length_factor = 0.95
        elif length > 2:
            length_factor = 0.9
        else:
            length_factor = 0.85

        # Capitalization factor
        if entity.text[0].isupper():
            cap_factor = 1.0
        else:
            cap_factor = 0.9

        # Combine factors
        confidence = base_confidence * length_factor * cap_factor

        # Ensure confidence is between 0 and 1
        confidence = max(0.0, min(1.0, confidence))

        return round(confidence, 2)

    def get_entity_statistics(
        self,
        extraction_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate statistics across multiple extractions.

        Args:
            extraction_results: List of extraction results from extract_entities_batch

        Returns:
            Statistics dictionary with:
                - total_documents: Number of documents processed
                - total_entities: Total entities extracted
                - unique_entities: Number of unique entity texts
                - entity_type_distribution: Count by entity type
                - avg_entities_per_doc: Average entities per document
                - top_entities: Most frequent entities

        Examples:
            >>> ner = NERExtractor()
            >>> docs = [{"text": "Apple in California"}, {"text": "Microsoft in Washington"}]
            >>> results = ner.extract_entities_batch(docs)
            >>> stats = ner.get_entity_statistics(results)
            >>> stats['total_documents'] == 2
            True
        """
        if not extraction_results:
            return {
                "total_documents": 0,
                "total_entities": 0,
                "unique_entities": 0,
                "entity_type_distribution": {},
                "avg_entities_per_doc": 0.0,
                "top_entities": []
            }

        # Collect all entities
        all_entities = []
        entity_type_counts = Counter()
        entity_text_counts = Counter()

        for result in extraction_results:
            entities = result.get('entities', [])
            all_entities.extend(entities)

            for entity in entities:
                entity_type_counts[entity['label']] += 1
                entity_text_counts[(entity['text'], entity['label'])] += 1

        # Calculate statistics
        total_entities = len(all_entities)
        unique_entities = len(entity_text_counts)
        total_documents = len(extraction_results)

        # Get top entities
        top_entities = [
            {
                "text": text,
                "label": label,
                "count": count
            }
            for (text, label), count in entity_text_counts.most_common(20)
        ]

        statistics = {
            "total_documents": total_documents,
            "total_entities": total_entities,
            "unique_entities": unique_entities,
            "entity_type_distribution": dict(entity_type_counts),
            "avg_entities_per_doc": round(total_entities / total_documents, 2) if total_documents > 0 else 0.0,
            "top_entities": top_entities
        }

        self.logger.info(
            f"Statistics: {total_entities} entities across "
            f"{total_documents} documents"
        )

        return statistics

    def save_results(
        self,
        results: Union[Dict[str, Any], List[Dict[str, Any]]],
        output_path: str
    ) -> None:
        """
        Save extraction results to JSON file.

        Args:
            results: Extraction results (single or list)
            output_path: Path to save JSON file

        Examples:
            >>> ner = NERExtractor()
            >>> result = ner.extract_entities("Test text")
            >>> ner.save_results(result, "output/entities.json")
        """
        try:
            save_json(results, output_path, indent=2)
            self.logger.info(f"Saved results to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise

    def filter_entities(
        self,
        entities: List[Dict[str, Any]],
        entity_types: Optional[List[str]] = None,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Filter entities by type and confidence.

        Args:
            entities: List of entity dictionaries
            entity_types: Entity types to keep (None = keep all)
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered entity list

        Examples:
            >>> ner = NERExtractor()
            >>> result = ner.extract_entities("Apple Inc. in California on January 1, 2024")
            >>> orgs = ner.filter_entities(result['entities'], entity_types=['ORG'])
            >>> all(e['label'] == 'ORG' for e in orgs)
            True
        """
        filtered = entities

        if entity_types:
            filtered = [e for e in filtered if e['label'] in entity_types]

        if min_confidence > 0:
            filtered = [e for e in filtered if e.get('confidence', 0) >= min_confidence]

        self.logger.debug(
            f"Filtered {len(entities)} entities to {len(filtered)} "
            f"(types={entity_types}, min_conf={min_confidence})"
        )

        return filtered


# Helper Functions

def visualize_entities(text: str, entities: List[Dict[str, Any]]) -> str:
    """
    Create a visual representation of entities in text.

    Highlights entities inline with their types.

    Args:
        text: Original text
        entities: List of extracted entities

    Returns:
        Text with entities highlighted

    Examples:
        >>> entities = [{"text": "Apple", "label": "ORG", "start_char": 0, "end_char": 5}]
        >>> result = visualize_entities("Apple is a company", entities)
        >>> "[Apple](ORG)" in result
        True
    """
    if not entities:
        return text

    # Sort entities by start position (reverse to replace from end to beginning)
    sorted_entities = sorted(entities, key=lambda e: e['start_char'], reverse=True)

    # Replace entities with highlighted version
    highlighted_text = text
    for entity in sorted_entities:
        start = entity['start_char']
        end = entity['end_char']
        entity_text = entity['text']
        label = entity['label']

        # Create highlighted version
        highlight = f"[{entity_text}]({label})"

        # Replace in text
        highlighted_text = highlighted_text[:start] + highlight + highlighted_text[end:]

    return highlighted_text


def merge_overlapping_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Handle overlapping entity spans by keeping the longest/highest confidence.

    Args:
        entities: List of entities

    Returns:
        Merged entity list with no overlaps

    Examples:
        >>> entities = [
        ...     {"text": "New York", "start_char": 0, "end_char": 8, "confidence": 0.9},
        ...     {"text": "York", "start_char": 4, "end_char": 8, "confidence": 0.7}
        ... ]
        >>> merged = merge_overlapping_entities(entities)
        >>> len(merged)
        1
        >>> merged[0]['text']
        'New York'
    """
    if not entities:
        return []

    # Sort by start position
    sorted_entities = sorted(entities, key=lambda e: (e['start_char'], -e['end_char']))

    merged = []
    current_end = -1

    for entity in sorted_entities:
        # Check if this entity overlaps with the previous one
        if entity['start_char'] >= current_end:
            # No overlap, add to merged list
            merged.append(entity)
            current_end = entity['end_char']
        else:
            # Overlap detected - keep the one with higher confidence
            if entity.get('confidence', 0) > merged[-1].get('confidence', 0):
                merged[-1] = entity
                current_end = entity['end_char']

    logger = logging.getLogger(__name__)
    logger.debug(f"Merged {len(entities)} entities to {len(merged)} (removed {len(entities) - len(merged)} overlaps)")

    return merged


def group_entities_by_type(entities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group entities by their type.

    Args:
        entities: List of entity dictionaries

    Returns:
        Dictionary mapping entity types to entity lists

    Examples:
        >>> entities = [
        ...     {"text": "Apple", "label": "ORG"},
        ...     {"text": "California", "label": "GPE"},
        ...     {"text": "Microsoft", "label": "ORG"}
        ... ]
        >>> grouped = group_entities_by_type(entities)
        >>> len(grouped['ORG'])
        2
    """
    grouped = defaultdict(list)

    for entity in entities:
        label = entity.get('label', 'UNKNOWN')
        grouped[label].append(entity)

    return dict(grouped)


# Module-level logger
logger = logging.getLogger(__name__)
