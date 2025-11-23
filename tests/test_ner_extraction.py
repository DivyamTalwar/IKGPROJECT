"""
Unit tests for NER Extraction module.
"""

import pytest
from src.ner_extraction import (
    NERExtractor,
    visualize_entities,
    merge_overlapping_entities,
    group_entities_by_type
)


class TestNERExtractor:
    """Test suite for NERExtractor class."""

    @pytest.fixture
    def ner_extractor(self):
        """Create NER extractor instance for testing."""
        return NERExtractor(model_name="en_core_web_lg")

    def test_initialization(self, ner_extractor):
        """Test NER extractor initializes correctly."""
        assert ner_extractor is not None
        assert ner_extractor.model_name == "en_core_web_lg"
        assert ner_extractor.nlp is not None

    def test_extract_entities_basic(self, ner_extractor):
        """Test basic entity extraction."""
        text = "Elon Musk is the CEO of Tesla."
        result = ner_extractor.extract_entities(text)

        assert 'entities' in result
        assert 'entity_count' in result
        assert len(result['entities']) > 0
        assert result['text'] == text

    def test_extract_entities_empty_text(self, ner_extractor):
        """Test extraction with empty text."""
        result = ner_extractor.extract_entities("")

        assert result['entities'] == []
        assert result['entity_count'] == {}

    def test_extract_entities_with_person(self, ner_extractor):
        """Test extraction finds PERSON entities."""
        text = "Barack Obama was the president."
        result = ner_extractor.extract_entities(text)

        person_entities = [e for e in result['entities'] if e['label'] == 'PERSON']
        assert len(person_entities) > 0

    def test_extract_entities_with_organization(self, ner_extractor):
        """Test extraction finds ORG entities."""
        text = "Microsoft is a technology company."
        result = ner_extractor.extract_entities(text)

        org_entities = [e for e in result['entities'] if e['label'] == 'ORG']
        assert len(org_entities) > 0

    def test_batch_processing(self, ner_extractor):
        """Test batch entity extraction."""
        documents = [
            {"id": "doc1", "text": "Apple Inc. is based in California."},
            {"id": "doc2", "text": "Google was founded in 1998."}
        ]

        results = ner_extractor.extract_entities_batch(documents)

        assert len(results) == 2
        assert all('entities' in r for r in results)
        assert all('id' in r for r in results)

    def test_batch_processing_empty(self, ner_extractor):
        """Test batch processing with empty list."""
        results = ner_extractor.extract_entities_batch([])
        assert results == []

    def test_entity_statistics(self, ner_extractor):
        """Test entity statistics calculation."""
        documents = [
            {"text": "Apple and Microsoft are companies."},
            {"text": "Google and Amazon are also companies."}
        ]

        results = ner_extractor.extract_entities_batch(documents)
        stats = ner_extractor.get_entity_statistics(results)

        assert 'total_documents' in stats
        assert 'total_entities' in stats
        assert 'unique_entities' in stats
        assert stats['total_documents'] == 2

    def test_filter_entities_by_type(self, ner_extractor):
        """Test filtering entities by type."""
        text = "Elon Musk founded Tesla in 2003."
        result = ner_extractor.extract_entities(text)

        persons = ner_extractor.filter_entities(
            result['entities'],
            entity_types=['PERSON']
        )

        assert all(e['label'] == 'PERSON' for e in persons)

    def test_filter_entities_by_confidence(self, ner_extractor):
        """Test filtering entities by confidence."""
        text = "Apple is a company."
        result = ner_extractor.extract_entities(text)

        high_conf = ner_extractor.filter_entities(
            result['entities'],
            min_confidence=0.8
        )

        assert all(e.get('confidence', 0) >= 0.8 for e in high_conf)


class TestHelperFunctions:
    """Test suite for helper functions."""

    def test_visualize_entities(self):
        """Test entity visualization."""
        text = "Apple is a company"
        entities = [
            {"text": "Apple", "label": "ORG", "start_char": 0, "end_char": 5}
        ]

        result = visualize_entities(text, entities)
        assert "[Apple](ORG)" in result

    def test_visualize_entities_empty(self):
        """Test visualization with no entities."""
        text = "No entities here"
        result = visualize_entities(text, [])
        assert result == text

    def test_merge_overlapping_entities(self):
        """Test merging overlapping entities."""
        entities = [
            {
                "text": "New York",
                "start_char": 0,
                "end_char": 8,
                "confidence": 0.9
            },
            {
                "text": "York",
                "start_char": 4,
                "end_char": 8,
                "confidence": 0.7
            }
        ]

        merged = merge_overlapping_entities(entities)
        assert len(merged) == 1
        assert merged[0]['text'] == 'New York'

    def test_merge_non_overlapping_entities(self):
        """Test merging keeps non-overlapping entities."""
        entities = [
            {"text": "Apple", "start_char": 0, "end_char": 5, "confidence": 0.9},
            {"text": "Google", "start_char": 10, "end_char": 16, "confidence": 0.9}
        ]

        merged = merge_overlapping_entities(entities)
        assert len(merged) == 2

    def test_group_entities_by_type(self):
        """Test grouping entities by type."""
        entities = [
            {"text": "Apple", "label": "ORG"},
            {"text": "Google", "label": "ORG"},
            {"text": "California", "label": "GPE"}
        ]

        grouped = group_entities_by_type(entities)

        assert 'ORG' in grouped
        assert 'GPE' in grouped
        assert len(grouped['ORG']) == 2
        assert len(grouped['GPE']) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
