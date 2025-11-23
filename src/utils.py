"""
Utility Functions Module

Provides helper functions for data loading, preprocessing, logging, and configuration
management for the KG construction pipeline.
"""

import json
import csv
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from datetime import datetime


def setup_logging(
    log_file: str = "kg_construction.log",
    level: int = logging.INFO,
    console_output: bool = True
) -> None:
    """
    Configure logging for the project.

    Args:
        log_file: Path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to output logs to console

    Examples:
        >>> setup_logging("my_app.log", logging.DEBUG)
    """
    handlers = [logging.FileHandler(log_file)]

    if console_output:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")


def load_json(file_path: Union[str, Path]) -> Any:
    """
    Load data from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data (dict, list, etc.)

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON

    Examples:
        >>> data = load_json("data/entities.json")
    """
    file_path = Path(file_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded JSON from {file_path}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {file_path}: {e}")
        raise


def save_json(
    data: Any,
    file_path: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False
) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save (must be JSON serializable)
        file_path: Output file path
        indent: JSON indentation level
        ensure_ascii: Whether to escape non-ASCII characters

    Examples:
        >>> save_json({"key": "value"}, "output.json")
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        logging.info(f"Successfully saved JSON to {file_path}")
    except Exception as e:
        logging.error(f"Error saving JSON to {file_path}: {e}")
        raise


def load_csv(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load CSV file as list of dictionaries.

    Args:
        file_path: Path to CSV file

    Returns:
        List of dictionaries (one per row)

    Examples:
        >>> data = load_csv("data/entities.csv")
    """
    file_path = Path(file_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        logging.info(f"Successfully loaded {len(data)} rows from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading CSV from {file_path}: {e}")
        raise


def save_csv(
    data: List[Dict[str, Any]],
    file_path: Union[str, Path],
    fieldnames: Optional[List[str]] = None
) -> None:
    """
    Save data to CSV file.

    Args:
        data: List of dictionaries to save
        file_path: Output file path
        fieldnames: Column names (auto-detected if None)

    Examples:
        >>> save_csv([{"name": "John", "age": 30}], "output.csv")
    """
    if not data:
        logging.warning("No data to save to CSV")
        return

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        fieldnames = list(data[0].keys())

    try:
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        logging.info(f"Successfully saved {len(data)} rows to {file_path}")
    except Exception as e:
        logging.error(f"Error saving CSV to {file_path}: {e}")
        raise


def load_config(config_path: Union[str, Path] = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary

    Examples:
        >>> config = load_config("config/config.yaml")
        >>> model_name = config['ner']['model']
    """
    config_path = Path(config_path)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info(f"Successfully loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Invalid YAML in {config_path}: {e}")
        raise


def load_text_file(file_path: Union[str, Path]) -> str:
    """
    Load text from file.

    Args:
        file_path: Path to text file

    Returns:
        File contents as string
    """
    file_path = Path(file_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logging.info(f"Successfully loaded text from {file_path}")
        return text
    except Exception as e:
        logging.error(f"Error loading text from {file_path}: {e}")
        raise


def save_text_file(text: str, file_path: Union[str, Path]) -> None:
    """
    Save text to file.

    Args:
        text: Text content to save
        file_path: Output file path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logging.info(f"Successfully saved text to {file_path}")
    except Exception as e:
        logging.error(f"Error saving text to {file_path}: {e}")
        raise


def clean_text(text: str) -> str:
    """
    Clean and normalize text.

    Performs the following operations:
    - Removes extra whitespace
    - Normalizes line breaks
    - Removes control characters

    Args:
        text: Input text

    Returns:
        Cleaned text

    Examples:
        >>> clean_text("Hello   world\\n\\n\\nTest")
        'Hello world\\nTest'
    """
    if not text:
        return ""

    # Remove control characters except newline and tab
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)

    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)

    # Normalize line breaks
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words.

    Simple whitespace tokenization.

    Args:
        text: Input text

    Returns:
        List of tokens

    Examples:
        >>> tokenize_text("Hello world!")
        ['Hello', 'world!']
    """
    return text.split()


def normalize_entity_text(text: str, entity_type: str = None) -> str:
    """
    Normalize entity text for matching.

    Args:
        text: Entity text
        entity_type: Entity type (optional)

    Returns:
        Normalized text

    Examples:
        >>> normalize_entity_text("Tesla Inc.")
        'tesla'
        >>> normalize_entity_text("  New York  ")
        'new york'
    """
    if not text:
        return ""

    # Convert to lowercase
    normalized = text.lower()

    # Remove common suffixes for organizations
    if entity_type == "ORGANIZATION":
        suffixes = ["inc.", "llc", "corp.", "ltd.", "co.", "corporation", "company"]
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()

    # Remove extra whitespace
    normalized = ' '.join(normalized.split())

    # Remove special characters but keep spaces
    normalized = re.sub(r'[^\w\s-]', '', normalized)

    return normalized.strip()


def merge_entities(entity_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge duplicate entities based on normalized text.

    Keeps the entity with highest confidence for each unique normalized text.

    Args:
        entity_list: List of entity dictionaries

    Returns:
        Deduplicated entity list

    Examples:
        >>> entities = [
        ...     {"text": "Tesla", "label": "ORG", "confidence": 0.9},
        ...     {"text": "Tesla Inc.", "label": "ORG", "confidence": 0.8}
        ... ]
        >>> merged = merge_entities(entities)
        >>> len(merged)
        1
    """
    if not entity_list:
        return []

    # Group by normalized text and type
    entity_groups = {}

    for entity in entity_list:
        normalized = normalize_entity_text(
            entity.get('text', ''),
            entity.get('label', '')
        )
        key = (normalized, entity.get('label', ''))

        if key not in entity_groups:
            entity_groups[key] = []
        entity_groups[key].append(entity)

    # Keep highest confidence entity from each group
    merged = []
    for entities in entity_groups.values():
        best_entity = max(entities, key=lambda e: e.get('confidence', 0))
        merged.append(best_entity)

    return merged


def time_function(func):
    """
    Decorator to time function execution.

    Logs the execution time of the decorated function.

    Args:
        func: Function to decorate

    Returns:
        Wrapped function

    Examples:
        >>> @time_function
        ... def slow_function():
        ...     time.sleep(1)
        >>> slow_function()
    """
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        logger = logging.getLogger(func.__module__)
        logger.info(f"{func.__name__} took {end - start:.2f} seconds")

        return result

    return wrapper


def create_entity_id(entity_text: str, entity_type: str) -> str:
    """
    Create a unique identifier for an entity.

    Args:
        entity_text: Entity text
        entity_type: Entity type

    Returns:
        Unique entity ID

    Examples:
        >>> create_entity_id("Elon Musk", "PERSON")
        'pers_elon_musk'
        >>> create_entity_id("Tesla Inc.", "ORGANIZATION")
        'orga_tesla'
    """
    # Normalize text
    normalized = normalize_entity_text(entity_text, entity_type)

    # Replace spaces with underscores
    normalized = normalized.replace(' ', '_')

    # Remove any remaining special characters
    normalized = re.sub(r'[^\w]', '', normalized)

    # Create type prefix (first 4 letters)
    prefix = entity_type.lower()[:4]

    # Create ID
    entity_id = f"{prefix}_{normalized}"

    return entity_id


def get_timestamp() -> str:
    """
    Get current timestamp as string.

    Returns:
        ISO format timestamp

    Examples:
        >>> timestamp = get_timestamp()
        >>> '2024' in timestamp
        True
    """
    return datetime.now().isoformat()


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        directory: Directory path

    Returns:
        Path object for the directory

    Examples:
        >>> ensure_dir("output/graphs")
        PosixPath('output/graphs')
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.

    Args:
        items: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks

    Examples:
        >>> chunk_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    chunks = []
    for i in range(0, len(items), chunk_size):
        chunks.append(items[i:i + chunk_size])
    return chunks


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """
    Flatten a nested list.

    Args:
        nested_list: List of lists

    Returns:
        Flattened list

    Examples:
        >>> flatten_list([[1, 2], [3, 4], [5]])
        [1, 2, 3, 4, 5]
    """
    return [item for sublist in nested_list for item in sublist]


# Module-level logger
logger = logging.getLogger(__name__)
