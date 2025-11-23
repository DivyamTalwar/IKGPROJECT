"""
Knowledge Graph Construction Module

Builds graph databases from extracted triples using Neo4j or NetworkX.
Provides unified interface for both graph database backends.
"""

from typing import List, Dict, Any, Optional, Union
import logging
from abc import ABC, abstractmethod
from collections import Counter
import json


class KGConstructor(ABC):
    """
    Abstract base class for Knowledge Graph constructors.

    Defines the interface that all KG constructors must implement.
    """

    @abstractmethod
    def create_node(self, entity_id: str, entity_type: str, properties: Dict[str, Any]) -> None:
        """Create a node in the graph."""
        pass

    @abstractmethod
    def create_relationship(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        properties: Dict[str, Any]
    ) -> None:
        """Create a relationship between nodes."""
        pass

    @abstractmethod
    def build_graph(self, triples: List[Dict[str, Any]]) -> Any:
        """Build the complete graph from triples."""
        pass

    @abstractmethod
    def export_graph(self, output_path: str, format: str = "graphml") -> None:
        """Export graph to file."""
        pass

    @abstractmethod
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        pass

    @abstractmethod
    def clear_graph(self) -> None:
        """Clear all nodes and relationships."""
        pass


class Neo4jKGConstructor(KGConstructor):
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j"
    ):
        self.logger = logging.getLogger(__name__)
        self.uri = uri
        self.database = database

        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(uri, auth=(user, password))

            # Test connection
            with self.driver.session(database=database) as session:
                session.run("RETURN 1")

            self.logger.info(f"Connected to Neo4j at {uri}")
        except ImportError:
            self.logger.error("neo4j package not installed. Install with: pip install neo4j")
            raise
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def create_node(
        self,
        entity_id: str,
        entity_type: str,
        properties: Dict[str, Any]
    ) -> None:
        """
        Create a node in Neo4j.

        Uses MERGE to avoid duplicates. Updates properties if node exists.

        Args:
            entity_id: Unique entity identifier
            entity_type: Node label (Person, Organization, etc.)
            properties: Node properties
        """
        query = f"""
        MERGE (n:{entity_type} {{id: $id}})
        SET n += $properties
        RETURN n
        """

        with self.driver.session(database=self.database) as session:
            session.run(query, id=entity_id, properties=properties)

        self.logger.debug(f"Created/updated node: {entity_type}({entity_id})")

    def create_relationship(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        properties: Dict[str, Any]
    ) -> None:
        """
        Create a relationship in Neo4j.

        Creates nodes if they don't exist, then creates relationship.

        Args:
            subject_id: Source node ID
            predicate: Relationship type
            object_id: Target node ID
            properties: Relationship properties
        """
        query = f"""
        MATCH (a {{id: $subject_id}})
        MATCH (b {{id: $object_id}})
        MERGE (a)-[r:{predicate}]->(b)
        SET r += $properties
        RETURN r
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                subject_id=subject_id,
                object_id=object_id,
                properties=properties
            )

        self.logger.debug(f"Created relationship: {subject_id}-[{predicate}]->{object_id}")

    def build_graph(self, triples: List[Dict[str, Any]], batch_size: int = 1000) -> None:
        """
        Build complete graph from triples.

        Uses batch operations for efficiency.

        Args:
            triples: List of triple dictionaries
            batch_size: Number of operations per batch
        """
        if not triples:
            self.logger.warning("No triples to build graph from")
            return

        self.logger.info(f"Building graph from {len(triples)} triples")

        # Extract unique nodes
        nodes_by_id = {}
        for triple in triples:
            # Add subject node
            subj_id = triple['subject_id']
            if subj_id not in nodes_by_id:
                nodes_by_id[subj_id] = {
                    'id': subj_id,
                    'type': triple['subject_type'],
                    'name': triple['subject']
                }

            # Add object node
            obj_id = triple['object_id']
            if obj_id not in nodes_by_id:
                nodes_by_id[obj_id] = {
                    'id': obj_id,
                    'type': triple['object_type'],
                    'name': triple['object']
                }

        # Create nodes in batches
        nodes = list(nodes_by_id.values())
        self.logger.info(f"Creating {len(nodes)} unique nodes")

        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            self._create_nodes_batch(batch)

            if (i + batch_size) % (batch_size * 10) == 0:
                self.logger.info(f"Created {min(i + batch_size, len(nodes))}/{len(nodes)} nodes")

        # Create relationships in batches
        self.logger.info(f"Creating {len(triples)} relationships")

        for i in range(0, len(triples), batch_size):
            batch = triples[i:i + batch_size]
            self._create_relationships_batch(batch)

            if (i + batch_size) % (batch_size * 10) == 0:
                self.logger.info(f"Created {min(i + batch_size, len(triples))}/{len(triples)} relationships")

        self.logger.info("Graph construction complete")

    def _create_nodes_batch(self, nodes: List[Dict[str, Any]]) -> None:
        query = """
        UNWIND $nodes AS node
        CALL apoc.merge.node([node.type], {id: node.id}, {name: node.name}, {}) YIELD node AS n
        RETURN count(n)
        """

        fallback_query = """
        UNWIND $nodes AS node
        CALL {
            WITH node
            MERGE (n {id: node.id})
            SET n.name = node.name
            WITH n, node
            CALL apoc.create.addLabels(n, [node.type]) YIELD node AS labeled
            RETURN labeled
        }
        RETURN count(*)
        """

        # Simple query that works without APOC
        simple_query = """
        UNWIND $nodes AS node
        MERGE (n {id: node.id})
        SET n.name = node.name, n.type = node.type
        RETURN count(n)
        """

        with self.driver.session(database=self.database) as session:
            try:
                session.run(simple_query, nodes=nodes)
            except Exception as e:
                self.logger.warning(f"Batch node creation failed, falling back to individual creation: {e}")
                # Fallback: create nodes individually
                for node in nodes:
                    self.create_node(node['id'], node['type'], {'name': node['name']})

    def _create_relationships_batch(self, triples: List[Dict[str, Any]]) -> None:
        """Create multiple relationships in a single transaction."""
        query = """
        UNWIND $triples AS triple
        MATCH (a {id: triple.subject_id})
        MATCH (b {id: triple.object_id})
        CALL apoc.create.relationship(a, triple.predicate, {
            confidence: triple.confidence,
            context: triple.context
        }, b) YIELD rel
        RETURN count(rel)
        """

        # Simple query that works without APOC
        with self.driver.session(database=self.database) as session:
            for triple in triples:
                rel_query = f"""
                MATCH (a {{id: $subject_id}})
                MATCH (b {{id: $object_id}})
                MERGE (a)-[r:{triple['predicate']}]->(b)
                SET r.confidence = $confidence
                RETURN r
                """

                session.run(
                    rel_query,
                    subject_id=triple['subject_id'],
                    object_id=triple['object_id'],
                    confidence=triple.get('confidence', 0.5)
                )

    def clear_graph(self) -> None:
        """Clear all nodes and relationships from the graph."""
        query = "MATCH (n) DETACH DELETE n"

        with self.driver.session(database=self.database) as session:
            session.run(query)

        self.logger.info("Cleared all nodes and relationships from graph")

    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get graph statistics.

        Returns:
            Dictionary with node counts, edge counts, etc.
        """
        stats = {}

        with self.driver.session(database=self.database) as session:
            # Total node count
            result = session.run("MATCH (n) RETURN count(n) AS count")
            stats['total_nodes'] = result.single()['count']

            # Total relationship count
            result = session.run("MATCH ()-[r]->() RETURN count(r) AS count")
            stats['total_relationships'] = result.single()['count']

            # Node types distribution
            result = session.run("""
                MATCH (n)
                RETURN n.type AS type, count(*) AS count
                ORDER BY count DESC
            """)
            stats['node_type_distribution'] = {
                record['type']: record['count']
                for record in result
            }

            # Relationship types distribution
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(*) AS count
                ORDER BY count DESC
            """)
            stats['relationship_type_distribution'] = {
                record['type']: record['count']
                for record in result
            }

        self.logger.info(f"Graph stats: {stats['total_nodes']} nodes, {stats['total_relationships']} relationships")

        return stats

    def export_graph(self, output_path: str, format: str = "json") -> None:
        """
        Export graph to file.

        Args:
            output_path: Output file path
            format: Export format (json, graphml)
        """
        if format == "json":
            # Export as JSON
            graph_data = {
                'nodes': [],
                'edges': []
            }

            with self.driver.session(database=self.database) as session:
                # Get all nodes
                result = session.run("MATCH (n) RETURN n")
                for record in result:
                    node = record['n']
                    graph_data['nodes'].append({
                        'id': node.get('id'),
                        'type': node.get('type'),
                        'name': node.get('name'),
                        'properties': dict(node)
                    })

                # Get all relationships
                result = session.run("MATCH (a)-[r]->(b) RETURN a.id AS source, type(r) AS type, b.id AS target, r")
                for record in result:
                    graph_data['edges'].append({
                        'source': record['source'],
                        'target': record['target'],
                        'type': record['type'],
                        'properties': dict(record['r'])
                    })

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2)

            self.logger.info(f"Exported graph to {output_path} (JSON)")

        else:
            raise ValueError(f"Unsupported export format: {format}")

    def close(self) -> None:
        """Close Neo4j connection."""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            self.logger.info("Closed Neo4j connection")


class NetworkXKGConstructor(KGConstructor):
    """
    Knowledge Graph constructor using NetworkX.

    Pure Python graph implementation, no external database required.
    Ideal for smaller graphs and prototyping.

    Attributes:
        graph: NetworkX MultiDiGraph instance

    Examples:
        >>> kg = NetworkXKGConstructor()
        >>> triples = [{
        ...     "subject": "Tesla", "subject_id": "orga_tesla",
        ...     "predicate": "LOCATED_IN",
        ...     "object": "California", "object_id": "gpe_california",
        ...     "subject_type": "ORG", "object_type": "GPE",
        ...     "confidence": 0.9
        ... }]
        >>> graph = kg.build_graph(triples)
        >>> stats = kg.get_graph_stats()
        >>> stats['total_nodes'] > 0
        True
    """

    def __init__(self):
        """Initialize NetworkX graph."""
        import networkx as nx
        self.graph = nx.MultiDiGraph()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized NetworkX KG constructor")

    def create_node(
        self,
        entity_id: str,
        entity_type: str,
        properties: Dict[str, Any]
    ) -> None:
        """
        Add node to NetworkX graph.

        Args:
            entity_id: Unique entity identifier
            entity_type: Node type
            properties: Node attributes
        """
        self.graph.add_node(
            entity_id,
            type=entity_type,
            **properties
        )

        self.logger.debug(f"Created node: {entity_type}({entity_id})")

    def create_relationship(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        properties: Dict[str, Any]
    ) -> None:
        """
        Add edge to NetworkX graph.

        Args:
            subject_id: Source node
            predicate: Edge type
            object_id: Target node
            properties: Edge attributes
        """
        self.graph.add_edge(
            subject_id,
            object_id,
            key=predicate,
            type=predicate,
            **properties
        )

        self.logger.debug(f"Created edge: {subject_id}-[{predicate}]->{object_id}")

    def build_graph(self, triples: List[Dict[str, Any]]) -> 'nx.MultiDiGraph':
        """
        Build NetworkX graph from triples.

        Args:
            triples: List of triple dictionaries

        Returns:
            NetworkX graph object
        """
        if not triples:
            self.logger.warning("No triples to build graph from")
            return self.graph

        self.logger.info(f"Building graph from {len(triples)} triples")

        # Add all nodes and edges
        for triple in triples:
            # Create subject node
            self.create_node(
                triple['subject_id'],
                triple['subject_type'],
                {'name': triple['subject']}
            )

            # Create object node
            self.create_node(
                triple['object_id'],
                triple['object_type'],
                {'name': triple['object']}
            )

            # Create relationship
            rel_props = {
                'confidence': triple.get('confidence', 0.5),
                'context': triple.get('metadata', {}).get('context', ''),
            }
            self.create_relationship(
                triple['subject_id'],
                triple['predicate'],
                triple['object_id'],
                rel_props
            )

        self.logger.info(
            f"Graph built: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

        return self.graph

    def clear_graph(self) -> None:
        """Clear all nodes and edges from the graph."""
        self.graph.clear()
        self.logger.info("Cleared graph")

    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Calculate graph statistics.

        Returns:
            Dictionary with statistics
        """
        import networkx as nx

        # Basic stats
        total_nodes = self.graph.number_of_nodes()
        total_edges = self.graph.number_of_edges()

        # Node type distribution
        node_types = Counter(
            self.graph.nodes[n].get('type', 'UNKNOWN')
            for n in self.graph.nodes()
        )

        # Edge type distribution
        edge_types = Counter(
            data.get('type', 'UNKNOWN')
            for u, v, data in self.graph.edges(data=True)
        )

        # Density
        density = nx.density(self.graph)

        # Connected components
        num_components = nx.number_weakly_connected_components(self.graph)

        # Average degree
        if total_nodes > 0:
            avg_degree = sum(dict(self.graph.degree()).values()) / total_nodes
        else:
            avg_degree = 0

        stats = {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'density': round(density, 4),
            'num_connected_components': num_components,
            'average_degree': round(avg_degree, 2),
            'node_type_distribution': dict(node_types),
            'edge_type_distribution': dict(edge_types)
        }

        self.logger.info(f"Graph stats: {total_nodes} nodes, {total_edges} edges")

        return stats

    def export_graph(self, output_path: str, format: str = "graphml") -> None:
        """
        Export NetworkX graph to file.

        Args:
            output_path: Output file path
            format: Format (graphml, gexf, json, pickle)
        """
        import networkx as nx
        from pathlib import Path

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if format == "graphml":
            nx.write_graphml(self.graph, output_path)
        elif format == "gexf":
            nx.write_gexf(self.graph, output_path)
        elif format == "json":
            from networkx.readwrite import json_graph
            data = json_graph.node_link_data(self.graph)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        elif format == "pickle":
            nx.write_gpickle(self.graph, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Exported graph to {output_path} ({format})")


# Cypher Query Templates for Neo4j

CREATE_NODE_QUERY = """
MERGE (n:{label} {{id: $id}})
SET n += $properties
RETURN n
"""

CREATE_RELATIONSHIP_QUERY = """
MATCH (a {{id: $subject_id}})
MATCH (b {{id: $object_id}})
MERGE (a)-[r:{rel_type}]->(b)
SET r += $properties
RETURN r
"""

GRAPH_STATS_QUERY = """
MATCH (n)
WITH labels(n) as labels
UNWIND labels as label
RETURN label, count(*) as count
ORDER BY count DESC
"""

# Module logger
logger = logging.getLogger(__name__)
