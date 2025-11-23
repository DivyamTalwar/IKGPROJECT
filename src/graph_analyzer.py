"""
Graph Analysis Module

Provides comprehensive analysis of Knowledge Graphs including
centrality metrics, structural analysis, and community detection.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import networkx as nx
from collections import Counter, defaultdict
import logging
from .utils import save_json, time_function


class GraphAnalyzer:
    """
    Analyzes Knowledge Graphs and generates insights.

    Works with both Neo4j and NetworkX graphs, providing
    comprehensive metrics and analysis.

    Attributes:
        graph: Graph object (NetworkX)
        graph_type: Type of graph ("networkx" or "neo4j")

    Examples:
        >>> import networkx as nx
        >>> G = nx.DiGraph()
        >>> G.add_edge("A", "B")
        >>> analyzer = GraphAnalyzer(G)
        >>> stats = analyzer.basic_statistics()
        >>> stats['total_nodes']
        2
    """

    def __init__(self, graph: Any, graph_type: str = "networkx"):
        """
        Initialize analyzer.

        Args:
            graph: Graph object (NetworkX graph or Neo4j driver)
            graph_type: Type of graph ("networkx" or "neo4j")
        """
        self.graph = graph
        self.graph_type = graph_type
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"GraphAnalyzer initialized for {graph_type} graph")

    @time_function
    def analyze_complete(self) -> Dict[str, Any]:
        """
        Perform complete graph analysis.

        Returns:
            Comprehensive analysis dictionary containing:
                - basic_stats
                - centrality_metrics
                - structural_analysis
                - degree_distribution
                - communities (if enabled)

        Examples:
            >>> import networkx as nx
            >>> G = nx.karate_club_graph()
            >>> analyzer = GraphAnalyzer(G)
            >>> analysis = analyzer.analyze_complete()
            >>> 'basic_stats' in analysis
            True
        """
        self.logger.info("Starting complete graph analysis")

        analysis = {
            "basic_stats": self.basic_statistics(),
            "centrality_metrics": self.centrality_analysis(top_n=20),
            "structural_analysis": self.structural_analysis(),
            "degree_distribution": self.degree_distribution(),
        }

        # Add community detection (can be slow for large graphs)
        try:
            analysis["communities"] = self.community_detection()
        except Exception as e:
            self.logger.warning(f"Community detection failed: {e}")
            analysis["communities"] = []

        self.logger.info("Complete analysis finished")

        return analysis

    def basic_statistics(self) -> Dict[str, Any]:
        """
        Calculate basic graph statistics.

        Returns:
            Basic stats dictionary with:
                - total_nodes
                - total_edges
                - graph_density
                - average_degree
                - is_directed
                - is_connected

        Examples:
            >>> import networkx as nx
            >>> G = nx.complete_graph(5)
            >>> analyzer = GraphAnalyzer(G)
            >>> stats = analyzer.basic_statistics()
            >>> stats['total_nodes']
            5
        """
        if self.graph_type != "networkx":
            raise ValueError("Basic statistics only supported for NetworkX graphs")

        # Basic counts
        total_nodes = self.graph.number_of_nodes()
        total_edges = self.graph.number_of_edges()

        # Density
        if total_nodes > 1:
            density = nx.density(self.graph)
        else:
            density = 0.0

        # Average degree
        if total_nodes > 0:
            degrees = dict(self.graph.degree())
            avg_degree = sum(degrees.values()) / total_nodes
        else:
            avg_degree = 0.0

        # Directed/undirected
        is_directed = self.graph.is_directed()

        # Connectivity
        if is_directed:
            is_connected = nx.is_weakly_connected(self.graph)
        else:
            is_connected = nx.is_connected(self.graph)

        stats = {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "graph_density": round(density, 4),
            "average_degree": round(avg_degree, 2),
            "is_directed": is_directed,
            "is_connected": is_connected
        }

        self.logger.debug(f"Basic stats: {total_nodes} nodes, {total_edges} edges")

        return stats

    @time_function
    def centrality_analysis(self, top_n: int = 20) -> Dict[str, Any]:
        if self.graph_type != "networkx":
            raise ValueError("Centrality analysis only supported for NetworkX graphs")

        self.logger.info("Calculating centrality metrics")

        metrics = {}

        # Degree Centrality
        try:
            degree_cent = nx.degree_centrality(self.graph)
            top_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:top_n]
            metrics['top_degree_nodes'] = [
                {"node": node, "centrality": round(score, 4)}
                for node, score in top_degree
            ]
        except Exception as e:
            self.logger.warning(f"Degree centrality failed: {e}")
            metrics['top_degree_nodes'] = []

        # PageRank
        try:
            pagerank = nx.pagerank(self.graph, alpha=0.85, max_iter=100)
            top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_n]
            metrics['top_pagerank_nodes'] = [
                {"node": node, "score": round(score, 4)}
                for node, score in top_pagerank
            ]
        except Exception as e:
            self.logger.warning(f"PageRank failed: {e}")
            metrics['top_pagerank_nodes'] = []

        # Betweenness Centrality (can be slow for large graphs)
        if self.graph.number_of_nodes() < 1000:
            try:
                betweenness = nx.betweenness_centrality(self.graph)
                top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_n]
                metrics['top_betweenness_nodes'] = [
                    {"node": node, "centrality": round(score, 4)}
                    for node, score in top_betweenness
                ]
            except Exception as e:
                self.logger.warning(f"Betweenness centrality failed: {e}")
                metrics['top_betweenness_nodes'] = []
        else:
            self.logger.info("Skipping betweenness centrality for large graph")
            metrics['top_betweenness_nodes'] = []

        # Closeness Centrality (only for smaller graphs)
        if self.graph.number_of_nodes() < 500:
            try:
                closeness = nx.closeness_centrality(self.graph)
                top_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:top_n]
                metrics['top_closeness_nodes'] = [
                    {"node": node, "centrality": round(score, 4)}
                    for node, score in top_closeness
                ]
            except Exception as e:
                self.logger.warning(f"Closeness centrality failed: {e}")
                metrics['top_closeness_nodes'] = []
        else:
            metrics['top_closeness_nodes'] = []

        self.logger.info("Centrality analysis complete")

        return metrics

    def structural_analysis(self) -> Dict[str, Any]:
        """
        Analyze graph structure.

        Returns:
            Structural analysis dictionary with:
                - connected_components
                - largest_component_size
                - diameter (if applicable)
                - average_path_length (if applicable)
                - clustering_coefficient

        Examples:
            >>> import networkx as nx
            >>> G = nx.karate_club_graph()
            >>> analyzer = GraphAnalyzer(G)
            >>> structure = analyzer.structural_analysis()
            >>> structure['connected_components'] >= 1
            True
        """
        if self.graph_type != "networkx":
            raise ValueError("Structural analysis only supported for NetworkX graphs")

        self.logger.info("Performing structural analysis")

        analysis = {}

        # Connected components
        if self.graph.is_directed():
            num_components = nx.number_weakly_connected_components(self.graph)
            components = list(nx.weakly_connected_components(self.graph))
        else:
            num_components = nx.number_connected_components(self.graph)
            components = list(nx.connected_components(self.graph))

        analysis['connected_components'] = num_components
        analysis['largest_component_size'] = len(max(components, key=len)) if components else 0

        # Get largest component
        if components:
            largest_cc = max(components, key=len)
            largest_subgraph = self.graph.subgraph(largest_cc)

            # Diameter (only for connected graphs)
            if not self.graph.is_directed() and nx.is_connected(largest_subgraph):
                try:
                    analysis['diameter'] = nx.diameter(largest_subgraph)
                except:
                    analysis['diameter'] = None
            else:
                analysis['diameter'] = None

            # Average path length
            if not self.graph.is_directed() and nx.is_connected(largest_subgraph):
                try:
                    analysis['average_path_length'] = round(
                        nx.average_shortest_path_length(largest_subgraph), 2
                    )
                except:
                    analysis['average_path_length'] = None
            else:
                analysis['average_path_length'] = None

        # Clustering coefficient
        try:
            if self.graph.is_directed():
                # Convert to undirected for clustering
                undirected = self.graph.to_undirected()
                analysis['average_clustering_coefficient'] = round(
                    nx.average_clustering(undirected), 4
                )
            else:
                analysis['average_clustering_coefficient'] = round(
                    nx.average_clustering(self.graph), 4
                )
        except:
            analysis['average_clustering_coefficient'] = None

        # Transitivity
        try:
            analysis['transitivity'] = round(nx.transitivity(self.graph), 4)
        except:
            analysis['transitivity'] = None

        self.logger.info("Structural analysis complete")

        return analysis

    @time_function
    def community_detection(
        self,
        algorithm: str = "louvain",
        min_community_size: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Detect communities in the graph.

        Args:
            algorithm: Community detection algorithm ("louvain" or "label_propagation")
            min_community_size: Minimum size for a community to be included

        Returns:
            List of community dictionaries with:
                - id: Community identifier
                - size: Number of nodes
                - nodes: List of node IDs
                - top_nodes: Most central nodes in community

        Examples:
            >>> import networkx as nx
            >>> G = nx.karate_club_graph()
            >>> analyzer = GraphAnalyzer(G)
            >>> communities = analyzer.community_detection()
            >>> len(communities) > 0
            True
        """
        if self.graph_type != "networkx":
            raise ValueError("Community detection only supported for NetworkX graphs")

        self.logger.info(f"Detecting communities using {algorithm} algorithm")

        # Convert to undirected for community detection
        if self.graph.is_directed():
            undirected_graph = self.graph.to_undirected()
        else:
            undirected_graph = self.graph

        communities_list = []

        try:
            if algorithm == "louvain":
                import community as community_louvain
                partition = community_louvain.best_partition(undirected_graph)

                # Group nodes by community
                community_nodes = defaultdict(list)
                for node, comm_id in partition.items():
                    community_nodes[comm_id].append(node)

            elif algorithm == "label_propagation":
                communities = nx.algorithms.community.label_propagation_communities(undirected_graph)
                community_nodes = {i: list(comm) for i, comm in enumerate(communities)}

            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            # Build community info
            for comm_id, nodes in community_nodes.items():
                if len(nodes) < min_community_size:
                    continue

                # Get subgraph for this community
                subgraph = undirected_graph.subgraph(nodes)

                # Find top nodes by degree in subgraph
                degrees = dict(subgraph.degree())
                top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]

                community_info = {
                    "id": comm_id,
                    "size": len(nodes),
                    "nodes": nodes[:100],  # Limit to first 100 for large communities
                    "top_nodes": [
                        {"node": node, "degree": deg}
                        for node, deg in top_nodes
                    ]
                }

                communities_list.append(community_info)

            # Sort by size
            communities_list.sort(key=lambda c: c['size'], reverse=True)

            self.logger.info(f"Detected {len(communities_list)} communities")

        except ImportError:
            self.logger.error("Community detection library not installed. Install python-louvain: pip install python-louvain")
            return []
        except Exception as e:
            self.logger.error(f"Community detection failed: {e}")
            return []

        return communities_list

    def degree_distribution(self) -> Dict[str, Any]:
        """
        Analyze degree distribution.

        Returns:
            Degree distribution stats with:
                - mean_degree
                - median_degree
                - max_degree
                - min_degree
                - degree_histogram

        Examples:
            >>> import networkx as nx
            >>> G = nx.complete_graph(5)
            >>> analyzer = GraphAnalyzer(G)
            >>> dist = analyzer.degree_distribution()
            >>> dist['mean_degree'] > 0
            True
        """
        if self.graph_type != "networkx":
            raise ValueError("Degree distribution only supported for NetworkX graphs")

        degrees = [d for n, d in self.graph.degree()]

        if not degrees:
            return {
                "mean_degree": 0,
                "median_degree": 0,
                "max_degree": 0,
                "min_degree": 0,
                "degree_histogram": {}
            }

        import statistics

        # Calculate statistics
        mean_degree = statistics.mean(degrees)
        median_degree = statistics.median(degrees)
        max_degree = max(degrees)
        min_degree = min(degrees)

        # Create histogram
        degree_counts = Counter(degrees)

        distribution = {
            "mean_degree": round(mean_degree, 2),
            "median_degree": median_degree,
            "max_degree": max_degree,
            "min_degree": min_degree,
            "degree_histogram": dict(sorted(degree_counts.items())[:20])  # Top 20 degrees
        }

        return distribution

    def export_analysis(self, analysis: Dict[str, Any], output_path: str) -> None:
        """
        Export analysis results to JSON.

        Args:
            analysis: Analysis results dictionary
            output_path: Output file path

        Examples:
            >>> import networkx as nx
            >>> G = nx.complete_graph(3)
            >>> analyzer = GraphAnalyzer(G)
            >>> analysis = analyzer.analyze_complete()
            >>> analyzer.export_analysis(analysis, "analysis.json")
        """
        save_json(analysis, output_path, indent=2)
        self.logger.info(f"Exported analysis to {output_path}")

    def get_node_neighborhood(
        self,
        node_id: str,
        depth: int = 1
    ) -> Dict[str, Any]:
        """
        Get neighborhood information for a specific node.

        Args:
            node_id: Node identifier
            depth: Neighborhood depth (hops)

        Returns:
            Dictionary with neighborhood info

        Examples:
            >>> import networkx as nx
            >>> G = nx.star_graph(5)
            >>> analyzer = GraphAnalyzer(G)
            >>> neighborhood = analyzer.get_node_neighborhood(0, depth=1)
            >>> len(neighborhood['neighbors']) > 0
            True
        """
        if node_id not in self.graph:
            return {"error": f"Node {node_id} not found in graph"}

        # Get neighbors at different depths
        neighbors_by_depth = {}

        current_level = {node_id}
        visited = {node_id}

        for d in range(1, depth + 1):
            next_level = set()
            for node in current_level:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        next_level.add(neighbor)
                        visited.add(neighbor)

            neighbors_by_depth[f"depth_{d}"] = list(next_level)
            current_level = next_level

        # Get node properties
        node_props = dict(self.graph.nodes[node_id])

        # Get edge info
        edges_out = list(self.graph.out_edges(node_id, data=True))
        edges_in = list(self.graph.in_edges(node_id, data=True))

        neighborhood = {
            "node_id": node_id,
            "properties": node_props,
            "degree": self.graph.degree(node_id),
            "out_degree": self.graph.out_degree(node_id),
            "in_degree": self.graph.in_degree(node_id),
            "neighbors": list(self.graph.neighbors(node_id)),
            "neighbors_by_depth": neighbors_by_depth,
            "total_reachable": len(visited) - 1
        }

        return neighborhood


# Helper Functions

def calculate_degree_centrality(graph: nx.Graph) -> Dict[str, float]:
    """
    Calculate degree centrality for all nodes.

    Args:
        graph: NetworkX graph

    Returns:
        Dictionary mapping nodes to centrality scores
    """
    return nx.degree_centrality(graph)


def calculate_pagerank(graph: nx.Graph, alpha: float = 0.85) -> Dict[str, float]:
    """
    Calculate PageRank for all nodes.

    Args:
        graph: NetworkX graph
        alpha: Damping parameter

    Returns:
        Dictionary mapping nodes to PageRank scores
    """
    return nx.pagerank(graph, alpha=alpha)


def calculate_betweenness_centrality(graph: nx.Graph) -> Dict[str, float]:
    """
    Calculate betweenness centrality.

    Args:
        graph: NetworkX graph

    Returns:
        Dictionary mapping nodes to betweenness scores
    """
    return nx.betweenness_centrality(graph)


def detect_communities_louvain(graph: nx.Graph) -> Dict[Any, int]:
    """
    Detect communities using Louvain algorithm.

    Args:
        graph: NetworkX graph (undirected)

    Returns:
        Dictionary mapping nodes to community IDs
    """
    try:
        import community as community_louvain
        return community_louvain.best_partition(graph)
    except ImportError:
        logging.error("python-louvain not installed")
        return {}


def get_subgraph_for_entity(
    graph: nx.Graph,
    entity_id: str,
    depth: int = 2
) -> nx.Graph:
    """
    Extract subgraph around an entity.

    Args:
        graph: Full graph
        entity_id: Central entity
        depth: Hop distance

    Returns:
        Subgraph containing entity and neighbors

    Examples:
        >>> import networkx as nx
        >>> G = nx.star_graph(10)
        >>> subgraph = get_subgraph_for_entity(G, 0, depth=1)
        >>> 0 in subgraph.nodes()
        True
    """
    if entity_id not in graph:
        return nx.Graph()

    # BFS to find nodes within depth
    nodes = {entity_id}
    current_level = {entity_id}

    for _ in range(depth):
        next_level = set()
        for node in current_level:
            neighbors = set(graph.neighbors(node))
            next_level.update(neighbors)
        nodes.update(next_level)
        current_level = next_level

    # Extract subgraph
    subgraph = graph.subgraph(nodes).copy()

    return subgraph


# Module logger
logger = logging.getLogger(__name__)
