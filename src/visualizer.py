"""
Graph Visualization Module

Creates visual representations of Knowledge Graphs using
various layouts and rendering engines for both static and interactive output.
"""

from typing import Optional, List, Dict, Any, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging
from pathlib import Path
from collections import Counter


# Default color palette for entity types
DEFAULT_COLORS = {
    "PERSON": "#FF6B6B",
    "ORGANIZATION": "#4ECDC4",
    "ORG": "#4ECDC4",
    "LOCATION": "#45B7D1",
    "GPE": "#45B7D1",
    "LOC": "#45B7D1",
    "PRODUCT": "#FFA07A",
    "EVENT": "#95E1D3",
    "DATE": "#F38181",
    "MONEY": "#A8E6CF",
    "PERCENT": "#FFD3B6",
    "DEFAULT": "#CCCCCC"
}

# Layout algorithms available
LAYOUT_ALGORITHMS = {
    "spring": nx.spring_layout,
    "circular": nx.circular_layout,
    "kamada_kawai": nx.kamada_kawai_layout,
    "shell": nx.shell_layout,
    "spectral": nx.spectral_layout,
    "random": nx.random_layout,
}


class GraphVisualizer:
    """
    Visualizes Knowledge Graphs in multiple formats.

    Supports static plots (PNG, SVG, PDF) and interactive HTML visualizations.

    Attributes:
        graph: NetworkX graph object
        color_map: Dictionary mapping entity types to colors

    Examples:
        >>> import networkx as nx
        >>> G = nx.karate_club_graph()
        >>> viz = GraphVisualizer(G)
        >>> viz.plot_full_graph("output.png")
    """

    def __init__(
        self,
        graph: nx.Graph,
        color_map: Optional[Dict[str, str]] = None
    ):
        """
        Initialize visualizer.

        Args:
            graph: NetworkX graph object
            color_map: Custom color mapping for node types
        """
        self.graph = graph
        self.color_map = color_map or DEFAULT_COLORS
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"GraphVisualizer initialized for graph with {graph.number_of_nodes()} nodes")

    def plot_full_graph(
        self,
        output_path: str,
        layout: str = "spring",
        figsize: Tuple[int, int] = (20, 16),
        node_color_attr: str = "type",
        node_size_attr: Optional[str] = None,
        with_labels: bool = True,
        dpi: int = 300
    ) -> None:
        """
        Create static visualization of full graph.

        Args:
            output_path: Output file path (PNG, PDF, SVG)
            layout: Layout algorithm name
            figsize: Figure size (width, height)
            node_color_attr: Node attribute for coloring
            node_size_attr: Node attribute for sizing (optional)
            with_labels: Whether to show node labels
            dpi: Resolution for raster formats

        Examples:
            >>> import networkx as nx
            >>> G = nx.karate_club_graph()
            >>> viz = GraphVisualizer(G)
            >>> viz.plot_full_graph("graph.png", layout="spring")
        """
        self.logger.info(f"Creating full graph visualization with {layout} layout")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Get layout
        if layout in LAYOUT_ALGORITHMS:
            pos = LAYOUT_ALGORITHMS[layout](self.graph, k=0.5, iterations=50)
        else:
            self.logger.warning(f"Unknown layout {layout}, using spring")
            pos = nx.spring_layout(self.graph, k=0.5, iterations=50)

        # Get node colors
        node_colors = self._get_node_colors(node_color_attr)

        # Get node sizes
        node_sizes = self._get_node_sizes(node_size_attr)

        # Draw network
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8,
            ax=ax
        )

        nx.draw_networkx_edges(
            self.graph,
            pos,
            alpha=0.3,
            edge_color='gray',
            arrows=True,
            arrowsize=10,
            ax=ax
        )

        if with_labels and self.graph.number_of_nodes() < 100:
            # Only show labels for smaller graphs
            labels = {
                node: self.graph.nodes[node].get('name', str(node))
                for node in self.graph.nodes()
            }
            nx.draw_networkx_labels(
                self.graph,
                pos,
                labels,
                font_size=8,
                ax=ax
            )

        # Add title
        ax.set_title(
            f"Knowledge Graph ({self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges)",
            fontsize=16,
            fontweight='bold'
        )
        ax.axis('off')

        # Add legend for node types
        self._add_legend(ax, node_color_attr)

        plt.tight_layout()

        # Save figure
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved graph visualization to {output_path}")

    def create_interactive_html(
        self,
        output_path: str,
        height: str = "750px",
        width: str = "100%",
        notebook: bool = False
    ) -> None:
        """
        Create interactive HTML visualization using Pyvis.

        Args:
            output_path: Output HTML file path
            height: Visualization height
            width: Visualization width
            notebook: Whether rendering in Jupyter notebook

        Examples:
            >>> import networkx as nx
            >>> G = nx.karate_club_graph()
            >>> viz = GraphVisualizer(G)
            >>> viz.create_interactive_html("graph.html")
        """
        try:
            from pyvis.network import Network
        except ImportError:
            self.logger.error("pyvis not installed. Install with: pip install pyvis")
            return

        self.logger.info("Creating interactive HTML visualization")

        # Create network
        net = Network(
            height=height,
            width=width,
            directed=self.graph.is_directed(),
            notebook=notebook
        )

        # Add nodes
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            node_type = node_data.get('type', 'DEFAULT')
            node_name = node_data.get('name', str(node))

            color = self.color_map.get(node_type, self.color_map['DEFAULT'])

            net.add_node(
                node,
                label=node_name,
                title=f"{node_type}: {node_name}",
                color=color,
                size=20
            )

        # Add edges
        for source, target, data in self.graph.edges(data=True):
            edge_type = data.get('type', '')
            confidence = data.get('confidence', 0.5)

            net.add_edge(
                source,
                target,
                title=edge_type,
                value=confidence * 10,  # Scale for visualization
                label=edge_type
            )

        # Set physics options
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 200,
              "springConstant": 0.04,
              "damping": 0.09
            },
            "maxVelocity": 50,
            "minVelocity": 0.1,
            "solver": "barnesHut",
            "timestep": 0.5
          },
          "nodes": {
            "font": {
              "size": 14
            }
          },
          "edges": {
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.5
              }
            },
            "smooth": {
              "type": "continuous"
            }
          }
        }
        """)

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(output_path)

        self.logger.info(f"Saved interactive visualization to {output_path}")

    def plot_subgraph(
        self,
        central_node: str,
        depth: int = 2,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ) -> None:
        """
        Visualize subgraph around a central node.

        Args:
            central_node: Central node ID
            depth: Neighborhood depth
            output_path: Output file path (optional)
            figsize: Figure size

        Examples:
            >>> import networkx as nx
            >>> G = nx.star_graph(10)
            >>> viz = GraphVisualizer(G)
            >>> viz.plot_subgraph(0, depth=1, output_path="subgraph.png")
        """
        if central_node not in self.graph:
            self.logger.error(f"Node {central_node} not found in graph")
            return

        self.logger.info(f"Creating subgraph visualization for {central_node}")

        # Extract subgraph
        nodes = {central_node}
        current_level = {central_node}

        for _ in range(depth):
            next_level = set()
            for node in current_level:
                neighbors = set(self.graph.neighbors(node))
                next_level.update(neighbors)
            nodes.update(next_level)
            current_level = next_level

        subgraph = self.graph.subgraph(nodes)

        # Create visualization
        fig, ax = plt.subplots(figsize=figsize)

        pos = nx.spring_layout(subgraph, k=0.9, iterations=50)

        # Color central node differently
        node_colors = []
        for node in subgraph.nodes():
            if node == central_node:
                node_colors.append('#FF0000')  # Red for central node
            else:
                node_type = subgraph.nodes[node].get('type', 'DEFAULT')
                node_colors.append(self.color_map.get(node_type, self.color_map['DEFAULT']))

        # Draw network
        nx.draw_networkx_nodes(
            subgraph,
            pos,
            node_color=node_colors,
            node_size=500,
            alpha=0.8,
            ax=ax
        )

        nx.draw_networkx_edges(
            subgraph,
            pos,
            alpha=0.5,
            edge_color='gray',
            arrows=True,
            arrowsize=15,
            ax=ax
        )

        # Labels
        labels = {
            node: subgraph.nodes[node].get('name', str(node))
            for node in subgraph.nodes()
        }
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=10, ax=ax)

        ax.set_title(f"Subgraph around {central_node} (depth={depth})", fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved subgraph to {output_path}")
        else:
            plt.show()

    def plot_degree_distribution(
        self,
        output_path: str,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Plot degree distribution histogram.

        Args:
            output_path: Output file path
            figsize: Figure size

        Examples:
            >>> import networkx as nx
            >>> G = nx.barabasi_albert_graph(100, 3)
            >>> viz = GraphVisualizer(G)
            >>> viz.plot_degree_distribution("degree_dist.png")
        """
        self.logger.info("Creating degree distribution plot")

        degrees = [d for n, d in self.graph.degree()]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Histogram
        ax1.hist(degrees, bins=30, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Degree', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Degree Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Log-log plot
        degree_counts = Counter(degrees)
        degrees_sorted = sorted(degree_counts.keys())
        counts = [degree_counts[d] for d in degrees_sorted]

        ax2.loglog(degrees_sorted, counts, 'bo-', alpha=0.7)
        ax2.set_xlabel('Degree (log scale)', fontsize=12)
        ax2.set_ylabel('Frequency (log scale)', fontsize=12)
        ax2.set_title('Degree Distribution (Log-Log)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, which='both')

        plt.tight_layout()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved degree distribution to {output_path}")

    def plot_community_structure(
        self,
        communities: List[set],
        output_path: str,
        layout: str = "spring",
        figsize: Tuple[int, int] = (20, 16)
    ) -> None:
        """
        Visualize detected communities.

        Args:
            communities: List of community node sets
            output_path: Output file path
            layout: Layout algorithm
            figsize: Figure size

        Examples:
            >>> import networkx as nx
            >>> G = nx.karate_club_graph()
            >>> communities = list(nx.algorithms.community.greedy_modularity_communities(G))
            >>> viz = GraphVisualizer(G)
            >>> viz.plot_community_structure(communities, "communities.png")
        """
        self.logger.info(f"Creating community structure visualization ({len(communities)} communities)")

        fig, ax = plt.subplots(figsize=figsize)

        # Get layout
        if layout in LAYOUT_ALGORITHMS:
            pos = LAYOUT_ALGORITHMS[layout](self.graph, k=0.5, iterations=50)
        else:
            pos = nx.spring_layout(self.graph, k=0.5, iterations=50)

        # Generate colors for communities
        import matplotlib.cm as cm
        colors = cm.rainbow([i / len(communities) for i in range(len(communities))])

        # Create node to community mapping
        node_to_community = {}
        for comm_id, community in enumerate(communities):
            for node in community:
                node_to_community[node] = comm_id

        # Get node colors based on community
        node_colors = [
            colors[node_to_community.get(node, 0)]
            for node in self.graph.nodes()
        ]

        # Draw network
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=200,
            alpha=0.8,
            ax=ax
        )

        nx.draw_networkx_edges(
            self.graph,
            pos,
            alpha=0.2,
            edge_color='gray',
            arrows=True,
            arrowsize=10,
            ax=ax
        )

        ax.set_title(
            f"Community Structure ({len(communities)} communities)",
            fontsize=16,
            fontweight='bold'
        )
        ax.axis('off')

        plt.tight_layout()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved community visualization to {output_path}")

    def create_dashboard(
        self,
        output_dir: str,
        communities: Optional[List[set]] = None
    ) -> None:
        """
        Create complete visualization dashboard.

        Generates multiple visualizations and saves them to output directory.

        Args:
            output_dir: Output directory for all visualizations
            communities: Optional community structure

        Examples:
            >>> import networkx as nx
            >>> G = nx.karate_club_graph()
            >>> viz = GraphVisualizer(G)
            >>> viz.create_dashboard("output/dashboard")
        """
        self.logger.info(f"Creating visualization dashboard in {output_dir}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Full graph visualization
        self.plot_full_graph(
            str(output_path / "full_graph.png"),
            layout="spring"
        )

        # 2. Interactive HTML
        self.create_interactive_html(
            str(output_path / "interactive_graph.html")
        )

        # 3. Degree distribution
        self.plot_degree_distribution(
            str(output_path / "degree_distribution.png")
        )

        # 4. Community structure (if provided)
        if communities:
            self.plot_community_structure(
                communities,
                str(output_path / "community_structure.png")
            )

        # 5. Circular layout
        self.plot_full_graph(
            str(output_path / "circular_layout.png"),
            layout="circular"
        )

        self.logger.info(f"Dashboard created in {output_dir}")

    def _get_node_colors(self, attr: str = "type") -> List[str]:
        """Get list of node colors based on attribute."""
        colors = []
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get(attr, 'DEFAULT')
            color = self.color_map.get(node_type, self.color_map['DEFAULT'])
            colors.append(color)
        return colors

    def _get_node_sizes(self, attr: Optional[str] = None) -> List[float]:
        """Get list of node sizes based on attribute or degree."""
        if attr and attr in next(iter(self.graph.nodes(data=True)))[1]:
            # Use specified attribute
            sizes = [
                self.graph.nodes[node].get(attr, 1) * 100
                for node in self.graph.nodes()
            ]
        else:
            # Use degree
            degrees = dict(self.graph.degree())
            max_degree = max(degrees.values()) if degrees else 1

            sizes = [
                100 + (degrees[node] / max_degree) * 400
                for node in self.graph.nodes()
            ]

        return sizes

    def _add_legend(self, ax, attr: str = "type") -> None:
        """Add legend showing node type colors."""
        # Get unique types in graph
        types_in_graph = set()
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get(attr, 'DEFAULT')
            types_in_graph.add(node_type)

        # Create legend elements
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.color_map.get(t, self.color_map['DEFAULT']), label=t)
            for t in sorted(types_in_graph)
        ]

        if legend_elements:
            ax.legend(
                handles=legend_elements,
                loc='upper right',
                fontsize=10,
                title="Node Types"
            )


# Module logger
logger = logging.getLogger(__name__)
