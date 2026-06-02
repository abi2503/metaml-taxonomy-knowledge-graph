from pathlib import Path
import pickle
import networkx as nx
from pyvis.network import Network

GRAPH_CACHE_PATH = Path("data/taxonomy_graph.gpickle")

def load_graph(path: Path) -> nx.DiGraph:
    try:
        return nx.read_gpickle(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

def main():
    if not GRAPH_CACHE_PATH.exists():
        raise FileNotFoundError(f"Graph cache not found at: {GRAPH_CACHE_PATH}")

    G = load_graph(GRAPH_CACHE_PATH)

    # IMPORTANT: big graphs will freeze your browser
    # So start with a subset
    nodes = list(G.nodes())[:500]
    H = G.subgraph(nodes).copy()

    net = Network(height="800px", width="100%", directed=True, notebook=False)

    # Add nodes with nicer labels
    for node_id, attrs in H.nodes(data=True):
        label = attrs.get("name") or node_id
        ntype = attrs.get("label", "unknown")

        net.add_node(
            node_id,
            label=label,
            title=f"{node_id} ({ntype})",  # shows on hover
        )

    # Add edges
    for u, v, attrs in H.edges(data=True):
        rel = attrs.get("relation", "")
        net.add_edge(u, v, title=rel)

    # Physics makes it interactive (drag nodes)
    net.toggle_physics(True)

    # ✅ Write HTML (this avoids the template bug)
    net.write_html("taxonomy_graph.html", open_browser=True, notebook=False)
    print("✅ Wrote taxonomy_graph.html and opened it in your browser.")

if __name__ == "__main__":
    main()
