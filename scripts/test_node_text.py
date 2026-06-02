from pathlib import Path
import sys
import pickle
import networkx as nx

# ✅ Add src/ so we can import rag_metaml as a package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from rag_metaml.node_text import node_to_text  # ✅ package import (not relative)

GRAPH_PATH = ROOT / "data" / "taxonomy_graph.gpickle"


def load_graph(path: Path) -> nx.DiGraph:
    try:
        return nx.read_gpickle(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    g = load_graph(GRAPH_PATH)

    # Grab 1 domain, 1 dimension, 1 area to test
    samples = []
    for node_id, attrs in g.nodes(data=True):
        if attrs.get("label") == "domain":
            samples.append(node_id)
            break

    for node_id, attrs in g.nodes(data=True):
        if attrs.get("label") == "dimension":
            samples.append(node_id)
            break

    for node_id, attrs in g.nodes(data=True):
        if attrs.get("label") == "area":
            samples.append(node_id)
            break

    for nid in samples:
        print("\n" + "-" * 80)
        print("NODE:", nid, "|", g.nodes[nid].get("name"))
        print(node_to_text(g, nid))
