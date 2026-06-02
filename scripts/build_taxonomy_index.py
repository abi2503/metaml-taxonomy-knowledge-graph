from pathlib import Path
import sys
import pickle
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from rag_metaml.taxonomy_index import build_taxonomy_index

GRAPH_PATH = ROOT / "data" / "taxonomy_graph.gpickle"


def load_graph(path: Path) -> nx.DiGraph:
    try:
        return nx.read_gpickle(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Graph cache missing: {GRAPH_PATH}")

    graph = load_graph(GRAPH_PATH)

    # limit=None means "index all nodes"
    build_taxonomy_index(graph, limit=None)
