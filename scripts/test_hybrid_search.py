import os
os.environ["USE_TF"] = "0"  

from pathlib import Path
import sys
import pickle
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from rag_metaml.hybrid_search import hybrid_search

GRAPH_PATH = ROOT / "data" / "taxonomy_graph.gpickle"


def load_graph(path: Path) -> nx.DiGraph:
    try:
        return nx.read_gpickle(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    g = load_graph(GRAPH_PATH)

    query = "Algorithmic Trading"
    result = hybrid_search(
        g,
        query=query,
        top_k=5,
        expand_hops=2,
        direction="both",
        label_filter=None,  # try "area" later
    )

    print("\nQUERY:", result["query"])
    print("Counts:", result["counts"])

    print("\n" + "=" * 90)
    print("SEEDS (from vector search)")
    print("=" * 90)
    for s in result["seeds"]:
        print(f"- #{s['rank']} {s['node_id']} | {s['label']} | {s['name']} | dist={s['distance']:.4f}")
        print(f"  preview: {s['doc_preview']}")

    print("\n" + "=" * 90)
    print("CANDIDATES (seeds + graph neighbors)")
    print("=" * 90)

    # show first 25
    for c in result["candidates"][:25]:
        path_str = " > ".join(c["path"] or [])
        print(f"- {c['node_id']} | {c['label']} | {c['name']}")
        print(f"  path: {path_str}")
