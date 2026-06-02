from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # project root
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import argparse
import pickle
from typing import List, Optional

import networkx as nx

from rag_metaml.graph_utils import (
    find_nodes_by_name,
    find_nodes_contains,
    get_parents,
    get_children,
    siblings,
    get_taxonomy_path,
    get_subtree,
)

GRAPH_CACHE_PATH = Path("data/taxonomy_graph.gpickle")


def load_graph(path: Path) -> nx.DiGraph:
    """Load cached graph (supports nx.gpickle or pickle fallback)."""
    try:
        return nx.read_gpickle(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def node_display(graph: nx.DiGraph, node_id: str) -> str:
    """Nice display string for a node."""
    attrs = graph.nodes[node_id]
    name = attrs.get("name") or ""
    label = attrs.get("label") or ""
    return f"{node_id} | {label} | {name}"


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_list(graph: nx.DiGraph, title: str, nodes: List[str], limit: int = 15):
    print(f"\n{title} ({len(nodes)}):")
    for n in nodes[:limit]:
        if n in graph:
            print("  -", node_display(graph, n))
        else:
            print("  -", n)
    if len(nodes) > limit:
        print(f"  ... (+{len(nodes) - limit} more)")


def inspect_node(graph: nx.DiGraph, node_id: str, subtree_depth: int = 2, limit: int = 15):
    """Print structure around a node in the taxonomy graph."""
    if node_id not in graph:
        print(f"❌ Node not found: {node_id}")
        return

    print_section("NODE")
    print(node_display(graph, node_id))

    print_section("TAXONOMY PATH")
    path = get_taxonomy_path(graph, node_id)
    if path:
        print(" > ".join(path))
    else:
        print("(no path found)")

    print_section("PARENTS / CHILDREN / SIBLINGS")
    print_list(graph, "Parents", get_parents(graph, node_id), limit=limit)
    print_list(graph, "Children", get_children(graph, node_id), limit=limit)
    print_list(graph, "Siblings", siblings(graph, node_id), limit=limit)

    print_section(f"SUBTREE PREVIEW (max_depth={subtree_depth})")
    subtree = get_subtree(graph, node_id, max_depth=subtree_depth)
    print_list(graph, "Descendants", subtree, limit=limit)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a taxonomy node (domain/dimension/area) in the knowledge graph."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--node_id", type=str, help='Direct node id like "area:42" or "domain:1"')
    group.add_argument("--name", type=str, help='Name like "FinTech", "Trading & Markets", "Stablecoins"')

    parser.add_argument(
        "--label",
        type=str,
        choices=["domain", "dimension", "area"],
        default=None,
        help="Optional filter when searching by --name",
    )
    parser.add_argument(
        "--contains",
        action="store_true",
        help="Use substring matching instead of exact name match.",
    )
    parser.add_argument(
        "--pick",
        type=int,
        default=0,
        help="If multiple matches, choose which one (0-based index).",
    )
    parser.add_argument(
        "--subtree_depth",
        type=int,
        default=2,
        help="Depth for subtree preview.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Max items to print per section (parents/children/siblings/subtree).",
    )
    parser.add_argument(
        "--graph_path",
        type=str,
        default=str(GRAPH_CACHE_PATH),
        help="Path to taxonomy_graph.gpickle",
    )

    args = parser.parse_args()

    graph_path = Path(args.graph_path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph cache not found: {graph_path}")

    graph = load_graph(graph_path)
    print(f"✅ Loaded graph: nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")

    if args.node_id:
        inspect_node(graph, args.node_id, subtree_depth=args.subtree_depth, limit=args.limit)
        return

    # If using name:
    print_section("SEARCH")
    print(f"Query name: {args.name}")
    print(f"Label filter: {args.label}")
    print(f"Mode: {'contains' if args.contains else 'exact'}")

    if args.contains:
        matches = find_nodes_contains(graph, args.name, label=args.label)
    else:
        matches = find_nodes_by_name(graph, args.name, label=args.label)

    if not matches:
        print("\n❌ No matches found.")
        return

    print(f"\n✅ Matches found: {len(matches)}")
    for i, n in enumerate(matches[:30]):
        print(f"  [{i}] {node_display(graph, n)}")
    if len(matches) > 30:
        print(f"  ... (+{len(matches) - 30} more)")

    pick = args.pick
    if pick < 0 or pick >= len(matches):
        print(f"\n❌ Invalid --pick {pick}. Must be between 0 and {len(matches)-1}.")
        return

    chosen = matches[pick]
    inspect_node(graph, chosen, subtree_depth=args.subtree_depth, limit=args.limit)


if __name__ == "__main__":
    main()
