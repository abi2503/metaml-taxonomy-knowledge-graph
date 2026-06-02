import networkx as nx


def build_node_text(graph: nx.DiGraph, node_id: str) -> str:
    node = graph.nodes[node_id]

    label = node.get("label")
    name = node.get("name")

    if label == "domain":
        return f"Domain: {name}"

    if label == "dimension":
        parents = list(graph.predecessors(node_id))
        if parents:
            parent_name = graph.nodes[parents[0]].get("name")
            return f"{parent_name} > {name}"

        return name

    if label == "area":
        parents = list(graph.predecessors(node_id))
        if parents:
            parent_name = graph.nodes[parents[0]].get("name")
            return f"{parent_name} > {name}"

        return name

    return name