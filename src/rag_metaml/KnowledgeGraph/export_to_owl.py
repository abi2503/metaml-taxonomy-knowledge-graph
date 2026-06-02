from rdflib import Graph as RDFGraph, Namespace, URIRef, RDF, RDFS, Literal
from rdflib.namespace import OWL
import networkx as nx

# Support both: run as script (python export_to_owl.py) and import as package
try:
    from .graph import build_taxonomy_graph
except ImportError:
    import sys
    from pathlib import Path
    src_dir = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(src_dir))
    from rag_metaml.KnowledgeGraph.graph import build_taxonomy_graph




EX = Namespace("http://metaml.com/ontology/")

def export_graph_to_owl(nx_graph: nx.DiGraph, output_file: str = "metaml.owl"):
    rdf_graph = RDFGraph()
    rdf_graph.bind("ex", EX)
    rdf_graph.bind("owl", OWL)

    # Ontology
    rdf_graph.add((URIRef(EX), RDF.type, OWL.Ontology))

    # Classes
    rdf_graph.add((EX.Domain, RDF.type, OWL.Class))
    rdf_graph.add((EX.Dimension, RDF.type, OWL.Class))
    rdf_graph.add((EX.Area, RDF.type, OWL.Class))

    # Properties
    rdf_graph.add((EX.hasDimension, RDF.type, OWL.ObjectProperty))
    rdf_graph.add((EX.hasArea, RDF.type, OWL.ObjectProperty))

    rdf_graph.add((EX.hasDimension, RDFS.domain, EX.Domain))
    rdf_graph.add((EX.hasDimension, RDFS.range, EX.Dimension))

    rdf_graph.add((EX.hasArea, RDFS.domain, EX.Dimension))
    rdf_graph.add((EX.hasArea, RDFS.range, EX.Area))

    # Nodes
    for node, data in nx_graph.nodes(data=True):
        label = data.get("label")
        name = data.get("name") or node

        node_uri = URIRef(EX + node.replace(":", "_"))

        if label == "domain":
            rdf_graph.add((node_uri, RDF.type, EX.Domain))
        elif label == "dimension":
            rdf_graph.add((node_uri, RDF.type, EX.Dimension))
        elif label == "area":
            rdf_graph.add((node_uri, RDF.type, EX.Area))

        rdf_graph.add((node_uri, RDFS.label, Literal(name)))

    # Edges
    for u, v, edge_data in nx_graph.edges(data=True):
        relation = edge_data.get("relation")

        u_uri = URIRef(EX + u.replace(":", "_"))
        v_uri = URIRef(EX + v.replace(":", "_"))

        if relation == "HAS_DIMENSION":
            rdf_graph.add((u_uri, EX.hasDimension, v_uri))
        elif relation == "HAS_AREA":
            rdf_graph.add((u_uri, EX.hasArea, v_uri))

    rdf_graph.serialize(destination=output_file, format="xml")
    print(f"✅ OWL exported to {output_file}")


# 🔥 MAIN ENTRY POINT
if __name__ == "__main__":
    print("🔄 Building graph...")
    graph = build_taxonomy_graph()

    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    print("🔄 Exporting to OWL...")
    export_graph_to_owl(graph, "metaml.owl")

    print("✅ Done!")