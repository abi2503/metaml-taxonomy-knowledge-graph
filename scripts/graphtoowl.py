
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.rag_metaml.KnowledgeGraph.graph import build_taxonomy_graph
from src.rag_metaml.KnowledgeGraph.export_to_owl import export_graph_to_owl

graph = build_taxonomy_graph()

export_graph_to_owl(graph, "metaml.owl")