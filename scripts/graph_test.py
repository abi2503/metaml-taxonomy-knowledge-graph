# scripts/graph_test.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rag_metaml.graph import build_taxonomy_graph, get_neighbors, describe_path

if __name__ == "__main__":
    print("=" * 80)
    print("Testing Knowledge Graph Construction")
    print("=" * 80)
    
    graph = build_taxonomy_graph()
    
    print(f"\n Graph built successfully!")
    print(f"   Nodes: {graph.number_of_nodes()}")
    print(f"   Edges: {graph.number_of_edges()}")
    

    print("\n" + "=" * 80)
    print("Sample Nodes:")
    print("=" * 80)
    node_count = 0
    for node_id, data in graph.nodes(data=True):
        if node_count < 10:
            print(f"  {node_id}: {data.get('name', 'N/A')} (label: {data.get('label')})")
            node_count += 1
    

    print("\n" + "=" * 80)
    print("Testing get_neighbors()")
    print("=" * 80)

    sample_area = None
    for node_id, data in graph.nodes(data=True):
        if data.get('label') == 'area':
            sample_area = node_id
            break
    
    if sample_area:
        print(f"\nTesting with node: {sample_area}")
        print(f"  Node name: {graph.nodes[sample_area].get('name')}")
        
        # Get neighbors (down - children)
        neighbors_down = get_neighbors(graph, sample_area, depth=1, direction="down")
        print(f"\n  Children (depth=1): {len(neighbors_down)}")
        for n in neighbors_down[:5]:
            print(f"    - {n}: {graph.nodes[n].get('name')}")
        

        neighbors_up = get_neighbors(graph, sample_area, depth=1, direction="up")
        print(f"\n  Parents (depth=1): {len(neighbors_up)}")
        for n in neighbors_up[:5]:
            print(f"    - {n}: {graph.nodes[n].get('name')}")
        
        # Get neighbors (both)
        neighbors_both = get_neighbors(graph, sample_area, depth=2, direction="both")
        print(f"\n  All neighbors (depth=2): {len(neighbors_both)}")
    
    print("\n" + "=" * 80)
    print("Testing describe_path()")
    print("=" * 80)
    
    domain_nodes = [n for n, d in graph.nodes(data=True) if d.get('label') == 'domain']
    area_nodes = [n for n, d in graph.nodes(data=True) if d.get('label') == 'area']
    
    if domain_nodes and area_nodes:
        source = domain_nodes[0]
        target = area_nodes[0]
        
        print(f"\nFinding path from: {source} ({graph.nodes[source].get('name')})")
        print(f"              to: {target} ({graph.nodes[target].get('name')})")
        
        path_names = describe_path(graph, source, target)
        if path_names:
            print(f"\n Path found:")
            print(f"   {' > '.join(path_names)}")
        else:
            print(f"\n No path found (may not be directly connected)")
    
    print("\n" + "=" * 80)
    print(" Graph test completed!")
    print("=" * 80)



