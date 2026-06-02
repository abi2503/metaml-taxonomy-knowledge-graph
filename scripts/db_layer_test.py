# scripts/db_layer_test.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rag_metaml.db import get_engine, fetch_domains, fetch_dimensions, fetch_areas
from src.rag_metaml.hierarchy import build_hierarchical_rows

if __name__ == "__main__":
    eng = get_engine()

    domains = fetch_domains(eng)
    dimensions = fetch_dimensions(eng)
    areas = fetch_areas(eng)

    print(f"Domains: {len(domains)}")
    print(f"Dimensions: {len(dimensions)}")
    print(f"Areas: {len(areas)}")

    rows = build_hierarchical_rows(domains, dimensions, areas)

    # Print a couple of sample rows to verify shapes & strings
    for r in rows[:5]:
        print(f"AREA_ID={r['AREA_ID']}, PATH={r['hierarchical_text']}")
