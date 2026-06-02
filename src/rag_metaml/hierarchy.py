from typing import Dict,List,Any
from .logging_setup import logger

def _index_by_id(row: List[Dict[str,Any]],key:str)->Dict[int,Dict[str,Any]]:
    """Utility: turn a list of dicts into a dict keyed by an integer id field."""
    return {r[key]: r for r in row}


def _build_area_path(area_id:int,areas_by_id:Dict[int,Dict[str,Any]])->List[str]:
    """
    Walk parent pointers up to the root, collecting area_name(s).
    Returns the path from root→leaf as a list of names.
    """
    path=[]
    current_id=area_id
    safety=0
    while current_id is not None and current_id in areas_by_id:
        node=areas_by_id[current_id]
        path.append(node["area_name"])
        current_id=node["area_parent_id"]
        safety+=1
        if safety > 2000:  # extremely deep or cyclic graph guard
            logger.warning(f"Aborting path build (cycle?) at AREA_ID={area_id}")
            break
    
    # Reverse to get root→leaf order (currently leaf→root)
    path.reverse()
    return path


def build_hierarchical_rows(
    domains: List[Dict[str, Any]],
    dimensions: List[Dict[str, Any]],
    areas: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Create hierarchical rows ready for embedding/indexing in Chroma.

    Output rows look like:
    {
      'AREA_ID': int,
      'AREA_NAME': str,
      'DIMENSION_ID': int,
      'DIMENSION_NAME': str,
      'DOMAIN_ID': int,
      'DOMAIN_NAME': str,
      'hierarchical_text': 'Domain > Dimension > A > B > C'
    }
    """
    domains_by_id = _index_by_id(domains, "domain_id")
    dimensions_by_id = _index_by_id(dimensions, "dimension_id")
    areas_by_id = _index_by_id(areas, "area_id")

    out: List[Dict[str, Any]] = []

    for a in areas:
        # --- Parse & validate area_id and dimension_id
        try:
            area_id = int(a["area_id"])
        except (KeyError, TypeError, ValueError):
            logger.warning(f"Bad/missing area_id in row: {a}")
            continue

        try:
            dim_id = int(a["dimension_id"])
        except (KeyError, TypeError, ValueError):
            logger.warning(f"Area {area_id} missing/invalid dimension_id: {a.get('dimension_id')}")
            continue

        dim = dimensions_by_id.get(dim_id)
        if not dim:
            logger.warning(f"Area {area_id} references missing dimension {dim_id}. Skipping.")
            continue

        # --- Get domain_id from the dimension (tolerate None/invalid)
        dom_id_raw = dim.get("domain_id")
        if dom_id_raw is None:
            logger.warning(f"Dimension {dim_id} has no domain_id. Skipping area {area_id}.")
            continue
        try:
            dom_id = int(dom_id_raw)
        except (TypeError, ValueError):
            logger.warning(f"Dimension {dim_id} has invalid domain_id={dom_id_raw}. Skipping area {area_id}.")
            continue

        dom = domains_by_id.get(dom_id)
        if not dom:
            # Fallback: if dimension carried a domain_name, synthesize a minimal domain row
            dom_name_fallback = str(dim.get("domain_name") or "").strip()
            if not dom_name_fallback:
                logger.warning(f"Missing domain row for domain_id={dom_id}. Skipping area {area_id}.")
                continue
            dom = {"domain_id": dom_id, "domain_name": dom_name_fallback}

        # --- Build the area path (root → leaf) using AREA_PARENT_ID links
        area_path = _build_area_path(area_id, areas_by_id)  # e.g., ["Digital/Virtual Banking", "Banking Platforms"]

        # --- Compose the full hierarchical text
        full_path_parts = [
            str(dom.get("domain_name", "")).strip(),
            str(dim.get("dimension_name", "")).strip(),
            *area_path
        ]
        full_path = " > ".join([p for p in full_path_parts if p])

        out.append({
            "AREA_ID": area_id,
            "AREA_NAME": a.get("area_name"),
            "DIMENSION_ID": dim_id,
            "DIMENSION_NAME": dim.get("dimension_name"),
            "DOMAIN_ID": dom_id,
            "DOMAIN_NAME": dom.get("domain_name"),
            "hierarchical_text": full_path,
        })

    logger.info(f"Built hierarchical_text for {len(out)} areas.")
    return out