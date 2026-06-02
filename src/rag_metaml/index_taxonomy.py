import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List,Dict,Any
from .config import settings
from .db import get_engine,fetch_domains,fetch_dimensions,fetch_areas
from .hierarchy import build_hierarchical_rows
from .embeddings import Embedder
from .logging_setup import logger
from .chroma_client import get_chroma_client

def _batched(seq,batch_size:int):
    """
    Embeds everything in blocks. It is a generator that yields the start index and the batch of items.
    """
    for i in range(0,len(seq),batch_size):
        yield i, seq[i:i+batch_size]


def build_or_refresh_taxonomy_index(batch_size:int=512)-> int:
    """
    Fetch taxonomy from DB, build hierarchical strings in Python ,
    embed and upsert into Chroma.
    Returns: number of upserted rows
    """
    engine=get_engine()
    domains=fetch_domains(engine)
    dimensions=fetch_dimensions(engine)
    areas=fetch_areas(engine)

    # Build hierarchical rows (Domain > Dimension > ... > Area)
    rows=build_hierarchical_rows(domains,dimensions,areas)
    if not rows:
        logger.warning("No taxonomy rows to index.")
        return 0
    

    #Chroma collection

    #persistent_chroma_client=chromadb.PersistentClient(path=settings.chroma_dir,settings=ChromaSettings(allow_reset=True))
    persistent_chroma_client=get_chroma_client()
    chroma_collection=persistent_chroma_client.get_or_create_collection(name=settings.taxonomy_collection,metadata={"hnsw:space": "cosine"})

    #Prepare ids/docs/metas
    '''
    IDs: area:{AREA_ID} makes IDs stable—re-running upserts 
    will update the same records instead of creating duplicates.

    docs: exactly what gets embedded/searched.

    metas: searchable/filterable metadata for downstream filtering 
    (e.g., where={"DOMAIN_NAME": "FinTech"} during queries). 
    text_type lets you distinguish this collection from others (e.g., docs vs. hierarchy).
    '''

    ids:List[str]=[f'area:{r["AREA_ID"]}' for r in rows]
    docs:List[str]=[r["hierarchical_text"] for r in rows]
    metas: List[Dict[str, Any]] = [{
    "AREA_ID": r["AREA_ID"],
    "AREA_NAME": r["AREA_NAME"],
    "DIMENSION_ID": r["DIMENSION_ID"],
    "DIMENSION_NAME": r["DIMENSION_NAME"],
    "DOMAIN_ID": r["DOMAIN_ID"],
    "DOMAIN_NAME": r["DOMAIN_NAME"],
    "text_type": "hierarchy",} for r in rows]

    embedder=Embedder()
    total=0
    for start,chunk in _batched(docs,batch_size):
        end=start+len(chunk)
        embs=embedder.encode(chunk)
        #Creates a collection that embeds batchwise the chunks required
        chroma_collection.upsert(ids=ids[start:end],documents=docs[start:end],metadatas=metas[start:end],embeddings=embs)
        total+=len(chunk)
        logger.info(f"Upserted {end} / {len(docs)} taxonomy rows.")
    
    logger.info(f"Completed taxonomy index build: {total} rows into '{settings.taxonomy_collection}'.")
    return total
    

if __name__ == "__main__":
    build_or_refresh_taxonomy_index()


