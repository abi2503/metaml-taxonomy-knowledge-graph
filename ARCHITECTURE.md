# RAG MetaML Architecture - Step by Step Guide

## 🎯 Overview
This is a **Retrieval-Augmented Generation (RAG)** system for Meta ML taxonomy. It combines:
- **Vector Search** (semantic similarity via embeddings)
- **Knowledge Graph** (structured relationships via NetworkX)

## 📊 Data Flow

```
MySQL Database
    ↓
[1] Fetch Taxonomy Data (domains, dimensions, areas)
    ↓
[2] Build Hierarchical Text ("Domain > Dimension > Area > SubArea")
    ↓
[3] Generate Embeddings (SentenceTransformer)
    ↓
[4] Store in ChromaDB (vector database)
    ↓
[5] Build Knowledge Graph (NetworkX) - parallel track
    ↓
[6] Query & Retrieve (semantic search + graph traversal)
```

## 🏗️ Component Breakdown

### Step 1: Database Layer (`db.py`)
**Purpose**: Connect to MySQL and fetch taxonomy data

**Functions**:
- `get_engine()` - Creates SQLAlchemy connection
- `fetch_domains()` - Gets all domains
- `fetch_dimensions()` - Gets dimensions with domain info
- `fetch_areas()` - Gets areas with parent relationships

**Data Structure**:
- **Domains**: Top-level categories (e.g., "FinTech", "HealthTech")
- **Dimensions**: Sub-categories within domains (e.g., "Banking", "Digital Health Systems")
- **Areas**: Specific topics with parent-child relationships

### Step 2: Hierarchy Builder (`hierarchy.py`)
**Purpose**: Convert flat database records into hierarchical text paths

**Key Function**: `build_hierarchical_rows()`
- Takes domains, dimensions, areas from DB
- Builds full path: `"FinTech > Banking > Digital/Virtual Banking > Banking Platforms"`
- Handles parent-child relationships in areas

**Output Format**:
```python
{
    'AREA_ID': 123,
    'AREA_NAME': 'Banking Platforms',
    'DIMENSION_ID': 45,
    'DIMENSION_NAME': 'Banking',
    'DOMAIN_ID': 1,
    'DOMAIN_NAME': 'FinTech',
    'hierarchical_text': 'FinTech > Banking > Digital/Virtual Banking > Banking Platforms'
}
```

### Step 3: Embeddings (`embeddings.py`)
**Purpose**: Convert text to numerical vectors for semantic search

**Class**: `Embedder`
- Uses SentenceTransformer model (default: `all-MiniLM-L6-v2`)
- Normalizes embeddings for cosine similarity
- Returns float32 numpy arrays

**Why**: Enables semantic search - "stable coins" matches "cryptocurrency" even if words differ

### Step 4: Vector Database (`chroma_client.py` + `index_taxonomy.py`)
**Purpose**: Store and search embeddings

**Process**:
1. Create ChromaDB collection
2. Embed all hierarchical texts
3. Store with metadata (IDs, names, etc.)
4. Query by semantic similarity

**Collection Structure**:
- **IDs**: `"area:{AREA_ID}"` (stable identifiers)
- **Documents**: Hierarchical text strings
- **Metadatas**: All taxonomy fields for filtering
- **Embeddings**: Vector representations

### Step 5: Knowledge Graph (`graph.py`)
**Purpose**: Represent taxonomy as a graph for relationship traversal

**Graph Structure**:
```
domain:{id} --HAS_DIMENSION--> dimension:{id} --HAS_AREA--> area:{id} --HAS_CHILD--> area:{id}
```

**Key Functions**:
- `build_taxonomy_graph()` - Builds graph from DB, caches to disk
- `get_neighbors()` - Find related nodes (parents, children, siblings)
- `describe_path()` - Get path between two nodes

**Use Cases**:
- Expand search results with related concepts
- Navigate taxonomy relationships
- Explain connections between concepts

## 🔄 Current State vs. Potential

### ✅ What Works Now
1. Database fetching ✓
2. Hierarchy building ✓ (just fixed bug)
3. Embedding generation ✓
4. ChromaDB indexing ✓
5. Vector search queries ✓
6. Graph construction ✓

### ⚠️ What's Missing
1. **Integration**: Graph is built but NOT used in retrieval
2. **Hybrid Search**: No combination of vector + graph search
3. **Query Interface**: No unified query function

## 🚀 Next Steps

1. **Test Each Component** - Verify everything works
2. **Create Query Function** - Combine vector search + graph expansion
3. **Add Graph-Based Retrieval** - Use neighbors to expand results
4. **Build End-to-End Pipeline** - Complete RAG system

## 📝 Example Query Flow (Future)

```
User Query: "stable coins"
    ↓
[1] Vector Search (ChromaDB) → Top 5 similar areas
    ↓
[2] Graph Expansion → Get neighbors of top results
    ↓
[3] Re-rank → Combine similarity + graph proximity
    ↓
[4] Return Results + Explanations
```

