# LLM Training Plan for Taxonomy Path Prediction

## 🎯 High-Level Understanding

### What You Want to Build

**Goal**: Train an open-source LLM to predict the correct taxonomy path when given a user query.

**Example**:
- **User Query**: "I want to learn about cryptocurrency trading platforms"
- **LLM Output**: `FinTech > Trading & Markets > Cryptocurrency Trading > Trading Platforms`

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MySQL Database                            │
│  (domains, dimensions, areas with relationships)            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Knowledge Graph Builder                         │
│  • Fetches data from DB                                     │
│  • Builds NetworkX graph (domain→dimension→area)           │
│  • Understands relationships (HAS_DIMENSION, HAS_AREA)     │
│  • Auto-updates when DB changes                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Training Data Generator                         │
│  • Extracts all taxonomy paths from graph                   │
│  • Creates query-path pairs                                 │
│  • Augments with synonyms/variations                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              LLM Fine-Tuning                                 │
│  • Format: "Query: {user_query}\nPath: {taxonomy_path}"    │
│  • Fine-tune open-source model (Llama, Mistral, etc.)       │
│  • Model learns to map queries → paths                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Inference Engine                                 │
│  • User asks: "stable coins"                                │
│  • Model predicts: "FinTech > Banking > Cryptocurrency..."  │
│  • Returns exact taxonomy path                               │
└─────────────────────────────────────────────────────────────┘
```

### Key Concepts Explained

#### 1. **Knowledge Graph** (What you have)
- **What it is**: A graph structure showing relationships between concepts
- **Why it's useful**: 
  - Understands hierarchy (parent-child relationships)
  - Can traverse relationships (find related concepts)
  - Updates automatically when database changes
- **Your current graph**: `domain → dimension → area → child_area`

#### 2. **Fine-Tuning an LLM** (What you need to add)
- **What it is**: Teaching a pre-trained language model your specific task
- **Why fine-tune**: 
  - Pre-trained models don't know your taxonomy
  - Need to learn the mapping: query → taxonomy path
  - Better than RAG for exact path prediction
- **How it works**: 
  - Show model many examples: "Query: X → Path: Y"
  - Model learns the pattern
  - Can then predict paths for new queries

#### 3. **Dynamic Updates**
- **Problem**: Database changes → Need to retrain model?
- **Solution**: 
  - Knowledge graph rebuilds automatically
  - Generate new training data from updated graph
  - Optionally: Incremental fine-tuning or full retrain

### Why This Approach?

**Traditional RAG** (what you have now):
- Searches for similar concepts
- Returns top matches
- May not give exact path

**Fine-Tuned LLM** (what you want):
- Directly predicts the path
- Understands query intent
- Can handle variations in wording
- More accurate for exact path prediction

**Best of Both Worlds**:
- Use knowledge graph to understand relationships
- Use LLM to predict paths from queries
- Graph ensures paths are valid and up-to-date



