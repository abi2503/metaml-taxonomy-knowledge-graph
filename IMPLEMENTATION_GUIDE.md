# Step-by-Step Implementation Guide

## 🎓 Beginner-Friendly Guide to Building Your LLM Taxonomy System

---

## Phase 1: Understanding & Setup (Week 1)

### Step 1.1: Understand Your Data Structure

**What you need to know:**
- Your database has 3 levels: **Domains → Dimensions → Areas**
- Each area has a full path: `"FinTech > Banking > Digital Banking > Mobile Banking"`
- The knowledge graph captures these relationships

**Action**: Run this to see your data:
```bash
python scripts/db_layer_test.py
```

**Expected Output**:
```
Domains: 10
Dimensions: 50
Areas: 500
AREA_ID=123, PATH=FinTech > Banking > Digital Banking > Mobile Banking
```

**Intuition**: Think of it like a family tree - Domains are grandparents, Dimensions are parents, Areas are children. The graph shows who is related to whom.

---

### Step 1.2: Verify Knowledge Graph Works

**What you need to know:**
- The graph is built from your database
- It automatically updates when DB changes
- It understands relationships (parent-child, domain-dimension-area)

**Action**: Create and run graph test:
```bash
python scripts/graph_test.py
```

**Expected Output**:
```
✅ Graph built successfully!
   Nodes: 560
   Edges: 550
```

**Intuition**: The graph is like a map of your taxonomy. Each node is a concept, each edge is a relationship. When you add new data to the database, the map updates automatically.

---

## Phase 2: Training Data Preparation (Week 1-2)

### Step 2.1: Extract All Taxonomy Paths from Graph

**Goal**: Get all possible paths from your knowledge graph

**What to do**:
1. Traverse the graph from root (domains) to leaves (areas)
2. Collect all full paths
3. Store as: `"Domain > Dimension > Area > SubArea"`

**Code Structure**:
```python
def extract_all_paths(graph):
    """
    Extract all paths from domain to area in the graph.
    Returns list of full path strings.
    """
    paths = []
    # Find all domain nodes (roots)
    domains = [n for n, d in graph.nodes(data=True) if d.get('label') == 'domain']
    
    for domain in domains:
        # Find all areas reachable from this domain
        areas = [n for n in graph.nodes() if graph.has_path(domain, n) 
                 and graph.nodes[n].get('label') == 'area']
        
        for area in areas:
            # Get the full path
            path_nodes = nx.shortest_path(graph, domain, area)
            path_names = [graph.nodes[n].get('name') for n in path_nodes]
            full_path = " > ".join(path_names)
            paths.append(full_path)
    
    return paths
```

**Intuition**: You're collecting all the "addresses" in your taxonomy. Like collecting all street addresses in a city - you need to know all possible paths.

---

### Step 2.2: Generate Query-Path Training Pairs

**Goal**: Create training examples: `(user_query, taxonomy_path)`

**The Challenge**: 
- You have paths, but need queries that users would ask
- Example: Path = "FinTech > Banking > Cryptocurrency"
- Query could be: "crypto banking", "digital currency", "bitcoin finance"

**Approach 1: Use Area Names as Queries** (Simple)
```python
# Direct mapping
"cryptocurrency" → "FinTech > Banking > Cryptocurrency"
"mobile banking" → "FinTech > Banking > Mobile Banking"
```

**Approach 2: Generate Variations** (Better)
```python
# Use synonyms and variations
"crypto" → "FinTech > Banking > Cryptocurrency"
"digital currency" → "FinTech > Banking > Cryptocurrency"
"bitcoin" → "FinTech > Banking > Cryptocurrency"
```

**Approach 3: Use LLM to Generate Queries** (Best)
```python
# Use GPT/Claude to generate realistic queries
path = "FinTech > Banking > Cryptocurrency"
# Generate: "What is cryptocurrency?", "crypto trading", etc.
```

**Code Structure**:
```python
def generate_training_pairs(graph):
    """
    Generate (query, path) pairs for training.
    """
    training_data = []
    paths = extract_all_paths(graph)
    
    for path in paths:
        # Extract key terms from path
        area_name = path.split(" > ")[-1]  # Last element
        
        # Generate query variations
        queries = [
            area_name.lower(),  # Direct
            f"what is {area_name.lower()}",  # Question form
            f"{area_name.lower()} explained",  # Explanation form
            # Add more variations...
        ]
        
        for query in queries:
            training_data.append({
                "query": query,
                "path": path
            })
    
    return training_data
```

**Intuition**: You're creating flashcards. One side has the user's question, the other side has the answer (taxonomy path). The more variations you create, the better the model learns.

---

### Step 2.3: Format Data for LLM Training

**Goal**: Convert to format LLM expects

**Format**: Each example should be:
```
### Instruction:
Given a user query, predict the taxonomy path.

### Query:
{user_query}

### Response:
{taxonomy_path}
```

**Code Structure**:
```python
def format_for_training(training_pairs):
    """
    Format data for fine-tuning.
    """
    formatted = []
    
    for pair in training_pairs:
        text = f"""### Instruction:
Given a user query, predict the taxonomy path.

### Query:
{pair['query']}

### Response:
{pair['path']}"""
        
        formatted.append({"text": text})
    
    return formatted
```

**Intuition**: You're writing a textbook for the LLM. Each page has a question and answer. The model reads all pages and learns the pattern.

---

## Phase 3: Model Selection & Setup (Week 2)

### Step 3.1: Choose an Open-Source Model

**Options** (ranked by ease):

1. **Llama 2/3** (7B-13B) - Most popular, good balance
   - Pros: Well-documented, good performance
   - Cons: Requires GPU, larger model size

2. **Mistral 7B** - Efficient, good performance
   - Pros: Smaller, faster, good quality
   - Cons: Still needs GPU

3. **Phi-2/3** (Microsoft) - Small, efficient
   - Pros: Can run on CPU, small size
   - Cons: Less capable than larger models

4. **T5-base** (Google) - Text-to-text, perfect for this task
   - Pros: Designed for text generation tasks
   - Cons: Older architecture

**Recommendation for Beginners**: Start with **Mistral 7B** or **Llama 2 7B**

**Intuition**: Think of models like cars - bigger ones are more powerful but need more fuel (GPU). Start with a medium-sized one that's easy to drive.

---

### Step 3.2: Set Up Training Environment

**Requirements**:
- Python 3.10+
- CUDA-capable GPU (recommended) or use CPU (slower)
- Libraries: `transformers`, `datasets`, `peft`, `bitsandbytes`

**Installation**:
```bash
pip install transformers datasets peft bitsandbytes accelerate
pip install torch torchvision torchaudio  # For GPU support
```

**Intuition**: You're setting up your workshop. You need the right tools (libraries) and a good workspace (GPU if possible).

---

## Phase 4: Fine-Tuning (Week 2-3)

### Step 4.1: Load Pre-trained Model

**Code Structure**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-v0.1"  # or "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Use GPU if available
    load_in_4bit=True,  # Use quantization to save memory
)
```

**Intuition**: You're getting a pre-trained model (like a student who already knows English) and you're going to teach it your specific subject (taxonomy paths).

---

### Step 4.2: Prepare Dataset

**Code Structure**:
```python
from datasets import Dataset

# Your formatted training data
training_data = format_for_training(generate_training_pairs(graph))

# Convert to HuggingFace dataset
dataset = Dataset.from_list(training_data)

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

**Intuition**: You're preparing your flashcards in a format the student (model) can easily read and understand.

---

### Step 4.3: Configure Training

**Code Structure**:
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./taxonomy_model",
    num_train_epochs=3,  # How many times to go through data
    per_device_train_batch_size=4,  # Batch size
    gradient_accumulation_steps=4,  # Accumulate gradients
    learning_rate=2e-4,  # Learning speed
    warmup_steps=100,  # Warmup period
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
)
```

**Intuition**: These are your teaching settings - how fast to teach, how many times to repeat, when to check progress.

---

### Step 4.4: Train the Model

**Code Structure**:
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start training
trainer.train()

# Save the fine-tuned model
trainer.save_model("./taxonomy_model_final")
```

**Intuition**: This is the actual teaching process. The model reads all your examples and learns the pattern. It's like a student studying flashcards until they can answer correctly.

---

## Phase 5: Inference & Testing (Week 3)

### Step 5.1: Load Fine-Tuned Model

**Code Structure**:
```python
from transformers import pipeline

# Load your fine-tuned model
model_path = "./taxonomy_model_final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Create pipeline
taxonomy_predictor = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
```

---

### Step 5.2: Test with User Queries

**Code Structure**:
```python
def predict_taxonomy_path(user_query):
    """
    Predict taxonomy path from user query.
    """
    prompt = f"""### Instruction:
Given a user query, predict the taxonomy path.

### Query:
{user_query}

### Response:
"""
    
    result = taxonomy_predictor(
        prompt,
        max_new_tokens=100,
        temperature=0.1,  # Low temperature for more deterministic output
    )
    
    # Extract the path from the response
    response = result[0]['generated_text']
    path = extract_path_from_response(response)
    
    return path

# Test
query = "stable coins"
predicted_path = predict_taxonomy_path(query)
print(f"Query: {query}")
print(f"Predicted Path: {predicted_path}")
```

**Expected Output**:
```
Query: stable coins
Predicted Path: FinTech > Banking > Cryptocurrency > Stablecoins
```

---

## Phase 6: Dynamic Updates (Week 4)

### Step 6.1: Monitor Database Changes

**Goal**: Detect when database changes and retrain/update model

**Approach 1: Scheduled Refresh**
```python
# Run daily/weekly
def refresh_model():
    # 1. Rebuild graph from DB
    graph = build_taxonomy_graph(cache_path=None)  # Force rebuild
    
    # 2. Regenerate training data
    training_data = generate_training_pairs(graph)
    
    # 3. Fine-tune model with new data
    # (Option A: Full retrain)
    # (Option B: Continue training from previous checkpoint)
```

**Approach 2: Incremental Updates**
```python
# Only retrain on new/changed paths
def incremental_update(old_graph, new_graph):
    # Find differences
    new_paths = set(extract_all_paths(new_graph))
    old_paths = set(extract_all_paths(old_graph))
    
    added_paths = new_paths - old_paths
    
    if added_paths:
        # Fine-tune only on new paths
        new_training_data = generate_training_pairs_for_paths(added_paths)
        # Continue training model
```

**Intuition**: When your taxonomy changes (like adding new categories), you need to update the model's knowledge. It's like updating a textbook when new information comes out.

---

## Phase 7: Integration (Week 4)

### Step 7.1: Create API/Interface

**Code Structure**:
```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/predict-taxonomy")
def predict_taxonomy(query: str):
    """
    API endpoint for taxonomy prediction.
    """
    path = predict_taxonomy_path(query)
    
    # Validate path exists in current graph
    graph = build_taxonomy_graph()
    is_valid = validate_path_in_graph(path, graph)
    
    return {
        "query": query,
        "predicted_path": path,
        "is_valid": is_valid
    }
```

---

## 📊 Complete Flow Diagram

```
User Query
    ↓
[LLM Predicts Path]
    ↓
"FinTech > Banking > Cryptocurrency"
    ↓
[Validate in Knowledge Graph]
    ↓
[Return Path + Related Concepts]
    ↓
Response to User
```

---

## 🎯 Key Takeaways for Beginners

1. **Knowledge Graph** = Your source of truth (what paths exist)
2. **Training Data** = Examples showing query → path mapping
3. **Fine-Tuning** = Teaching the model your specific task
4. **Inference** = Using the trained model to predict paths
5. **Dynamic Updates** = Keeping model current with database changes

---

## 📝 Next Steps

1. ✅ Verify knowledge graph works
2. ✅ Extract all paths from graph
3. ✅ Generate training data
4. ⏳ Choose and set up model
5. ⏳ Fine-tune model
6. ⏳ Test predictions
7. ⏳ Set up dynamic updates
8. ⏳ Create API/interface

---

## 🚀 Quick Start Checklist

- [ ] Database connection working
- [ ] Knowledge graph building correctly
- [ ] Can extract all paths
- [ ] Training data generated
- [ ] Model selected and environment set up
- [ ] Fine-tuning completed
- [ ] Testing with sample queries
- [ ] Dynamic update mechanism in place


