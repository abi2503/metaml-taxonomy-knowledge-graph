# Testing Guide - Step by Step

## Prerequisites

### 1. Database Setup
You need MySQL running with the taxonomy data. 

**Option A: Use existing MySQL**
- Copy `.env.example` to `.env`
- Fill in your MySQL credentials
- Ensure MySQL server is running

**Option B: Test without database (mock mode)**
- We can create mock data to test other components
- Database connection will be skipped

### 2. Python Dependencies
```bash
pip install -r requirements.txt
```

## Step-by-Step Testing

### ✅ Step 1: Configuration Test
```bash
python scripts/smoke_test.py
```
**Expected**: Prints all configuration values

### ⚠️ Step 2: Database Connection Test
```bash
python scripts/db_layer_test.py
```
**Expected**: 
- Connects to MySQL
- Fetches domains, dimensions, areas
- Prints counts and sample hierarchical paths

**If fails**: 
- Check MySQL is running: `mysql -u root -p`
- Verify `.env` file has correct credentials
- Check database exists: `SHOW DATABASES;`

### Step 3: Embeddings Test
```bash
python scripts/embeddings_test.py
```
**Expected**: 
- Loads SentenceTransformer model
- Generates embeddings
- Shows similarity scores (related concepts should be closer)

### Step 4: ChromaDB Indexing Test
```bash
python scripts/index_taxonomy_test.py
```
**Expected**:
- Builds/refreshes taxonomy index
- Stores embeddings in ChromaDB
- Performs a sample query ("stable coins")
- Shows top 5 matches

### Step 5: Knowledge Graph Test
```bash
python scripts/graph_test.py  # (we'll create this)
```
**Expected**:
- Builds NetworkX graph
- Tests neighbor finding
- Tests path description

## Current Status

- ✅ Code structure is correct
- ✅ Bug in hierarchy.py is fixed
- ⚠️ Need MySQL connection configured
- ⏳ Components ready to test once DB is connected



