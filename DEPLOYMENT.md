# MetaML Taxonomy — Run locally & deploy

## Architecture (two processes)

| Process | Role | Default URL |
|---------|------|-------------|
| **FastAPI** (`index/api.py`) | UC1 search, UC2 graph, UC3–UC5 analytics | `http://localhost:8000` |
| **Streamlit** (`frontend/app.py`) | UI; calls API via `API_BASE_URL` | `http://localhost:8501` |

The graph is loaded from `data/taxonomy_graph.gpickle`. Embeddings are built in memory at API startup (`all-MiniLM-L6-v2`, ~622 area nodes).

---

## Run locally

**Important:** Always run commands from the **project root** (`rag_metaml/`), not from `frontend/`.  
Running `uvicorn` inside `frontend/` causes `ModuleNotFoundError: No module named 'index'`.

### 1. One-time setup

```bash
cd /path/to/rag_metaml
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional; edit if you use MySQL scripts
```

Confirm `data/taxonomy_graph.gpickle` exists (~90 KB).

### 2. Terminal A — API (uvicorn)

```bash
cd /path/to/rag_metaml
source .venv/bin/activate
bash start.sh
```

Or manually:

```bash
PYTHONPATH=. uvicorn index.api:app --reload --port 8000 --host 127.0.0.1
```

Wait until you see `[api] Ready.` and `Application startup complete.`  
Smoke test:

```bash
curl http://127.0.0.1:8000/health
```

### 3. Terminal B — Streamlit UI

```bash
cd /path/to/rag_metaml
source .venv/bin/activate
streamlit run frontend/app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). Use **Check API** in the header to verify connectivity.

### Optional: point UI at a remote API

```bash
export API_BASE_URL=https://your-api.example.com
streamlit run frontend/app.py
```

---

## Push to GitHub

```bash
cd /path/to/rag_metaml
git init
git add .
git commit -m "Initial commit: MetaML taxonomy API and Streamlit UI"
git branch -M main
git remote add origin https://github.com/YOUR_USER/rag_metaml.git
git push -u origin main
```

Do **not** commit `.env` (listed in `.gitignore`). Commit `data/taxonomy_graph.gpickle` unless you rebuild it in CI.

---

## Deploy (recommended split)

This project is **not a good fit for Vercel** as a whole:

- **Streamlit** needs a long-lived Python process; Vercel is serverless with short timeouts.
- **FastAPI + sentence-transformers** needs ~1 GB RAM, a multi-second cold start, and a large dependency tree — beyond typical Vercel serverless limits.

Use **two hosts**: API on a container/PaaS service, UI on **Streamlit Community Cloud** (free tier works well).

### A. Deploy API — Render (example)

1. [render.com](https://render.com) → **New → Web Service** → connect your GitHub repo.
2. Settings:
   - **Root directory:** (repo root)
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `PYTHONPATH=. uvicorn index.api:app --host 0.0.0.0 --port $PORT`
   - **Instance type:** at least **512 MB–1 GB** RAM (embeddings + PyTorch stack).
3. After deploy, copy the public URL (e.g. `https://rag-metaml-api.onrender.com`).
4. Verify: `curl https://YOUR_URL/health`

**Alternatives:** Railway, Fly.io, Google Cloud Run (Docker), AWS ECS — same start command and memory requirements.

`render.yaml` in this repo is optional infrastructure-as-code for the same setup.

### B. Deploy UI — Streamlit Community Cloud

1. [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub.
2. **New app** → select repo `rag_metaml`.
3. **Main file path:** `frontend/app.py`
4. **Secrets** (Settings → Secrets), TOML:

```toml
API_BASE_URL = "https://YOUR-RENDER-API.onrender.com"
```

5. Deploy. The app reads `API_BASE_URL` via `os.getenv` (injected from secrets on Cloud).

### C. About Vercel

| Component | Vercel? | Why |
|-----------|---------|-----|
| Streamlit `frontend/app.py` | No | Not serverless; needs persistent process |
| FastAPI + embeddings | Not practical | Model size, RAM, startup time, 10s/60s limits |

If you must use Vercel, you would only host a **thin static front-end** or a **pre-warmed API elsewhere** — not this Streamlit + in-process embedding stack.

---

## Environment variables (production)

| Variable | Used by | Description |
|----------|---------|-------------|
| `API_BASE_URL` | Streamlit | Public FastAPI base URL (no trailing slash) |
| `GRAPH_CACHE_PATH` | API | Optional override for `.gpickle` path |
| `PORT` | Render/Railway | Set by host; pass to uvicorn `--port` |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No module named 'index'` | Run uvicorn from **project root** with `PYTHONPATH=.` |
| `Pickle not found` | Ensure `data/taxonomy_graph.gpickle` exists or set `GRAPH_CACHE_PATH` |
| Streamlit: "Cannot reach the API" | Start API first; check `API_BASE_URL` / firewall |
| API exits / OOM on cloud | Increase RAM; first request after idle may be slow on free tier |
| Segfault loading model locally | Use project `.venv`; avoid mixing system Python with torch |

---

## API routes (reference)

- `GET /health` — status + graph/embedder stats  
- `POST /search` — UC1 semantic search  
- `GET /node/{id}`, `/neighbors/{id}`, `/subtree/{id}`, `/search-name` — UC2  
- `POST /classify` — UC4  
- `POST /compare` — UC3  
- `GET /coverage`, `/overlaps` — UC5  

Interactive docs when API is running: `http://localhost:8000/docs`
