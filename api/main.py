"""
Vercel serverless ASGI entrypoint.

Re-exports the FastAPI app from index.api so all routes (/health, /search, …)
are served at the deployment root via vercel.json rewrites.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from index.api import app  # noqa: F401
