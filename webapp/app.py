"""
DataMind AI — FastAPI backend
Serves the Chatbot UI and exposes a POST /api/chat endpoint
that calls the RAG chain (ChromaDB retriever + Ollama LLM).
"""

import os
import sys
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup — make sure `src/` is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Loads CHROMA_API_KEY / CHROMA_TENANT / CHROMA_DATABASE from a local .env
# file. In production, set these as platform secrets instead — this call
# is a no-op if no .env file exists (e.g. on a deployed host).
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from src.rag_chain import get_answer_with_sources  # noqa: E402
from src.chroma_client import get_cloud_client  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("datamind")

# ---------------------------------------------------------------------------
# Lifespan — warm-up the embedding model on startup so the first query is fast
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("DataMind AI server starting …")
    logger.info("RAG chain ready  (model imports will happen on first query)")
    yield
    logger.info("DataMind AI server shutting down.")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DataMind AI",
    description="RAG-powered ML/Data Science chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the UI (and any local dev tools) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str

class SourceInfo(BaseModel):
    video_title: str
    start: float | None = None
    end: float | None = None
    chunk_index: int | None = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo] = []

# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Accept a user query, retrieve relevant chunks from ChromaDB,
    generate an answer via Ollama, and return the result with sources.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    logger.info(f"Query received: {query[:120]}…")

    try:
        result = get_answer_with_sources(query, k=5)
    except Exception as e:
        logger.error(f"RAG chain error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(e)}",
        )

    # Build source list from retrieved documents
    sources = []
    for doc in result.get("sources", []):
        meta = doc.metadata
        sources.append(SourceInfo(
            video_title=meta.get("video_title", "Unknown"),
            start=meta.get("start"),
            end=meta.get("end"),
            chunk_index=meta.get("chunk_index"),
        ))

    answer_text = result.get("answer", "Sorry, I couldn't generate an answer.")
    logger.info(f"Answer length: {len(answer_text)} chars, sources: {len(sources)}")

    return ChatResponse(answer=answer_text, sources=sources)

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    """
    Confirms the server is up AND that Chroma Cloud credentials are valid
    and reachable — useful right after deploying, to catch a missing/typo'd
    env var immediately instead of only discovering it on the first user query.
    """
    try:
        get_cloud_client().heartbeat()
        chroma_status = "connected"
    except Exception as e:
        chroma_status = f"error: {e}"

    return {"status": "ok", "service": "DataMind AI", "chroma_cloud": chroma_status}

# ---------------------------------------------------------------------------
# Static files — serve the Chatbot UI
# Mount AFTER API routes so /api/* takes priority.
# ---------------------------------------------------------------------------
UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Chatbot_UI")

@app.get("/")
async def serve_index():
    """Serve the main index.html for the root URL."""
    return FileResponse(os.path.join(UI_DIR, "index.html"))

app.mount("/", StaticFiles(directory=UI_DIR), name="static")

# ---------------------------------------------------------------------------
# Entrypoint — lets this run as `python app.py` on platforms that expect
# that, in addition to the usual `uvicorn app:app` for local dev.
# Most hosts (Render, Railway, Hugging Face Spaces) inject the port to
# bind via the PORT environment variable.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
