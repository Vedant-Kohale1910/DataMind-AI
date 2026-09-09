
# 🔷 DataMind AI — ML & Data Science RAG Chatbot

> A Retrieval-Augmented Generation chatbot that answers Machine Learning and Data Science questions, grounded entirely in a knowledge base built from real ML tutorial video transcripts.

---

## ◻️ Overview

DataMind AI is an end-to-end RAG (Retrieval-Augmented Generation) application that turns a playlist of Machine Learning tutorial videos into a queryable knowledge base. Instead of relying purely on an LLM's memorized training data, every answer is grounded in transcript chunks retrieved from a vector database — so responses stay accurate, on-topic, and traceable back to a source video and timestamp.

The project covers the full pipeline: transcribing raw video audio, splitting the text into semantically meaningful chunks, embedding and indexing those chunks in a cloud vector store, retrieving the most relevant ones for a given question, and generating a natural-language answer with an LLM — all wrapped in a neumorphic chat interface and deployed publicly.

**✨ Highlights**
- Fully grounded answers — no hallucinated ML facts, only what's in the transcripts
- Cloud-native vector search (Chroma Cloud) — no local vector index to manage
- Lightweight production footprint — hosted embeddings API instead of loading models in-process
- Clean, responsive neumorphic UI with dark/light themes, built from scratch in vanilla HTML/CSS/JS

---

## 🔗 Try the Chatbot

**[datamind-ai-650f.onrender.com](https://datamind-ai-650f.onrender.com)**

> Hosted on Render's free tier — the first request after a period of inactivity may take ~30–50 seconds to wake up.

---

## ◻️ How It Works

The project is built as a sequence of independent, well-scoped stages — from raw video to a deployed chatbot.

**1. Speech-to-Text Transcription**
ML tutorial videos are converted to audio with `ffmpeg`, then transcribed using **OpenAI Whisper (large-v2)**, run on GPU in Google Colab. Each video's output is saved as a structured JSON file containing timestamped text segments — the raw material the rest of the pipeline builds on.

**2. Semantic Chunking**
The raw Whisper segments are re-chunked using **LangChain's `RecursiveCharacterTextSplitter`** (700-character chunks, 100-character overlap) so each chunk is a coherent, semantically meaningful unit rather than an arbitrary text slice. Original start/end timestamps are re-mapped onto every new chunk, so timestamp-accurate source citations survive the re-chunking step.

**3. Embedding & Indexing**
Each chunk is embedded using **`sentence-transformers/all-MiniLM-L6-v2`**, called through **Hugging Face's hosted Inference API** (`HuggingFaceEndpointEmbeddings`) rather than loaded locally — this keeps the production app lightweight enough to run on a free-tier server. The resulting vectors, along with video title, chunk index, and timestamp metadata, are upserted into a collection on **Chroma Cloud** using deterministic IDs, so the pipeline can be re-run safely without creating duplicates.

**4. Retrieval**
At query time, a **similarity-search retriever** built on the same Chroma Cloud collection pulls the top-k most relevant chunks for the user's question, each carrying its source video title and timestamp as metadata.

**5. LLM Answer Generation**
The retrieved chunks are passed as context into a **LangChain RAG chain**, which prompts an LLM to synthesize one coherent answer from across all the excerpts (rather than restating a single chunk verbatim). In production this LLM is **Groq's hosted `openai/gpt-oss-20b`** for fast, free inference; the same chain also supports a local **Ollama** model for offline development. The prompt explicitly instructs the model to render any math as LaTeX and to keep source citations out of the generated text — citations are attached separately from the retrieved chunk metadata.

**6. Web Application**
A **FastAPI** backend exposes a `POST /api/chat` endpoint (query in, grounded answer + sources out) and a `GET /api/health` check that verifies the Chroma Cloud connection. It serves a custom-built **neumorphic chat UI** (HTML5, CSS3, vanilla JS) featuring a dark/light theme toggle, an animated 3D icon sphere, markdown-rendered answers, and expandable source citations.

**7. Deployment**
The codebase is version-controlled on **GitHub** and deployed on **Render** as a Python web service, which auto-redeploys on every push to `main`.

---

## ◻️ Tech Stack

| Layer                    | Technology                                                         |
| ------------------------ | -------------------------------------------------------------------|
| Speech-to-Text            | OpenAI Whisper (`large-v2`), Google Colab (GPU)                    |
| Chunking                  | LangChain Text Splitters (`RecursiveCharacterTextSplitter`)        |
| Embeddings                | `sentence-transformers/all-MiniLM-L6-v2` via HF Inference API      |
| Vector Database           | ChromaDB (Chroma Cloud)                                            |
| Retrieval & Orchestration | LangChain (retriever + RAG chain)                                  |
| LLM                       | Groq — `openai/gpt-oss-20b` (Ollama supported for local dev)       |
| Backend                   | FastAPI, Uvicorn, Pydantic                                         |
| Frontend                  | HTML5, CSS3, Vanilla JavaScript                                    |
| Deployment                | Render                                                              |
| Version Control           | Git, GitHub                                                        |

---

## ◻️ How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Vedant-Kohale1910/DataMind-AI.git
cd DataMind-AI
```

**2. Create a virtual environment and install dependencies**
```bash
cd webapp
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Configure environment variables**

Create a `.env` file in the **project root** (one level above `webapp/`):
```
CHROMA_API_KEY=your_chroma_cloud_api_key
CHROMA_TENANT=your_chroma_tenant_id
CHROMA_DATABASE=your_chroma_database_name
HF_API_TOKEN=your_huggingface_api_token
GROQ_API_KEY=your_groq_api_key
LLM_PROVIDER=groq
```
- `CHROMA_API_KEY` / `CHROMA_TENANT` / `CHROMA_DATABASE` — from your [Chroma Cloud](https://www.trychroma.com) project
- `HF_API_TOKEN` — a free "Read" token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- `GROQ_API_KEY` — a free key from [console.groq.com/keys](https://console.groq.com/keys)
- `LLM_PROVIDER` — set to `groq` to use the hosted model, or omit it and run [Ollama](https://ollama.com) locally with `llama3.1:8b` pulled for a fully offline setup

**4. Start the app**
```bash
python app.py
```

**5. Open in browser**
```
http://127.0.0.1:8000
```

> Note: the chatbot connects to an existing Chroma Cloud collection, so no local data ingestion is needed just to chat. Rebuilding the knowledge base from your own videos means running the `src/` pipeline scripts in order — transcription → chunking → `embed_and_index.py` — which requires your own audio source files.

---

## ◻️ Deployment

The app is deployed on **Render** as a Python web service, configured to auto-deploy on every push to `main`.

Render settings:
- **Root Directory:** `webapp`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `python app.py`
- **Environment Variables:** `CHROMA_API_KEY`, `CHROMA_TENANT`, `CHROMA_DATABASE`, `HF_API_TOKEN`, `GROQ_API_KEY`, `LLM_PROVIDER=groq`

Two design choices keep this comfortably within Render's free-tier 512MB RAM limit:
- **Hosted embeddings, not local ones** — calling Hugging Face's Inference API instead of loading `sentence-transformers` + `torch` in-process avoids the memory spike that a locally loaded embedding model would cause.
- **Hosted LLM, not local ones** — Groq's API serves the `openai/gpt-oss-20b` model remotely, so the deployed server never needs to load or run an LLM itself.

---

## Author

**Vedant Kohale** — [GitHub](https://github.com/Vedant-Kohale1910)