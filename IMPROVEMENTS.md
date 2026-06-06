# RAG Chatbot — Current State & Improvement Roadmap

> Assessment date: 2026-06-06

## Current State

**Architecture:** FastAPI backend (`main.py`) + Bootstrap single-page frontend (`index.html`). Multi-tenant design — each "chatbot" gets its own Pinecone serverless index and upload folder.

**Pipeline:** upload PDF/DOCX/TXT → extract text → sentence-split into ~500-char chunks → Cohere `embed-english-v3.0` embeddings → Pinecone → on query, retrieve top-3 chunks → Cohere `generate` for the answer. Chat history saved as JSON files.

**Status: the app currently won't start.** See Tier 1 blockers below.

---

## Tier 1 — Correctness & Security (do first)

### 1.1 Missing `static/` directory (startup crash)
- `main.py:37` mounts `StaticFiles(directory="static")` and `main.py:279` reads `static/index.html`, but `index.html` sits in the repo root and `static/` doesn't exist. FastAPI raises an error at startup.
- **Fix:** create `static/` and move `index.html` into it.

### 1.2 `requirments.txt` is for a different project
- It lists `faiss-cpu`, `sentence-transformers`, `openai`, `jinja2`, but the code imports `cohere`, `pinecone`, `python-dotenv`, `PyPDF2`, `python-docx`.
- **Fix:** rename to `requirements.txt` and list the actual dependencies:
  `fastapi`, `uvicorn`, `cohere`, `pinecone`, `python-dotenv`, `pypdf`, `python-docx`, `python-multipart`.

### 1.3 🚨 `.env` with API keys is committed to git
- The Cohere and Pinecone keys are in the repo history (commit `173d75f`) and `.env` is still tracked.
- **Fix:**
  1. Rotate both API keys in the Cohere and Pinecone dashboards (manual step).
  2. `git rm --cached .env`
  3. Add a `.gitignore` containing `.env`, `uploaded_files/`, `chat_history/`, `__pycache__/`.

### 1.4 Path traversal vulnerability
- `main.py:338` uses `file.filename` raw in the save path — a crafted filename like `..\..\evil.txt` escapes the upload folder.
- **Fix:** sanitize with `os.path.basename()` plus a whitelist regex.

### 1.5 Deprecated APIs
- Cohere `generate` (`main.py:447`) is a legacy endpoint → migrate to the Cohere `chat` API.
- `PyPDF2` is unmaintained → replace with `pypdf`.

### 1.6 Blocking calls in async routes
- `time.sleep(20)` and synchronous Cohere/Pinecone SDK calls block the event loop.
- **Fix:** use async clients or run blocking work in a threadpool (`run_in_executor` / `asyncio.to_thread`).

### 1.7 Wide-open CORS with credentials
- `main.py:30-31` allows all origins with `allow_credentials=True`.
- **Fix:** restrict `allow_origins` to known hosts before deploying.

### 1.8 Duplicate vectors on re-upload
- Re-uploading a file re-upserts everything, duplicating chunks in the index.
- **Fix:** store `filename` in vector metadata and delete old vectors for that file before re-upserting.

---

## Tier 2 — Retrieval Quality (what makes RAG actually good)

### 2.1 Chunking with overlap
- 500 chars with no overlap loses context at chunk boundaries.
- **Fix:** ~512–1024 tokens per chunk with 10–20% overlap, or semantic/recursive chunking.

### 2.2 Batch embeddings
- The embed API is called once *per chunk* (`main.py:360-367`); Cohere accepts up to 96 texts per call.
- **Fix:** batch the chunks — uploads become ~50× faster and cheaper.

### 2.3 Reranking
- Retrieve top-20–50 candidates, then rerank to top-3–5 (Cohere Rerank API).
- Single biggest retrieval-quality lift available.

### 2.4 Hybrid search
- Combine dense vectors with keyword/BM25 (Pinecone sparse-dense) so exact terms like product codes and names still match.

### 2.5 Richer metadata + citations
- Store filename, page number, and chunk position so answers can cite sources. The API already returns `context` but the UI never shows it.

### 2.6 One index + namespaces instead of one index per chatbot
- Pinecone's free tier caps at 5 indexes; namespaces scale to thousands of chatbots and avoid the 20-second index-creation wait (`main.py:214`).

---

## Tier 3 — Generation & UX

- **Conversation memory:** each `/ask` is stateless; pass recent chat history into the prompt so follow-up questions work.
- **Grounding instructions:** tell the model to answer *only* from context and say "I don't know" otherwise; have it cite which chunk it used.
- **Streaming responses:** SSE or WebSocket so answers appear token by token.
- **Show sources in the UI:** display the retrieved chunks under each answer.

---

## Tier 4 — Engineering Hygiene

- Pydantic request models instead of raw `Request.json()`.
- Proper logging instead of `print`.
- Tests (pytest + httpx for the API routes).
- A `README.md` with setup instructions.
- A database (SQLite is fine) instead of JSON files for chat history — current file writes have race conditions.

---

## Evaluation — the part most people skip

A "perfect" RAG system isn't a feature list — it's measured:

1. Build a golden set of ~20–50 question/expected-answer pairs from your own documents.
2. Track **retrieval hit-rate** (is the right chunk in the top-k?) and **answer faithfulness** (does the answer stick to the context?).
3. Re-run the eval after every change to chunking, reranking, or prompts — otherwise you're tuning blind.

---

## Suggested Order of Work

1. Tier 1.3 (rotate keys — do this *now* if the repo is public)
2. Tier 1.1 + 1.2 (get the app running)
3. Remaining Tier 1 items
4. Tier 2.2 (batching — quick win), then 2.1, 2.3
5. Build the eval set, then iterate on Tier 2/3 guided by metrics
6. Tier 4 as ongoing hygiene
