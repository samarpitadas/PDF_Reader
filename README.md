# PDF RAG Chat

Chat with one or more PDFs per conversation. Retrieval combines dense
embeddings and BM25 full-text search in Milvus, fuses both rankings with
RRF, reranks the result with Jina, and answers with a locally hosted
Llama 3.1 model — every claim in the answer is tagged with a `[Sn]`
citation back to a retrieved chunk.

## Prerequisites

Besides Python, this app needs **two services running** and **one API key**:

| Requirement | Why | Where |
|---|---|---|
| Python 3.10+ | App runtime | — |
| A running Milvus instance | Vector + BM25 chunk store | Docker (below) or [Zilliz Cloud](https://zilliz.com/cloud) |
| Ollama, running locally | Embeddings (`nomic-embed-text`) + generation (`llama3.1`) | [ollama.com](https://ollama.com) |
| Jina API key | Reranking (`jina-reranker-v3.5`) — **required to ask questions**, not just to upload PDFs | [jina.ai](https://jina.ai/reranker/) (free tier available) |

`pymilvus` in `requirements.txt` is only the **client library** — it does
not start a Milvus server for you. If `localhost:19530` isn't reachable,
every upload and query will fail.

## 1. Start Milvus

Milvus standalone runs as a Docker container under the hood, so **Docker
Desktop must be installed and running** before either script below works.

```powershell
# Windows
Invoke-WebRequest https://raw.githubusercontent.com/milvus-io/milvus/refs/heads/master/scripts/standalone_embed.bat -OutFile standalone.bat
.\standalone.bat start
```

```bash
# macOS / Linux
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

Either script pulls the Milvus image, starts it, and exposes the service
at `http://localhost:19530` (with its admin UI at `:9091`), matching the
app's default `MILVUS_URI` — no `.env` changes needed if you're running
locally. Confirm it's up before moving on:

```powershell
docker ps
```

You should see a container named `milvus-standalone` (plus `etcd` and
`minio` helper containers) with status `Up`. Other commands the script
supports: `.\standalone.bat stop` and `.\standalone.bat delete` (removes
the container and its data — use this if you need to reset the vector
store from scratch).

If you'd rather not run Docker locally, point `MILVUS_URI`/`MILVUS_TOKEN`
at a [Zilliz Cloud](https://zilliz.com/cloud) instance instead — no code
change needed, just set the env vars (see step 4).

## 2. Install and start Ollama, pull the models

```powershell
# Windows
irm https://ollama.com/install.ps1 | iex
ollama pull llama3.1
ollama pull nomic-embed-text
```

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1
ollama pull nomic-embed-text
```

The installer starts the Ollama service in the background. Verify it's
listening before continuing:

```bash
curl http://localhost:11434/api/tags
```

If that doesn't return JSON, start it manually with `ollama serve`.

> `llama3.1` at default settings is a large, CPU-heavy model — expect
> generation latencies in the 30–170s range per question on modest
> hardware (see `evaluation/metrics.json` for real numbers). If that's
> too slow, pull a smaller tag (e.g. `llama3.1:8b-instruct-q4_0`) and
> update the model name in `app.py`'s `invoke_llama_with_metrics`.

## 3. Set up the Python environment

```powershell
# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 4. Configure environment variables

Create a `.env` file in the project root:

```ini
# Required
JINA_API_KEY=your_jina_api_key_here

# Optional — defaults shown
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=
MILVUS_COLLECTION=pdf_rag_chunks
DENSE_VECTOR_DIM=768
OLLAMA_API_URL=http://localhost:11434/api/generate
```

Without `JINA_API_KEY`, PDF uploads still work, but every question will
raise `RuntimeError: JINA_API_KEY is not set` — reranking is a hard
dependency in the query path.

## 5. Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`), attach
one or more PDFs in the chat input, and ask a question.

## Project structure

```
.
├── app.py                       # Streamlit app: UI, ingestion, RAG pipeline
├── requirements.txt
├── .env                         # you create this (not committed)
├── rag_data/
│   ├── chat_history.db          # SQLite: chats, messages, documents
│   └── chats/<chat_id>/         # raw uploaded PDFs, per chat
├── LOG/<YYYY-MM-DD>/rag_<date>.txt   # structured JSONL logs (uploads + queries)
└── evaluation/
    ├── questions.json           # 24-question grounded eval set
    ├── run_evaluation.py        # drives the eval through the live pipeline
    ├── results.json             # per-question output (generated)
    └── metrics.json             # aggregated metrics (generated)
```

## Running the evaluation suite

```powershell
# Fresh chat, indexes both source PDFs, then runs all 24 questions
.\venv\Scripts\python evaluation\run_evaluation.py --bootstrap `
  --pdf "path\to\Provisional Placement Policy-2026.pdf" `
  --pdf "path\to\Assignment4_Report.pdf"

# Reuse an already-indexed chat
.\venv\Scripts\python evaluation\run_evaluation.py --chat-id <CHAT_ID>

# Resume from a specific question, keeping earlier saved results
.\venv\Scripts\python evaluation\run_evaluation.py --chat-id <CHAT_ID> --start-at 14

# Smoke test with only the first N questions
.\venv\Scripts\python evaluation\run_evaluation.py --chat-id <CHAT_ID> --limit 3
```

Results stream into `evaluation/results.json` question-by-question (so a
crash mid-run doesn't lose earlier answers), and `evaluation/metrics.json`
holds the aggregated correctness/citation/latency metrics, broken down by
category.

## Troubleshooting

- **`JINA_API_KEY is not set`** — add it to `.env` and restart the app.
- **Connection refused on `:19530`** — Milvus isn't running; see step 1.
  Check `docker ps` first — if Docker Desktop itself isn't running,
  `standalone.bat start` fails before Milvus ever comes up.
- **`ReadTimeout` from `localhost:11434`** — Ollama generation exceeded
  the app's fixed 180s timeout (`invoke_llama_with_metrics`). Either use
  a smaller/faster model or raise the `timeout=180` value in `app.py`.
- **Uploads work but questions fail** — almost always a missing or
  invalid `JINA_API_KEY`.
