import os
import time
import uuid
import hashlib
import sqlite3
import tempfile
import shutil
import json
import traceback
from datetime import datetime

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import requests
import re

from dotenv import load_dotenv
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

APP_DIR = "rag_data"
DB_PATH = os.path.join(APP_DIR, "chat_history.db")
CHATS_DIR = os.path.join(APP_DIR, "chats")

# Retrieval + reranking configuration
RETRIEVAL_K = 12
RERANK_K = 6
JINA_RERANKER_MODEL = "jina-reranker-v3.5"
JINA_API_URL = "https://api.jina.ai/v1/rerank"
JINA_API_KEY = os.getenv("JINA_API_KEY")

# Structured logging configuration.
# Logs are written as JSON Lines inside:
# LOG/YYYY-MM-DD/rag_YYYY-MM-DD.txt
LOG_DIR = "LOG"
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")

os.makedirs(LOG_DIR, exist_ok=True)

# Chunk configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

os.makedirs(APP_DIR, exist_ok=True)
os.makedirs(CHATS_DIR, exist_ok=True)


# ============================================================
# STRUCTURED LOGGING
# ============================================================

def _json_safe(value):
    """Convert common runtime objects into JSON-serializable values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def get_log_paths():
    """Return today's dated log folder and JSONL-in-TXT log file."""
    today = datetime.now().strftime("%Y-%m-%d")
    dated_folder = os.path.join(LOG_DIR, today)
    os.makedirs(dated_folder, exist_ok=True)
    log_file = os.path.join(
        dated_folder,
        f"rag_{today}.txt"
    )
    return dated_folder, log_file


def write_structured_log(event_type, **data):
    """
    Append one structured JSON record to the current day's .txt log.

    The file remains .txt as requested, while each line is valid JSON,
    making it both human-readable and machine-parseable.
    """
    _, log_file = get_log_paths()

    record = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "event": event_type,
        **_json_safe(data)
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def ns_to_seconds(value):
    """Convert Ollama's nanosecond durations to seconds."""
    if value is None:
        return None
    try:
        return float(value) / 1_000_000_000
    except (TypeError, ValueError):
        return None


def invoke_llama_with_metrics(prompt_text):
    """
    Call Ollama directly so the response includes token counts and
    stage-level generation timings exposed by Ollama.
    """
    stage_start = time.perf_counter()

    response = requests.post(
        OLLAMA_API_URL,
        json={
            "model": "llama3.1",
            "prompt": prompt_text,
            "stream": False
        },
        timeout=180
    )
    response.raise_for_status()

    payload = response.json()
    wall_time = time.perf_counter() - stage_start

    metrics = {
        "model": payload.get("model", "llama3.1"),
        "prompt_tokens": payload.get("prompt_eval_count"),
        "completion_tokens": payload.get("eval_count"),
        "total_tokens": None,
        "ollama_total_duration_s": ns_to_seconds(
            payload.get("total_duration")
        ),
        "ollama_load_duration_s": ns_to_seconds(
            payload.get("load_duration")
        ),
        "ollama_prompt_eval_duration_s": ns_to_seconds(
            payload.get("prompt_eval_duration")
        ),
        "ollama_eval_duration_s": ns_to_seconds(
            payload.get("eval_duration")
        ),
        "llama_wall_clock_latency_s": wall_time
    }

    if (
        metrics["prompt_tokens"] is not None
        and metrics["completion_tokens"] is not None
    ):
        metrics["total_tokens"] = (
            metrics["prompt_tokens"]
            + metrics["completion_tokens"]
        )

    return payload.get("response", ""), metrics


def format_log_location():
    _, log_file = get_log_paths()
    return log_file



# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PDF RAG Chat",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    return sqlite3.connect(DB_PATH)


def init_database():

    conn = get_connection()
    cursor = conn.cursor()

    # --------------------------------------------------------
    # Chats
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # --------------------------------------------------------
    # Messages
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(chat_id) REFERENCES chats(id)
        )
    """)

    # --------------------------------------------------------
    # Documents
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(chat_id) REFERENCES chats(id)
        )
    """)

    conn.commit()
    conn.close()


def migrate_database_schema():
    # Handle databases created by older versions of the app.
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(chats)")
    chat_columns = {row[1] for row in cursor.fetchall()}

    if "created_at" not in chat_columns:
        cursor.execute("ALTER TABLE chats ADD COLUMN created_at TIMESTAMP")
        cursor.execute(
            "UPDATE chats SET created_at = CURRENT_TIMESTAMP "
            "WHERE created_at IS NULL"
        )

    if "updated_at" not in chat_columns:
        cursor.execute("ALTER TABLE chats ADD COLUMN updated_at TIMESTAMP")
        cursor.execute(
            "UPDATE chats SET updated_at = CURRENT_TIMESTAMP "
            "WHERE updated_at IS NULL"
        )

    cursor.execute("PRAGMA table_info(documents)")
    document_columns = {row[1] for row in cursor.fetchall()}

    if "uploaded_at" not in document_columns:
        cursor.execute("ALTER TABLE documents ADD COLUMN uploaded_at TIMESTAMP")
        cursor.execute(
            "UPDATE documents SET uploaded_at = CURRENT_TIMESTAMP "
            "WHERE uploaded_at IS NULL"
        )

    conn.commit()
    conn.close()


init_database()
migrate_database_schema()


# ============================================================
# CHAT DATABASE FUNCTIONS
# ============================================================

def create_chat():

    chat_id = str(uuid.uuid4())

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO chats (id, title, created_at, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """,
        (chat_id, "New Chat")
    )

    conn.commit()
    conn.close()

    # Create folder for this chat
    chat_folder = os.path.join(CHATS_DIR, chat_id)

    os.makedirs(chat_folder, exist_ok=True)
    os.makedirs(
        os.path.join(chat_folder, "chroma"),
        exist_ok=True
    )

    return chat_id


def get_all_chats():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, title, created_at, updated_at
        FROM chats
        ORDER BY updated_at DESC
    """)

    chats = cursor.fetchall()

    conn.close()

    return chats


def get_chat(chat_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, title, created_at, updated_at
        FROM chats
        WHERE id = ?
    """, (chat_id,))

    chat = cursor.fetchone()

    conn.close()

    if chat is None:
        return None

    return {
        "id": chat[0],
        "title": chat[1],
        "created_at": chat[2],
        "updated_at": chat[3]
    }


def update_chat_timestamp(chat_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE chats
        SET updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (chat_id,))

    conn.commit()
    conn.close()


def update_chat_title(chat_id, title):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE chats
        SET title = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (title, chat_id))

    conn.commit()
    conn.close()


def delete_chat(chat_id):

    conn = get_connection()
    cursor = conn.cursor()

    # Delete messages
    cursor.execute("""
        DELETE FROM messages
        WHERE chat_id = ?
    """, (chat_id,))

    # Delete documents
    cursor.execute("""
        DELETE FROM documents
        WHERE chat_id = ?
    """, (chat_id,))

    # Delete chat
    cursor.execute("""
        DELETE FROM chats
        WHERE id = ?
    """, (chat_id,))

    conn.commit()
    conn.close()

    # Delete files and Chroma database
    chat_folder = os.path.join(CHATS_DIR, chat_id)

    if os.path.exists(chat_folder):
        shutil.rmtree(chat_folder)


# ============================================================
# MESSAGE FUNCTIONS
# ============================================================

def save_message(chat_id, role, content):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO messages (chat_id, role, content, created_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (chat_id, role, content)
    )

    conn.commit()
    conn.close()

    update_chat_timestamp(chat_id)


def get_messages(chat_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT role, content, created_at
        FROM messages
        WHERE chat_id = ?
        ORDER BY id ASC
    """, (chat_id,))

    messages = cursor.fetchall()

    conn.close()

    return [
        {
            "role": row[0],
            "content": row[1],
            "created_at": row[2]
        }
        for row in messages
    ]


# ============================================================
# DOCUMENT FUNCTIONS
# ============================================================

def save_document(
    chat_id,
    filename,
    file_path,
    file_hash
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO documents
        (
            chat_id,
            filename,
            file_path,
            file_hash,
            uploaded_at
        )
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (
            chat_id,
            filename,
            file_path,
            file_hash
        )
    )

    conn.commit()
    conn.close()


def get_documents(chat_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            filename,
            file_path,
            file_hash,
            uploaded_at
        FROM documents
        WHERE chat_id = ?
        ORDER BY uploaded_at ASC
    """, (chat_id,))

    documents = cursor.fetchall()

    conn.close()

    return [
        {
            "id": row[0],
            "filename": row[1],
            "file_path": row[2],
            "file_hash": row[3],
            "uploaded_at": row[4]
        }
        for row in documents
    ]


def document_exists(chat_id, file_hash):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id
        FROM documents
        WHERE chat_id = ?
        AND file_hash = ?
    """, (chat_id, file_hash))

    result = cursor.fetchone()

    conn.close()

    return result is not None


# ============================================================
# AI MODELS
# ============================================================

@st.cache_resource
def load_embeddings():

    return OllamaEmbeddings(
        model="nomic-embed-text"
    )


@st.cache_resource
def load_llm():

    return OllamaLLM(
        model="llama3.1"
    )


embeddings = load_embeddings()
llm = load_llm()

if not JINA_API_KEY:
    st.warning(
        "JINA_API_KEY is not set. Uploads can still be added, "
        "but questions need JINA_API_KEY for reranking."
    )


# ============================================================
# PDF PROCESSING
# ============================================================

def calculate_file_hash(file_bytes):

    return hashlib.md5(file_bytes).hexdigest()


def extract_pdf_documents(file_path):

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    return documents


def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


# ============================================================
# CHROMA DATABASE
# ============================================================

def get_chroma_path(chat_id):

    return os.path.join(
        CHATS_DIR,
        chat_id,
        "chroma"
    )


def get_vector_store(chat_id):

    chroma_path = get_chroma_path(chat_id)

    return Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings
    )


def make_stable_chunk_id(chat_id, file_hash, page, chunk_index, chunk_text):
    raw = f"{chat_id}|{file_hash}|{page}|{chunk_index}|{chunk_text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def add_documents_to_vector_store(chat_id, chunks, chunk_ids):
    vector_store = get_vector_store(chat_id)
    vector_store.add_documents(
        documents=chunks,
        ids=chunk_ids
    )
    return vector_store


def make_unique_stored_filename(chat_id, filename, file_hash):
    safe_filename = os.path.basename(filename)
    chat_folder = os.path.join(CHATS_DIR, chat_id)
    candidate = os.path.join(chat_folder, safe_filename)

    if not os.path.exists(candidate):
        return safe_filename

    stem, ext = os.path.splitext(safe_filename)
    return f"{stem}_{file_hash[:10]}{ext}"


# ============================================================
# PROCESS ONE PDF
# ============================================================

def process_pdf(chat_id, file_bytes, filename):
    total_start = time.perf_counter()
    upload_id = str(uuid.uuid4())
    stage_timings = {}

    # --------------------------------------------------------
    # Stage 1: MD5 duplicate detection
    # --------------------------------------------------------
    stage_start = time.perf_counter()

    file_hash = calculate_file_hash(file_bytes)

    stage_timings["md5_hash_s"] = (
        time.perf_counter() - stage_start
    )

    if document_exists(chat_id, file_hash):
        stage_timings["total_s"] = time.perf_counter() - total_start

        write_structured_log(
            "pdf_duplicate",
            upload_id=upload_id,
            chat_id=chat_id,
            original_filename=filename,
            md5=file_hash,
            file_size_bytes=len(file_bytes),
            stage_timings=stage_timings,
            status="duplicate_skipped",
            log_file=format_log_location()
        )

        return get_vector_store(chat_id), False, 0

    chat_folder = os.path.join(CHATS_DIR, chat_id)
    os.makedirs(chat_folder, exist_ok=True)

    # --------------------------------------------------------
    # Stage 2: Unique filename + disk write
    # --------------------------------------------------------
    stage_start = time.perf_counter()

    safe_filename = make_unique_stored_filename(
        chat_id,
        filename,
        file_hash
    )
    document_path = os.path.join(
        chat_folder,
        safe_filename
    )

    with open(document_path, "wb") as f:
        f.write(file_bytes)

    stage_timings["file_write_s"] = (
        time.perf_counter() - stage_start
    )

    # --------------------------------------------------------
    # Stage 3: PDF extraction
    # --------------------------------------------------------
    stage_start = time.perf_counter()

    documents = extract_pdf_documents(
        document_path
    )

    stage_timings["pdf_extraction_s"] = (
        time.perf_counter() - stage_start
    )

    # --------------------------------------------------------
    # Stage 4: Chunking
    # --------------------------------------------------------
    stage_start = time.perf_counter()

    chunks = split_documents(documents)

    stage_timings["chunking_s"] = (
        time.perf_counter() - stage_start
    )

    # --------------------------------------------------------
    # Stage 5: Metadata + stable chunk IDs
    # --------------------------------------------------------
    stage_start = time.perf_counter()

    chunk_ids = []

    for chunk_index, chunk in enumerate(chunks):
        page = chunk.metadata.get("page")

        chunk.metadata["chat_id"] = chat_id
        chunk.metadata["source_file"] = safe_filename
        chunk.metadata["file_hash"] = file_hash
        chunk.metadata["page"] = page
        chunk.metadata["chunk_index"] = chunk_index

        chunk_id = make_stable_chunk_id(
            chat_id,
            file_hash,
            page,
            chunk_index,
            chunk.page_content
        )

        chunk.metadata["chunk_id"] = chunk_id
        chunk_ids.append(chunk_id)

    stage_timings["metadata_and_chunk_ids_s"] = (
        time.perf_counter() - stage_start
    )

    # --------------------------------------------------------
    # Stage 6: Chroma indexing
    # --------------------------------------------------------
    stage_start = time.perf_counter()

    vector_store = add_documents_to_vector_store(
        chat_id,
        chunks,
        chunk_ids
    )

    stage_timings["chroma_indexing_s"] = (
        time.perf_counter() - stage_start
    )

    # --------------------------------------------------------
    # Stage 7: SQLite document registration
    # --------------------------------------------------------
    stage_start = time.perf_counter()

    save_document(
        chat_id,
        safe_filename,
        document_path,
        file_hash
    )

    stage_timings["database_save_s"] = (
        time.perf_counter() - stage_start
    )

    stage_timings["total_s"] = (
        time.perf_counter() - total_start
    )

    write_structured_log(
        "pdf_upload",
        upload_id=upload_id,
        chat_id=chat_id,
        original_filename=filename,
        stored_filename=safe_filename,
        file_path=document_path,
        md5=file_hash,
        file_size_bytes=len(file_bytes),
        page_count=len(documents),
        chunk_count=len(chunks),
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        chunk_ids=chunk_ids,
        pages=[
            chunk.metadata.get("page") + 1
            if chunk.metadata.get("page") is not None
            else None
            for chunk in chunks
        ],
        stage_timings=stage_timings,
        status="indexed",
        log_file=format_log_location()
    )

    return (
        vector_store,
        True,
        stage_timings["total_s"]
    )


# ============================================================
# LOAD EXISTING VECTOR DATABASE
# ============================================================

def load_existing_vector_store(chat_id):

    documents = get_documents(chat_id)

    chroma_path = get_chroma_path(chat_id)

    if not documents:
        return None

    if not os.path.exists(chroma_path):
        return None

    return Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings
    )


# ============================================================
# RAG RESPONSE
# ============================================================

def jina_rerank(query, documents, top_n=RERANK_K):
    if not documents:
        return [], {}, 0.0

    if not JINA_API_KEY:
        raise RuntimeError(
            "JINA_API_KEY is not set. Add it to your environment."
        )

    stage_start = time.perf_counter()

    response = requests.post(
        JINA_API_URL,
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": JINA_RERANKER_MODEL,
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": min(top_n, len(documents)),
            "return_documents": False
        },
        timeout=60
    )
    response.raise_for_status()

    payload = response.json()
    stage_latency = time.perf_counter() - stage_start

    reranked = []

    for result in payload.get("results", []):
        index = result.get("index")

        if index is None or index >= len(documents):
            continue

        doc = documents[index]
        doc.metadata["jina_score"] = float(
            result.get("relevance_score", 0.0)
        )
        reranked.append(doc)

    jina_usage = payload.get("usage", {})

    jina_metrics = {
        "model": JINA_RERANKER_MODEL,
        "candidate_count": len(documents),
        "returned_count": len(reranked),
        "latency_s": stage_latency,
        "usage": jina_usage
    }

    return reranked, jina_metrics, stage_latency


def assess_evidence(answer, docs):
    if not docs:
        return "NO RETRIEVED EVIDENCE"

    phrases = [
        "i couldn't find that information",
        "cannot find that information",
        "not found in the uploaded",
        "not available in the provided",
        "insufficient information"
    ]

    if any(p in answer.lower() for p in phrases):
        return "NOT FOUND IN UPLOADED PDFs"

    return "SUPPORTED BY RETRIEVED PDF EVIDENCE"


def detect_cross_document_conflicts(docs):
    # Conservative warning: flag different numeric evidence across
    # different source PDFs. This is explicitly a warning, not a
    # claim that every numeric difference is a true contradiction.
    source_text = {}

    for doc in docs:
        source = doc.metadata.get("source_file", "Unknown PDF")
        source_text.setdefault(source, []).append(doc.page_content)

    sources = list(source_text.keys())
    warnings = []

    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            a = " ".join(source_text[sources[i]])
            b = " ".join(source_text[sources[j]])

            nums_a = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", a))
            nums_b = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", b))

            if nums_a and nums_b and nums_a != nums_b:
                warnings.append(
                    f"Possible conflicting numeric evidence between "
                    f"{sources[i]} and {sources[j]}. Verify the source/date."
                )

    return warnings


def generate_response(vector_store, query, chat_history, chat_id):
    total_start = time.perf_counter()
    request_id = str(uuid.uuid4())
    stage_timings = {}

    # --------------------------------------------------------
    # Stage 1: Chroma retrieval
    # --------------------------------------------------------
    stage_start = time.perf_counter()

    candidate_docs = vector_store.similarity_search(
        query,
        k=RETRIEVAL_K
    )

    stage_timings["chroma_retrieval_s"] = (
        time.perf_counter() - stage_start
    )

    # --------------------------------------------------------
    # Stage 2: Jina reranking
    # --------------------------------------------------------
    stage_start = time.perf_counter()

    matching_docs, jina_metrics, _ = jina_rerank(
        query,
        candidate_docs,
        top_n=RERANK_K
    )

    stage_timings["jina_rerank_s"] = (
        time.perf_counter() - stage_start
    )

    # --------------------------------------------------------
    # Stage 3: Build citation-aware context
    # --------------------------------------------------------
    stage_start = time.perf_counter()

    context_parts = []
    source_records = []

    for source_index, doc in enumerate(matching_docs, start=1):
        source_file = doc.metadata.get(
            "source_file",
            "Unknown PDF"
        )
        page = doc.metadata.get("page")
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        jina_score = doc.metadata.get("jina_score", 0.0)

        if page is not None:
            source_info = f"{source_file}, page {page + 1}"
        else:
            source_info = source_file

        citation = f"[S{source_index}]"

        context_parts.append(
            f"{citation} Source: {source_info}\n"
            f"Chunk ID: {chunk_id}\n"
            f"{doc.page_content}"
        )

        source_records.append({
            "citation": citation,
            "source_file": source_file,
            "page": page + 1 if page is not None else None,
            "chunk_id": chunk_id,
            "jina_relevance_score": jina_score,
            "chunk_text": doc.page_content
        })

    context = "\n\n".join(context_parts)

    history_text = ""

    for message in chat_history:
        history_text += (
            f"{message['role'].upper()}: "
            f"{message['content']}\n"
        )

    stage_timings["context_build_s"] = (
        time.perf_counter() - stage_start
    )

    # --------------------------------------------------------
    # Stage 4: Prompt construction
    # --------------------------------------------------------
    stage_start = time.perf_counter()

    prompt_template = ChatPromptTemplate.from_template(
        """
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the provided PDF evidence.

IMPORTANT RULES:

1. Use the PDF context as the primary source.
2. Do not invent information.
3. Every factual claim from a PDF MUST include [S1], [S2], etc.
4. If the evidence does not establish the answer, say:
   "I couldn't find that information in the uploaded PDFs."
5. If PDFs disagree, explicitly mention the conflict and cite both sources.
6. Never create a citation that is not in the supplied evidence.
7. Use conversation history only for follow-up context.
8. Keep the answer clear and concise.

Previous conversation:
{history}

PDF evidence:
{context}

Current question:
{input}

Answer:
"""
    )

    prompt_value = prompt_template.invoke({
        "history": history_text,
        "context": context,
        "input": query
    })
    prompt_text = prompt_value.to_string()

    stage_timings["prompt_construction_s"] = (
        time.perf_counter() - stage_start
    )

    # --------------------------------------------------------
    # Stage 5: Llama generation + token accounting
    # --------------------------------------------------------
    stage_start = time.perf_counter()

    answer, llama_metrics = invoke_llama_with_metrics(
        prompt_text
    )

    stage_timings["llama_generation_s"] = (
        time.perf_counter() - stage_start
    )

    # --------------------------------------------------------
    # Stage 6: Evidence validation
    # --------------------------------------------------------
    stage_start = time.perf_counter()

    evidence_status = assess_evidence(
        answer,
        matching_docs
    )

    stage_timings["evidence_check_s"] = (
        time.perf_counter() - stage_start
    )

    # --------------------------------------------------------
    # Stage 7: Cross-document conflict check
    # --------------------------------------------------------
    stage_start = time.perf_counter()

    conflicts = detect_cross_document_conflicts(
        matching_docs
    )

    stage_timings["conflict_check_s"] = (
        time.perf_counter() - stage_start
    )

    total_time = time.perf_counter() - total_start
    stage_timings["total_pipeline_s"] = total_time

    # --------------------------------------------------------
    # Structured query log
    # --------------------------------------------------------
    write_structured_log(
        "rag_query",
        request_id=request_id,
        chat_id=chat_id,
        query=query,
        chat_history_message_count=len(chat_history),
        retrieval={
            "requested_k": RETRIEVAL_K,
            "candidate_count": len(candidate_docs),
            "stage_latency_s": stage_timings[
                "chroma_retrieval_s"
            ]
        },
        reranking=jina_metrics,
        final_context={
            "requested_k": RERANK_K,
            "selected_count": len(matching_docs),
            "sources": source_records
        },
        llama={
            **llama_metrics,
            "prompt_characters": len(prompt_text),
            "answer_characters": len(answer)
        },
        evidence_status=evidence_status,
        conflicts=conflicts,
        stage_timings=stage_timings,
        prompt=prompt_text,
        answer=answer,
        log_file=format_log_location()
    )

    return (
        answer,
        stage_timings["chroma_retrieval_s"],
        stage_timings["jina_rerank_s"],
        total_time,
        matching_docs,
        evidence_status,
        conflicts,
        stage_timings,
        llama_metrics,
        request_id
    )


# ============================================================
# SESSION STATE
# ============================================================

if "current_chat_id" not in st.session_state:

    chats = get_all_chats()

    if chats:

        st.session_state.current_chat_id = chats[0][0]

    else:

        st.session_state.current_chat_id = create_chat()


if "vector_store" not in st.session_state:

    st.session_state.vector_store = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📚 PDF RAG")

    # --------------------------------------------------------
    # New Chat
    # --------------------------------------------------------

    if st.button(
        "＋ New Chat",
        use_container_width=True
    ):

        new_chat_id = create_chat()

        st.session_state.current_chat_id = new_chat_id
        st.session_state.vector_store = None

        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # Chat History
    # --------------------------------------------------------

    st.subheader("Chats")

    chats = get_all_chats()

    if chats:

        for chat in chats:

            chat_id = chat[0]
            title = chat[1]

            # Highlight current chat
            if chat_id == st.session_state.current_chat_id:

                button_label = f"▶ {title}"

            else:

                button_label = f"  {title}"

            if st.button(
                button_label,
                key=f"chat_{chat_id}",
                use_container_width=True
            ):

                st.session_state.current_chat_id = chat_id

                st.session_state.vector_store = (
                    load_existing_vector_store(
                        chat_id
                    )
                )

                st.rerun()

    else:

        st.caption("No conversations yet.")

    st.divider()

    # --------------------------------------------------------
    # Current Chat Actions
    # --------------------------------------------------------

    if st.button(
        "🗑️ Delete Current Chat",
        use_container_width=True
    ):

        current_chat_id = (
            st.session_state.current_chat_id
        )

        delete_chat(
            current_chat_id
        )

        remaining_chats = get_all_chats()

        if remaining_chats:

            st.session_state.current_chat_id = (
                remaining_chats[0][0]
            )

        else:

            st.session_state.current_chat_id = (
                create_chat()
            )

        st.session_state.vector_store = None

        st.rerun()


# ============================================================
# CURRENT CHAT
# ============================================================

current_chat = get_chat(
    st.session_state.current_chat_id
)

if current_chat is None:

    st.session_state.current_chat_id = create_chat()

    current_chat = get_chat(
        st.session_state.current_chat_id
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    f"📚 {current_chat['title']}"
)

st.caption(
    "Chat with one or more uploaded PDFs"
)


# ============================================================
# CURRENT CHAT DOCUMENTS
# ============================================================

current_documents = get_documents(
    st.session_state.current_chat_id
)

if current_documents:

    st.markdown("### 📄 Documents in this chat")

    document_names = [
        document["filename"]
        for document in current_documents
    ]

    st.write(
        " • ".join(document_names)
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

messages = get_messages(
    st.session_state.current_chat_id
)

for message in messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT + PDF ATTACHMENTS
# ============================================================

# Streamlit returns a ChatInputValue when accept_file is enabled.
# It contains .text for the question and .files for attachments.
# "multiple" allows several PDFs to be attached in one submission.
chat_submission = st.chat_input(
    "Ask something about your PDFs...",
    accept_file="multiple",
    file_type=["pdf"]
)


if chat_submission:

    current_chat_id = (
        st.session_state.current_chat_id
    )

    user_query = chat_submission.text.strip()
    uploaded_files = chat_submission.files

    # --------------------------------------------------------
    # PROCESS NEW PDF ATTACHMENTS FIRST
    # --------------------------------------------------------

    # Every attachment is processed into this chat's existing
    # ChromaDB collection. Therefore a later upload is ADDED to
    # the PDFs already stored for this chat; it does not replace
    # the previous documents.
    for uploaded_file in uploaded_files:

        file_bytes = uploaded_file.getvalue()

        file_hash = calculate_file_hash(
            file_bytes
        )

        # Ignore the same PDF only when its MD5 already exists
        # in this chat. The same PDF may be uploaded in another
        # chat because each chat has its own document context.
        if document_exists(
            current_chat_id,
            file_hash
        ):

            st.info(
                f"Already added to this chat: "
                f"{uploaded_file.name}"
            )

            continue

        with st.spinner(
            f"Adding {uploaded_file.name}..."
        ):

            (
                vector_store,
                added,
                processing_time
            ) = process_pdf(
                current_chat_id,
                file_bytes,
                uploaded_file.name
            )

        # process_pdf adds the chunks to the chat-specific
        # persistent ChromaDB collection immediately.
        st.session_state.vector_store = vector_store

        if added:

            st.success(
                f"Added {uploaded_file.name} "
                f"in {processing_time:.2f}s"
            )

    # --------------------------------------------------------
    # REFRESH DOCUMENT LIST
    # --------------------------------------------------------

    current_documents = get_documents(
        current_chat_id
    )

    # --------------------------------------------------------
    # CREATE/UPDATE TITLE FROM FIRST PDF
    # --------------------------------------------------------

    current_chat = get_chat(
        current_chat_id
    )

    if (
        current_chat["title"] == "New Chat"
        and current_documents
    ):

        first_name = current_documents[0]["filename"]

        title = os.path.splitext(
            first_name
        )[0][:40]

        update_chat_title(
            current_chat_id,
            title
        )

    write_structured_log(
        "chat_submission",
        chat_id=current_chat_id,
        query=user_query,
        attachment_count=len(uploaded_files),
        attachments=[
            {
                "filename": uploaded_file.name,
                "size_bytes": uploaded_file.size
            }
            for uploaded_file in uploaded_files
        ],
        log_file=format_log_location()
    )

    # --------------------------------------------------------
    # FILE-ONLY SUBMISSION
    # --------------------------------------------------------

    # A PDF can be submitted without a question. It is stored
    # in the current chat, and the user can ask about it later.
    if not user_query:

        st.rerun()

    # --------------------------------------------------------
    # LOAD CHAT'S VECTOR STORE IF NECESSARY
    # --------------------------------------------------------

    vector_store = (
        st.session_state.vector_store
    )

    if vector_store is None:

        vector_store = load_existing_vector_store(
            current_chat_id
        )

        st.session_state.vector_store = vector_store

    if vector_store is None:

        st.warning(
            "Please upload at least one PDF "
            "before asking a question."
        )

        st.stop()

    # --------------------------------------------------------
    # GET PREVIOUS MESSAGES
    # --------------------------------------------------------

    previous_messages = get_messages(
        current_chat_id
    )

    # --------------------------------------------------------
    # DISPLAY + SAVE USER QUESTION
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(user_query)

        if uploaded_files:

            st.caption(
                "📎 Attached: "
                + ", ".join(
                    uploaded_file.name
                    for uploaded_file in uploaded_files
                )
            )

    save_message(
        current_chat_id,
        "user",
        user_query
    )

    # --------------------------------------------------------
    # GENERATE ANSWER FROM THE COMPLETE CHAT CONTEXT
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            (
                answer,
                retrieval_time,
                rerank_time,
                total_time,
                matching_docs,
                evidence_status,
                conflicts,
                stage_timings,
                llama_metrics,
                request_id
            ) = generate_response(
                vector_store,
                user_query,
                previous_messages,
                current_chat_id
            )

        st.markdown(answer)

        # ----------------------------------------------------
        # TIMING
        # ----------------------------------------------------

        st.caption(
            f"⏱️ {total_time:.2f}s · "
            f"Chroma: {retrieval_time:.2f}s · "
            f"Jina: {rerank_time:.2f}s · "
            f"Llama: {stage_timings.get('llama_generation_s', 0):.2f}s"
        )

        prompt_tokens = llama_metrics.get("prompt_tokens")
        completion_tokens = llama_metrics.get("completion_tokens")
        total_tokens = llama_metrics.get("total_tokens")

        st.caption(
            f"🔢 Tokens — Prompt: {prompt_tokens if prompt_tokens is not None else 'N/A'} · "
            f"Completion: {completion_tokens if completion_tokens is not None else 'N/A'} · "
            f"Total: {total_tokens if total_tokens is not None else 'N/A'}"
        )

        st.caption(
            f"Evidence status: **{evidence_status}**"
        )

        with st.expander("📊 Stage-level latency & token details"):
            for stage_name, stage_time in stage_timings.items():
                st.write(
                    f"**{stage_name}**: {stage_time:.4f}s"
                    if isinstance(stage_time, (int, float))
                    else f"**{stage_name}**: {stage_time}"
                )

            st.write(
                f"**Request ID:** `{request_id}`"
            )
            st.write(
                f"**Log file:** `{format_log_location()}`"
            )
            st.json({
                "llama": llama_metrics,
                "jina": {
                    "model": JINA_RERANKER_MODEL,
                    "candidate_count": RETRIEVAL_K,
                    "selected_count": len(matching_docs)
                }
            })

        if conflicts:
            for conflict in conflicts:
                st.warning(f"⚠️ {conflict}")

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------
        if matching_docs:
            with st.expander("📎 Sources used"):
                for source_index, doc in enumerate(
                    matching_docs,
                    start=1
                ):
                    source_file = doc.metadata.get(
                        "source_file",
                        "Unknown PDF"
                    )
                    page = doc.metadata.get("page")
                    chunk_id = doc.metadata.get(
                        "chunk_id",
                        "unknown"
                    )
                    jina_score = doc.metadata.get(
                        "jina_score",
                        0.0
                    )

                    if page is not None:
                        source = (
                            f"[S{source_index}] "
                            f"{source_file} — Page {page + 1}"
                        )
                    else:
                        source = (
                            f"[S{source_index}] {source_file}"
                        )

                    st.write(source)
                    st.caption(
                        f"Jina relevance: {jina_score:.4f} · "
                        f"Chunk ID: {chunk_id}"
                    )

    # --------------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # --------------------------------------------------------

    save_message(
        current_chat_id,
        "assistant",
        answer
    )

    # --------------------------------------------------------
    # AUTOMATICALLY UPDATE TITLE FROM FIRST QUESTION
    # --------------------------------------------------------

    current_chat = get_chat(
        current_chat_id
    )

    # If the chat title was generated from the first PDF, replace
    # it with the first question after the first real conversation.
    pdf_title = ""

    if current_documents:

        pdf_title = os.path.splitext(
            current_documents[0]["filename"]
        )[0][:40]

    if current_chat["title"] in ("New Chat", pdf_title):

        title = user_query.strip()

        if len(title) > 40:

            title = title[:40] + "..."

        update_chat_title(
            current_chat_id,
            title
        )

    st.rerun()


# ============================================================
# EMPTY STATE
# ============================================================

if (
    not messages
    and not current_documents
):

    st.info(
        "Use the 📎 attachment button inside the chat input to add one or more PDFs, then type your question and press Enter."
    )