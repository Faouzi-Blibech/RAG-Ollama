
import os
import sqlite3
import glob
import shutil
import re
import time
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
from pathlib import Path
import requests          # NEW
import sqlite_vec
from sqlite_vec import serialize_float32
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF

# ---------- CONFIG ----------
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1:8b"

DB_DOCS    = "my_docs.db"
DB_QUERIES = "queries.db"
DATA_DIR   = "data"
TOP_K      = 3
os.makedirs(DATA_DIR, exist_ok=True)

USE_RAM = bool(os.getenv("RAM", ""))
emb_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- OLLAMA HELPER ----------
def chat_ollama(messages: list[dict]) -> str:
    """Send messages to the local Ollama server and return the assistant reply."""
    payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": False}
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]

# ---------- PERFORMANCE HELPERS ----------
def tweak_sqlite(conn: sqlite3.Connection, ram=False):
    if ram:
        conn.execute("PRAGMA temp_store=memory")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

# ---------- BENCHMARK LOGGER ----------
def log_time(task: str, t0: float):
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()} | {task} | {time.time()-t0:.3f}s\n")

# ---------- EMBEDDING ----------
def get_embedding(text: str) -> list[float]:
    return emb_model.encode(text, normalize_embeddings=True).tolist()

# ---------- DATABASE ----------
def init_docs_db(ram=False):
    conn = sqlite3.connect(":memory:" if ram else DB_DOCS)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS docs
        USING vec0(embedding float[384], file TEXT, content TEXT);
    """)
    tweak_sqlite(conn, ram)
    conn.commit()
    return conn

def init_queries_db(ram=False):
    conn = sqlite3.connect(":memory:" if ram else DB_QUERIES)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS queries(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            question TEXT,
            answer TEXT
        );
        CREATE TABLE IF NOT EXISTS summaries(
            file TEXT PRIMARY KEY,
            summary TEXT
        );
    """)
    tweak_sqlite(conn, ram)
    conn.commit()
    return conn

# ---------- AUTO-ADAPTIVE CHUNKING ----------
def chunk(text: str, max_words: int = None):
    total_words = len(text.split())
    if max_words is None:
        max_words = 1000 if total_words > 1000 else total_words + 1
    paragraphs = re.split(r"\n{2,}", text.strip())
    buf, buf_len = [], 0
    for p in paragraphs:
        p_w = len(p.split())
        if buf_len + p_w > max_words and buf:
            yield "\n\n".join(buf).strip()
            buf, buf_len = [p], p_w
        else:
            buf.append(p)
            buf_len += p_w
    if buf:
        yield "\n\n".join(buf).strip()

# ---------- INGESTION ----------
def ingest_file(path: str):
    t0 = time.time()
    conn = init_docs_db(ram=USE_RAM)
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    log_time("load_text", t0)

    for part in chunk(text):
        emb = get_embedding(part)
        conn.execute(
            "INSERT INTO docs(embedding, file, content) VALUES (?,?,?)",
            (serialize_float32(emb), os.path.basename(path), part),
        )
    conn.commit()
    log_time("insert_to_db", t0)
    conn.close()
    print(f"✅ Ingested {os.path.basename(path)}")

def ingest_all():
    for path in glob.glob(os.path.join(DATA_DIR, "*")):
        if os.path.isfile(path):
            ingest_file(path)

# ---------- RETRIEVAL ----------
def ask(question: str, save=True):
    t0 = time.time()
    conn = init_docs_db(ram=USE_RAM)
    q_emb = serialize_float32(get_embedding(question))
    log_time("embedding_q", t0)

    rows = conn.execute("""
        SELECT file, content, distance
        FROM docs
        WHERE embedding MATCH ? AND k = ?
        ORDER BY distance;
    """, (q_emb, TOP_K)).fetchall()
    log_time("retrieve", t0)
    conn.close()

    context = "\n\n".join(r[1] for r in rows)
    prompt = (
        "Answer the question using only the context below. "
        "Otherwise say I don't know!\n\n"
        f"{context}\n\nQ: {question}\nA:"
    )
    answer = chat_ollama([{"role": "user", "content": prompt}])
    log_time("llm_answer", t0)
    if save:
        save_query(question, answer)
    try:
        answer = chat_ollama([{"role": "user", "content": prompt}])
    except Exception as e:
        print(" LLM error:", e)
        answer = "  Local model unavailable."

    print(answer)
    if save:
        save_query(question, answer)
    return answer

def save_query(q: str, a: str):
    conn = init_queries_db(ram=USE_RAM)
    conn.execute(
        "INSERT INTO queries(timestamp, question, answer) VALUES (?,?,?)",
        (datetime.utcnow().isoformat(), q, a)
    )
    conn.commit()
    conn.close()

# ---------- SUMMARIZER ----------
def summarize(book_file: str):
    t0 = time.time()
    conn = init_queries_db(ram=USE_RAM)
    row = conn.execute("SELECT summary FROM summaries WHERE file=?", (book_file,)).fetchone()
    if row:
        print("📘 Cached summary:\n", row[0])
        return

    conn_docs = init_docs_db(ram=USE_RAM)
    rows = conn_docs.execute(
        "SELECT content FROM docs WHERE file=? ORDER BY rowid", (book_file,)
    ).fetchall()
    full_text = "\n".join(r[0] for r in rows)[:6000]
    prompt = f"Summarize the following book in ~200 words:\n\n{full_text}"
    summary = chat_ollama([{"role": "user", "content": prompt}])
    conn.execute(
        "INSERT OR REPLACE INTO summaries(file, summary) VALUES (?,?)",
        (book_file, summary)
    )
    conn.commit()
    conn.close()
    log_time("summarize", t0)
    print("📘 Summary:\n", summary)

# ---------- PDF → TXT ----------
def pdf_to_txt(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text

# ---------- UPLOAD ----------
def upload_file():
    root = tk.Tk()
    root.withdraw()
    root.update()
    path = filedialog.askopenfilename(
        title="Select book",
        filetypes=[("Books", "*.pdf *.txt"), ("All files", "*.*")]
    )
    root.destroy()

    if not path:
        print("❌ No file selected")
        return

    src_path = Path(path).expanduser().resolve()
    ext = src_path.suffix.lower()

    if ext == ".pdf":
        t0 = time.time()
        print("📄 Converting PDF → Markdown …")
        text = pdf_to_txt(str(src_path))
        elapsed = time.time() - t0
        print(f"⏱️  PDF→Markdown: {elapsed:.3f}s")
        log_time("pdf_to_markdown", t0)

        dst_path = Path(DATA_DIR) / f"{src_path.stem}.txt"
        dst_path.write_text(text, encoding="utf-8")
    elif ext == ".txt":
        dst_path = Path(DATA_DIR) / src_path.name
        shutil.copy2(src_path, dst_path)
    else:
        print("❌ Only .pdf or .txt allowed")
        return

    t1 = time.time()
    ingest_file(str(dst_path))
    elapsed_i = time.time() - t1
    print(f"⏱️  Total upload+ingest: {elapsed_i:.3f}s")

# ---------- HISTORY ----------
def show_history():
    conn = init_queries_db(ram=USE_RAM)
    rows = conn.execute(
        "SELECT id, timestamp, question FROM queries ORDER BY id DESC LIMIT 20"
    ).fetchall()
    conn.close()
    for rid, ts, q in rows:
        print(f"{rid:>3} | {ts[:16]} | {q}")

def replay_all():
    conn = init_queries_db(ram=USE_RAM)
    rows = conn.execute(
        "SELECT timestamp, question, answer FROM queries ORDER BY id"
    ).fetchall()
    conn.close()
    if not rows:
        print("📭 No saved questions.")
        return
    for ts, q, a in rows:
        print("-" * 60)
        print(f"🕒 {ts}")
        print(f"Q: {q}")
        print(f"A: {a}")
    print("-" * 60)

# ---------- CLI ----------
if __name__ == "__main__":
    if not os.path.exists(DB_DOCS) and not USE_RAM:
        print("First run: ingesting documents...")
        ingest_all()

    conn = init_docs_db(ram=USE_RAM)
    books = sorted(set(r[0] for r in conn.execute("SELECT DISTINCT file FROM docs").fetchall()))
    conn.close()
    print("\n📚 Available books:")
    for b in books:
        print(" •", b)
    print("-" * 40)

    while True:
        q = input(
            "\nAsk me anything, or:\n"
            "  upload         – pick a book via file dialog\n"
            "  summarize <book>\n"
            "  history        – list past questions\n"
            "  replay         – show full history with answers\n"
            "  q              – quit\n> "
        ).strip()

        if q.lower() == "q":
            break
        elif q.lower() == "upload":
            upload_file()

        elif q.lower().startswith("summarize"):
            pattern = q[9:].strip()
            conn = init_docs_db(ram=USE_RAM)
            all_files = [r[0] for r in conn.execute("SELECT DISTINCT file FROM docs").fetchall()]
            conn.close()

            candidates = [f for f in all_files if f == pattern]
            if not candidates:
                candidates = [f for f in all_files if pattern.lower() in f.lower()]

            if len(candidates) == 1:
                summarize(candidates[0])
            elif len(candidates) > 1:
                print("Multiple matches:", ", ".join(candidates))
            else:
                print("No book matches:", pattern)

        elif q == "history":
            show_history()
        elif q == "replay":
            replay_all()
        else:
            ask(q)
