from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os, sys, datetime, shutil, sqlite3
from pathlib import Path

# Import the helpers you already wrote
sys.path.insert(0, str(Path(__file__).parent))
from test import ask, upload_file, summarize

UPLOAD_FOLDER = Path("data")
UPLOAD_FOLDER.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def index():
    return send_from_directory(".", "frontend.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"status": "error", "message": "no file"}), 400

    filename = secure_filename(file.filename)
    dst = UPLOAD_FOLDER / filename
    file.save(dst)                       # 1. save original

    try:
        upload_file(str(dst))            # 2. PDF→Markdown→TXT & ingest
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask_endpoint():
    data = request.get_json()
    answer = ask(data["question"])
    return jsonify({"answer": answer})

@app.route("/history")
def history():
    conn = sqlite3.connect("queries.db")
    rows = conn.execute(
        "SELECT id, datetime(timestamp,'localtime'), question FROM queries ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return jsonify(rows)

# NEW: delete a history row
@app.route("/history/<int:row_id>", methods=["DELETE"])
def delete_history(row_id):
    conn = sqlite3.connect("queries.db")
    conn.execute("DELETE FROM queries WHERE id=?", (row_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok"})

@app.route("/summarize/<path:book>")
def summarize_book(book):
    summarize(book)
    conn = sqlite3.connect("queries.db")
    summary = conn.execute("SELECT summary FROM summaries WHERE file=?", (book,)).fetchone()[0]
    conn.close()
    return jsonify({"summary": summary})

if __name__ == "__main__":
    app.run(debug=True)