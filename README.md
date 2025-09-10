# Local RAG System with LLM Integration

A Retrieval-Augmented Generation (RAG) system that combines document ingestion, vector search, and Large Language Model capabilities for intelligent question-answering from local documents.

## 🌟 Features

- **Dual LLM Support**: Local Ollama models and remote API integration (Kimi/Moonshot)
- **Smart Document Processing**: PDF to text conversion with PyMuPDF
- **Adaptive Chunking**: Automatic text segmentation based on document size
- **Vector Search**: Semantic search using sentence transformers and SQLite-Vec
- **Web Interface**: Modern dark-themed frontend with file upload
- **Performance Optimization**: WAL mode, RAM optimization, and comprehensive benchmarking
- **Query History**: Persistent storage and management of Q&A sessions
- **Document Summarization**: AI-powered book/document summaries

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   Local LLM     │
│   (HTML/JS)     │◄──►│   (Flask)       │◄──►│   (Ollama)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                        ┌─────────────────┐
                        │   SQLite + Vec  │
                        │   (Embeddings)  │
                        └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Ollama (for local LLM)
- SQLite with vector extension

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-system
cd rag-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start Ollama service:
```bash
ollama serve
ollama pull llama3.1:8b
```

4. Run the application:
```bash
# Backend
python backend.py

# CLI version
python test2.py
```

5. Access web interface at `http://localhost:5000`

## 📁 Project Structure

```
WORKINGRAG/
├── backend.py          # Flask web server
├── frontend.html       # Web interface
├── test2.py           # Main CLI application (Local Ollama)
├── data/              # Document storage
├── my_docs.db         # Vector database
├── queries.db         # Query history
└── log.txt           # Performance benchmarks
```

## 🔧 Configuration

Key configuration options in `test2.py`:

```python
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1:8b"
TOP_K = 3  # Number of retrieved chunks
USE_RAM = bool(os.getenv("RAM", ""))  # In-memory database
```

## 📊 Performance Features

- **Benchmark Logging**: Detailed timing for each operation
- **Database Optimization**: WAL mode and memory storage options
- **Adaptive Chunking**: Dynamic chunk sizing based on document length
- **Efficient Retrieval**: Vector similarity search with configurable TOP_K

## 🎯 Usage Examples

### CLI Interface
```bash
python test2.py
> upload  # Select PDF/TXT file via dialog
> What is the main topic of this document?
> summarize filename.pdf
> history  # View past queries
```

### Web Interface
1. Upload documents via the web interface
2. Ask questions in natural language
3. View query history in the sidebar
4. Generate document summaries

## 🛠️ Technical Implementation

### Vector Embeddings
- **Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Storage**: SQLite with vec0 extension
- **Similarity**: Cosine similarity search

### Document Processing
- **PDF Support**: PyMuPDF for text extraction
- **Text Chunking**: Paragraph-aware adaptive chunking
- **Encoding**: UTF-8 with error handling

### LLM Integration
- **Local**: Ollama with Llama 3.1 8B
- **Remote**: Kimi/Moonshot API support
- **Fallback**: Graceful error handling

## 📈 Performance Metrics

The system logs detailed performance metrics to `log.txt`:
- Document ingestion time
- Embedding generation time
- Vector search latency
- LLM response time

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Sentence Transformers for embedding models
- SQLite-Vec for vector search capabilities
- Ollama for local LLM hosting
- PyMuPDF for PDF processing

---

Built during internship at Aptum - July 2025

