# 📚 AI Research Assistant

A Retrieval-Augmented Generation (RAG) system for querying and analyzing research documents. It ingests PDFs and other documents, indexes them with a hybrid (dense + sparse) retriever, reranks the results, and generates grounded, cited answers using an LLM — all through a Streamlit interface.

## Features

- **Document ingestion** — extracts text from PDFs (via PyMuPDF and pdfplumber) and HTML (via BeautifulSoup), with automatic encoding detection.
- **Chunking pipeline** — splits documents into overlapping, cleaned text chunks with page and section metadata.
- **Hybrid retrieval** — combines dense vector search (FAISS + sentence-transformer embeddings) with sparse BM25 keyword search, fused with Reciprocal Rank Fusion (RRF).
- **Cross-encoder reranking** — reorders candidates for relevance using a BGE reranker model.
- **Grounded generation** — builds a source-cited context window and generates answers via an LLM (Groq by default; LangChain/LangGraph included as dependencies).
- **Evaluation suite** — Recall@K, Precision@K, and related retrieval metrics over dev/test query sets.
- **Streamlit UI** — a simple web app for asking questions over your indexed documents.

## Project Structure

```
Ai-research-assistant/
├── config/
│   ├── config.yaml          # Main project configuration
│   └── logging.conf         # Logging configuration
├── notebooks/
│   └── exploration.ipynb    # Exploratory analysis
├── scripts/
│   ├── build_index.py       # Build the document index
│   ├── evaluate.py          # Run retrieval evaluation
│   └── run_demo.py          # Run a quick demo query
├── src/
│   ├── data/                # Loading, cleaning, chunking, validation
│   ├── retrieval/           # Dense, sparse, hybrid retrievers + reranker
│   ├── generation/          # LLM interface + context/citation builder
│   ├── evaluation/          # Metrics and evaluator
│   └── ui/                  # Streamlit application
├── streamlit/
│   └── app.py                # Streamlit entry point
├── packages.txt              # System-level dependencies
├── requirements.txt           # Python dependencies
└── setup.py
```

## Getting Started

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com/) (or another supported LLM provider)

### Installation

```bash
git clone https://github.com/Tauhid-Topu-007/Ai-research-assistant.git
cd Ai-research-assistant
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root with your API key:

```
GROQ_API_KEY=your_api_key_here
```

Adjust chunking, retrieval, and model settings in `config/config.yaml` as needed.

### Usage

1. **Add documents** — place PDFs/documents in `data/raw/`.
2. **Build the index**:
   ```bash
   python scripts/build_index.py
   ```
3. **Run the app**:
   ```bash
   streamlit run streamlit/app.py
   ```
4. **Evaluate retrieval quality** (optional):
   ```bash
   python scripts/evaluate.py
   ```

## How It Works

1. Documents are loaded and split into overlapping chunks (`src/data`).
2. Chunks are embedded and indexed in FAISS, and separately indexed with BM25 (`src/retrieval`).
3. At query time, dense and sparse results are fused (RRF) and reranked with a cross-encoder.
4. The top chunks are assembled into a cited context and passed to the LLM for answer generation (`src/generation`).
5. Retrieval quality can be benchmarked against labeled query sets with Recall@K / Precision@K (`src/evaluation`).

## Tech Stack

| Component | Library |
|---|---|
| PDF parsing | PyMuPDF, pdfplumber |
| Embeddings | sentence-transformers (`BAAI/bge-base-en-v1.5`) |
| Vector search | FAISS |
| Keyword search | BM25 (custom implementation) |
| Reranking | `BAAI/bge-reranker-base` |
| LLM | Groq (`llama-3.1-8b-instant`) via LangChain/LangGraph |
| UI | Streamlit |

## Contributing

Issues and pull requests are welcome. Please open an issue to discuss significant changes before submitting a PR.

## License

No license has been specified for this repository yet.
