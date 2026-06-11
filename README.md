# RAG Playground

Hands-on RAG (Retrieval-Augmented Generation) learning playground — build every component from scratch in Python.

**Inspired by [rag-from-scratch](https://github.com/pguso/rag-from-scratch)** but built as an original Python project following the progressive, exercise-driven format of [dynamodb-playground](https://github.com/danwa/dynamodb-playground).

## Quick Start

### Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com)** — local LLM runtime

### Setup

```bash
# 1. Install Ollama (if not already)
# macOS:   brew install ollama
# Windows: winget install Ollama.Ollama
# Linux:   curl -fsSL https://ollama.com/install.sh | sh

# 2. Start Ollama
ollama serve

# 3. Clone and install
git clone https://github.com/danwa/rag-playground.git
cd rag-playground
pip install -e ".[dev]"

# 4. Pull models and verify setup
python scripts/check_ollama.py
python scripts/pull_models.py
python scripts/setup.py
```

### Run an Exercise

```bash
python -m rag_playground.m02_data_loading.exercise
python -m rag_playground.m03_chunking.exercise
python -m rag_playground.m04_embeddings.exercise
# ... etc.
```

### Run the Web App

```bash
# Terminal 1: Backend
fastapi dev server/main.py

# Terminal 2: Frontend
cd frontend && npm install && npm run dev
```

Open `http://localhost:5173` to interact with the RAG pipeline through a UI.

## Learning Path

| # | Module | What You'll Learn |
|---|--------|-------------------|
| 01 | **LLM Basics** | Local chat completions, streaming, prompt engineering |
| 02 | **Data Loading** | Document abstraction, file I/O, metadata extraction |
| 03 | **Chunking** | Splitting strategies, overlap, context windows |
| 04 | **Embeddings** | Vector representations, cosine similarity |
| 05 | **Vector Store** | In-memory k-NN search, metadata filtering, persistence |
| 06 | **Retrieval** | Query expansion, multi-query fusion, reranking |
| 07 | **RAG Pipeline** | End-to-end: load→chunk→embed→store→retrieve→generate |
| 08 | **Evaluation** | Precision/Recall/MRR/NDCG, faithfulness scoring |

Each module contains:
- **`concept.md`** — detailed explanation with diagrams and gotchas
- **Source files** — well-documented Python implementations
- **`exercise.py`** — interactive CLI walkthrough
- **Tests** — unit tests with mocked providers

## Architecture

```
.env → ModelRegistry → Provider (ABC)
                          ├── OllamaProvider  (local, free)
                          └── ...             (extensible)

RAG Pipeline:
  Load → Chunk → Embed → Store → Retrieve → Augment → Generate
```

## Model Configuration

Default: **Qwen 2.5 7B** via Ollama. Switch models by changing `.env`:

```bash
# Use a different model
CHAT_MODEL=llama3.2
CHAT_MODEL=qwen2.5:14b
CHAT_MODEL=mistral:7b
```

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=rag_playground --cov-report=term

# Lint
ruff check src/ tests/
ruff format --check src/ tests/
```

## License

MIT
