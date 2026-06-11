# RAG Playground

> Hands-on RAG learning playground — build every component of Retrieval-Augmented Generation from scratch in Python.
> No frameworks, no cloud APIs — just you, Python, and a local LLM.

## How It Works

```
                          RAG Pipeline
   ┌─────────┐    ┌─────────┐    ┌───────────┐    ┌──────────┐
   │  Load   │───▶│  Chunk  │───▶│  Embed    │───▶│  Store   │
   │Documents│    │  Text   │    │  Vectors  │    │ Vectors  │
   └─────────┘    └─────────┘    └───────────┘    └─────┬────┘
                                                        │
   ┌─────────┐    ┌──────────┐    ┌───────────┐         │
   │ Answer  │◀───│ Generate │◀───│ Augment   │◀────────┘
   │         │    │  (LLM)   │    │ Prompt    │    Retrieve
   └─────────┘    └──────────┘    └───────────┘    Top-K Chunks
                                                        ▲
                                              ┌─────────┴─────────┐
                                              │  User Query       │
                                              │  "What is RAG?"   │
                                              └───────────────────┘
```

## Quick Start

**Prerequisites:** Python 3.11+ and [Ollama](https://ollama.com)

```bash
# 1. Install Ollama
#    macOS:   brew install ollama
#    Windows: winget install Ollama.Ollama
#    Linux:   curl -fsSL https://ollama.com/install.sh | sh

# 2. Start Ollama (keep this running)
ollama serve

# 3. Clone, install, and pull models
git clone https://github.com/DanWangDev/rag-playground.git
cd rag-playground
pip install -e ".[dev]"
python scripts/pull_models.py
python scripts/setup.py

# 4. Run your first exercise
python -m rag_playground.m02_data_loading.exercise
```

## Learning Path (8 Modules)

Each module is a self-contained lesson with concept docs, source code, an interactive exercise, and tests.

```
 Module 02 ──▶ Module 03 ──▶ Module 04 ──▶ Module 05 ──▶ Module 06 ──▶ Module 07 ──▶ Module 08
   Load         Chunk         Embed         Store        Retrieve       Pipeline      Evaluate
                                                              │
 Module 01 ──────────────────────────────────────────────────┘
   LLM Basics (runs in parallel)
```

| # | Module | What You Build | Key Concepts |
|---|--------|---------------|-------------|
| 01 | **LLM Basics** | Chat + streaming + prompt engineering | Tokens, temperature, system prompts, chain-of-thought |
| 02 | **Data Loading** | File loaders (txt, md, directories) | Document abstraction, metadata, encoding |
| 03 | **Chunking** | 3 splitting strategies + overlap | Character, recursive, token-aware splitting |
| 04 | **Embeddings** | Vector generation + similarity | Cosine similarity, nearest neighbors, nomic-embed-text |
| 05 | **Vector Store** | In-memory k-NN search engine | Brute-force search, threshold, metadata filter, persistence |
| 06 | **Retrieval** | Advanced retrieval strategies | Query expansion, multi-query fusion, LLM reranking |
| 07 | **RAG Pipeline** | Complete end-to-end RAG | Retrieve → augment → generate, streaming, citations |
| 08 | **Evaluation** | Quality metrics | Precision@K, Recall@K, MRR, NDCG, faithfulness |

### Each module includes:
- **`concept.md`** — thorough explanation with ASCII diagrams, tables, and gotchas
- **Source files** — well-documented Python with "Key Learning Points" annotations
- **`exercise.py`** — interactive CLI walkthrough (step-by-step)
- **Tests** — unit tests with mocked providers (run offline, no Ollama needed)

## Architecture

```
                    ┌──────────────────────────────┐
                    │        .env  (config)        │
                    │  CHAT_MODEL=qwen2.5:7b       │
                    │  EMBED_MODEL=nomic-embed-text│
                    └─────────────┬────────────────┘
                                  │
                    ┌─────────────▼────────────────┐
                    │      ModelProvider (ABC)      │
                    │  chat()  chat_stream()        │
                    │  embed()  health_check()      │
                    └─────────────┬────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
    ┌─────────▼─────────┐ ┌──────▼──────┐ ┌─────────▼─────────┐
    │  OllamaProvider   │ │  (future)   │ │  (future)         │
    │  local, free      │ │  OpenAI     │ │  vLLM             │
    └───────────────────┘ └─────────────┘ └───────────────────┘

    Zero dependencies on LangChain, LlamaIndex, or any RAG framework.
    Every component is built from scratch — you understand every line.
```

## Model Configuration

Default model: **Qwen 2.5 7B** via Ollama. Switch by changing one line in `.env`:

```bash
CHAT_MODEL=qwen2.5:7b    # default (7B params, ~5GB RAM)
CHAT_MODEL=llama3.2       # lighter, faster
CHAT_MODEL=qwen2.5:14b    # more capable, needs ~10GB RAM
CHAT_MODEL=mistral:7b      # alternative 7B

EMBED_MODEL=nomic-embed-text       # 768 dimensions, ~274MB
EMBED_MODEL=mxbai-embed-large      # 1024 dimensions, ~669MB
```

Future: add an OpenAI-compatible provider by implementing the `ModelProvider` ABC and setting `CHAT_PROVIDER=openai`.

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Language | Python 3.11+ | Native RAG/ML ecosystem |
| LLM Runtime | Ollama | One-command install, local, free |
| HTTP | httpx | Modern async HTTP client |
| Validation | Pydantic | Runtime type checking |
| CLI | Rich | Beautiful terminal output |
| Server | FastAPI | Async, auto OpenAPI docs, built-in SSE |
| Frontend | React + Vite | Interactive RAG testing UI |
| Tests | pytest | Fixtures, async, parametrize |
| Lint | Ruff | Fast Python linter + formatter |

## Development

```bash
pytest                          # Run all tests (no Ollama needed)
pytest --cov=rag_playground     # With coverage
ruff check src/ tests/          # Lint
ruff format src/ tests/         # Format
```

## License

MIT
