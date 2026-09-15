# IMR RAG

A retrieval-augmented question-answering system over **42,241 California Department of Managed Health Care (DMHC) Independent Medical Review (IMR) determinations**. Ask a natural-language question like *"Why are mental health admissions often denied?"* and get a grounded, cited answer drawn from real insurance dispute records.

Built as a portfolio project to practice RAG fundamentals, vector search, and modern Python deployment patterns — with an emphasis on **grounding and abstention**, informed by prior research on LLM hallucination.

## What it does

Given a question, the system:

1. Embeds the question into a vector
2. Retrieves the most semantically similar IMR records from a vector database (with optional metadata filtering, e.g. by diagnosis category)
3. Passes the retrieved records to an LLM with a prompt designed to answer **only** from the provided evidence
4. Returns a concise answer that cites each claim by the record's reference ID — or abstains when the question falls outside the dataset

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.12 |
| API | FastAPI |
| Vector store | Qdrant |
| Embeddings | sentence-transformers (`BAAI/bge-small-en-v1.5`, 384-dim) |
| LLM | `openai/gpt-oss-20b` via Groq |
| Containerization | Docker + Docker Compose |

## Quickstart

**Prerequisites:** Docker Desktop, and a free [Groq API key](https://console.groq.com).

```bash
# 1. Clone and enter the project
git clone https://github.com/AnoshaRehan/insurance-claim-denial-rag-imr-explorer.git
cd insurance-claim-denial-rag-imr-explorer

# 2. Create your environment file and add your Groq key
cp .env.example .env
# then edit .env and set GROQ_API_KEY=...

# 3. Start the stack (API + Qdrant)
docker compose up --build

# 4. Build the search index (first run only; ~15-20 min to embed 42k records)
#    In a second terminal, with a local venv:
pip install -e ".[dev]"
python -m scripts.build_index --recreate
```

Once the index is built, the API is live at `http://localhost:8000`. Open `http://localhost:8000/docs` for interactive documentation.

### Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Why are mental health admissions often denied?"}'
```

### Command-line interface

You can also query without the API:

```bash
python -m scripts.ask "What are common reasons cancer treatment is denied?"
python -m scripts.ask "How are eating disorder appeals decided?" --filter-diagnosis "Mental Disorder"
```

## Architecture

```
                       ┌──────────────────────────────┐
   Question ──────────▶│  FastAPI  (/ask, /search)     │
                       └──────────────┬───────────────┘
                                      │
                        embed question │
                                      ▼
                       ┌──────────────────────────────┐
                       │  sentence-transformers        │
                       │  (BAAI/bge-small-en-v1.5)     │
                       └──────────────┬───────────────┘
                                      │ query vector
                                      ▼
                       ┌──────────────────────────────┐
                       │  Qdrant  (42,241 vectors)     │
                       │  semantic search + metadata   │
                       │  filtering                    │
                       └──────────────┬───────────────┘
                                      │ top-k records
                                      ▼
                       ┌──────────────────────────────┐
                       │  Prompt builder               │
                       │  (grounding + citation rules) │
                       └──────────────┬───────────────┘
                                      │ prompt
                                      ▼
                       ┌──────────────────────────────┐
                       │  LLM via Groq (gpt-oss-20b)   │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                        Cited answer + source records
```

### Project layout

```
app/
├── config.py            # Settings via environment variables
├── main.py              # FastAPI entrypoint
├── data/
│   └── loader.py        # Cache-on-first-use dataset loader
├── rag/
│   ├── embedder.py      # sentence-transformers wrapper
│   ├── vectorstore.py   # Qdrant client wrapper
│   ├── prompt.py        # Prompt template (grounding + citations)
│   ├── generator.py     # LLM client (Groq)
│   └── pipeline.py      # End-to-end: retrieve -> prompt -> generate
└── api/
    ├── routes.py        # /ask, /search, /health
    └── schemas.py       # Pydantic request/response models

scripts/
├── build_index.py       # Embed records and load into Qdrant
├── ask.py               # CLI to query the RAG system
├── test_search.py       # CLI for raw retrieval (no LLM)
└── inspect_data.py      # Explore the dataset
```

## How it works

**Retrieval.** Each IMR record's structured fields (diagnosis, treatment, determination) and free-text findings are combined and embedded into a 384-dimensional vector, stored in Qdrant. At query time, the question is embedded the same way, and Qdrant returns the nearest records by cosine similarity. Because Qdrant supports metadata filtering, queries can be scoped — e.g. *only* records where `diagnosis_category = "Mental Disorder"* — before ranking.

**Generation.** Retrieved records are formatted into a prompt that instructs the model to answer strictly from the provided evidence, cite each claim by reference ID, and state plainly when the records don't support an answer. The model runs on Groq for fast inference.

**Abstention.** Retrieval always returns the nearest vectors, even for off-topic questions. A minimum-similarity threshold flags low-confidence results, and the prompt design leads the model to decline rather than fabricate when the retrieved records don't actually address the question.

## Design decisions

**Cache-on-first-use data loader.** The DMHC dataset is fetched from the state's open-data portal (which redirects to S3) on first use, cached locally, and reused thereafter — with a `--refresh` flag to force re-download. Anyone who clones the repo is up and running without manual data setup.

**CPU-only PyTorch.** The default `pip install torch` pulls a CUDA (GPU) build that bloated the Docker image to ~9.7 GB. Since the container runs CPU-only inference, switching to the CPU-only torch build cut the image to ~2.2 GB (a 78% reduction).

**Multi-stage Docker build.** Dependencies are installed in a builder stage; the runtime image copies only the finished environment, leaving build tools behind for a leaner final image.

**Provider-agnostic LLM client.** The generator is isolated behind a small interface, so the inference backend is a configuration change rather than a code change. The system currently uses Groq; swapping to OpenAI or a local Ollama setup is a drop-in.

**Configuration via environment variables.** All settings load from the environment (with a local `.env` for development), so the same image runs unchanged in local Docker and in cloud deployment — only the injected variables differ.

## Data

The dataset is published by the California DMHC on the [CHHS Open Data Portal](https://data.chhs.ca.gov/dataset/independent-medical-review-imr-determinations-trend). Each record documents an independent review of an insurance coverage denial, including the diagnosis, requested treatment, the health plan's original decision, the reviewer's findings, and the final determination (upheld or overturned). It is public and contains no personally identifying information.

## Project status

**Done:**
- [x] Data ingestion with cache-on-first-use loader (handles S3 redirects)
- [x] Semantic search over 42k records with Qdrant + metadata filtering
- [x] RAG pipeline with grounding, citations, and abstention
- [x] FastAPI service (`/ask`, `/search`, `/health`) with auto-generated docs
- [x] Full containerization with Docker Compose (CPU-only torch, ~2.2 GB image)

**Planned:**
- [ ] CI/CD with GitHub Actions (lint + test on every push)
- [ ] Cloud deployment with a public demo URL
- [ ] Evaluation suite measuring answer faithfulness and relevancy
- [ ] Calibrate the retrieval confidence threshold on a labeled query set

## License

MIT