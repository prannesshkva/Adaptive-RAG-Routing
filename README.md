# Adaptive RAG Routing

<p align="center">
  <strong>Intelligent Dynamic Routing Engine for Modular Retrieval-Augmented Generation</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge&logo=github&logoColor=white" alt="Status" />
  <img src="https://img.shields.io/badge/Python-3.10+-18181B?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Architecture-Modular_RAG-18181B?style=for-the-badge" alt="Architecture" />
  <img src="https://img.shields.io/badge/License-MIT-18181B?style=for-the-badge" alt="License" />
</p>

---

## Overview

**Adaptive-RAG-Routing** is a high-performance orchestration layer designed to dynamically route natural language queries to optimal retrieval and reasoning pipelines. Rather than applying a single uniform retrieval strategy to every user inquiry, this architecture performs query decomposition and complexity classification to select among specialized RAG modalities—including **Hybrid RAG (Dense + BM25 with Reciprocal Rank Fusion)**, **Corrective RAG (Self-Reflective CRAG)**, Context-Aware Generation (CAG), dense semantic search, Multi-Query expansion, and Maximal Marginal Relevance (MMR) re-ranking.

---

## Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-18181B?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-18181B?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Pydantic-18181B?style=for-the-badge&logo=pydantic&logoColor=white" alt="Pydantic" />
  <img src="https://img.shields.io/badge/FAISS-18181B?style=for-the-badge&logo=meta&logoColor=white" alt="FAISS" />
  <img src="https://img.shields.io/badge/Qdrant-18181B?style=for-the-badge&logo=database&logoColor=white" alt="Qdrant" />
  <img src="https://img.shields.io/badge/BM25-18181B?style=for-the-badge&logo=apache&logoColor=white" alt="BM25" />
  <img src="https://img.shields.io/badge/Sentence_Transformers-18181B?style=for-the-badge&logo=huggingface&logoColor=white" alt="Sentence Transformers" />
  <img src="https://img.shields.io/badge/Uvicorn-18181B?style=for-the-badge&logo=gunicorn&logoColor=white" alt="Uvicorn" />
</p>

---

## Key Routing Modalities

| Modality | Router Module | Description | Primary Use Case & Triggers |
| :--- | :--- | :--- | :--- |
| **Hybrid RAG (RRF)** | `app/api/routers/hybridrag.py` | Dense vector (FAISS) + Sparse BM25 lexical fusion via Reciprocal Rank Fusion ($k=60$) | Complex, procedural, architectural, or technical queries (≥6 words, technical terms) |
| **Corrective RAG (CRAG)** | `app/api/routers/crag.py` | Self-reflective retrieval with confidence scoring, noise filtering, and query reformulation | Verification, accuracy check, confirmation queries (`"verify"`, `"is it true"`, `"confirm"`) |
| **Standard Dense RAG** | `app/api/routers/rag.py` | Dense vector indexing with cosine similarity gating | Direct factual and conceptual inquiries (`"define"`, `"what is"`, `"why"`, `"explain"`) |
| **Sparse Lexical RAG** | `app/api/routers/vlrag.py` | Exact token BM25Okapi search with POS filtering and lemmatization | Exact identifiers, error codes, part numbers, short terms (`len < 4`, codes with `_`) |
| **Multi-Query RAG** | `app/api/routers/multirag.py` | Multi-perspective query generation & synonym expansion | Broad, comparative, or multi-faceted inquiries (`"different"`, `"expand"`, `"perspectives"`) |
| **MMR Diversity RAG** | `app/api/routers/mmrrag.py` | Maximal Marginal Relevance diversified retrieval ($\lambda=0.7$) | Exploratory queries prone to redundancy (`"diverse"`, `"variety"`, `"types of"`) |
| **CAG (Context-Aware)** | `app/api/routers/cag.py` | In-memory LRU cache with TTL + optional Redis backing | Repeated inquiries & cached high-frequency states |
| **Direct Generation** | `app/api/endpoints/query.py` | Zero-retrieval direct parametric generation | Conversational pleasantries, direct instructions, or math (`"aa"` prefix, greetings) |

---

## System Architecture

```
                                  User Query
                                      │
                                      ▼
                  [ Semantic Intent Classifier & Adaptive Router ]
                                      │
    ┌──────────────┬──────────────┬───┴──────────┬──────────────┬──────────────┬──────────────┐
    │              │              │              │              │              │              │
    ▼              ▼              ▼              ▼              ▼              ▼              ▼
[ Hybrid RAG ] [ Corrective ] [ Dense RAG ] [ Sparse BM25 ] [ Multi-Query ] [ MMR RAG ] [ Cached CAG ]
 (Dense+Sparse)   (CRAG)        (FAISS/Qdrant)   (VLRAG)      (Expansion)   (Diversity)   (Fast Cache)
    │              │              │              │              │              │              │
    └──────────────┴──────────────┼──────────────┴──────────────┴──────────────┴──────────────┘
                                  ▼
                   [ Context Extraction & Synthesis ]
                                  │
                                  ▼
                       Final Grounded Response
```

---

## Deep Dive: Advanced RAG Strategies

### 1. Hybrid RAG with Reciprocal Rank Fusion (RRF)
Combines dense semantic vector retrieval (FAISS) with sparse lexical BM25 matching (BM25Okapi). Results from both retrieval heads are fused using Reciprocal Rank Fusion:
$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{60 + \text{rank}_m(d)}$$
This overcomes the classic trade-off where dense search misses precise identifiers (like error codes or version strings) while sparse search misses semantic paraphrasing.

### 2. Corrective RAG (CRAG)
Acts as an automated quality gatekeeper that evaluates retrieval confidence before passing context to the generator:
* **CORRECT (Confidence $\ge 0.60$)**: High-confidence match. Strips non-informative sentences and delivers clean context.
* **AMBIGUOUS ($0.40 \le \text{Confidence} < 0.60$)**: Partial coverage. Automatically refines the query, extracts key entities, re-retrieves, and merges results.
* **INCORRECT (Confidence $< 0.40$)**: Low relevance. Flags context insufficiency to prevent hallucinated answers.

---

## Repository Structure

```
Adaptive-RAG-Routing/
├── app/
│   ├── api/
│   │   ├── endpoints/
│   │   │   ├── ingest.py            # Unified PDF & text ingestion pipeline
│   │   │   └── query.py             # Adaptive routing classifier & query endpoint
│   │   └── routers/
│   │       ├── hybridrag.py         # Hybrid Dense + Sparse BM25 with RRF
│   │       ├── crag.py              # Corrective RAG with self-reflection
│   │       ├── rag.py               # Standard Dense Vector RAG (Qdrant)
│   │       ├── vlrag.py             # Sparse Lexical BM25 (Vectorless RAG)
│   │       ├── multirag.py          # Multi-Query Expansion RAG
│   │       ├── mmrrag.py            # Maximal Marginal Relevance (Diversity RAG)
│   │       └── cag.py               # Cache-Augmented Generation (CAG)
│   ├── core/
│   │   ├── config.py                # Configuration and environment settings
│   │   └── embeddings.py            # Production-ready normalized text embeddings
│   ├── models/
│   │   └── schemas.py               # Pydantic schemas (Query, Document)
│   ├── services/
│   │   └── llm_service.py           # LLM session lifecycle and fallback handler
│   └── main.py                      # FastAPI application entrypoint & router wiring
├── tests/
│   └── test_routing.py              # Comprehensive classification & integration tests
├── requirements.txt                 # Production dependencies
└── README.md
```

---

## Getting Started

### Prerequisites
* Python 3.10+
* Virtual environment tool (`venv` or `conda`)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/prannesshkva/Adaptive-RAG-Routing.git
   cd Adaptive-RAG-Routing
   ```

2. **Initialize virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the API service:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Access Interactive Documentation:**
   * Swagger UI: `http://localhost:8000/docs`
   * ReDoc: `http://localhost:8000/redoc`

---

## API Usage Examples

### 1. Ingest Documents
Ingest raw text:
```bash
curl -X POST "http://localhost:8000/ingest/text" \
     -H "Content-Type: application/json" \
     -d '{"text": "FastAPI is a modern, high-performance web framework for building APIs with Python."}'
```
Or upload a PDF document:
```bash
curl -X POST "http://localhost:8000/ingest" \
     -F "file=@document.pdf"
```

### 2. Query with Automatic Adaptive Routing
```bash
curl -X POST "http://localhost:8000/adaptive_query" \
     -H "Content-Type: application/json" \
     -d '{"message": "how does the FastAPI framework implement modern web APIs with python"}'
```

Response payload includes clear telemetry:
```json
{
  "data": "FastAPI is a modern, fast web framework for building APIs with Python based on standard Python type hints.",
  "routing_strategy": "hybridrag",
  "reason": "Complex or technical query combining concepts and terminology; routed to Hybrid RRF."
}
```

### 3. Explicit Strategy Override
You can optionally bypass the classifier and specify any strategy directly:
```bash
curl -X POST "http://localhost:8000/adaptive_query" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "verify if FastAPI uses Python type hints",
       "strategy": "crag",
       "explain": true
     }'
```

---

## Testing

Run the test suite to verify routing classification and end-to-end retrieval across all engines:
```bash
python tests/test_routing.py
```

---

## Author & Attribution

* **Architect**: [Prannessh KVA](https://github.com/prannesshkva)
* **Institution**: Lovely Professional University
* **Focus**: Advanced Cognitive Architectures, Retrieval Systems, and Distributed AI

---

## License

This project is licensed under the MIT License.
