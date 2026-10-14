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

**Adaptive-RAG-Routing** is a high-performance orchestration layer designed to dynamically route natural language queries to optimal retrieval and reasoning pipelines. Rather than applying a single uniform retrieval strategy to every user inquiry, this architecture performs query decomposition and complexity classification to select among specialized RAG modalities—including Context-Aware Generation (CAG), dense semantic search, multi-source vector retrieval, Maximal Marginal Relevance (MMR) re-ranking, and vision-language indexing.

---

## Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-18181B?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-18181B?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Pydantic-18181B?style=for-the-badge&logo=pydantic&logoColor=white" alt="Pydantic" />
  <img src="https://img.shields.io/badge/PyTorch-18181B?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/LangChain-18181B?style=for-the-badge&logo=chainlink&logoColor=white" alt="LangChain" />
  <img src="https://img.shields.io/badge/Uvicorn-18181B?style=for-the-badge&logo=gunicorn&logoColor=white" alt="Uvicorn" />
  <img src="https://img.shields.io/badge/Vector_DB-18181B?style=for-the-badge&logo=databricks&logoColor=white" alt="Vector DB" />
</p>

---

## Key Routing Modalities

| Modality | Router | Description | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **CAG** | `app.api.routers.cag` | Context-Aware Generation without heavy retrieval overhead | High-context single-turn prompts & cached semantic states |
| **Standard RAG** | `app.api.routers.rag` | Dense vector indexing with similarity threshold gating | Direct factual retrieval over domain documentation |
| **VL-RAG** | `app.api.routers.vlrag` | Vision-Language multi-modal retrieval | Queries paired with visual evidence, charts, and diagrams |
| **Multi-RAG** | `app.api.routers.multirag` | Distributed multi-corpus fan-out aggregation | Cross-domain inquiries requiring consensus from distinct indices |
| **MMR-RAG** | `app.api.routers.mmrrag` | Maximal Marginal Relevance diversified retrieval | Exploratory queries prone to redundant context saturation |

---

## System Architecture

```
User Query
    │
    ▼
[Semantic Intent Classifier & Query Router]
    │
    ├─── Simple Query (Cached / Direct) ───────────► Context-Aware Generation (CAG)
    ├─── Single-Corpus Factoid ────────────────────► Dense Vector Search (RAG)
    ├─── Multi-Source Cross-Corpus ────────────────► Federated Ingestion (Multi-RAG)
    ├─── Diverse Exploratory Search ───────────────► Maximal Marginal Relevance (MMR)
    └─── Multimodal Evidence ──────────────────────► Vision-Language RAG (VL-RAG)
    │
    ▼
[Response Synthesizer & Confidence Evaluator]
    │
    ▼
Final Grounded Response
```

---

## Repository Structure

```
Adaptive-RAG-Routing/
├── app/
│   ├── api/
│   │   ├── endpoints/       # Ingestion and query API controllers
│   │   │   ├── ingest.py
│   │   │   └── query.py
│   │   └── routers/         # Strategy routing pipelines
│   │       ├── cag.py       # Context-Aware Generation router
│   │       ├── mmrrag.py    # Maximal Marginal Relevance router
│   │       ├── multirag.py  # Multi-corpus vector router
│   │       ├── rag.py       # Standard dense retrieval router
│   │       └── vlrag.py     # Vision-Language multi-modal router
│   ├── core/                # Configuration, logging, and security definitions
│   ├── models/              # Pydantic schemas and serialization definitions
│   ├── services/            # LLM session lifecycle and vector store drivers
│   │   └── llm_service.py
│   └── main.py              # Application entrypoint & FastAPI wiring
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
   pip install fastapi uvicorn pydantic torch sentence-transformers
   ```

4. **Launch the API service:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Access Interactive Documentation:**
   * Swagger UI: `http://localhost:8000/docs`
   * ReDoc: `http://localhost:8000/redoc`

---

## Author & Attribution

* **Architect**: [Prannessh KVA](https://github.com/prannesshkva)
* **Institution**: Lovely Professional University
* **Focus**: Advanced Cognitive Architectures, Retrieval Systems, and Distributed AI

---

## License

This project is licensed under the MIT License.
