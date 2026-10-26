# Adaptive RAG Routing

<p align="center">
  <strong>Intelligent Dynamic Routing Engine for Modular Retrieval-Augmented Generation</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge&logo=github&logoColor=white" alt="Status" />
  <img src="https://img.shields.io/badge/Python-3.10+-18181B?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Architecture-Modular_RAG-18181B?style=for-the-badge" alt="Architecture" />
  <img src="https://img.shields.io/badge/FastAPI-2.0.0-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/License-MIT-18181B?style=for-the-badge" alt="License" />
</p>

<p align="center">
  <a href="#system-architecture">Architecture</a> •
  <a href="#mathematical-foundations">Mathematical Foundations</a> •
  <a href="#key-routing-modalities">Routing Modalities</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#api-reference">API Reference</a>
</p>

---

## Overview

**Adaptive-RAG-Routing** is a high-performance orchestration layer engineered to dynamically route natural language queries to optimal retrieval and reasoning pipelines. Rather than applying a single uniform retrieval strategy to every user inquiry, this architecture performs real-time query decomposition, lexical density analysis, and semantic complexity classification to select among specialized RAG modalities:

* **Hybrid RAG (RRF)**: Fuses dense semantic embeddings (FAISS) and sparse lexical tokens (BM25Okapi) via Reciprocal Rank Fusion ($k = 60$).
* **Corrective RAG (CRAG)**: Self-reflective retrieval with automated confidence scoring, contextual noise stripping, and hallucination guardrails.
* **Standard Dense RAG**: Fast cosine-distance vector indexing backed by Qdrant.
* **Sparse Lexical RAG (VLRAG)**: Pure token-based BM25Okapi search with POS lemmatization for alphanumeric identifiers and error codes.
* **Multi-Query Expansion**: Multi-perspective query generation to overcome lexical mismatch in exploratory prompts.
* **Maximal Marginal Relevance (MMR)**: Re-ranking engine ($\lambda = 0.70$) that penalizes redundant context chunks to maximize prompt entropy.
* **Cache-Augmented Generation (CAG)**: In-memory LRU cache with TTL for zero-latency responses on repeated queries.
* **Direct Generation**: Parametric LLM bypass for greetings, casual chat, and direct mathematical expressions.

---

## Dynamic Architecture & Execution Flow

<p align="center">
  <img src="assets/adaptive_rag_animation.svg" alt="Adaptive RAG Routing Architecture & Dynamic Execution Flow" width="100%" />
</p>

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

| Modality | Router Module | Mathematical / Algorithmic Core | Primary Triggers & Classification Rule |
| :--- | :--- | :--- | :--- |
| **Hybrid RAG (RRF)** | `app/api/routers/hybridrag.py` | Dense vector (FAISS) + Sparse BM25 via Reciprocal Rank Fusion ($k = 60$) | Complex technical queries ($\ge 6$ words), architectural or procedural terms (`"how does"`, `"architecture"`, `"steps"`) |
| **Corrective RAG (CRAG)** | `app/api/routers/crag.py` | Self-reflective dual confidence metric ($\Gamma \ge 0.60$) with sentence noise stripping | Verification and fact-checking queries (`"verify"`, `"is it true"`, `"confirm"`, `"check if"`) |
| **Standard Dense RAG** | `app/api/routers/rag.py` | Cosine similarity vector search via Qdrant in-memory client | Definitions and direct conceptual inquiries (`"define"`, `"what is"`, `"why"`, `"explain"`) |
| **Sparse Lexical RAG** | `app/api/routers/vlrag.py` | BM25Okapi token matching with NLTK POS tagging and WordNet lemmatization | Exact codes, IDs, alphanumeric strings (`_` character, mixed digits/letters, quotes, $\le 3$ words) |
| **Multi-Query RAG** | `app/api/routers/multirag.py` | Synonym generation & multi-perspective query expansion | Broad, comparative, or multi-faceted queries (`"different"`, `"expand"`, `"perspectives"`, `"compare"`) |
| **MMR Diversity RAG** | `app/api/routers/mmrrag.py` | Maximal Marginal Relevance diversified selection ($\lambda = 0.70$, $K = 5$) | Diversity and taxonomy queries (`"diverse"`, `"variety"`, `"types of"`, `"range of"`) |
| **Cache-Augmented (CAG)** | `app/api/routers/cag.py` | In-memory LRU cache with TTL + semantic key normalization | Repeated inquiries & cached high-frequency states (Default fallback) |
| **Direct Generation** | `app/api/endpoints/query.py` | Zero-retrieval direct parametric LLM generation | Explicit bypass prefix (`"aa"`), conversational greetings, or pure arithmetic |

---

## Mathematical Foundations

### 1. Dynamic Routing Decision Boundary
The adaptive classifier inspects the input query $q$ across lexical density, structural tokens, and semantic intent to evaluate an assignment function $\mathcal{R}(q)$:

$$
\mathcal{R}(q) \in \{\text{CRAG}, \text{Dense}, \text{MultiQuery}, \text{MMR}, \text{Hybrid}, \text{BM25}, \text{CAG}, \text{Direct}\}
$$

The priority cascade ensures that deterministic signals (e.g., verification predicates and alphanumeric identifiers) preempt general vector retrieval, avoiding costly vector index lookups for non-vector tasks.

---

### 2. Hybrid RAG: Reciprocal Rank Fusion (RRF)
Dense vector search excels at broad semantic understanding but misses exact alphanumeric identifiers (part numbers, error codes, version flags). Conversely, sparse BM25 matches exact lexical tokens but fails on conceptual paraphrasing. 

To unify both retrieval heads without requiring calibrated probability scores, results are merged via **Reciprocal Rank Fusion**:

$$
\operatorname{RRF\_Score}(d \in \mathcal{D}) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{k + r_m(d)}
$$

where:
* $k = 60$ is the smoothing constant that balances high-ranking dominance against lower-ranked consensus.
* $r_m(d) \in \{1, 2, \dots, N\}$ represents the 1-based ordinal rank of document $d$ within retrieval system $m$.
* Documents appearing in both retrieval spaces accumulate score additive bonuses, pushing consensus items to the top of the context window.

---

### 3. Sparse Lexical Scoring: BM25Okapi
For exact identifier and code retrieval, the corpus is tokenized, stripped of non-informative stopwords, and normalized via WordNet POS-lemmatization. Document relevance is scored via the BM25Okapi formula:

$$
\operatorname{Score}_{\mathrm{BM25}}(D, Q) = \sum_{t \in Q} \operatorname{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\operatorname{avgdl}}\right)}
$$

where the Inverse Document Frequency (IDF) is calculated as:

$$
\operatorname{IDF}(t) = \ln \left( \frac{N - n(t) + 0.5}{n(t) + 0.5} + 1 \right)
$$

* $f(t, D)$ is the term frequency of token $t$ in candidate chunk $D$.
* $|D|$ is the length of document $D$ in words, and $\operatorname{avgdl}$ is the average document length across the corpus.
* $k_1 = 1.5$ regulates term saturation non-linearity.
* $b = 0.75$ controls document length normalization.

---

### 4. Maximal Marginal Relevance (MMR)
Standard top-$K$ semantic similarity frequently populates the prompt window with near-duplicate passages. **MMR** maximizes information entropy by balancing candidate query relevance with candidate diversity against already-selected context:

$$
\operatorname{MMR}(Q, \mathcal{D}, \mathcal{S}) \equiv \underset{d_i \in \mathcal{D} \setminus \mathcal{S}}{\operatorname{argmax}} \left[ \lambda \cdot \operatorname{Sim}_1(d_i, Q) - (1 - \lambda) \cdot \max_{d_j \in \mathcal{S}} \operatorname{Sim}_2(d_i, d_j) \right]
$$

where:
* $\operatorname{Sim}_1(d_i, Q) = \mathbf{e}_{d_i}^{\top} \mathbf{e}_Q$ is the cosine similarity between candidate document $d_i$ and query $Q$.
* $\operatorname{Sim}_2(d_i, d_j) = \mathbf{e}_{d_i}^{\top} \mathbf{e}_{d_j}$ measures pairwise semantic similarity between candidate $d_i$ and already selected document $d_j \in \mathcal{S}$.
* $\lambda = 0.70$ provides the optimal trade-off (70% query relevance, 30% novelty diversification).

---

### 5. Corrective RAG (CRAG) Confidence Scoring & Guardrails
When fact-checking or verification queries are received, CRAG acts as an automated validation firewall to eliminate hallucinations. 

The retrieval confidence $\Gamma(q, d, s)$ integrates dense semantic similarity $s$ with query-document keyword overlap:

$$
\Gamma(q, d, s) = 0.70 \cdot \max(0, s) + 0.30 \cdot \left( \frac{|\mathcal{K}(q) \cap \mathcal{W}(d)|}{|\mathcal{K}(q)|} \right)
$$

where $\mathcal{K}(q)$ denotes filtered non-stopword query keywords, and $\mathcal{W}(d)$ denotes the unique vocabulary of document $d$.

The evaluation pipeline branches across three deterministic thresholds:

$$
\operatorname{Action}(\Gamma) = \begin{cases} 
\text{CORRECT} & \text{if } \Gamma \ge 0.60 \implies \text{Contextual noise stripping (retains top salient sentences)} \\ 
\text{AMBIGUOUS} & \text{if } 0.40 \le \Gamma \lt 0.60 \implies \text{Query reformulation \& entity-directed re-retrieval} \\ 
\text{INCORRECT} & \text{if } \Gamma \lt 0.40 \implies \text{Context insufficiency flag (Prevents hallucinated answers)} 
\end{cases}
$$

---

### 6. Normalized Dense Vector Embeddings
All dense vector embeddings $\mathbf{v} \in \mathbb{R}^D$ are $L_2$-normalized during inference:

$$
\mathbf{e} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2} \implies \|\mathbf{e}\|_2 = 1.0
$$

Under unit-norm constraint, the cosine similarity between query embedding $\mathbf{e}_q$ and document embedding $\mathbf{e}_d$ simplifies to a single matrix dot product:

$$
\operatorname{Sim}_{\cos}(\mathbf{e}_q, \mathbf{e}_d) = \frac{\mathbf{e}_q \cdot \mathbf{e}_d}{\|\mathbf{e}_q\|_2 \|\mathbf{e}_d\|_2} = \mathbf{e}_q^{\top} \mathbf{e}_d
$$

This allows FAISS (`IndexFlatIP`) and Qdrant (`Distance.COSINE`) to execute high-throughput inner product scans with zero runtime vector norm divisions.

---

## Repository Structure

```
Adaptive-RAG-Routing/
├── assets/
│   └── adaptive_rag_animation.svg   # Motion graphics SVG architecture diagram
├── app/
│   ├── api/
│   │   ├── endpoints/
│   │   │   ├── ingest.py            # Unified PDF & text ingestion pipeline
│   │   │   └── query.py             # Adaptive routing classifier & query endpoint
│   │   └── routers/
│   │       ├── hybridrag.py         # Hybrid Dense + Sparse BM25 with RRF (k=60)
│   │       ├── crag.py              # Corrective RAG with self-reflection & noise stripping
│   │       ├── rag.py               # Standard Dense Vector RAG (Qdrant in-memory)
│   │       ├── vlrag.py             # Sparse Lexical BM25 (Vectorless RAG)
│   │       ├── multirag.py          # Multi-Query Expansion RAG
│   │       ├── mmrrag.py            # Maximal Marginal Relevance (Diversity RAG, λ=0.7)
│   │       └── cag.py               # Cache-Augmented Generation (CAG)
│   ├── core/
│   │   ├── config.py                # Configuration and environment settings
│   │   └── embeddings.py            # Normalized text embeddings with resilient fallback
│   ├── models/
│   │   └── schemas.py               # Pydantic validation schemas (Query, Document)
│   ├── services/
│   │   └── llm_service.py           # LLM session lifecycle and fallback handler
│   └── main.py                      # FastAPI application entrypoint & router catalog
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

## API Reference

### 1. Ingest Documents

Ingest raw text:
```bash
curl -X POST "http://localhost:8000/ingest/text" \
     -H "Content-Type: application/json" \
     -d '{"text": "FastAPI is a modern, high-performance web framework for building APIs with Python based on standard Python type hints."}'
```

Upload a PDF document:
```bash
curl -X POST "http://localhost:8000/ingest" \
     -F "file=@document.pdf"
```

---

### 2. Query with Automatic Adaptive Routing

```bash
curl -X POST "http://localhost:8000/adaptive_query" \
     -H "Content-Type: application/json" \
     -d '{"message": "how does the FastAPI framework implement modern web APIs with python"}'
```

Response payload includes routing telemetry:
```json
{
  "data": "FastAPI leverages standard Python type hints and Pydantic models to automatically validate request schemas and generate OpenAPI specifications.",
  "routing_strategy": "hybridrag",
  "reason": "Complex or technical query combining concepts and terminology; routed to Hybrid RRF."
}
```

---

### 3. Explicit Strategy Override

You can bypass the classifier and enforce any retrieval strategy directly:
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

Execute the comprehensive test suite to verify routing classification, FAISS/BM25 indexing, and CRAG threshold gating:

```bash
python tests/test_routing.py
```

Expected output:
```text
--- Testing Adaptive Routing Classification ---
[PASS] 'verify if quantum encryption is unbreakable' -> Expected: crag, Got: crag
[PASS] 'ERR_404_NOT_FOUND' -> Expected: vlrag, Got: vlrag
[PASS] 'how does the transformer attention mechanism work' -> Expected: hybridrag, Got: hybridrag
[PASS] 'define entropy' -> Expected: rag, Got: rag
All classification tests passed successfully!

--- Testing Document Ingestion & Routing End-to-End ---
[PASS] CRAG correctly detected out-of-domain query and prevented hallucination!
All end-to-end integration tests passed successfully!
```

---

## Author & Attribution

* **Architect**: [Prannessh KVA](https://github.com/prannesshkva)
* **Profile**: [github.com/prannesshkva](https://github.com/prannesshkva)
* **Focus**: Non-Euclidean SSMs, Differentiable Agentic Systems, and Modular RAG Architectures

---

## License

This project is licensed under the MIT License.
