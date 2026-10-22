from fastapi import FastAPI
from app.services.llm_service import init_session, close_session
from app.api.endpoints import ingest, query
from app.api.routers import rag, vlrag, cag, multirag, mmrrag, hybridrag, crag

app = FastAPI(
    title="Adaptive RAG Routing API",
    description="Production-ready multi-strategy Adaptive RAG system with dynamic query classification, Hybrid RRF, and Corrective RAG.",
    version="2.0.0"
)

# Core Ingestion and Adaptive Query Endpoints
app.include_router(ingest.router)
app.include_router(query.router)

# Individual Strategy Routers
app.include_router(rag.router)
app.include_router(vlrag.router)
app.include_router(cag.router)
app.include_router(multirag.router)
app.include_router(mmrrag.router)
app.include_router(hybridrag.router)
app.include_router(crag.router)

@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "online",
        "service": "Adaptive-RAG-Routing",
        "version": "2.0.0",
        "endpoints": {
            "adaptive_query": "/adaptive_query",
            "ingest_file": "/ingest",
            "ingest_text": "/ingest/text",
            "routes_metadata": "/routes"
        }
    }

@app.get("/routes", tags=["Routing Catalog"])
async def list_routes():
    """
    Catalog of all production RAG routing strategies available in the system.
    """
    return {
        "strategies": [
            {
                "id": "hybridrag",
                "name": "Hybrid RAG (RRF)",
                "description": "Combines Dense Vector (FAISS) and Sparse Lexical (BM25) search using Reciprocal Rank Fusion.",
                "triggers": ["Technical", "Architectural", "Detailed queries", "6+ words", "Domain terminology"]
            },
            {
                "id": "crag",
                "name": "Corrective RAG",
                "description": "Self-reflective RAG with confidence evaluation, noise-stripping, and query reformulation fallback.",
                "triggers": ["Verification", "Fact checking", "Confirmation", "'verify'", "'is it true'", "'confirm'"]
            },
            {
                "id": "vlrag",
                "name": "Sparse Lexical BM25 (VLRAG)",
                "description": "Token-based exact BM25Okapi search with lemmatization for identifiers and codes.",
                "triggers": ["Short queries (<4 words)", "Error codes", "Product IDs", "Quoted phrases"]
            },
            {
                "id": "rag",
                "name": "Dense Vector RAG",
                "description": "Cosine similarity dense vector search via normalized embeddings for conceptual matching.",
                "triggers": ["Definitions", "'define'", "'why'", "'what is'", "'explain'"]
            },
            {
                "id": "multirag",
                "name": "Multi-Query Expansion RAG",
                "description": "Expands queries with synonyms and multi-perspective variants to improve recall.",
                "triggers": ["Broad topics", "Comparative", "'different'", "'expand'", "'perspectives'", "'aspects'"]
            },
            {
                "id": "mmrrag",
                "name": "Maximal Marginal Relevance (MMRRAG)",
                "description": "Balances query relevance and candidate diversity to eliminate redundant passages.",
                "triggers": ["Diversity requests", "'diverse'", "'variety'", "'types of'", "'range of'"]
            },
            {
                "id": "cag",
                "name": "Cache-Augmented Generation (CAG)",
                "description": "Sub-millisecond retrieval cache with TTL and semantic key normalization for repetitive queries.",
                "triggers": ["Default fallback", "Frequent queries"]
            },
            {
                "id": "direct_llm",
                "name": "Direct LLM Generation",
                "description": "Bypasses retrieval for conversational pleasantries, direct instructions, and greetings.",
                "triggers": ["'aa' prefix", "Greetings"]
            }
        ]
    }

@app.on_event("startup")
async def startup():
    await init_session()
    # Initialize in-memory indexes
    await rag.init()
    await hybridrag.init()
    await crag.init()

@app.on_event("shutdown")
async def shutdown():
    await close_session()
