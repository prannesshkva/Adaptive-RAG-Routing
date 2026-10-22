import faiss, re, asyncio
import numpy as np
from fastapi import APIRouter, HTTPException
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple

from app.models.schemas import Query, Document
from app.core.embeddings import get_embedder

router = APIRouter(prefix="/crag", tags=["Corrective RAG (CRAG)"])

embedder = get_embedder()
index = None
documents_store = []
executor = ThreadPoolExecutor(max_workers=4)

CONFIDENCE_CORRECT_THRESHOLD = 0.60
CONFIDENCE_AMBIGUOUS_THRESHOLD = 0.40


def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 30) -> List[str]:
    words = clean_text(text).split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


async def embed_texts(texts: List[str]):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor,
        lambda: embedder.encode(texts, normalize_embeddings=True).astype("float32")
    )


def extract_keywords(query: str) -> List[str]:
    stopwords = {
        "what", "is", "the", "a", "an", "how", "why", "can", "you", "tell",
        "me", "about", "which", "where", "when", "does", "do", "for", "in",
        "of", "on", "to", "and", "or", "verify", "check", "confirm", "true"
    }
    words = clean_text(query).split()
    return [w for w in words if w not in stopwords and len(w) > 2]


def evaluate_confidence(query: str, doc_text: str, vector_score: float) -> float:
    """
    Evaluates retrieval confidence by fusing cosine vector score with keyword match ratio.
    """
    q_keywords = extract_keywords(query)
    doc_words = set(clean_text(doc_text).split())
    if not q_keywords:
        return float(vector_score)
    
    keyword_overlap = sum(1 for kw in q_keywords if kw in doc_words) / len(q_keywords)
    # Weighted harmonic fusion of vector semantic similarity and lexical presence
    confidence = (0.7 * max(0.0, float(vector_score))) + (0.3 * keyword_overlap)
    return min(1.0, max(0.0, confidence))


def strip_noise_sentences(query: str, doc_text: str, top_sentences: int = 3) -> str:
    """
    Sentence-level contextual refinement: extracts the sentences most relevant to the query.
    """
    sentences = re.split(r'(?<=[.!?])\s+', doc_text.strip())
    if len(sentences) <= top_sentences:
        return doc_text

    q_keywords = set(extract_keywords(query))
    scored_sentences = []
    for s in sentences:
        s_words = set(clean_text(s).split())
        overlap = len(q_keywords.intersection(s_words))
        scored_sentences.append((overlap, s))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    selected = [s for _, s in scored_sentences[:top_sentences]]
    return " ".join(selected)


@router.post("/init")
async def init():
    global index, documents_store
    dim = embedder.dim
    index = faiss.IndexFlatIP(dim)
    documents_store = []
    return {"status": "initialized", "dim": dim}


@router.post("/add")
async def add(doc: Document):
    global index, documents_store
    if index is None:
        await init()

    chunks = chunk_text(doc.text)
    if not chunks:
        return {"status": "no content"}

    embeddings = await embed_texts(chunks)
    index.add(embeddings)
    documents_store.extend(chunks)
    return {"status": "added", "chunks": len(chunks), "total_documents": len(documents_store)}


@router.post("/query")
async def query(q: Query, top_k: int = 5):
    """
    Corrective RAG retrieval:
    1. Retrieve candidates
    2. Evaluate confidence
    3. Self-correct: Refine (CORRECT), Re-query with key terms (AMBIGUOUS), or Flag (INCORRECT)
    """
    global index, documents_store
    if index is None or len(documents_store) == 0:
        return {"data": ["No documents available in knowledge base."], "status": "empty", "confidence": 0.0}

    query_str = q.message.strip()
    if not query_str:
        raise HTTPException(status_code=400, detail="Empty query")

    # Initial retrieval
    query_vec = await embed_texts([query_str])
    scores, indices = index.search(query_vec, min(top_k * 2, len(documents_store)))

    candidates = []
    for rank, idx in enumerate(indices[0]):
        if 0 <= idx < len(documents_store):
            sim_score = float(scores[0][rank])
            chunk = documents_store[idx]
            conf = evaluate_confidence(query_str, chunk, sim_score)
            candidates.append({"text": chunk, "confidence": conf, "vector_score": sim_score})

    if not candidates:
        return {"data": ["No relevant context found."], "status": "incorrect", "confidence": 0.0}

    best_confidence = max(c["confidence"] for c in candidates)

    # Decision Path 1: CORRECT (High Confidence) -> Strip noise and return focused context
    if best_confidence >= CONFIDENCE_CORRECT_THRESHOLD:
        refined_docs = [
            strip_noise_sentences(query_str, c["text"])
            for c in candidates if c["confidence"] >= CONFIDENCE_AMBIGUOUS_THRESHOLD
        ][:top_k]
        return {
            "source": "corrective_rag",
            "status": "correct",
            "confidence": round(best_confidence, 4),
            "data": refined_docs if refined_docs else [candidates[0]["text"]]
        }

    # Decision Path 2: AMBIGUOUS (Moderate Confidence) -> Query decomposition / rewrite & re-search
    elif best_confidence >= CONFIDENCE_AMBIGUOUS_THRESHOLD:
        keywords = extract_keywords(query_str)
        refined_query = " ".join(keywords) if keywords else query_str
        ref_vec = await embed_texts([refined_query])
        ref_scores, ref_indices = index.search(ref_vec, top_k)
        
        merged_results = [candidates[0]["text"]]
        for idx in ref_indices[0]:
            if 0 <= idx < len(documents_store):
                text = documents_store[idx]
                if text not in merged_results:
                    merged_results.append(text)

        return {
            "source": "corrective_rag",
            "status": "ambiguous_reformulated",
            "confidence": round(best_confidence, 4),
            "refined_query": refined_query,
            "data": merged_results[:top_k]
        }

    # Decision Path 3: INCORRECT (Low Confidence) -> Guardrail against hallucination
    else:
        return {
            "source": "corrective_rag",
            "status": "incorrect",
            "confidence": round(best_confidence, 4),
            "data": [
                "INSUFFICIENT CONTEXT: The available documents do not contain high-confidence information regarding this query."
            ]
        }
