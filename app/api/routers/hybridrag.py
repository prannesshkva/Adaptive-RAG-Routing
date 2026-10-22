import faiss, re, asyncio
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi
from fastapi import APIRouter, HTTPException
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

from app.models.schemas import Query, Document
from app.core.embeddings import get_embedder

router = APIRouter(prefix="/hybridrag", tags=["Hybrid RAG (RRF)"])

embedder = get_embedder()
dense_index = None
sparse_model = None
documents_store = []
tokenized_corpus = []
executor = ThreadPoolExecutor(max_workers=4)

try:
    stop_words = set(stopwords.words("english"))
except Exception:
    stop_words = set()

lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> List[str]:
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


def tokenize(text: str) -> List[str]:
    words = clean_text(text).split()
    tokens = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return tokens if tokens else words


async def embed_texts(texts: List[str]):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor,
        lambda: embedder.encode(texts, normalize_embeddings=True).astype("float32")
    )


@router.post("/init")
async def init():
    global dense_index, sparse_model, documents_store, tokenized_corpus
    dim = embedder.dim
    dense_index = faiss.IndexFlatIP(dim)
    documents_store = []
    tokenized_corpus = []
    sparse_model = None
    return {"status": "initialized", "dim": dim}


@router.post("/add")
async def add(doc: Document):
    global dense_index, sparse_model, documents_store, tokenized_corpus
    if dense_index is None:
        await init()

    chunks = chunk_text(doc.text)
    if not chunks:
        return {"status": "no content"}

    # 1. Dense FAISS indexing
    embeddings = await embed_texts(chunks)
    dense_index.add(embeddings)

    # 2. Sparse BM25 indexing
    for chunk in chunks:
        documents_store.append(chunk)
        tokenized_corpus.append(tokenize(chunk))

    sparse_model = BM25Okapi(tokenized_corpus)
    return {"status": "added", "chunks": len(chunks), "total_documents": len(documents_store)}


def reciprocal_rank_fusion(dense_ranks: Dict[int, int], sparse_ranks: Dict[int, int], k: int = 60) -> List[int]:
    all_doc_ids = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))
    rrf_scores = {}

    for doc_id in all_doc_ids:
        score = 0.0
        if doc_id in dense_ranks:
            score += 1.0 / (k + dense_ranks[doc_id])
        if doc_id in sparse_ranks:
            score += 1.0 / (k + sparse_ranks[doc_id])
        rrf_scores[doc_id] = score

    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda did: rrf_scores[did], reverse=True)
    return sorted_doc_ids


@router.post("/query")
async def query(q: Query, top_k: int = 5):
    global dense_index, sparse_model, documents_store
    if dense_index is None or len(documents_store) == 0:
        return {"data": []}

    query_str = q.message.strip()
    if not query_str:
        raise HTTPException(status_code=400, detail="Empty query")

    candidate_k = min(len(documents_store), top_k * 3)

    # 1. Retrieve Dense Candidates (FAISS)
    query_vec = await embed_texts([query_str])
    scores, indices = dense_index.search(query_vec, candidate_k)
    dense_ranks = {int(idx): rank + 1 for rank, idx in enumerate(indices[0]) if 0 <= idx < len(documents_store)}

    # 2. Retrieve Sparse Candidates (BM25)
    query_tokens = tokenize(query_str)
    sparse_ranks = {}
    if sparse_model and query_tokens:
        bm25_scores = sparse_model.get_scores(query_tokens)
        top_sparse_idx = np.argsort(bm25_scores)[::-1][:candidate_k]
        sparse_ranks = {int(idx): rank + 1 for rank, idx in enumerate(top_sparse_idx) if bm25_scores[idx] > 0}

    # 3. Reciprocal Rank Fusion
    fused_order = reciprocal_rank_fusion(dense_ranks, sparse_ranks, k=60)
    top_indices = fused_order[:top_k]

    results = [documents_store[i] for i in top_indices if i < len(documents_store)]
    return {"source": "hybrid_rrf", "data": results}
