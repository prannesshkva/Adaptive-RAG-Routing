import faiss, re, asyncio
import numpy as np
from fastapi import APIRouter, HTTPException
from concurrent.futures import ThreadPoolExecutor

from app.models.schemas import Query, Document
from app.core.embeddings import get_embedder

router = APIRouter(prefix="/mmrrag", tags=["MMRRAG"])
embedder = get_embedder()
index = None
documents_store = []
embeddings_store = []
executor = ThreadPoolExecutor(max_workers=4)

def process(text: str):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    words = text.lower().strip().split()
    chunks = []
    for i in range(0, len(words), 200):
        chunk = " ".join(words[i:i + 200])
        if chunk:
            chunks.append(chunk)
    return chunks

async def embed_texts(texts):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, lambda: embedder.encode(texts, normalize_embeddings=True).astype("float32"))

def mmr(query_embedding, doc_embeddings, top_k=5, lambda_param=0.5):
    if len(doc_embeddings) == 0:
        return []
    doc_embeddings = np.array(doc_embeddings)
    query_sims = np.dot(doc_embeddings, query_embedding.T).flatten()
    selected = []
    candidates = list(range(len(doc_embeddings)))
    while len(selected) < min(top_k, len(doc_embeddings)):
        if not selected:
            best_idx = np.argmax(query_sims)
        else:
            selected_embeddings = doc_embeddings[selected]
            cross_sims = np.dot(doc_embeddings[candidates], selected_embeddings.T)
            max_sim_to_selected = np.max(cross_sims, axis=1)
            mmr_scores = lambda_param * query_sims[candidates] - (1 - lambda_param) * max_sim_to_selected
            best_idx = candidates[np.argmax(mmr_scores)]
        selected.append(best_idx)
        candidates.remove(best_idx)
    return selected

@router.post("/init")
async def init():
    global index, documents_store, embeddings_store
    dim = embedder.dim
    index = faiss.IndexFlatIP(dim)
    documents_store = []
    embeddings_store = []
    return {"status": "initialized", "dim": dim}

@router.post("/add")
async def add(doc: Document):
    global index, documents_store, embeddings_store
    if index is None:
        await init()
    chunks = process(doc.text)
    if not chunks:
        return {"status": "no content"}
    embeddings = await embed_texts(chunks)
    index.add(embeddings)
    documents_store.extend(chunks)
    embeddings_store.extend(embeddings.tolist())
    return {"status": "added", "chunks": len(chunks)}

@router.post("/query")
async def query(q: Query):
    global documents_store, embeddings_store
    if len(documents_store) == 0:
        return {"data": []}
    query_vec = (await embed_texts([q.message]))[0]
    selected_indices = mmr(query_vec, embeddings_store, top_k=5, lambda_param=0.7)
    results = [documents_store[i] for i in selected_indices]
    return {"data": results}
