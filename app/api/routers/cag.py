import faiss, pickle, re, asyncio, time
import numpy as np
from fastapi import APIRouter, HTTPException
from concurrent.futures import ThreadPoolExecutor
from app.models.schemas import Query, Document
from app.core.embeddings import get_embedder

router = APIRouter(prefix="/cag", tags=["CAG"])
embedder = get_embedder()
index = None
documents_store = []
executor = ThreadPoolExecutor(max_workers=4)

# In-memory resilient cache with TTL support
_memory_cache = {}
_CACHE_TTL = 3600

# Optional Redis connection with safe fallback
redis_client = None
try:
    import redis
    r = redis.Redis(host="localhost", port=6379, db=0, socket_timeout=0.1)
    r.ping()
    redis_client = r
except Exception:
    redis_client = None

def get_from_cache(key: str):
    if redis_client is not None:
        try:
            cached = redis_client.get(key)
            if cached:
                return pickle.loads(cached)
        except Exception:
            pass
    if key in _memory_cache:
        val, expiry = _memory_cache[key]
        if time.time() < expiry:
            return val
        else:
            del _memory_cache[key]
    return None

def set_in_cache(key: str, val, ttl: int = _CACHE_TTL):
    if redis_client is not None:
        try:
            redis_client.set(key, pickle.dumps(val), ex=ttl)
            return
        except Exception:
            pass
    _memory_cache[key] = (val, time.time() + ttl)

def process(text: str):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    words = text.lower().strip().split()
    chunks = []
    for i in range(0, len(words), 500):
        chunk = " ".join(words[i:i + 500])
        if chunk:
            chunks.append(chunk)
    return chunks

async def embed_texts(texts):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor,
        lambda: embedder.encode(texts, normalize_embeddings=True).astype("float32")
    )

@router.post("/init")
async def init():
    global index, documents_store
    dim = embedder.dim
    index = faiss.IndexFlatIP(dim)
    documents_store = []
    _memory_cache.clear()
    return {"status": "initialized", "dim": dim}

@router.post("/add")
async def add(doc: Document):
    global index, documents_store
    if index is None:
        await init()
    chunks = process(doc.text)
    if not chunks:
        return {"status": "no content"}
    embeddings = await embed_texts(chunks)
    index.add(embeddings)
    documents_store.extend(chunks)
    return {"status": "added", "chunks": len(chunks)}

@router.post("/query")
async def query(q: Query):
    global index, documents_store
    if index is None or len(documents_store) == 0:
        return {"source": "none", "data": []}
    key = f"cache:{q.message.strip().lower()}"
    cached = get_from_cache(key)
    if cached is not None:
        return {"source": "cache", "data": cached}
    query_vec = await embed_texts([q.message])
    scores, indices = index.search(query_vec, min(5, len(documents_store)))
    results = [documents_store[i] for i in indices[0] if 0 <= i < len(documents_store)]
    set_in_cache(key, results)
    return {"source": "faiss", "data": results}
