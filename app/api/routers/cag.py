import faiss,redis,pickle,re,asyncio
import numpy as np
from fastapi import APIRouter,HTTPException
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
from app.models.schemas import Query, Document

router=APIRouter(prefix="/cag",tags=["CAG"])
embedder=SentenceTransformer("BAAI/bge-base-en-v1.5")
index=None
documents_store=[]
cache=redis.Redis(host="localhost",port=6379,db=0)
executor=ThreadPoolExecutor(max_workers=4)

def process(text:str):
    text=re.sub(r"[^a-zA-Z0-9\s]","",text)
    text=re.sub(r"\s+"," ",text)
    words=text.lower().strip().split()
    chunks=[]
    for i in range(0,len(words),500):
        chunk=" ".join(words[i:i+500])
        if chunk:
            chunks.append(chunk)
    return chunks

async def embed_texts(texts):
    loop=asyncio.get_event_loop()
    return await loop.run_in_executor(executor,lambda:embedder.encode(texts,normalize_embeddings=True).astype("float32"))

@router.post("/init")
async def init():
    global index,documents_store
    if index is None:
        dim=len((await embed_texts(["test"]))[0])
        index=faiss.IndexFlatIP(dim)
        documents_store=[]
    return {"status":"initialized"}

@router.post("/add")
async def add(doc:Document):
    global index,documents_store
    if index is None:
        raise HTTPException(status_code=400,detail="FAISS not initialized")
    chunks=process(doc.text)
    if not chunks:
        return {"status":"no content"}
    embeddings=await embed_texts(chunks)
    index.add(embeddings)
    documents_store.extend(chunks)
    return {"status":"added","chunks":len(chunks)}

@router.post("/query")
async def query(q:Query):
    global index,documents_store
    if index is None:
        raise HTTPException(status_code=400,detail="FAISS not initialized")
    key=f"cache:{q.message}"
    cached=cache.get(key)
    if cached:
        return {"source":"cache","data":pickle.loads(cached)}
    query_vec=await embed_texts([q.message])
    scores,indices=index.search(query_vec,5)
    results=[documents_store[i] for i in indices[0] if i<len(documents_store)]
    await add_cache(q.message,results)
    return {"source":"faiss","data":results}

async def add_cache(query,answer):
    key=f"cache:{query}"
    cache.set(key,pickle.dumps(answer),ex=3600)
