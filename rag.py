from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import re, asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List

router = APIRouter(prefix="/rag", tags=["RAG"])

class Query(BaseModel):
    message: str

class Document(BaseModel):
    text: str

client = QdrantClient(":memory:")
embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
collection_name = "REX AI"
executor = ThreadPoolExecutor(max_workers=4)

current_id = 0

def get_next_id():
    global current_id
    current_id += 1
    return current_id

def process(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    words = text.lower().strip().split()
    chunks = []
    chunk_size = 200
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

async def embed_texts(texts: List[str]):
    if not texts:
        return []
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor,
        lambda: embedder.encode(texts, normalize_embeddings=True)
    )

async def ensure_collection():
    if not client.collection_exists(collection_name):
        test_vec = await embed_texts(["test"])
        dim = len(test_vec[0])
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

@router.post("/init")
async def init():
    await ensure_collection()
    return {"status": "ready"}

@router.post("/add")
async def add(doc: Document):
    await ensure_collection()
    chunks = process(doc.text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Empty or invalid text")

    embeddings = (await embed_texts(chunks)).tolist()
    points = []

    for i, chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id=get_next_id(),
                vector=embeddings[i],
                payload={"text": chunk}
            )
        )

    client.upsert(collection_name=collection_name, points=points)
    return {"status": "added", "chunks": len(chunks)}

@router.post("/query")
async def query(q: Query):
    await ensure_collection()

    if not q.message or not q.message.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    query_vec = (await embed_texts([q.message]))[0].tolist()

    result = client.query_points(
        collection_name=collection_name,
        query=query_vec,
        limit=5,
        with_payload=True
    )

    points = result.points if hasattr(result, "points") else result

    if not points:
        return {"data": []}

    return {
        "data": [
            {
                "text": p.payload.get("text", ""),
                "score": p.score
            }
            for p in points
        ]
    }
