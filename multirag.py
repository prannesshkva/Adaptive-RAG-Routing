import faiss, re, asyncio
import numpy as np
import nltk
from nltk.corpus import wordnet
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor

nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)

class Query(BaseModel):
    message: str

class Document(BaseModel):
    text: str

router = APIRouter(prefix="/multirag", tags=["MultiRAG"])
embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
index = None
documents_store = []
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
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: embedder.encode(texts, normalize_embeddings=True).astype("float32"))

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lm in syn.lemmas():
            if lm.name().isalpha():
                synonyms.add(lm.name())
    return list(synonyms)

def expand_query(query: str):
    tokens = nltk.word_tokenize(query)
    pos_tags = nltk.pos_tag(tokens)
    queries = [query]
    for word, tag in pos_tags:
        if tag.startswith('N') or tag.startswith('V') or tag.startswith('J'):
            syns = get_synonyms(word)
            for syn in syns[:2]:
                if syn.lower() != word.lower():
                    new_query = query.replace(word, syn)
                    queries.append(new_query)
    return list(set(queries))[:3]

@router.post("/init")
async def init():
    global index, documents_store
    if index is None:
        dim = len((await embed_texts(["test"]))[0])
        index = faiss.IndexFlatIP(dim)
        documents_store = []
    return {"status": "initialized"}

@router.post("/add")
async def add(doc: Document):
    global index, documents_store
    if index is None:
        raise HTTPException(status_code=400, detail="FAISS not initialized")
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
    if index is None:
        raise HTTPException(status_code=400, detail="FAISS not initialized")
    queries = expand_query(q.message)
    query_vecs = await embed_texts(queries)
    all_results = []
    for vec in query_vecs:
        scores, indices = index.search(np.array([vec]), 3)
        for i in indices[0]:
            if i < len(documents_store):
                all_results.append(documents_store[i])
    unique_results = []
    for res in all_results:
        if res not in unique_results:
            unique_results.append(res)
    return {"data": unique_results[:5]}
