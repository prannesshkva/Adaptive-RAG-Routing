import faiss, re, asyncio
import numpy as np
import nltk
from fastapi import APIRouter, HTTPException
from concurrent.futures import ThreadPoolExecutor

from app.models.schemas import Query, Document
from app.core.embeddings import get_embedder

router = APIRouter(prefix="/multirag", tags=["MultiRAG"])
embedder = get_embedder()
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
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, lambda: embedder.encode(texts, normalize_embeddings=True).astype("float32"))

def get_synonyms(word: str) -> list:
    synonyms = set()
    try:
        from nltk.corpus import wordnet
        for syn in wordnet.synsets(word):
            for lm in syn.lemmas():
                if lm.name().isalpha() and lm.name().lower() != word.lower():
                    synonyms.add(lm.name().lower().replace("_", " "))
    except Exception:
        pass
    return list(synonyms)

def expand_query(query: str):
    words = query.strip().split()
    queries = [query]
    
    # Try POS tag expansion if available, else simple word synonym lookup
    try:
        tokens = nltk.word_tokenize(query)
        pos_tags = nltk.pos_tag(tokens)
    except Exception:
        pos_tags = [(w, 'NN') for w in words]

    for word, tag in pos_tags:
        if tag.startswith('N') or tag.startswith('V') or tag.startswith('J'):
            syns = get_synonyms(word)
            for syn in syns[:2]:
                new_query = query.replace(word, syn)
                if new_query not in queries:
                    queries.append(new_query)
    return queries[:3]

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
        return {"data": []}
    queries = expand_query(q.message)
    query_vecs = await embed_texts(queries)
    all_results = []
    for vec in query_vecs:
        scores, indices = index.search(np.array([vec]), min(3, len(documents_store)))
        for i in indices[0]:
            if i < len(documents_store):
                all_results.append(documents_store[i])
    unique_results = []
    for res in all_results:
        if res not in unique_results:
            unique_results.append(res)
    return {"data": unique_results[:5]}
