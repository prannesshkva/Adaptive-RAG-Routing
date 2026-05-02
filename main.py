from fastapi import FastAPI, UploadFile, File, Header
from pydantic import BaseModel
from pypdf import PdfReader
from cagrag import cag
from qdrantrag import rag
from vectorless import vlrag
from cagrag.cag import Document as CagDoc
from qdrantrag.rag import Document as RagDoc
from vectorless.vlrag import Document as VlDoc
import aiohttp
import os
import shutil
import re

app = FastAPI()

app.include_router(cag.router)
app.include_router(rag.router)
app.include_router(vlrag.router)

API_KEY = "naanthaandaleo"

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

CLOUD_URL = "https://13.235.9.116/chat"

session = None


def clean_response_strict(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<.*?>", "", text)
    text = text.replace("```", "")
    text = text.replace("**", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def optimize_query(query: str) -> str:
    query = query.strip().lower()
    stopwords = {
        "please", "can", "you", "tell", "me", "about",
        "explain", "what", "is", "the", "a", "an"
    }
    words = [w for w in query.split() if w not in stopwords]
    optimized = " ".join(words)
    return optimized if optimized else query


async def call_cloud_llm(prompt: str, model: str):
    try:
        async with session.post(
            CLOUD_URL,
            json={
                "prompt": prompt,
                "model": model
            },
            headers={
                "x-api-key": API_KEY
            },
            timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            data = await resp.json()
            return data.get("response", "")
    except Exception as e:
        print("CLOUD ERROR:", e)
        return "Error contacting cloud model"


class Query(BaseModel):
    message: str
    model: str
    session_id: str
    pdf: str | None = None


@app.on_event("startup")
async def startup():
    global session
    session = aiohttp.ClientSession()
    await cag.init()
    await rag.init()
    await vlrag.init()


@app.on_event("shutdown")
async def shutdown():
    await session.close()


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    reader = PdfReader(file_path)
    content = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            content += text + " "

    if not content.strip():
        return {"status": "error", "message": "No readable text found"}

    await cag.add(CagDoc(text=content, metadata={"source": file.filename}))
    await rag.add(RagDoc(text=content, metadata={"source": file.filename}))
    await vlrag.add(VlDoc(text=content, metadata={"source": file.filename}))

    return {"status": "success"}


@app.get("/documents")
async def list_documents():
    return {"files": os.listdir(UPLOAD_DIR)}


@app.post("/adaptive_query")
async def adaptive_query(q: Query, x_api_key: str = Header(default=None)):

    original_message = q.message.strip()
    optimized_query = optimize_query(original_message)
    q.message = optimized_query
    message_lower = optimized_query.lower()

    if original_message.lower().startswith("aa"):
        clean_msg = original_message[2:].strip()

        if not clean_msg:
            return {"data": "Provide message after AA"}

        prompt = f"Q: {clean_msg}\nAnswer clearly."

        response = await call_cloud_llm(prompt, q.model)
        return {"data": clean_response_strict(response)}

    try:
        if len(optimized_query.split()) < 4:
            result = await vlrag.query(q)
        elif any(word in message_lower for word in ["define", "why"]):
            result = await rag.query(q)
        else:
            result = await cag.query(q)
    except Exception:
        return {"data": "Error retrieving context"}

    context = result.get("data", "")

    if not context.strip():
        return {"data": "No relevant context found"}

    context = context[:2000]

    if q.pdf:
        context += f"\n[Document: {q.pdf}]"

    prompt = f"""
Context:
{context}

Q: {original_message}

Answer ONLY from context.
If missing, say: Insufficient data.
"""

    response = await call_cloud_llm(prompt, q.model)
    final_answer = clean_response_strict(response)

    return {"data": final_answer}
