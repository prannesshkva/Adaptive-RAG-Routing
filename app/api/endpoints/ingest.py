from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import shutil
from pypdf import PdfReader
from app.core.config import UPLOAD_DIR
from app.models.schemas import Document
from app.api.routers import cag, rag, vlrag, multirag, mmrrag, hybridrag, crag

router = APIRouter()

async def broadcast_document_to_routers(doc: Document) -> dict:
    """
    Ingests the document into all active production RAG engines simultaneously.
    """
    statuses = {}

    for name, rtr in [
        ("cag", cag),
        ("rag", rag),
        ("vlrag", vlrag),
        ("multirag", multirag),
        ("mmrrag", mmrrag),
        ("hybridrag", hybridrag),
        ("crag", crag),
    ]:
        try:
            res = await rtr.add(doc)
            statuses[name] = res.get("status", "added")
        except Exception as e:
            statuses[name] = f"error: {str(e)}"

    return statuses

@router.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload and index a PDF file across all RAG indices.
    """
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

    doc = Document(text=content, metadata={"source": file.filename})
    statuses = await broadcast_document_to_routers(doc)

    return {"status": "success", "file": file.filename, "indices": statuses}

@router.post("/ingest/text")
async def ingest_text(doc: Document):
    """
    Directly index raw text across all RAG indices.
    """
    if not doc.text or not doc.text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided")

    statuses = await broadcast_document_to_routers(doc)
    return {"status": "success", "indices": statuses}

@router.get("/documents")
async def list_documents():
    return {"files": os.listdir(UPLOAD_DIR)}
