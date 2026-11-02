import os
import shutil
import asyncio
from typing import Annotated
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pypdf import PdfReader

from app.core.config import UPLOAD_DIR
from app.models.schemas import Document
from app.api.routers import cag, rag, vlrag, multirag, mmrrag, hybridrag, crag

router = APIRouter(prefix="", tags=["Ingestion"])

def parse_pdf_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    return " ".join(pages_text).strip()


async def broadcast_document_to_routers(doc: Document) -> dict:
    """
    Ingests the document into all active production RAG engines concurrently.
    """
    router_map = [
        ("cag", cag),
        ("rag", rag),
        ("vlrag", vlrag),
        ("multirag", multirag),
        ("mmrrag", mmrrag),
        ("hybridrag", hybridrag),
        ("crag", crag),
    ]

    async def _add_safe(name, rtr):
        try:
            res = await rtr.add(doc)
            return name, res.get("status", "added")
        except Exception as e:
            return name, f"error: {str(e)}"

    # Fan out across all 7 strategy indexes in parallel
    tasks = [_add_safe(name, rtr) for name, rtr in router_map]
    results = await asyncio.gather(*tasks)

    return dict(results)


@router.post("/ingest", status_code=status.HTTP_201_CREATED)
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload and index a PDF file across all RAG indices.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported for file ingestion.",
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save file to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Offload blocking PDF reading & text extraction to worker thread
    content = await asyncio.to_thread(parse_pdf_text, file_path)

    if not content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No readable text found in uploaded PDF.",
        )

    doc = Document(text=content, metadata={"source": file.filename})
    statuses = await broadcast_document_to_routers(doc)

    return {
        "status": "success",
        "file": file.filename,
        "indices": statuses,
    }


@router.post("/ingest/text", status_code=status.HTTP_201_CREATED)
async def ingest_text(doc: Document):
    """
    Directly index raw text across all RAG indices.
    """
    if not doc.text or not doc.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty text provided",
        )

    statuses = await broadcast_document_to_routers(doc)
    return {"status": "success", "indices": statuses}


@router.get("/documents")
async def list_documents():
    return {"files": os.listdir(UPLOAD_DIR)}
