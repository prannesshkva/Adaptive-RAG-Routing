from fastapi import APIRouter, UploadFile, File
import os
import shutil
from pypdf import PdfReader
from app.core.config import UPLOAD_DIR
from app.models.schemas import Document
from app.api.routers import cag, rag, vlrag, multirag, mmrrag

router = APIRouter()

@router.post("/ingest")
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

    doc = Document(text=content, metadata={"source": file.filename})
    
    await cag.add(doc)
    await rag.add(doc)
    await vlrag.add(doc)
    await multirag.add(doc)
    await mmrrag.add(doc)

    return {"status": "success"}

@router.get("/documents")
async def list_documents():
    return {"files": os.listdir(UPLOAD_DIR)}
