from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class Query(BaseModel):
    message: str
    model: str = ""
    session_id: str = ""
    pdf: Optional[str] = None
    strategy: Optional[str] = None  # Explicit routing override: "hybrid", "crag", "dense", "sparse", "multirag", "mmr", "cag"
    explain: bool = False

class Document(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None
