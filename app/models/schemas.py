from pydantic import BaseModel

class Query(BaseModel):
    message: str
    model: str = ""
    session_id: str = ""
    pdf: str | None = None

class Document(BaseModel):
    text: str
    metadata: dict | None = None
