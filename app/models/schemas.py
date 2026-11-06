from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum

class RAGStrategy(str, Enum):
    AUTO = "auto"
    HYBRID = "hybrid"
    CRAG = "crag"
    DENSE = "dense"
    SPARSE = "sparse"
    MULTIRAG = "multirag"
    MMR = "mmr"
    CAG = "cag"

class Query(BaseModel):
    message: str = Field(description="The user's query or natural language question")
    model: str = Field(default="", description="Target LLM model ID if applicable")
    session_id: str = Field(default="", description="Session or conversation thread ID")
    pdf: Optional[str] = Field(default=None, description="Optional uploaded PDF identifier")
    strategy: Optional[str] = Field(
        default=None,
        description="Explicit RAG strategy override: 'hybrid', 'crag', 'dense', 'sparse', 'multirag', 'mmr', 'cag'. If omitted, Aurelio AI Semantic Router dynamically selects the optimal framework."
    )
    explain: bool = Field(default=False, description="Whether to include routing telemetry and confidence scores in the response")

class Document(BaseModel):
    text: str = Field(description="Raw document text content to index across all RAG frameworks")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Arbitrary metadata attributes for the document")
