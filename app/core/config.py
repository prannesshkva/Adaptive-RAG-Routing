import os

API_KEY = "naanthaandaleo"
UPLOAD_DIR = "uploaded_docs"
CLOUD_URL = "https://13.235.9.116/chat"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# =====================================================================
# LangSmith Observability & Telemetry Configuration
# =====================================================================
# Support both modern LANGSMITH_* and legacy LANGCHAIN_* environment variables
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "true"))
LANGSMITH_ENDPOINT = os.getenv(
    "LANGSMITH_ENDPOINT",
    os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
)
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", os.getenv("LANGCHAIN_API_KEY", ""))
LANGSMITH_PROJECT = os.getenv(
    "LANGSMITH_PROJECT",
    os.getenv("LANGCHAIN_PROJECT", "adaptive-rag-routing")
)

# Explicitly export into os.environ so all sub-processes, thread pools, and LangSmith client instances inherit them
os.environ["LANGSMITH_TRACING"] = str(LANGSMITH_TRACING).lower()
os.environ["LANGCHAIN_TRACING_V2"] = str(LANGSMITH_TRACING).lower()
os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT
os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT

if LANGSMITH_API_KEY:
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
