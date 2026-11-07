import os
from typing import Any, Callable, Dict, Optional
from app.core import config  # Ensures os.environ variables for LangSmith are loaded

# Attempt safe import of LangSmith traceable
try:
    from langsmith import traceable, Client
    _LANGSMITH_AVAILABLE = True
except ImportError:
    _LANGSMITH_AVAILABLE = False
    Client = None

    def traceable(
        name: Optional[str] = None,
        run_type: Optional[str] = None,
        tags: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Callable:
        """Fallback transparent decorator when langsmith is not installed."""
        def decorator(func: Callable) -> Callable:
            return func
        return decorator


def get_telemetry_status() -> Dict[str, Any]:
    """
    Returns current LangSmith telemetry status and environment configuration.
    """
    return {
        "provider": "LangSmith",
        "library_installed": _LANGSMITH_AVAILABLE,
        "tracing_enabled": os.getenv("LANGSMITH_TRACING", "true").lower() == "true",
        "endpoint": os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        "project": os.getenv("LANGSMITH_PROJECT", "adaptive-rag-routing"),
        "has_api_key": bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")),
    }


def get_langsmith_client() -> Optional[Any]:
    """
    Returns a configured LangSmith client instance if an API key is present.
    """
    if _LANGSMITH_AVAILABLE and Client is not None:
        api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
        if api_key:
            try:
                return Client(
                    api_url=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
                    api_key=api_key
                )
            except Exception:
                return None
    return None
