import aiohttp
import re
from typing import Optional, Any
from aiohttp import TCPConnector, ClientTimeout
from app.core.config import (
    CLOUD_URL,
    API_KEY,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_DEFAULT_MODEL
)
from app.core.telemetry import traceable

try:
    from openai import AsyncOpenAI
    from langsmith.wrappers import wrap_openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    wrap_openai = lambda client: client

session: aiohttp.ClientSession | None = None
groq_client: Optional[Any] = None


def get_groq_client():
    """
    Returns an AsyncOpenAI client wrapped with LangSmith wrap_openai telemetry,
    pointing to Groq API with openai/gpt-oss-120b.
    """
    global groq_client
    if groq_client is None and _OPENAI_AVAILABLE and GROQ_API_KEY:
        try:
            raw_client = AsyncOpenAI(
                api_key=GROQ_API_KEY,
                base_url=GROQ_BASE_URL
            )
            # Wrap client to capture token usage, inputs, outputs, and latency in LangSmith
            groq_client = wrap_openai(raw_client)
        except Exception:
            groq_client = None
    return groq_client


async def init_session():
    global session
    if session is None or session.closed:
        connector = TCPConnector(
            limit=100,            # Max total simultaneous connections
            limit_per_host=20,    # Max connections per host
            ssl=False             # Bypasses IP SSL certificate verification errors
        )
        session = aiohttp.ClientSession(connector=connector)


async def close_session():
    global session, groq_client
    if session and not session.closed:
        await session.close()
        session = None
    if groq_client and hasattr(groq_client, "close"):
        try:
            await groq_client.close()
        except Exception:
            pass
        groq_client = None


def clean_response_strict(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
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


def extract_fallback_summary(prompt: str) -> str:
    ctx_match = re.search(r"Context:\s*(.*?)\s*Q:", prompt, flags=re.DOTALL)
    if ctx_match:
        ctx_text = ctx_match.group(1).strip()
        if ctx_text:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ctx_text) if s.strip()]
            return " ".join(sentences[:2]) if sentences else ctx_text[:200]
    return "Context retrieved successfully. (Cloud LLM unavailable for final generation)"


@traceable(name="cloud_llm_inference", run_type="llm", tags=["llm", "generation"])
async def call_cloud_llm(prompt: str, model: str) -> str:
    # 1. Primary: High-speed Groq inference with wrap_openai telemetry if GROQ_API_KEY is configured
    client = get_groq_client()
    if client is not None:
        try:
            target_model = model if model and model not in ["default", "llama-3-8b"] else GROQ_DEFAULT_MODEL
            completion = await client.chat.completions.create(
                model=target_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a factual, concise RAG assistant. Answer strictly based on the provided context."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1024
            )
            if completion.choices and completion.choices[0].message.content:
                return completion.choices[0].message.content
        except Exception:
            # Graceful fallback to secondary cloud endpoint if Groq experiences rate limits or network issues
            pass

    # 2. Secondary: Resilient aiohttp cloud session fallback
    global session
    if session is None or session.closed:
        await init_session()

    timeout = ClientTimeout(total=30.0, connect=3.0, sock_read=27.0)

    try:
        async with session.post(
            CLOUD_URL,
            json={"prompt": prompt, "model": model},
            headers={"x-api-key": API_KEY, "Content-Type": "application/json"},
            timeout=timeout,
            ssl=False
        ) as resp:
            if resp.status == 200:
                data = await resp.json(content_type=None)
                return data.get("response", "")
            return extract_fallback_summary(prompt)

    except Exception:
        # Graceful zero-failure fallback
        return extract_fallback_summary(prompt)
