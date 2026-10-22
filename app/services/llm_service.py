import aiohttp
import re
from app.core.config import CLOUD_URL, API_KEY

session = None

async def init_session():
    global session
    if session is None or session.closed:
        session = aiohttp.ClientSession()

async def close_session():
    global session
    if session and not session.closed:
        await session.close()
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

def extract_fallback_summary(prompt: str) -> str:
    """
    Offline fallback: extracts the best matching sentence from the context
    if cloud LLM endpoint is unreachable.
    """
    ctx_match = re.search(r"Context:\s*(.*?)\s*Q:", prompt, flags=re.DOTALL)
    if ctx_match:
        ctx_text = ctx_match.group(1).strip()
        if ctx_text:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ctx_text) if s.strip()]
            return " ".join(sentences[:2]) if sentences else ctx_text[:200]
    return "Context retrieved successfully. (Cloud LLM unavailable for final generation)"

async def call_cloud_llm(prompt: str, model: str):
    global session
    if session is None or session.closed:
        session = aiohttp.ClientSession()

    try:
        async with session.post(
            CLOUD_URL,
            json={"prompt": prompt, "model": model},
            headers={"x-api-key": API_KEY},
            timeout=aiohttp.ClientTimeout(total=5)
        ) as resp:
            data = await resp.json()
            return data.get("response", "")
    except Exception as e:
        return extract_fallback_summary(prompt)
