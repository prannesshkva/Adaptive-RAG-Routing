import aiohttp
import re
from app.core.config import CLOUD_URL, API_KEY

session = None

async def init_session():
    global session
    session = aiohttp.ClientSession()

async def close_session():
    global session
    if session:
        await session.close()

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

async def call_cloud_llm(prompt: str, model: str):
    try:
        async with session.post(
            CLOUD_URL,
            json={"prompt": prompt, "model": model},
            headers={"x-api-key": API_KEY},
            timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            data = await resp.json()
            return data.get("response", "")
    except Exception as e:
        print("CLOUD ERROR:", e)
        return "Error contacting cloud model"
