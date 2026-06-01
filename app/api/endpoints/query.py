from fastapi import APIRouter, Header
from app.models.schemas import Query
from app.services.llm_service import call_cloud_llm, clean_response_strict, optimize_query
from app.api.routers import cag, rag, vlrag, multirag, mmrrag

router = APIRouter()

@router.post("/adaptive_query")
async def adaptive_query(q: Query, x_api_key: str = Header(default=None)):
    original_message = q.message.strip()
    optimized_query = optimize_query(original_message)
    q.message = optimized_query
    message_lower = optimized_query.lower()

    if original_message.lower().startswith("aa"):
        clean_msg = original_message[2:].strip()
        if not clean_msg:
            return {"data": "Provide message after AA"}
        prompt = f"Q: {clean_msg}\nAnswer clearly."
        response = await call_cloud_llm(prompt, q.model)
        return {"data": clean_response_strict(response)}

    try:
        if len(optimized_query.split()) < 4:
            result = await vlrag.query(q)
        elif any(word in message_lower for word in ["define", "why"]):
            result = await rag.query(q)
        elif any(word in message_lower for word in ["different", "expand"]):
            result = await multirag.query(q)
        elif any(word in message_lower for word in ["diverse", "variety"]):
            result = await mmrrag.query(q)
        else:
            result = await cag.query(q)
    except Exception:
        return {"data": "Error retrieving context"}

    context = result.get("data", "")
    if not context:
        return {"data": "No relevant context found"}
    if isinstance(context, list):
        context = " ".join(context)
        
    context = context[:2000]

    if q.pdf:
        context += f"\n[Document: {q.pdf}]"

    prompt = f"""
Context:
{context}

Q: {original_message}

Answer ONLY from context.
If missing, say: Insufficient data.
"""

    response = await call_cloud_llm(prompt, q.model)
    final_answer = clean_response_strict(response)

    return {"data": final_answer}
