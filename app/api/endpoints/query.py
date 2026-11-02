from typing import Annotated
from fastapi import APIRouter, Header, HTTPException, status
from app.models.schemas import Query
from app.core.config import API_KEY
from app.services.llm_service import call_cloud_llm, clean_response_strict, optimize_query
from app.api.routers import cag, rag, vlrag, multirag, mmrrag, hybridrag, crag

router = APIRouter(tags=["Adaptive Query"])

# Explicit routing map
STRATEGY_EXECUTORS = {
    "hybridrag": hybridrag.query,
    "crag": crag.query,
    "vlrag": vlrag.query,
    "multirag": multirag.query,
    "mmrrag": mmrrag.query,
    "rag": rag.query,
    "cag": cag.query,
}


def classify_routing_strategy(query_str: str, explicit_strategy: str | None = None) -> tuple[str, str]:
    if explicit_strategy:
        strategy_clean = explicit_strategy.lower().strip()
        overrides = {
            "hybrid": "hybridrag", "hybridrag": "hybridrag",
            "crag": "crag", "corrective": "crag",
            "vlrag": "vlrag", "sparse": "vlrag", "bm25": "vlrag",
            "rag": "rag", "dense": "rag", "vector": "rag",
            "multirag": "multirag", "multi": "multirag", "expansion": "multirag",
            "mmr": "mmrrag", "mmrrag": "mmrrag", "diversity": "mmrrag",
            "cag": "cag", "cache": "cag",
        }
        if strategy_clean in overrides:
            strat = overrides[strategy_clean]
            return strat, f"Explicitly requested {strat} strategy."

    msg_lower = query_str.lower().strip()
    words = msg_lower.split()

    # 1. Corrective RAG (CRAG) -> Verification, confirmation, fact-checking
    crag_triggers = ["verify", "is it true", "confirm", "fact check", "accurate", "validate", "check if", "is there proof"]
    if any(trigger in msg_lower for trigger in crag_triggers):
        return "crag", "Verification or fact-checking intent detected; routed to Corrective RAG."

    # 2. Standard Dense Vector RAG -> Definitions and conceptual inquiries
    dense_triggers = ["define", "why", "what is", "meaning of", "explain"]
    if any(trigger in msg_lower for trigger in dense_triggers):
        return "rag", "Conceptual definition query; routed to Dense Vector RAG."

    # 3. Multi-Query Expansion (MultiRAG) -> Multi-perspective, broad, or comparative queries
    multi_triggers = ["different", "expand", "perspectives", "aspects", "compare", "overview", "various ways"]
    if any(trigger in msg_lower for trigger in multi_triggers):
        return "multirag", "Multi-faceted or comparative query detected; routed to Multi-Query expansion."

    # 4. MMR Diversity (MMRRAG) -> Queries seeking variety, taxonomy, or diverse subsets
    mmr_triggers = ["diverse", "variety", "types of", "range of", "all kinds", "different kinds", "spectrum"]
    if any(trigger in msg_lower for trigger in mmr_triggers):
        return "mmrrag", "Diversity or categorization query detected; routed to Maximal Marginal Relevance (MMR)."

    # 5. Hybrid RAG (Dense + BM25 Fusion) -> Technical, procedural, or 6+ words
    hybrid_triggers = ["how does", "how to", "architecture", "mechanism", "steps", "implement", "process", "pipeline", "protocol"]
    if any(trigger in msg_lower for trigger in hybrid_triggers) or len(words) >= 6:
        return "hybridrag", "Complex or technical query combining concepts and terminology; routed to Hybrid RRF."

    # 6. Sparse Lexical BM25 (VLRAG) -> Exact codes, IDs, quoted phrases, short numeric strings
    has_code_or_id = any('_' in w or (any(c.isdigit() for c in w) and any(c.isalpha() for c in w)) for w in words)
    has_quotes = '"' in query_str or "'" in query_str
    if has_code_or_id or has_quotes or (len(words) <= 3 and any(c.isdigit() for c in query_str)):
        return "vlrag", "Exact code, ID, or alphanumeric string detected; routed to Sparse BM25."

    # 7. Default to CAG
    return "cag", "Standard query; routed to Cache-Augmented Generation."


@router.post("/adaptive_query")
async def adaptive_query(
    q: Query,
    x_api_key: Annotated[str | None, Header()] = None
):
    # Optional Security Key Guard: Enforce if provided or configured
    if API_KEY and x_api_key and x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid X-API-KEY header.",
        )

    original_message = q.message.strip()
    if not original_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty query message.",
        )

    # 1. Direct LLM bypass check (Raw message before mutation)
    if original_message.lower().startswith("aa"):
        clean_msg = original_message[2:].strip()
        if not clean_msg:
            return {"data": "Provide message after AA", "routing_strategy": "direct_llm"}
        prompt = f"Q: {clean_msg}\nAnswer clearly."
        response = await call_cloud_llm(prompt, q.model)
        return {
            "data": clean_response_strict(response),
            "routing_strategy": "direct_llm",
            "reason": "Direct LLM generation requested via prefix."
        }

    # 2. Routing Decision based on unmutated query
    selected_strategy, reason = classify_routing_strategy(original_message, q.strategy)

    # 3. Optimize query for retrieval
    q.message = optimize_query(original_message)

    # 4. Dispatch query to selected strategy router
    executor = STRATEGY_EXECUTORS.get(selected_strategy, cag.query)
    try:
        result = await executor(q)
    except Exception as e:
        return {
            "data": "Error retrieving context",
            "routing_strategy": selected_strategy,
            "error": str(e)
        }

    raw_context = result.get("data", "")
    if not raw_context:
        return {
            "data": "No relevant context found",
            "routing_strategy": selected_strategy,
            "reason": reason
        }

    # 5. Normalize context payload
    if isinstance(raw_context, list):
        extracted = [
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in raw_context
        ]
        context_str = "\n".join(extracted)
    else:
        context_str = str(raw_context)

    # Check for Corrective RAG insufficient context signal
    if "INSUFFICIENT CONTEXT" in context_str:
        return {
            "data": "Insufficient data in documents to reliably answer this query.",
            "routing_strategy": selected_strategy,
            "confidence": result.get("confidence", 0.0),
            "reason": reason
        }

    # Truncate and attach optional document metadata
    context_str = context_str[:3000]
    if q.pdf:
        context_str += f"\n[Document: {q.pdf}]"

    prompt = f"""Context:
{context_str}

Q: {original_message}

Answer ONLY from context.
If missing, say: Insufficient data.
"""

    response = await call_cloud_llm(prompt, q.model)
    final_answer = clean_response_strict(response)

    response_payload = {
        "data": final_answer,
        "routing_strategy": selected_strategy,
        "reason": reason
    }

    if q.explain:
        response_payload["context_snippets"] = raw_context
        if "confidence" in result:
            response_payload["confidence"] = result["confidence"]

    return response_payload
