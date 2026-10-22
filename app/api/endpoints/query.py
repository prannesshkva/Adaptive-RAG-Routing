from fastapi import APIRouter, Header
from app.models.schemas import Query
from app.services.llm_service import call_cloud_llm, clean_response_strict, optimize_query
from app.api.routers import cag, rag, vlrag, multirag, mmrrag, hybridrag, crag

router = APIRouter()

def classify_routing_strategy(query_str: str, explicit_strategy: str | None = None) -> tuple[str, str]:
    """
    Intelligent Adaptive RAG Classifier.
    Determines the most effective RAG strategy based on query semantics,
    lexical density, intent markers, and explicit overrides.
    Returns: (strategy_name, reason)
    """
    if explicit_strategy:
        strategy_clean = explicit_strategy.lower().strip()
        if strategy_clean in ["hybrid", "hybridrag"]:
            return "hybridrag", "Explicitly requested Hybrid RRF strategy."
        elif strategy_clean in ["crag", "corrective"]:
            return "crag", "Explicitly requested Corrective RAG strategy."
        elif strategy_clean in ["vlrag", "sparse", "bm25"]:
            return "vlrag", "Explicitly requested Sparse BM25 strategy."
        elif strategy_clean in ["rag", "dense", "vector"]:
            return "rag", "Explicitly requested Dense Vector strategy."
        elif strategy_clean in ["multirag", "multi", "expansion"]:
            return "multirag", "Explicitly requested Multi-Query expansion strategy."
        elif strategy_clean in ["mmr", "mmrrag", "diversity"]:
            return "mmrrag", "Explicitly requested MMR Diversity strategy."
        elif strategy_clean in ["cag", "cache"]:
            return "cag", "Explicitly requested Cache-Augmented strategy."

    msg_lower = query_str.lower().strip()
    words = msg_lower.split()

    # 1. Corrective RAG (CRAG) -> Verification, confirmation, fact-checking queries
    crag_triggers = ["verify", "is it true", "confirm", "fact check", "accurate", "validate", "check if", "is there proof"]
    if any(trigger in msg_lower for trigger in crag_triggers):
        return "crag", "Verification or fact-checking intent detected; routed to Corrective RAG with self-reflection."

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

    # 5. Hybrid RAG (Dense + BM25 Fusion) -> Technical, architectural, procedural, or balanced descriptive queries
    hybrid_triggers = ["how does", "how to", "architecture", "mechanism", "steps", "implement", "process", "pipeline", "protocol"]
    if any(trigger in msg_lower for trigger in hybrid_triggers) or len(words) >= 6:
        return "hybridrag", "Complex or technical query combining concepts and terminology; routed to Hybrid RRF."

    # 6. Sparse Lexical BM25 (VLRAG) -> Exact codes, IDs, quoted phrases, or short queries without questions
    has_code_or_id = any('_' in w or (any(c.isdigit() for c in w) and any(c.isalpha() for c in w)) for w in words)
    has_quotes = '"' in query_str or "'" in query_str
    if has_code_or_id or has_quotes or (len(words) <= 3 and any(c.isdigit() for c in query_str)):
        return "vlrag", "Exact code, ID, or alphanumeric string detected; routed to Sparse BM25."

    # 7. Default to CAG (Cache-Augmented Generation)
    return "cag", "Standard query; routed to Cache-Augmented Generation."


@router.post("/adaptive_query")
async def adaptive_query(q: Query, x_api_key: str = Header(default=None)):
    original_message = q.message.strip()
    optimized_query = optimize_query(original_message)
    q.message = optimized_query

    # Direct generation bypass for "aa" prefix
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

    # Classify the optimal RAG route
    selected_strategy, reason = classify_routing_strategy(original_message, q.strategy)

    try:
        if selected_strategy == "hybridrag":
            result = await hybridrag.query(q)
        elif selected_strategy == "crag":
            result = await crag.query(q)
        elif selected_strategy == "vlrag":
            result = await vlrag.query(q)
        elif selected_strategy == "multirag":
            result = await multirag.query(q)
        elif selected_strategy == "mmrrag":
            result = await mmrrag.query(q)
        elif selected_strategy == "rag":
            result = await rag.query(q)
        else:
            result = await cag.query(q)
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

    # Handle lists of context strings or dicts
    if isinstance(raw_context, list):
        extracted = []
        for item in raw_context:
            if isinstance(item, dict):
                extracted.append(item.get("text", str(item)))
            else:
                extracted.append(str(item))
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
