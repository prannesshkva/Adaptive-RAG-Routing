from fastapi import APIRouter, Header
from app.models.schemas import Query
from app.services.llm_service import call_cloud_llm, clean_response_strict, optimize_query
from app.services.semantic_decision_router import get_semantic_router
from app.core.telemetry import traceable, get_telemetry_status
from app.api.routers import cag, rag, vlrag, multirag, mmrrag, hybridrag, crag

router = APIRouter()

def fallback_heuristic_routing(query_str: str) -> tuple[str, str, float]:
    """
    Fast rule-based heuristic fallback if the semantic router is initializing.
    """
    msg_lower = query_str.lower().strip()
    words = msg_lower.split()

    if any(trigger in msg_lower for trigger in ["verify", "is it true", "confirm", "fact check", "validate"]):
        return "crag", "Fact-checking intent detected; routed to Corrective RAG.", 0.85
    if any(trigger in msg_lower for trigger in ["define", "why", "what is", "meaning of"]):
        return "rag", "Conceptual definition query; routed to Dense Vector RAG.", 0.80
    if any(trigger in msg_lower for trigger in ["different", "expand", "perspectives", "aspects", "compare"]):
        return "multirag", "Multi-perspective query detected; routed to Multi-Query expansion.", 0.80
    if any(trigger in msg_lower for trigger in ["diverse", "variety", "types of", "range of"]):
        return "mmrrag", "Diversity query detected; routed to Maximal Marginal Relevance.", 0.80
    if any(trigger in msg_lower for trigger in ["how does", "how to", "architecture", "mechanism", "protocol"]) or len(words) >= 6:
        return "hybridrag", "Technical or procedural query; routed to Hybrid RRF.", 0.80
    if any('_' in w for w in words) or (len(words) <= 3 and any(c.isdigit() for c in query_str)):
        return "vlrag", "Literal code or identifier detected; routed to Sparse BM25.", 0.95
    return "cag", "Standard query; routed to Cache-Augmented Generation.", 0.70


@traceable(name="classify_routing_strategy", run_type="chain", tags=["routing", "classifier"])
def classify_routing_strategy(query_str: str, explicit_strategy: str | None = None) -> tuple[str, str, float]:
    """
    Aurelio AI Semantic Decision Router with explicit override and resilient fallback.
    Returns: (strategy_name, reason, confidence)
    """
    # 1. Handle Explicit User / Bot Override
    if explicit_strategy and explicit_strategy.lower().strip() != "auto":
        strategy_clean = explicit_strategy.lower().strip()
        overrides = {
            "hybrid": ("hybridrag", "Explicitly requested Hybrid RRF strategy.", 1.0),
            "hybridrag": ("hybridrag", "Explicitly requested Hybrid RRF strategy.", 1.0),
            "crag": ("crag", "Explicitly requested Corrective RAG strategy.", 1.0),
            "corrective": ("crag", "Explicitly requested Corrective RAG strategy.", 1.0),
            "vlrag": ("vlrag", "Explicitly requested Sparse BM25 strategy.", 1.0),
            "sparse": ("vlrag", "Explicitly requested Sparse BM25 strategy.", 1.0),
            "bm25": ("vlrag", "Explicitly requested Sparse BM25 strategy.", 1.0),
            "rag": ("rag", "Explicitly requested Dense Vector strategy.", 1.0),
            "dense": ("rag", "Explicitly requested Dense Vector strategy.", 1.0),
            "vector": ("rag", "Explicitly requested Dense Vector strategy.", 1.0),
            "multirag": ("multirag", "Explicitly requested Multi-Query expansion strategy.", 1.0),
            "multi": ("multirag", "Explicitly requested Multi-Query expansion strategy.", 1.0),
            "mmr": ("mmrrag", "Explicitly requested MMR Diversity strategy.", 1.0),
            "mmrrag": ("mmrrag", "Explicitly requested MMR Diversity strategy.", 1.0),
            "cag": ("cag", "Explicitly requested Cache-Augmented strategy.", 1.0),
            "cache": ("cag", "Explicitly requested Cache-Augmented strategy.", 1.0)
        }
        if strategy_clean in overrides:
            return overrides[strategy_clean]

    # 2. Aurelio AI Semantic Decision Routing
    try:
        decision = get_semantic_router().route_sync(query_str)
        return decision["strategy"], decision["reason"], decision.get("confidence", 0.0)
    except Exception:
        # Resilient fallback if vector router encounters an unexpected issue
        return fallback_heuristic_routing(query_str)


@router.post("/adaptive_query")
@traceable(name="adaptive_query_pipeline", run_type="chain", tags=["pipeline", "adaptive-rag"])
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
            "decision_engine": "Direct Bypass",
            "reason": "Direct LLM generation requested via prefix."
        }

    # Classify the optimal RAG route using Aurelio AI Semantic Router
    selected_strategy, reason, confidence = classify_routing_strategy(original_message, q.strategy)

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
            "decision_engine": "Aurelio AI Semantic Router",
            "confidence": confidence,
            "error": str(e)
        }

    raw_context = result.get("data", "")
    if not raw_context:
        return {
            "data": "No relevant context found",
            "routing_strategy": selected_strategy,
            "decision_engine": "Aurelio AI Semantic Router",
            "confidence": confidence,
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
            "decision_engine": "Aurelio AI Semantic Router",
            "confidence": confidence,
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
        "decision_engine": "Aurelio AI Semantic Router",
        "confidence": confidence,
        "reason": reason
    }

    if q.explain:
        response_payload["context_snippets"] = raw_context
        response_payload["telemetry"] = get_telemetry_status()
        if "confidence" in result:
            response_payload["retrieval_confidence"] = result["confidence"]

    return response_payload
