import asyncio
import re
from typing import Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from app.core.telemetry import traceable

_ROUTER_INSTANCE = None


class AurelioSemanticRouter:
    """
    Aurelio AI Semantic Decision Router.
    Uses Aurelio AI's IndexedRouter with BAAI/bge-base-en-v1.5 to map
    user queries to specialized RAG engines in vector space.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self._router = None
        self._executor = ThreadPoolExecutor(max_workers=2)

    def _init_router(self):
        if self._router is None:
            from semantic_router import Route, IndexedRouter
            from semantic_router.encoders import HuggingFaceEncoder

            encoder = HuggingFaceEncoder(name=self.model_name)

            routes = [
                Route(name="crag", utterances=[
                    "verify if this statement is true",
                    "fact check the accuracy of this claim",
                    "is it true that",
                    "confirm whether there is proof",
                    "validate and verify accuracy"
                ]),
                Route(name="hybridrag", utterances=[
                    "how does the system architecture work",
                    "how does the transformer attention mechanism work in modern neural networks",
                    "technical implementation steps and mechanism",
                    "explain the pipeline architecture and protocol steps",
                    "how to implement distributed consensus in production"
                ]),
                Route(name="vlrag", utterances=[
                    "ERR_404_NOT_FOUND",
                    "error code 500 internal server",
                    "part number SKU identifier lookup",
                    "ip address 192.168.1.1",
                    "exact literal identifier code"
                ]),
                Route(name="rag", utterances=[
                    "what is the definition of entropy",
                    "define entropy in thermodynamics",
                    "explain why leaves change color",
                    "conceptual meaning of this concept",
                    "what does this concept mean"
                ]),
                Route(name="multirag", utterances=[
                    "compare different perspectives and trade-offs",
                    "various aspects and multiple viewpoints",
                    "different ways to train language models",
                    "compare and expand on gradient descent"
                ]),
                Route(name="mmrrag", utterances=[
                    "diverse variety of machine learning algorithms",
                    "broad spectrum of types and categories",
                    "taxonomy of options and diverse examples",
                    "variety of renewable energy options",
                    "diverse types of database indexing"
                ])
            ]

            self._router = IndexedRouter(encoder=encoder, routes=routes, top_k=5)

    @traceable(name="aurelio_semantic_routing", run_type="chain", tags=["aurelio-ai", "semantic-router"])
    def route_sync(self, query: str) -> Dict[str, Any]:
        """
        Evaluates the query through Aurelio AI's IndexedRouter.
        Returns selected strategy, confidence, and human-readable reason.
        """
        query_clean = query.strip()
        if not query_clean:
            return {
                "strategy": "cag",
                "confidence": 1.0,
                "reason": "Empty query defaulted to CAG.",
                "engine": "Aurelio AI Semantic Router"
            }

        # Fast literal check for codes or identifiers with underscores/digits
        words = query_clean.split()
        if any('_' in w and len(w) > 3 for w in words):
            return {
                "strategy": "vlrag",
                "confidence": 0.99,
                "reason": "Aurelio Router matched exact identifier syntax; routed to Sparse BM25.",
                "engine": "Aurelio AI Semantic Router"
            }

        self._init_router()
        res = self._router(query_clean)

        strategy = getattr(res, "name", None)
        confidence = float(getattr(res, "confidence", 0.0))

        if not strategy or confidence < 0.35:
            return {
                "strategy": "cag",
                "confidence": round(confidence, 3),
                "reason": "General or conversational inquiry; routed to Cache-Augmented Generation.",
                "engine": "Aurelio AI Semantic Router"
            }

        reasons = {
            "crag": f"Aurelio AI matched fact-checking and verification intent ({round(confidence * 100, 1)}% similarity).",
            "hybridrag": f"Aurelio AI matched technical architecture or procedural mechanism ({round(confidence * 100, 1)}% similarity).",
            "vlrag": f"Aurelio AI matched literal token or keyword lookup ({round(confidence * 100, 1)}% similarity).",
            "rag": f"Aurelio AI matched conceptual definition inquiry ({round(confidence * 100, 1)}% similarity).",
            "multirag": f"Aurelio AI matched multi-perspective or comparative exploration ({round(confidence * 100, 1)}% similarity).",
            "mmrrag": f"Aurelio AI matched taxonomy or diversity exploration ({round(confidence * 100, 1)}% similarity)."
        }

        return {
            "strategy": strategy,
            "confidence": round(confidence, 3),
            "reason": reasons.get(strategy, f"Aurelio AI routed to {strategy}."),
            "engine": "Aurelio AI Semantic Router"
        }

    async def route_async(self, query: str) -> Dict[str, Any]:
        """
        Asynchronous evaluation to keep FastAPI event loop unblocked.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.route_sync(query)
        )


def get_semantic_router() -> AurelioSemanticRouter:
    global _ROUTER_INSTANCE
    if _ROUTER_INSTANCE is None:
        _ROUTER_INSTANCE = AurelioSemanticRouter()
    return _ROUTER_INSTANCE
