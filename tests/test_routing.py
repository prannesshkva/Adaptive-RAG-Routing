import sys
import os
import asyncio

# Ensure app is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.schemas import Query, Document
from app.api.endpoints.query import classify_routing_strategy
from app.api.routers import rag, vlrag, cag, multirag, mmrrag, hybridrag, crag


def test_classification():
    test_cases = [
        ("verify if quantum encryption is unbreakable", "crag"),
        ("is it true that light speed is constant", "crag"),
        ("ERR_404_NOT_FOUND", "vlrag"),
        ("ip 192.168.1.1", "vlrag"),
        ("different ways to train language models", "multirag"),
        ("compare and expand on gradient descent", "multirag"),
        ("diverse types of database indexing", "mmrrag"),
        ("variety of renewable energy options", "mmrrag"),
        ("how does the transformer attention mechanism work in modern neural networks", "hybridrag"),
        ("architecture and protocol steps for distributed systems", "hybridrag"),
        ("define entropy", "rag"),
        ("why do leaves change color", "rag"),
        ("standard topic", "cag"),
    ]

    print("\n--- Testing Adaptive Routing Classification ---")
    all_passed = True
    for query_str, expected in test_cases:
        actual, reason = classify_routing_strategy(query_str)
        status = "PASS" if actual == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"[{status}] '{query_str}' -> Expected: {expected}, Got: {actual}")

    # Explicit override test
    override_actual, _ = classify_routing_strategy("hello", explicit_strategy="hybrid")
    assert override_actual == "hybridrag", f"Override failed: {override_actual}"
    print("[PASS] Explicit strategy override -> hybridrag")

    assert all_passed, "Some classification tests failed!"
    print("All classification tests passed successfully!")


async def test_routers_end_to_end():
    print("\n--- Testing Document Ingestion & Routing End-to-End ---")
    sample_text = (
        "The Attention Is All You Need paper introduced the Transformer architecture. "
        "Transformers rely entirely on self-attention mechanisms without recurrence or convolution. "
        "In contrast, recurrent neural networks process sequences step by step. "
        "Key hyper-parameters include hidden dimension, number of attention heads, and learning rate. "
        "The ERR_CODE_TRANSFORMER_01 occurs when head dimensions do not divide evenly."
    )
    doc = Document(text=sample_text, metadata={"topic": "transformer_architecture"})

    # 1. Test Ingestion across both new routers
    print("Ingesting into Hybrid RAG...")
    await hybridrag.init()
    h_res = await hybridrag.add(doc)
    print("Hybrid add result:", h_res)
    assert h_res.get("status") == "added"

    print("Ingesting into Corrective RAG...")
    await crag.init()
    c_res = await crag.add(doc)
    print("CRAG add result:", c_res)
    assert c_res.get("status") == "added"

    # 2. Query Hybrid RAG
    q_hybrid = Query(message="how does the transformer self-attention mechanism work without recurrence")
    h_query_res = await hybridrag.query(q_hybrid)
    print("Hybrid Query Result (count):", len(h_query_res.get("data", [])))
    assert len(h_query_res.get("data", [])) > 0, "Hybrid retrieval returned empty!"
    print("Hybrid top snippet:", h_query_res["data"][0][:80], "...")

    # 3. Query Corrective RAG (High confidence query)
    q_crag = Query(message="verify if transformer architecture uses self-attention")
    c_query_res = await crag.query(q_crag)
    print("CRAG Query Status:", c_query_res.get("status"), "Confidence:", c_query_res.get("confidence"))
    assert c_query_res.get("status") in ["correct", "ambiguous_reformulated"]
    assert len(c_query_res.get("data", [])) > 0

    # 4. Query Corrective RAG (Low confidence query out of domain)
    q_crag_irrelevant = Query(message="verify recipe for french chocolate soufflé baking temperature")
    c_irrelevant_res = await crag.query(q_crag_irrelevant)
    print("CRAG Irrelevant Query Status:", c_irrelevant_res.get("status"), "Confidence:", c_irrelevant_res.get("confidence"))
    assert c_irrelevant_res.get("status") == "incorrect"
    assert "INSUFFICIENT CONTEXT" in c_irrelevant_res["data"][0]
    print("[PASS] CRAG correctly detected out-of-domain query and prevented hallucination!")

    print("\nAll end-to-end integration tests passed successfully!")


if __name__ == "__main__":
    test_classification()
    asyncio.run(test_routers_end_to_end())
