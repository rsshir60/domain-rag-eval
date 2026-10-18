"""
tests/test_api.py

Async tests for FastAPI RAG endpoints using httpx AsyncClient.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport
from src.api.main import app


@pytest.fixture(scope="module")
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["indexed_chunks"] > 0


@pytest.mark.asyncio
async def test_rag_query_with_citations(client: AsyncClient):
    payload = {
        "query": "What are CloudScale's remaining performance obligations?",
        "top_k": 2,
        "use_reranker": False,
    }
    resp = await client.post("/rag/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert len(data["citations"]) > 0
    assert data["estimated_tokens"] > 0
    assert data["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_rag_benchmark_endpoint(client: AsyncClient):
    resp = await client.post("/rag/benchmark?config_name=API_Test&top_k=2")
    assert resp.status_code == 200
    data = resp.json()
    assert "hit_rate_at_k" in data
    assert "mrr" in data
    assert "faithfulness" in data
    assert "cost_per_1k_queries" in data
