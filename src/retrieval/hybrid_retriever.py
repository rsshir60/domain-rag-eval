"""
src/retrieval/hybrid_retriever.py

Hybrid retriever combining BM25 lexical search and Dense vector search
using Reciprocal Rank Fusion (RRF), followed by an optional Cross-Encoder reranking step.

RRF formula:
    RRF_score(d) = sum_{retriever in [dense, bm25]} ( 1.0 / (k + rank(d)) )
    where k = 60 by standard IR practice.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple
from loguru import logger
from sentence_transformers import CrossEncoder

from src.ingestion.chunker import Chunk
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever


class HybridRetriever:
    """
    Hybrid Lexical-Dense Retriever with RRF and Cross-Encoder Reranking.
    """

    def __init__(
        self,
        dense_model_name: str = "all-MiniLM-L6-v2",
        reranker_model_name: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rrf_k: int = 60,
    ):
        self.dense_retriever = DenseRetriever(model_name=dense_model_name)
        self.bm25_retriever = BM25Retriever()
        self.rrf_k = rrf_k
        self.reranker: Optional[CrossEncoder] = None

        if reranker_model_name:
            try:
                logger.info(f"Loading Cross-Encoder reranker: {reranker_model_name} ...")
                self.reranker = CrossEncoder(reranker_model_name)
            except Exception as e:
                logger.warning(f"Could not load cross-encoder {reranker_model_name}: {e}. Running without reranker.")
                self.reranker = None

    def index(self, chunks: List[Chunk]) -> None:
        self.dense_retriever.index(chunks)
        self.bm25_retriever.index(chunks)
        logger.info(f"Hybrid index built with {len(chunks)} chunks.")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        dense_k: int = 10,
        bm25_k: int = 10,
        use_reranker: bool = True,
    ) -> List[Tuple[Chunk, float, dict]]:
        """
        Executes hybrid retrieval:
          1. Retrieve top-N dense & BM25 candidates
          2. Compute Reciprocal Rank Fusion (RRF) scores
          3. Optionally re-rank top candidates with Cross-Encoder
        Returns list of (Chunk, final_score, score_breakdown).
        """
        t0 = time.perf_counter()

        # Step 1: Candidate Generation
        dense_hits = self.dense_retriever.search(query, top_k=dense_k)
        bm25_hits = self.bm25_retriever.search(query, top_k=bm25_k)

        # Step 2: Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}
        chunks_by_id: Dict[str, Chunk] = {}
        metadata_by_id: Dict[str, dict] = {}

        # Process dense ranks
        for rank, (chunk, score) in enumerate(dense_hits, start=1):
            cid = chunk.chunk_id
            chunks_by_id[cid] = chunk
            rrf_score = 1.0 / (self.rrf_k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_score
            metadata_by_id[cid] = {
                "dense_rank": rank,
                "dense_score": round(score, 4),
                "bm25_rank": None,
                "bm25_score": None,
            }

        # Process BM25 ranks
        for rank, (chunk, score) in enumerate(bm25_hits, start=1):
            cid = chunk.chunk_id
            chunks_by_id[cid] = chunk
            rrf_score = 1.0 / (self.rrf_k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_score
            if cid not in metadata_by_id:
                metadata_by_id[cid] = {
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": rank,
                    "bm25_score": round(score, 4),
                }
            else:
                metadata_by_id[cid]["bm25_rank"] = rank
                metadata_by_id[cid]["bm25_score"] = round(score, 4)

        for cid in rrf_scores:
            metadata_by_id[cid]["rrf_score"] = round(rrf_scores[cid], 5)

        # Sort by RRF score
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = [
            (chunks_by_id[cid], score, metadata_by_id[cid])
            for cid, score in sorted_candidates[: top_k * 2]
        ]

        # Step 3: Cross-Encoder Reranking
        if use_reranker and self.reranker and top_candidates:
            pairs = [[query, c.text] for c, _, _ in top_candidates]
            ce_scores = self.reranker.predict(pairs)
            reranked = []
            for i, (chunk, _, meta) in enumerate(top_candidates):
                meta["rerank_score"] = round(float(ce_scores[i]), 4)
                reranked.append((chunk, float(ce_scores[i]), meta))
            # Sort by cross-encoder score
            reranked = sorted(reranked, key=lambda x: x[1], reverse=True)[:top_k]
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.debug(f"Hybrid retrieval with reranker executed in {elapsed_ms:.2f}ms")
            return reranked

        # If no reranker, return top_k by RRF score
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"Hybrid retrieval (RRF only) executed in {elapsed_ms:.2f}ms")
        return [(c, s, m) for c, s, m in top_candidates[:top_k]]
