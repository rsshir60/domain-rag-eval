"""
src/retrieval/bm25_retriever.py

BM25 Okapi lexical retrieval component for exact keyword, ticker, and numeric match.
"""

from __future__ import annotations

import re
from typing import List, Tuple
from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk


def _tokenize(text: str) -> List[str]:
    # Lowercase and split on non-alphanumeric characters, keeping numbers & currency
    return re.findall(r"\b[\$\w\.\-]+\b", text.lower())


class BM25Retriever:
    """Exact lexical matching using BM25Okapi."""

    def __init__(self):
        self.chunks: List[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def index(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if self._bm25 is None or not self.chunks:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]
