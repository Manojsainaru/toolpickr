# BM25 lexical retriever for tool texts

from typing import List
from toolpickr.vectorstores.base import SearchResult


class BM25Retriever:
    """
    BM25 lexical retriever over tool text representations.

    Uses rank_bm25 for Okapi BM25 scoring. Built during the build() phase
    alongside the FAISS index, and used as the lexical component of the
    ensemble retriever.
    """

    def __init__(self, debug: bool = False):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 is required for BM25 retrieval. "
                "Install with: pip install rank-bm25"
            )

        self.debug = debug
        self._bm25 = None
        self._tool_names: List[str] = []
        self._tokenized_corpus: List[List[str]] = []

    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"[BM25Retriever - Debug] {msg}")

    def build(self, tool_names: List[str], tool_texts: List[str]) -> None:
        """
        Builds the BM25 index from tool texts.

        Args:
            tool_names: List of tool names (parallel to tool_texts).
            tool_texts: List of rendered tool text strings to index.
        """
        from rank_bm25 import BM25Okapi

        self._tool_names = tool_names
        # Simple whitespace + lowercase tokenization
        self._tokenized_corpus = [self._tokenize(text) for text in tool_texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self._log(f"BM25 index built with {len(tool_names)} documents.")

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Retrieves the top-k tools matching the query using BM25 scoring.

        Args:
            query:  Natural language query string.
            top_k:  Number of results to return.

        Returns:
            List of SearchResult with BM25 scores.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() first.")

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get indices sorted by score (descending)
        scored_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for idx in scored_indices:
            if scores[idx] > 0:  # Only include positive-scoring results
                results.append(
                    SearchResult(
                        tool_name=self._tool_names[idx],
                        score=float(scores[idx]),
                    )
                )

        self._log(f"BM25 retrieval found {len(results)} results for query: '{query}'")
        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + lowercase tokenization."""
        return text.lower().split()
