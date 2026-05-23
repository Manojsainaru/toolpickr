# Ensemble retriever — combines semantic (FAISS) and BM25 with weighted score fusion

from typing import List
from toolpickr.retrieval.flat import FlatRetriever
from toolpickr.retrieval.bm25 import BM25Retriever
from toolpickr.vectorstores.base import SearchResult


class EnsembleRetriever:
    """
    Combines semantic (embedding + FAISS) and BM25 (lexical) retrievers
    using weighted score fusion.

    Each retriever's scores are min-max normalized to [0, 1] before combining,
    so they're on the same scale regardless of their raw score ranges.

    Args:
        semantic_retriever: The FlatRetriever (embedding-based cosine similarity).
        bm25_retriever:     The BM25Retriever (lexical matching).
        semantic_weight:    Weight for semantic scores (default: 0.5).
        bm25_weight:        Weight for BM25 scores (default: 0.5).
        debug:              Enable debug logging.
    """

    def __init__(
        self,
        semantic_retriever: FlatRetriever,
        bm25_retriever: BM25Retriever,
        semantic_weight: float = 0.5,
        bm25_weight: float = 0.5,
        debug: bool = False,
    ):
        self.semantic_retriever = semantic_retriever
        self.bm25_retriever = bm25_retriever
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.debug = debug

    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"[EnsembleRetriever - Debug] {msg}")

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Retrieves tools using both semantic and BM25 retrievers,
        fuses scores with configurable weights, and returns the top_k results.

        The fusion process:
        1. Get results from both retrievers (each asks for more than top_k to
           ensure good coverage after fusion).
        2. Min-max normalize scores from each retriever to [0, 1].
        3. Combine: final_score = semantic_weight * norm_semantic + bm25_weight * norm_bm25
        4. Sort by final_score and return top_k.

        Args:
            query:  Natural language query string.
            top_k:  Number of final results to return.

        Returns:
            List of SearchResult with fused scores, sorted descending.
        """
        # Fetch more candidates than top_k from each retriever for better fusion
        fetch_k = max(top_k * 3, 20)

        self._log(f"Semantic retrieval (top {fetch_k})...")
        semantic_results = self.semantic_retriever.retrieve(query, top_k=fetch_k)

        self._log(f"BM25 retrieval (top {fetch_k})...")
        bm25_results = self.bm25_retriever.retrieve(query, top_k=fetch_k)

        # Normalize scores
        semantic_normalized = self._normalize_scores(semantic_results)
        bm25_normalized = self._normalize_scores(bm25_results)

        self._log(
            f"Fusing {len(semantic_normalized)} semantic + {len(bm25_normalized)} BM25 results "
            f"(weights: semantic={self.semantic_weight}, bm25={self.bm25_weight})"
        )

        # Build a combined score map: tool_name -> weighted score
        combined_scores: dict[str, float] = {}

        for tool_name, score in semantic_normalized.items():
            combined_scores[tool_name] = (
                combined_scores.get(tool_name, 0.0)
                + self.semantic_weight * score
            )

        for tool_name, score in bm25_normalized.items():
            combined_scores[tool_name] = (
                combined_scores.get(tool_name, 0.0)
                + self.bm25_weight * score
            )

        # Sort by combined score and take top_k
        sorted_tools = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results = [
            SearchResult(tool_name=name, score=score)
            for name, score in sorted_tools
        ]

        self._log(f"Ensemble returning {len(results)} results.")
        if self.debug:
            for r in results:
                self._log(f"  {r.tool_name}: {r.score:.4f}")

        return results

    @staticmethod
    def _normalize_scores(results: List[SearchResult]) -> dict[str, float]:
        """
        Min-max normalizes scores to [0, 1].
        Returns a dict of tool_name -> normalized_score.
        """
        if not results:
            return {}

        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        normalized = {}
        for r in results:
            if score_range > 0:
                normalized[r.tool_name] = (r.score - min_score) / score_range
            else:
                # All scores are the same — assign 1.0 to all
                normalized[r.tool_name] = 1.0

        return normalized
