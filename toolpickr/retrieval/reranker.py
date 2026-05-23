# Reranker — re-scores candidates using a cross-encoder model

import os
from typing import List, Optional
from toolpickr.vectorstores.base import SearchResult

# Default reranker model (local, no API key required)
DEFAULT_RERANKER_MODEL = "cross_encoder/ms-marco-MiniLM-L-6-v2"

# Maps provider names to the env vars checked for API keys
_RERANKER_API_KEY_ENV_VARS = {
    "jinaai": ["JINA_API_KEY"],
    "cohere": ["COHERE_API_KEY"],
}


def _resolve_reranker_api_key(provider: str, explicit_key: Optional[str] = None) -> Optional[str]:
    """Resolves the API key for a reranker provider."""
    if explicit_key:
        return explicit_key
    env_var_names = _RERANKER_API_KEY_ENV_VARS.get(provider, [])
    for var_name in env_var_names:
        key = os.environ.get(var_name)
        if key:
            return key
    return None


def parse_reranker_model_string(model_string: str) -> tuple[str, str]:
    """
    Parses a 'provider/model' string for reranker models.

    Examples:
        "cross_encoder/ms-marco-MiniLM-L-6-v2"  → ("cross_encoder", "ms-marco-MiniLM-L-6-v2")
        "jinaai/jina-reranker-v2-base-multilingual" → ("jinaai", "jina-reranker-v2-base-multilingual")
    """
    if "/" not in model_string:
        raise ValueError(
            f"Invalid reranker model string: '{model_string}'. "
            f"Expected format: 'provider/model' (e.g., 'cross_encoder/ms-marco-MiniLM-L-6-v2')."
        )
    provider, model_name = model_string.split("/", 1)
    return provider.strip().lower(), model_name.strip()


class Reranker:
    """
    Re-scores retrieval candidates using a cross-encoder model.

    Takes the top_p candidates from the ensemble retriever and re-ranks them,
    returning the top_k best results based on cross-encoder relevance scores.

    Supports:
      - "cross_encoder/model_name" — uses sentence-transformers CrossEncoder (local)
      - Future: "jinaai/...", "cohere/..." etc.

    Args:
        model_string:  Format "provider/model" (e.g., "cross_encoder/ms-marco-MiniLM-L-6-v2").
        api_key:       Optional explicit API key (overrides env var auto-discovery).
        debug:         Enable debug logging.
    """

    def __init__(
        self,
        model_string: str = DEFAULT_RERANKER_MODEL,
        api_key: Optional[str] = None,
        debug: bool = False,
    ):
        self.debug = debug
        self.model_string = model_string

        provider, model_name = parse_reranker_model_string(model_string)
        self.provider = provider
        self.model_name = model_name

        resolved_key = _resolve_reranker_api_key(provider, api_key)

        if provider == "cross_encoder":
            self._init_cross_encoder(model_name)
        else:
            raise ValueError(
                f"Unsupported reranker provider: '{provider}'. "
                f"Supported: 'cross_encoder'. "
                f"Example: 'cross_encoder/ms-marco-MiniLM-L-6-v2'."
            )

    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"[Reranker - Debug] {msg}")

    def _init_cross_encoder(self, model_name: str) -> None:
        """Initializes a local cross-encoder model from sentence-transformers."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking. "
                "Install with: pip install sentence-transformers"
            )

        self._log(f"Loading cross-encoder model: {model_name}")
        # CrossEncoder expects the full model path, e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2"
        full_model_name = f"cross-encoder/{model_name}"
        self._model = CrossEncoder(full_model_name)

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        tool_texts: dict[str, str],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Re-ranks candidate tools using the cross-encoder model.

        Args:
            query:       The original user query.
            candidates:  List of SearchResult from the ensemble retriever (top_p).
            tool_texts:  Dict mapping tool_name -> rendered tool text for scoring.
            top_k:       Number of final results to return after reranking.

        Returns:
            List of SearchResult re-scored and sorted by the reranker, trimmed to top_k.
        """
        if not candidates:
            return []

        self._log(f"Reranking {len(candidates)} candidates for query: '{query}'")

        # Build query-document pairs for the cross-encoder
        pairs = []
        valid_candidates = []
        for candidate in candidates:
            text = tool_texts.get(candidate.tool_name, "")
            if text:
                pairs.append((query, text))
                valid_candidates.append(candidate)

        if not pairs:
            self._log("No valid candidates with tool texts found.")
            return candidates[:top_k]

        # Score all pairs with the cross-encoder
        scores = self._model.predict(pairs)

        # Combine with candidates and sort by reranker score
        scored_candidates = list(zip(valid_candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        results = []
        for candidate, score in scored_candidates[:top_k]:
            results.append(
                SearchResult(
                    tool_name=candidate.tool_name,
                    score=float(score),
                )
            )

        self._log(f"Reranker returning {len(results)} results.")
        if self.debug:
            for r in results:
                self._log(f"  {r.tool_name}: {r.score:.4f}")

        return results
