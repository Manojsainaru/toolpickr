# SentenceTransformer embedding provider (local, no API key needed)

from typing import List
from toolpickr.embeddings.base import EmbeddingProvider


class SentenceTransformerEmbeddings(EmbeddingProvider):
    """
    Embedding provider using the sentence-transformers library.
    Runs locally — no API key required.

    Args:
        model: The model name from HuggingFace hub (e.g., "all-MiniLM-L6-v2").
    """

    # Maps commonly used models to their output dimensions for fast lookup.
    # If a model isn't here, we detect dimension dynamically on first embed.
    _KNOWN_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-MiniLM-L12-v2": 384,
        "all-mpnet-base-v2": 768,
        "all-distilroberta-v1": 768,
        "paraphrase-MiniLM-L6-v2": 384,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "multi-qa-MiniLM-L6-cos-v1": 384,
    }

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model
        self._model = SentenceTransformer(model)
        self._dimension = self._KNOWN_DIMENSIONS.get(model, None)

    @property
    def dimension(self) -> int:
        """Returns the embedding dimension. Auto-detects if not in the known list."""
        if self._dimension is None:
            # Detect dimension by encoding a dummy string
            dummy = self._model.encode(["test"])
            self._dimension = len(dummy[0])
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        """Embeds a single string into a vector."""
        embedding = self._model.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Embeds a list of strings efficiently using batched encoding."""
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [emb.tolist() for emb in embeddings]
