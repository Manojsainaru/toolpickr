# OpenAI embedding provider

import os
from typing import List, Optional
from toolpickr.embeddings.base import EmbeddingProvider


class OpenAIEmbeddings(EmbeddingProvider):
    """
    Embedding provider using the OpenAI Embeddings API.

    Supports all OpenAI embedding models including:
      - text-embedding-3-small  (1536 dims, cheapest)
      - text-embedding-3-large  (3072 dims, most capable)
      - text-embedding-ada-002  (1536 dims, legacy)

    API key is resolved in order:
      1. Explicit `api_key` parameter
      2. OPENAI_API_KEY environment variable

    Args:
        api_key: Optional OpenAI API key. Auto-discovered from OPENAI_API_KEY env var if not provided.
        model:   The embedding model name (default: "text-embedding-3-small").
    """

    # Maps known OpenAI embedding models to their default output dimensions
    _DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    # OpenAI's max batch size for embeddings is 2048 inputs per request
    _MAX_BATCH_SIZE = 2048

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required for OpenAI embeddings. "
                "Install with: pip install openai"
            )

        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI Embeddings requires an API key. "
                "Set OPENAI_API_KEY environment variable, "
                "or pass api_key explicitly."
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model

    @property
    def dimension(self) -> int:
        """Returns the embedding dimension for the configured model."""
        return self._DIMENSIONS.get(self.model, 1536)

    def embed_text(self, text: str) -> List[float]:
        """Embeds a single string into a vector."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str], batch_size: int = 2048) -> List[List[float]]:
        """
        Embeds a list of strings efficiently by processing them in batches.

        OpenAI supports up to 2048 inputs per request, so we batch accordingly.
        """
        batch_size = min(batch_size, self._MAX_BATCH_SIZE)
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            response = self.client.embeddings.create(
                input=batch,
                model=self.model,
            )

            # OpenAI returns embeddings in the same order as the input
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings
