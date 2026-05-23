from typing import List, Optional
from toolpickr.embeddings.base import EmbeddingProvider

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("google-genai is required. Please install it using 'pip install google-genai'")

class GoogleEmbeddings(EmbeddingProvider):
    # Maps known Google embedding models to their output dimensions
    _DIMENSIONS = {
        "gemini-embedding-001": 3072,
        "text-embedding-004": 768,
    }

    def __init__(self, api_key: str | None = None, model: str = "gemini-embedding-001"):
        """
        Initializes the Google Embedding provider. 
        If api_key is not passed, it automatically looks for GEMINI_API_KEY
        or GOOGLE_API_KEY environment variables (in that order).
        """
        import os
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google Embeddings requires an API key. "
                "Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable, "
                "or pass api_key explicitly."
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model

    @property
    def dimension(self) -> int:
        """Returns the embedding dimension for the configured model."""
        return self._DIMENSIONS.get(self.model, 3072)

    def embed_text(self, text: str) -> List[float]:
        """Embeds a single string into a vector."""
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )
        # response.embeddings is a list. We take the first one and return its values.
        return response.embeddings[0].values

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Embeds a list of strings efficiently by processing them in manageable batches."""
        all_embeddings = []
        
        # Iterate through the texts in increments of batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = self.client.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT"
                )
            )
            
            # Extract the actual values list out of each embedding object and append to results
            batch_embeddings = [emb.values for emb in response.embeddings]
            all_embeddings.extend(batch_embeddings)
        #print("ALL embeddings: ", all_embeddings)
        return all_embeddings