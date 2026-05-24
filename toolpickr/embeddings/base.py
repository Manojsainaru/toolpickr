from abc import ABC, abstractmethod
from typing import List

class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    All embedding providers must implement the methods defined in this class.
    """
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors produced by this provider."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text string."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text strings."""
        pass
