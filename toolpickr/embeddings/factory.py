# Embedding provider factory — parses "provider/model" strings and auto-discovers API keys.

import os
from typing import Optional
from toolpickr.embeddings.base import EmbeddingProvider

# Default embedding model (local, no API key required)
DEFAULT_EMBEDDING_MODEL = "sentence_transformers/all-MiniLM-L6-v2"

# Maps provider names to the env vars checked for API keys (in order of priority)
_API_KEY_ENV_VARS = {
    "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "cohere": ["COHERE_API_KEY"],
    "voyage": ["VOYAGE_API_KEY"],
}


def _resolve_api_key(provider: str, explicit_key: Optional[str] = None) -> Optional[str]:
    """
    Resolves the API key for a provider.
    Priority: explicit_key > environment variable > None.
    """
    if explicit_key:
        return explicit_key

    env_var_names = _API_KEY_ENV_VARS.get(provider, [])
    for var_name in env_var_names:
        key = os.environ.get(var_name)
        if key:
            return key

    return None


def parse_model_string(model_string: str) -> tuple[str, str]:
    """
    Parses a 'provider/model' string into (provider, model_name).

    Examples:
        "google/gemini-embedding-001"        → ("google", "gemini-embedding-001")
        "sentence_transformers/all-MiniLM-L6-v2" → ("sentence_transformers", "all-MiniLM-L6-v2")
        "openai/text-embedding-3-small"      → ("openai", "text-embedding-3-small")
    """
    if "/" not in model_string:
        raise ValueError(
            f"Invalid model string: '{model_string}'. "
            f"Expected format: 'provider/model' (e.g., 'google/gemini-embedding-001')."
        )
    provider, model_name = model_string.split("/", 1)
    return provider.strip().lower(), model_name.strip()


def create_embedding_provider(
    model_string: str = DEFAULT_EMBEDDING_MODEL,
    api_key: Optional[str] = None,
) -> EmbeddingProvider:
    """
    Creates an embedding provider from a 'provider/model' string.

    API keys are resolved in order:
      1. Explicit `api_key` parameter
      2. Environment variable (auto-discovered based on provider)
      3. None (for local providers like sentence_transformers)

    Args:
        model_string: Format "provider/model" (e.g., "google/gemini-embedding-001").
        api_key:      Optional explicit API key. Overrides env var auto-discovery.

    Returns:
        An initialized EmbeddingProvider instance.

    Raises:
        ValueError: If the provider is unsupported or API key is required but not found.
    """
    provider, model_name = parse_model_string(model_string)
    resolved_key = _resolve_api_key(provider, api_key)

    if provider == "sentence_transformers":
        from toolpickr.embeddings.sentence_transformers import SentenceTransformerEmbeddings
        return SentenceTransformerEmbeddings(model=model_name)

    elif provider == "google":
        from toolpickr.embeddings.google import GoogleEmbeddings
        if not resolved_key:
            raise ValueError(
                f"API key required for Google embeddings. "
                f"Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable, "
                f"or pass api_key explicitly."
            )
        return GoogleEmbeddings(api_key=resolved_key, model=model_name)

    elif provider == "openai":
        from toolpickr.embeddings.openai import OpenAIEmbeddings
        if not resolved_key:
            raise ValueError(
                f"API key required for OpenAI embeddings. "
                f"Set OPENAI_API_KEY environment variable, "
                f"or pass api_key explicitly."
            )
        return OpenAIEmbeddings(api_key=resolved_key, model=model_name)

    elif provider == "cohere":
        # TODO: Implement Cohere embeddings provider
        raise NotImplementedError(
            f"Cohere embedding provider is not yet implemented. "
            f"Contributions welcome!"
        )

    else:
        raise ValueError(
            f"Unsupported embedding provider: '{provider}'. "
            f"Supported: 'sentence_transformers', 'google', 'openai', 'cohere'."
        )
