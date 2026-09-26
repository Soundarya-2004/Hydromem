"""Search engine module for HydroMem.

Supports:
1. Fast keyword matching with token overlap scoring.
2. Vector similarity search via optional fastembed integration.
3. Pure Python cosine similarity calculation (zero heavy dependencies).
"""

import math
from typing import Any, Dict, List, Optional, Union

# Optional FastEmbed integration
try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculate cosine similarity between two numerical vectors.

    Implemented using Python standard library math (no numpy required).

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        float: Cosine similarity in [-1.0, 1.0].
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    sim = dot / (norm_a * norm_b)
    # Clip to valid cosine range
    return float(max(-1.0, min(1.0, sim)))


class EmbeddingEngine:
    """Optional wrapper for local text embeddings using FastEmbed."""

    def __init__(self, model_name: str = "BAAI/bge-small-en") -> None:
        """Initialize the embedding model.

        Args:
            model_name: Model identifier for FastEmbed.

        Raises:
            ImportError: If fastembed is not installed.
        """
        if not FASTEMBED_AVAILABLE:
            raise ImportError(
                "fastembed is required for vector embeddings. "
                "Install with: pip install 'hydromem[embed]' or pip install fastembed"
            )
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed_text(self, text: str) -> List[float]:
        """Compute numerical embedding vector for a single string.

        Args:
            text: Text to embed.

        Returns:
            List[float]: Embedding vector.
        """
        generator = self._model.embed([text])
        embedding_array = next(iter(generator))
        return [float(x) for x in embedding_array]


def keyword_match_score(query: str, text: str) -> float:
    """Compute normalized keyword overlap relevance score between query and text.

    Args:
        query: Search query string.
        text: Memory text.

    Returns:
        float: Relevance score in [0.0, 1.0].
    """
    if not query or not text:
        return 0.0

    query_lower = query.lower()
    text_lower = text.lower()

    # Exact substring match earns high score
    if query_lower in text_lower:
        return 1.0

    query_words = [w for w in query_lower.split() if w]
    if not query_words:
        return 0.0

    matches = sum(1 for w in query_words if w in text_lower)
    return float(matches / len(query_words))
