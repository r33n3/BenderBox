"""
Embedding Model Integration for BenderBox

Provides text embedding generation using sentence-transformers with:
- Offline mode support for air-gapped deployments
- Embedding caching to avoid recomputation
- Batch processing for efficiency
"""

import hashlib
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

from benderbox.config import EmbeddingConfig, get_config

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Text embedding model using sentence-transformers.

    Supports offline operation by loading models from local cache.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        offline_mode: Optional[bool] = None,
    ):
        """
        Initialize EmbeddingModel.

        Args:
            model_name_or_path: Model name (HuggingFace) or local path.
            cache_dir: Directory for caching downloaded models.
            offline_mode: If True, only use locally cached models.
        """
        config = get_config().embedding

        self.model_name_or_path = model_name_or_path or config.model_name_or_path
        self.cache_dir = cache_dir or config.cache_dir
        self.offline_mode = offline_mode if offline_mode is not None else config.offline_mode

        self._model = None
        self._dimension: Optional[int] = None

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise EmbeddingError(
                "sentence-transformers is not installed. Install with:\n"
                "  pip install sentence-transformers"
            )

        # Check for local model first
        local_path = Path(self.cache_dir) / self.model_name_or_path.replace("/", "_")
        if local_path.exists():
            logger.info(f"Loading embedding model from local cache: {local_path}")
            self._model = SentenceTransformer(str(local_path))
        elif self.offline_mode:
            # Try loading by name from default cache
            try:
                self._model = SentenceTransformer(
                    self.model_name_or_path,
                    cache_folder=self.cache_dir,
                )
            except Exception as e:
                raise EmbeddingError(
                    f"Embedding model not found locally and offline mode is enabled.\n"
                    f"Model: {self.model_name_or_path}\n"
                    f"Cache dir: {self.cache_dir}\n\n"
                    f"To download the model, run with offline_mode=False or:\n"
                    f"  python -c \"from sentence_transformers import SentenceTransformer; "
                    f"SentenceTransformer('{self.model_name_or_path}')\"\n\n"
                    f"Original error: {e}"
                )
        else:
            logger.info(f"Downloading embedding model: {self.model_name_or_path}")
            self._model = SentenceTransformer(
                self.model_name_or_path,
                cache_folder=self.cache_dir,
            )

        # Get embedding dimension
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            f"Embedding model loaded: {self.model_name_or_path} "
            f"(dimension: {self._dimension})"
        )

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text.

        Returns:
            Embedding vector as list of floats.
        """
        self._ensure_loaded()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts.
            batch_size: Batch size for processing.

        Returns:
            List of embedding vectors.
        """
        self._ensure_loaded()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        self._ensure_loaded()
        return self._dimension

    def save_local(self, path: Optional[str] = None) -> str:
        """
        Save model to local path for offline use.

        Args:
            path: Output path. Defaults to cache_dir/model_name.

        Returns:
            Path where model was saved.
        """
        self._ensure_loaded()

        if path is None:
            path = str(
                Path(self.cache_dir) / self.model_name_or_path.replace("/", "_")
            )

        Path(path).mkdir(parents=True, exist_ok=True)
        self._model.save(path)
        logger.info(f"Model saved to: {path}")
        return path

    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        loaded = self._model is not None
        return {
            "model_name_or_path": self.model_name_or_path,
            "cache_dir": self.cache_dir,
            "offline_mode": self.offline_mode,
            "loaded": loaded,
            "dimension": self._dimension if loaded else None,
        }


class EmbeddingCache:
    """
    Cache for embeddings to avoid recomputation.

    Uses content hash as key for cache lookup.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        max_size: Optional[int] = None,
        persist_path: Optional[str] = None,
    ):
        """
        Initialize EmbeddingCache.

        Args:
            embedding_model: Embedding model to use.
            max_size: Maximum cache size. Defaults to config value.
            persist_path: Optional path for persistent cache.
        """
        config = get_config().embedding

        self.embedding_model = embedding_model
        self.max_size = max_size or config.cache_size
        self.persist_path = persist_path

        # LRU cache
        self._cache: OrderedDict[str, List[float]] = OrderedDict()

        # Load persisted cache if available
        if persist_path:
            self._load_cache()

    def _hash_text(self, text: str) -> str:
        """Generate hash for text content."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _load_cache(self) -> None:
        """Load cache from disk."""
        if not self.persist_path:
            return

        cache_file = Path(self.persist_path)
        if not cache_file.exists():
            return

        try:
            import json

            with open(cache_file, "r") as f:
                data = json.load(f)
            self._cache = OrderedDict(data.get("cache", {}))
            logger.info(f"Loaded {len(self._cache)} cached embeddings")
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")

    def _save_cache(self) -> None:
        """Save cache to disk."""
        if not self.persist_path:
            return

        try:
            import json

            cache_file = Path(self.persist_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            with open(cache_file, "w") as f:
                json.dump({"cache": dict(self._cache)}, f)
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")

    def get(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding for text.

        Args:
            text: Input text.

        Returns:
            Cached embedding or None if not found.
        """
        key = self._hash_text(text)
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def embed(self, text: str) -> List[float]:
        """
        Get embedding for text, using cache if available.

        Args:
            text: Input text.

        Returns:
            Embedding vector.
        """
        # Check cache
        cached = self.get(text)
        if cached is not None:
            return cached

        # Generate embedding
        embedding = self.embedding_model.embed(text)

        # Add to cache
        self.put(text, embedding)

        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts, using cache where available.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors.
        """
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cached = self.get(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = self.embedding_model.embed_batch(uncached_texts)

            for i, embedding in zip(uncached_indices, new_embeddings):
                results[i] = embedding
                self.put(texts[i], embedding)

        return results

    def put(self, text: str, embedding: List[float]) -> None:
        """
        Add embedding to cache.

        Args:
            text: Input text.
            embedding: Embedding vector.
        """
        key = self._hash_text(text)

        # Evict if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        self._cache[key] = embedding

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        if self.persist_path:
            self._save_cache()

    def save(self) -> None:
        """Save cache to disk."""
        self._save_cache()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "persist_path": self.persist_path,
        }


class EmbeddingError(Exception):
    """Exception for embedding-related errors."""

    pass
