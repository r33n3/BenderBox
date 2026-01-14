"""
ChromaDB Vector Store for BenderBox

Provides semantic search capabilities using ChromaDB with:
- Multiple collections for different content types
- Async interface for non-blocking operations
- Integration with embedding model
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from benderbox.config import get_config
from benderbox.nlp.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document for vector store."""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @classmethod
    def create(
        cls,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> "Document":
        """Create a new document with auto-generated ID."""
        return cls(
            id=doc_id or str(uuid.uuid4()),
            content=content,
            metadata=metadata or {},
        )


@dataclass
class SearchResult:
    """Search result from vector store."""

    document: Document
    score: float
    distance: float


# Default collections for BenderBox
DEFAULT_COLLECTIONS = [
    "vulnerabilities",
    "jailbreaks",
    "reports",
    "code_patterns",
    "knowledge",
]


class VectorStore:
    """
    ChromaDB-based vector store for semantic search.

    Manages multiple collections:
    - vulnerabilities: Security vulnerability patterns
    - jailbreaks: Jailbreak techniques and examples
    - reports: Analysis report summaries
    - code_patterns: Suspicious code patterns
    - knowledge: General security knowledge
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        """
        Initialize VectorStore.

        Args:
            db_path: Path to ChromaDB storage. Uses config default if not provided.
            embedding_model: Embedding model instance. Creates one if not provided.
        """
        config = get_config().storage

        self.db_path = db_path or config.vector_store_path
        self._embedding_model = embedding_model
        self._client = None
        self._collections: Dict[str, Any] = {}

    async def initialize(self) -> None:
        """
        Explicitly initialize the vector store.

        Creates ChromaDB client and default collections.
        Safe to call multiple times (idempotent).

        This method can be called upfront to validate dependencies
        and create collections. If not called, initialization happens
        automatically on first use.

        Raises:
            VectorStoreError: If ChromaDB is not installed or initialization fails.
        """
        await asyncio.get_event_loop().run_in_executor(
            None, self._ensure_initialized
        )
        logger.debug("VectorStore initialized")

    def _ensure_initialized(self) -> None:
        """Ensure ChromaDB client is initialized."""
        if self._client is not None:
            return

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise VectorStoreError(
                "chromadb is not installed. Install with:\n"
                "  pip install chromadb"
            )

        # Ensure directory exists
        Path(self.db_path).mkdir(parents=True, exist_ok=True)

        # Initialize client with persistent storage
        self._client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.db_path,
                anonymized_telemetry=False,
            )
        )

        logger.info(f"ChromaDB initialized at: {self.db_path}")

        # Initialize default collections
        for collection_name in DEFAULT_COLLECTIONS:
            self._get_or_create_collection(collection_name)

    def _get_embedding_model(self) -> EmbeddingModel:
        """Get or create embedding model."""
        if self._embedding_model is None:
            self._embedding_model = EmbeddingModel()
        return self._embedding_model

    def _get_or_create_collection(self, name: str):
        """Get or create a collection."""
        self._ensure_initialized()

        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    async def add_document(
        self,
        doc: Document,
        collection: str = "knowledge",
    ) -> str:
        """
        Add a document to the vector store.

        Args:
            doc: Document to add.
            collection: Collection name.

        Returns:
            Document ID.
        """
        coll = self._get_or_create_collection(collection)

        # Generate embedding if not provided
        if doc.embedding is None:
            embedding_model = self._get_embedding_model()
            doc.embedding = await asyncio.get_event_loop().run_in_executor(
                None, embedding_model.embed, doc.content
            )

        # Add to collection
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: coll.add(
                ids=[doc.id],
                embeddings=[doc.embedding],
                documents=[doc.content],
                metadatas=[doc.metadata],
            ),
        )

        logger.debug(f"Added document {doc.id} to collection {collection}")
        return doc.id

    async def add_documents(
        self,
        docs: List[Document],
        collection: str = "knowledge",
    ) -> List[str]:
        """
        Add multiple documents to the vector store.

        Args:
            docs: Documents to add.
            collection: Collection name.

        Returns:
            List of document IDs.
        """
        if not docs:
            return []

        coll = self._get_or_create_collection(collection)
        embedding_model = self._get_embedding_model()

        # Generate embeddings for docs that don't have them
        texts_to_embed = []
        indices_to_embed = []

        for i, doc in enumerate(docs):
            if doc.embedding is None:
                texts_to_embed.append(doc.content)
                indices_to_embed.append(i)

        if texts_to_embed:
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None, embedding_model.embed_batch, texts_to_embed
            )
            for i, embedding in zip(indices_to_embed, embeddings):
                docs[i].embedding = embedding

        # Add all documents
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: coll.add(
                ids=[doc.id for doc in docs],
                embeddings=[doc.embedding for doc in docs],
                documents=[doc.content for doc in docs],
                metadatas=[doc.metadata for doc in docs],
            ),
        )

        logger.debug(f"Added {len(docs)} documents to collection {collection}")
        return [doc.id for doc in docs]

    async def search(
        self,
        query: str,
        collection: str = "knowledge",
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents.

        Args:
            query: Search query text.
            collection: Collection to search.
            top_k: Number of results to return.
            filter_metadata: Optional metadata filter.

        Returns:
            List of search results sorted by relevance.
        """
        coll = self._get_or_create_collection(collection)
        embedding_model = self._get_embedding_model()

        # Generate query embedding
        query_embedding = await asyncio.get_event_loop().run_in_executor(
            None, embedding_model.embed, query
        )

        # Build query params
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if filter_metadata:
            query_params["where"] = filter_metadata

        # Execute query
        results = await asyncio.get_event_loop().run_in_executor(
            None, lambda: coll.query(**query_params)
        )

        # Convert to SearchResult objects
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0.0
                # Convert distance to similarity score (cosine distance to similarity)
                score = 1.0 - distance

                doc = Document(
                    id=doc_id,
                    content=results["documents"][0][i] if results["documents"] else "",
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                )

                search_results.append(
                    SearchResult(document=doc, score=score, distance=distance)
                )

        return search_results

    async def get_document(
        self,
        doc_id: str,
        collection: str = "knowledge",
    ) -> Optional[Document]:
        """
        Get a document by ID.

        Args:
            doc_id: Document ID.
            collection: Collection name.

        Returns:
            Document if found, None otherwise.
        """
        coll = self._get_or_create_collection(collection)

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: coll.get(
                ids=[doc_id],
                include=["documents", "metadatas", "embeddings"],
            ),
        )

        if not result["ids"]:
            return None

        return Document(
            id=result["ids"][0],
            content=result["documents"][0] if result["documents"] else "",
            metadata=result["metadatas"][0] if result["metadatas"] else {},
            embedding=result["embeddings"][0] if result["embeddings"] else None,
        )

    async def delete_document(
        self,
        doc_id: str,
        collection: str = "knowledge",
    ) -> bool:
        """
        Delete a document by ID.

        Args:
            doc_id: Document ID.
            collection: Collection name.

        Returns:
            True if deleted, False if not found.
        """
        coll = self._get_or_create_collection(collection)

        # Check if exists
        existing = await self.get_document(doc_id, collection)
        if existing is None:
            return False

        await asyncio.get_event_loop().run_in_executor(
            None, lambda: coll.delete(ids=[doc_id])
        )

        logger.debug(f"Deleted document {doc_id} from collection {collection}")
        return True

    async def list_collections(self) -> List[str]:
        """
        List all collections.

        Returns:
            List of collection names.
        """
        self._ensure_initialized()
        collections = await asyncio.get_event_loop().run_in_executor(
            None, self._client.list_collections
        )
        return [c.name for c in collections]

    async def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """
        Get statistics for a collection.

        Args:
            collection: Collection name.

        Returns:
            Dictionary with collection statistics.
        """
        coll = self._get_or_create_collection(collection)

        count = await asyncio.get_event_loop().run_in_executor(
            None, coll.count
        )

        return {
            "name": collection,
            "count": count,
        }

    async def clear_collection(self, collection: str) -> int:
        """
        Clear all documents from a collection.

        Args:
            collection: Collection name.

        Returns:
            Number of documents deleted.
        """
        coll = self._get_or_create_collection(collection)

        # Get count before clearing
        count = await asyncio.get_event_loop().run_in_executor(
            None, coll.count
        )

        if count > 0:
            # Get all IDs and delete
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: coll.get(include=[])
            )
            if result["ids"]:
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: coll.delete(ids=result["ids"])
                )

        logger.info(f"Cleared {count} documents from collection {collection}")
        return count

    def persist(self) -> None:
        """Persist the database to disk."""
        if self._client is not None:
            self._client.persist()
            logger.info("Vector store persisted to disk")


class VectorStoreError(Exception):
    """Exception for vector store errors."""

    pass
