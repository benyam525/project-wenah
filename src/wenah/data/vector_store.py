"""
Vector store integration using ChromaDB.

Provides storage and retrieval of law document embeddings for RAG pipeline.
Optional dependency - provides stub implementation if not installed.
"""

from pathlib import Path
from typing import Any

from wenah.config import settings

# Lazy import for optional heavy dependency
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None
    ChromaSettings = None
    CHROMADB_AVAILABLE = False


class VectorStore:
    """
    ChromaDB-based vector store for civil rights law documents.

    Handles document storage, embedding, and similarity search for the
    RAG pipeline.
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
    ):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory for ChromaDB persistence.
                              Defaults to config setting.
            collection_name: Name of the ChromaDB collection.
                            Defaults to config setting.
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb is not installed. Install with: pip install chromadb")

        self.persist_directory = Path(persist_directory or settings.chroma_persist_directory)
        self.collection_name = collection_name or settings.chroma_collection_name

        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Civil rights law documents for compliance analysis"},
        )

    @classmethod
    def is_available(cls) -> bool:
        """Check if ChromaDB is available."""
        return CHROMADB_AVAILABLE

    @property
    def collection(self):
        """Get the underlying ChromaDB collection."""
        return self._collection

    def add_documents(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: List of unique document IDs
        """
        self._collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def add_documents_with_auto_embed(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        """
        Add documents and let ChromaDB handle embedding.

        Uses ChromaDB's default embedding function.

        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of unique document IDs
        """
        self._collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def query(
        self,
        query_texts: list[str] | None = None,
        query_embeddings: list[list[float]] | None = None,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Query the vector store for similar documents.

        Args:
            query_texts: Text queries (will be embedded automatically)
            query_embeddings: Pre-computed query embeddings
            n_results: Number of results to return
            where: Metadata filter
            where_document: Document content filter
            include: What to include in results (documents, metadatas, distances)

        Returns:
            Query results with documents, metadatas, and distances
        """
        include = include or ["documents", "metadatas", "distances"]

        return self._collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include,
        )

    def query_with_scores(
        self,
        query_text: str,
        n_results: int = 5,
        category_filter: str | None = None,
    ) -> list[tuple[str, dict[str, Any], float]]:
        """
        Query and return documents with relevance scores.

        Args:
            query_text: The query text
            n_results: Number of results to return
            category_filter: Optional category to filter by

        Returns:
            List of (document_text, metadata, relevance_score) tuples
        """
        where = {"category": category_filter} if category_filter else None

        results = self.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Combine results into tuples
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        # Convert distance to similarity score (1 - normalized_distance)
        # ChromaDB uses L2 distance by default
        scored_results = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            # Convert distance to similarity (higher is better)
            # This is a simple conversion; adjust based on your needs
            similarity = 1 / (1 + dist)
            scored_results.append((doc, meta, similarity))

        return scored_results

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """
        Get a specific document by ID.

        Args:
            doc_id: The document ID

        Returns:
            Document data or None if not found
        """
        results = self._collection.get(
            ids=[doc_id],
            include=["documents", "metadatas"],
        )

        if results["ids"]:
            return {
                "id": results["ids"][0],
                "document": results["documents"][0] if results["documents"] else None,
                "metadata": results["metadatas"][0] if results["metadatas"] else None,
            }
        return None

    def update_document(
        self,
        doc_id: str,
        document: str | None = None,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Update an existing document.

        Args:
            doc_id: The document ID to update
            document: New document text (optional)
            embedding: New embedding (optional)
            metadata: New metadata (optional)
        """
        self._collection.update(
            ids=[doc_id],
            documents=[document] if document else None,
            embeddings=[embedding] if embedding else None,
            metadatas=[metadata] if metadata else None,
        )

    def delete_documents(self, ids: list[str]) -> None:
        """
        Delete documents by ID.

        Args:
            ids: List of document IDs to delete
        """
        self._collection.delete(ids=ids)

    def count(self) -> int:
        """
        Get the total number of documents in the collection.

        Returns:
            Number of documents
        """
        return self._collection.count()

    def reset(self) -> None:
        """
        Reset the collection (delete all documents).

        Warning: This permanently deletes all data.
        """
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"description": "Civil rights law documents for compliance analysis"},
        )

    def get_all_categories(self) -> list[str]:
        """
        Get all unique categories in the collection.

        Returns:
            List of unique category values
        """
        # Get all documents with just metadata
        results = self._collection.get(include=["metadatas"])
        categories = set()
        for meta in results.get("metadatas", []):
            if meta and "category" in meta:
                categories.add(meta["category"])
        return list(categories)

    def get_documents_by_category(
        self,
        category: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get all documents in a specific category.

        Args:
            category: The category to filter by
            limit: Maximum number of documents to return

        Returns:
            List of document dictionaries
        """
        results = self._collection.get(
            where={"category": category},
            include=["documents", "metadatas"],
            limit=limit,
        )

        documents = []
        for i, doc_id in enumerate(results.get("ids", [])):
            documents.append(
                {
                    "id": doc_id,
                    "document": results["documents"][i] if results["documents"] else None,
                    "metadata": results["metadatas"][i] if results["metadatas"] else None,
                }
            )

        return documents


# Singleton instance for convenience
_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """
    Get the singleton vector store instance.

    Returns:
        VectorStore: The shared vector store instance
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
