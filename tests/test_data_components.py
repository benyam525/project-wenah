"""
Tests for data layer components (embeddings, vector store).

Tests are primarily mocked to avoid external dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from wenah.data.embeddings import (
    EmbeddingGenerator,
    LawDocumentChunker,
    get_embedding_generator,
)
from wenah.data.vector_store import (
    VectorStore,
    get_vector_store,
)


# =============================================================================
# EmbeddingGenerator Tests
# =============================================================================


class TestEmbeddingGenerator:
    """Tests for the EmbeddingGenerator class."""

    @patch("wenah.data.embeddings.SentenceTransformer")
    def test_init_with_model_name(self, mock_st):
        """Test initialization with custom model name."""
        generator = EmbeddingGenerator(model_name="custom-model")
        assert generator.model_name == "custom-model"

    @patch("wenah.data.embeddings.SentenceTransformer")
    def test_init_with_default_model(self, mock_st):
        """Test initialization with default model from settings."""
        generator = EmbeddingGenerator()
        assert generator.model_name is not None

    @patch("wenah.data.embeddings.SentenceTransformer")
    def test_lazy_model_loading(self, mock_st):
        """Test model is not loaded until accessed."""
        generator = EmbeddingGenerator(model_name="test-model")
        # Model should not be loaded yet
        mock_st.assert_not_called()

        # Access model property
        _ = generator.model
        mock_st.assert_called_once_with("test-model")

    @patch("wenah.data.embeddings.SentenceTransformer")
    def test_embed_text(self, mock_st):
        """Test embedding a single text."""
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2, 0.3])
        mock_st.return_value = mock_model

        generator = EmbeddingGenerator(model_name="test-model")
        result = generator.embed_text("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_model.encode.assert_called_once()

    @patch("wenah.data.embeddings.SentenceTransformer")
    def test_embed_texts(self, mock_st):
        """Test embedding multiple texts."""
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(
            tolist=lambda: [[0.1, 0.2], [0.3, 0.4]]
        )
        mock_st.return_value = mock_model

        generator = EmbeddingGenerator(model_name="test-model")
        result = generator.embed_texts(["text1", "text2"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]

    @patch("wenah.data.embeddings.SentenceTransformer")
    def test_embed_documents(self, mock_st):
        """Test adding embeddings to documents."""
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(
            tolist=lambda: [[0.1, 0.2], [0.3, 0.4]]
        )
        mock_st.return_value = mock_model

        generator = EmbeddingGenerator(model_name="test-model")
        docs = [
            {"id": "1", "content": "text1"},
            {"id": "2", "content": "text2"},
        ]

        result = generator.embed_documents(docs, text_field="content")

        assert "embedding" in result[0]
        assert "embedding" in result[1]
        assert result[0]["embedding"] == [0.1, 0.2]

    @patch("wenah.data.embeddings.SentenceTransformer")
    def test_embedding_dimension(self, mock_st):
        """Test getting embedding dimension."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        generator = EmbeddingGenerator(model_name="test-model")
        dim = generator.embedding_dimension

        assert dim == 384


class TestEmbeddingGeneratorSingleton:
    """Tests for embedding generator singleton."""

    @patch("wenah.data.embeddings.SentenceTransformer")
    def test_get_embedding_generator(self, mock_st):
        """Test singleton returns same instance."""
        # Reset singleton
        import wenah.data.embeddings as emb_module
        emb_module._embedding_generator = None

        gen1 = get_embedding_generator()
        gen2 = get_embedding_generator()

        assert gen1 is gen2


# =============================================================================
# LawDocumentChunker Tests
# =============================================================================


class TestLawDocumentChunker:
    """Tests for the LawDocumentChunker class."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        chunker = LawDocumentChunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        chunker = LawDocumentChunker(chunk_size=500, chunk_overlap=100)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 100

    def test_chunk_law_document_basic(self):
        """Test chunking a basic law document."""
        chunker = LawDocumentChunker()

        law_data = {
            "law": {
                "id": "test-law",
                "name": "Test Law",
                "category": "employment",
                "summary": "This is a test law summary.",
            }
        }

        chunks = chunker.chunk_law_document(law_data)

        assert len(chunks) >= 1
        assert chunks[0]["metadata"]["law_id"] == "test-law"
        assert "Test Law" in chunks[0]["content"]

    def test_chunk_law_document_with_protected_classes(self):
        """Test chunking law with protected classes."""
        chunker = LawDocumentChunker()

        law_data = {
            "law": {
                "id": "test-law",
                "name": "Test Law",
                "category": "employment",
                "protected_classes": [
                    {"id": "race", "name": "Race", "description": "Racial discrimination"},
                    {"id": "gender", "name": "Gender", "description": "Gender discrimination"},
                ],
            }
        }

        chunks = chunker.chunk_law_document(law_data)

        # Should have chunks for each protected class
        pc_chunks = [c for c in chunks if c["metadata"]["chunk_type"] == "protected_class"]
        assert len(pc_chunks) == 2

    def test_chunk_law_document_with_prohibited_practices(self):
        """Test chunking law with prohibited practices."""
        chunker = LawDocumentChunker()

        law_data = {
            "law": {
                "id": "test-law",
                "name": "Test Law",
                "category": "employment",
                "prohibited_practices": [
                    {
                        "id": "pp-001",
                        "name": "Discrimination",
                        "severity": "critical",
                        "description": "Direct discrimination is prohibited.",
                    },
                ],
            }
        }

        chunks = chunker.chunk_law_document(law_data)

        pp_chunks = [c for c in chunks if c["metadata"]["chunk_type"] == "prohibited_practice"]
        assert len(pp_chunks) >= 1
        assert pp_chunks[0]["metadata"].get("severity") == "critical"

    def test_chunk_law_document_with_safe_harbors(self):
        """Test chunking law with safe harbors."""
        chunker = LawDocumentChunker()

        law_data = {
            "law": {
                "id": "test-law",
                "name": "Test Law",
                "category": "employment",
                "safe_harbors": [
                    {
                        "id": "sh-001",
                        "name": "BFOQ",
                        "description": "Bona fide occupational qualification defense.",
                    },
                ],
            }
        }

        chunks = chunker.chunk_law_document(law_data)

        sh_chunks = [c for c in chunks if c["metadata"]["chunk_type"] == "safe_harbor"]
        assert len(sh_chunks) == 1

    def test_chunk_law_document_with_ai_considerations(self):
        """Test chunking law with AI considerations."""
        chunker = LawDocumentChunker()

        law_data = {
            "law": {
                "id": "test-law",
                "name": "Test Law",
                "category": "employment",
                "ai_ml_considerations": [
                    {
                        "id": "ai-001",
                        "name": "Algorithmic Bias",
                        "description": "AI systems must be tested for bias.",
                    },
                ],
            }
        }

        chunks = chunker.chunk_law_document(law_data)

        ai_chunks = [c for c in chunks if c["metadata"]["chunk_type"] == "ai_consideration"]
        assert len(ai_chunks) == 1

    def test_chunk_law_document_with_key_cases(self):
        """Test chunking law with key cases."""
        chunker = LawDocumentChunker()

        law_data = {
            "law": {
                "id": "test-law",
                "name": "Test Law",
                "category": "employment",
                "key_cases": [
                    {
                        "name": "Test v. Company",
                        "citation": "123 F.3d 456",
                        "holding": "Important holding",
                        "relevance": "Relevant to AI bias",
                    },
                ],
            }
        }

        chunks = chunker.chunk_law_document(law_data)

        case_chunks = [c for c in chunks if c["metadata"]["chunk_type"] == "case_law"]
        assert len(case_chunks) == 1

    def test_split_if_needed_short_content(self):
        """Test split returns single chunk for short content."""
        chunker = LawDocumentChunker(chunk_size=100)

        result = chunker._split_if_needed("Short content")

        assert len(result) == 1
        assert result[0] == "Short content"

    def test_split_if_needed_long_content(self):
        """Test split divides long content."""
        chunker = LawDocumentChunker(chunk_size=50, chunk_overlap=10)

        long_content = "This is a much longer content. " * 10

        result = chunker._split_if_needed(long_content)

        assert len(result) > 1

    def test_format_protected_class(self):
        """Test formatting protected class."""
        chunker = LawDocumentChunker()

        pc = {
            "id": "race",
            "name": "Race",
            "description": "Racial discrimination",
            "examples": ["Example 1", "Example 2"],
        }

        result = chunker._format_protected_class(pc, "Test Law")

        assert "Test Law" in result
        assert "Race" in result
        assert "Example 1" in result

    def test_format_prohibited_practice(self):
        """Test formatting prohibited practice."""
        chunker = LawDocumentChunker()

        pp = {
            "name": "Disparate Treatment",
            "severity": "critical",
            "description": "Direct discrimination",
            "elements": ["Intent", "Protected class"],
        }

        result = chunker._format_prohibited_practice(pp, "Test Law")

        assert "Test Law" in result
        assert "critical" in result
        assert "Intent" in result

    def test_format_safe_harbor(self):
        """Test formatting safe harbor."""
        chunker = LawDocumentChunker()

        sh = {
            "name": "BFOQ",
            "description": "Bona fide occupational qualification",
            "applies_to": ["employers"],
            "conditions": ["Must be necessary", "No alternative"],
        }

        result = chunker._format_safe_harbor(sh, "Test Law")

        assert "Test Law" in result
        assert "BFOQ" in result
        assert "employers" in result

    def test_format_ai_consideration(self):
        """Test formatting AI consideration."""
        chunker = LawDocumentChunker()

        ai = {
            "name": "Bias Testing",
            "description": "Must test for bias",
            "risks": ["Disparate impact", "Proxy discrimination"],
            "requirements": ["Regular testing", "Documentation"],
        }

        result = chunker._format_ai_consideration(ai, "Test Law")

        assert "Test Law" in result
        assert "Bias Testing" in result
        assert "Disparate impact" in result

    def test_format_key_case(self):
        """Test formatting key case."""
        chunker = LawDocumentChunker()

        case = {
            "name": "Test v. Corp",
            "citation": "123 F.3d 456",
            "holding": "Important decision",
            "relevance": "Applies to AI",
        }

        result = chunker._format_key_case(case, "Test Law")

        assert "Test Law" in result
        assert "Test v. Corp" in result
        assert "123 F.3d 456" in result

    def test_create_chunk_generates_id(self):
        """Test chunk creation generates proper ID."""
        chunker = LawDocumentChunker()

        chunk = chunker._create_chunk(
            content="Test content",
            law_id="law-001",
            law_name="Test Law",
            category="employment",
            section="summary",
            chunk_type="overview",
        )

        assert chunk["id"] == "law-001-summary"
        assert chunk["content"] == "Test content"
        assert chunk["metadata"]["law_id"] == "law-001"

    def test_create_chunk_with_subsection(self):
        """Test chunk creation with subsection."""
        chunker = LawDocumentChunker()

        chunk = chunker._create_chunk(
            content="Test content",
            law_id="law-001",
            law_name="Test Law",
            category="employment",
            section="protected_classes",
            subsection="race",
            chunk_type="protected_class",
        )

        assert chunk["id"] == "law-001-protected_classes-race"
        assert chunk["metadata"]["subsection"] == "race"


# =============================================================================
# VectorStore Tests
# =============================================================================


class TestVectorStore:
    """Tests for the VectorStore class."""

    @patch("wenah.data.vector_store.chromadb")
    @patch("wenah.data.vector_store.Path")
    def test_init(self, mock_path, mock_chromadb):
        """Test vector store initialization."""
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        store = VectorStore(
            persist_directory="/tmp/test",
            collection_name="test_collection",
        )

        assert store.collection_name == "test_collection"
        mock_chromadb.PersistentClient.assert_called_once()

    @patch("wenah.data.vector_store.chromadb")
    @patch("wenah.data.vector_store.Path")
    def test_add_documents(self, mock_path, mock_chromadb):
        """Test adding documents to store."""
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        store = VectorStore(persist_directory="/tmp/test")

        store.add_documents(
            documents=["doc1", "doc2"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            metadatas=[{"cat": "a"}, {"cat": "b"}],
            ids=["id1", "id2"],
        )

        mock_collection.add.assert_called_once_with(
            documents=["doc1", "doc2"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            metadatas=[{"cat": "a"}, {"cat": "b"}],
            ids=["id1", "id2"],
        )

    @patch("wenah.data.vector_store.chromadb")
    @patch("wenah.data.vector_store.Path")
    def test_query(self, mock_path, mock_chromadb):
        """Test querying documents."""
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["doc-1", "doc-2"]],
            "documents": [["content1", "content2"]],
            "metadatas": [[{"cat": "a"}, {"cat": "b"}]],
            "distances": [[0.1, 0.2]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        store = VectorStore(persist_directory="/tmp/test")
        results = store.query(
            query_embeddings=[[0.1, 0.2]],
            n_results=2,
        )

        assert "ids" in results
        assert results["ids"] == [["doc-1", "doc-2"]]

    @patch("wenah.data.vector_store.chromadb")
    @patch("wenah.data.vector_store.Path")
    def test_query_with_scores(self, mock_path, mock_chromadb):
        """Test querying with scores."""
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["doc-1"]],
            "documents": [["content1"]],
            "metadatas": [[{"category": "employment"}]],
            "distances": [[0.5]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        store = VectorStore(persist_directory="/tmp/test")
        results = store.query_with_scores(
            query_text="test query",
            n_results=1,
        )

        assert len(results) == 1
        doc, meta, score = results[0]
        assert doc == "content1"
        assert meta["category"] == "employment"
        assert 0 < score <= 1  # Score should be between 0 and 1

    @patch("wenah.data.vector_store.chromadb")
    @patch("wenah.data.vector_store.Path")
    def test_get_document(self, mock_path, mock_chromadb):
        """Test getting a document by ID."""
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["doc-1"],
            "documents": ["content1"],
            "metadatas": [{"category": "employment"}],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        store = VectorStore(persist_directory="/tmp/test")
        result = store.get_document("doc-1")

        assert result is not None
        assert result["id"] == "doc-1"
        assert result["document"] == "content1"

    @patch("wenah.data.vector_store.chromadb")
    @patch("wenah.data.vector_store.Path")
    def test_get_document_not_found(self, mock_path, mock_chromadb):
        """Test getting non-existent document."""
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        store = VectorStore(persist_directory="/tmp/test")
        result = store.get_document("nonexistent")

        assert result is None

    @patch("wenah.data.vector_store.chromadb")
    @patch("wenah.data.vector_store.Path")
    def test_count(self, mock_path, mock_chromadb):
        """Test counting documents."""
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        store = VectorStore(persist_directory="/tmp/test")
        count = store.count()

        assert count == 42

    @patch("wenah.data.vector_store.chromadb")
    @patch("wenah.data.vector_store.Path")
    def test_delete_documents(self, mock_path, mock_chromadb):
        """Test deleting documents."""
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        store = VectorStore(persist_directory="/tmp/test")
        store.delete_documents(["doc-1", "doc-2"])

        mock_collection.delete.assert_called_once_with(ids=["doc-1", "doc-2"])

    @patch("wenah.data.vector_store.chromadb")
    @patch("wenah.data.vector_store.Path")
    def test_reset(self, mock_path, mock_chromadb):
        """Test resetting collection."""
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        store = VectorStore(
            persist_directory="/tmp/test",
            collection_name="test_collection",
        )
        store.reset()

        mock_client.delete_collection.assert_called_once_with("test_collection")
        mock_client.create_collection.assert_called_once()

    @patch("wenah.data.vector_store.chromadb")
    @patch("wenah.data.vector_store.Path")
    def test_get_all_categories(self, mock_path, mock_chromadb):
        """Test getting all categories."""
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "metadatas": [
                {"category": "employment"},
                {"category": "housing"},
                {"category": "employment"},
            ]
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        store = VectorStore(persist_directory="/tmp/test")
        categories = store.get_all_categories()

        assert set(categories) == {"employment", "housing"}

    @patch("wenah.data.vector_store.chromadb")
    @patch("wenah.data.vector_store.Path")
    def test_get_documents_by_category(self, mock_path, mock_chromadb):
        """Test getting documents by category."""
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["doc-1", "doc-2"],
            "documents": ["content1", "content2"],
            "metadatas": [{"category": "employment"}, {"category": "employment"}],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        store = VectorStore(persist_directory="/tmp/test")
        docs = store.get_documents_by_category("employment")

        assert len(docs) == 2
        assert docs[0]["id"] == "doc-1"


# =============================================================================
# Integration Tests
# =============================================================================


class TestEmbeddingVectorStoreIntegration:
    """Integration tests for embedding and vector store."""

    @patch("wenah.data.vector_store.chromadb")
    @patch("wenah.data.vector_store.Path")
    @patch("wenah.data.embeddings.SentenceTransformer")
    def test_embed_and_store_workflow(self, mock_st, mock_path, mock_chromadb):
        """Test workflow of embedding documents and storing them."""
        # Setup embedding mock
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(
            tolist=lambda: [[0.1, 0.2], [0.3, 0.4]]
        )
        mock_st.return_value = mock_model

        # Setup vector store mock
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Create components
        generator = EmbeddingGenerator(model_name="test-model")
        store = VectorStore(persist_directory="/tmp/test")

        # Create documents
        documents = [
            {"id": "doc-1", "content": "First document"},
            {"id": "doc-2", "content": "Second document"},
        ]

        # Embed documents
        embedded_docs = generator.embed_documents(documents, text_field="content")

        # Verify embeddings added
        assert "embedding" in embedded_docs[0]
        assert "embedding" in embedded_docs[1]

        # Store documents
        store.add_documents(
            documents=[d["content"] for d in embedded_docs],
            embeddings=[d["embedding"] for d in embedded_docs],
            metadatas=[{"id": d["id"]} for d in embedded_docs],
            ids=[d["id"] for d in embedded_docs],
        )

        # Verify store called
        mock_collection.add.assert_called_once()
