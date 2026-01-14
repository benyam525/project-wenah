"""
Embedding generation for law documents.

Uses sentence-transformers to generate embeddings for the RAG pipeline.
Optional dependency - provides stub implementation if not installed.
"""

from typing import Any

from wenah.config import settings

# Lazy import for optional heavy dependency
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore[misc, assignment]
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EmbeddingGenerator:
    """
    Generates embeddings for text documents using sentence-transformers.

    Uses the configured embedding model to create vector representations
    of law documents for semantic search.
    """

    def __init__(self, model_name: str | None = None):
        """
        Initialize the embedding generator.

        Args:
            model_name: Name of the sentence-transformer model.
                       Defaults to config setting.
        """
        self.model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> Any:
        """Lazy load the model on first access."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install with: pip install sentence-transformers"
            )
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def is_available(self) -> bool:
        """Check if embedding generation is available."""
        return SENTENCE_TRANSFORMERS_AVAILABLE

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return list(embedding.tolist())

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [list(e) for e in embeddings.tolist()]

    def embed_documents(
        self,
        documents: list[dict[str, Any]],
        text_field: str = "content",
    ) -> list[dict[str, Any]]:
        """
        Add embeddings to document dictionaries.

        Args:
            documents: List of document dictionaries
            text_field: Field containing the text to embed

        Returns:
            Documents with 'embedding' field added
        """
        texts = [doc.get(text_field, "") for doc in documents]
        embeddings = self.embed_texts(texts)

        for doc, embedding in zip(documents, embeddings):
            doc["embedding"] = embedding

        return documents

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return int(self.model.get_sentence_embedding_dimension())


class LawDocumentChunker:
    """
    Chunks law documents into smaller pieces for embedding.

    Preserves context and metadata while creating appropriately-sized
    chunks for the vector store.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_law_document(
        self,
        law_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Chunk a law document into searchable pieces.

        Creates chunks for different sections of the law document,
        preserving structure and adding relevant metadata.

        Args:
            law_data: Parsed law document data

        Returns:
            List of chunk dictionaries with content and metadata
        """
        chunks = []
        law = law_data.get("law", {})
        law_id = law.get("id", "unknown")
        law_name = law.get("name", "Unknown Law")
        category = law.get("category", "unknown")

        # Chunk the summary
        if summary := law.get("summary"):
            chunks.append(
                self._create_chunk(
                    content=f"{law_name}\n\n{summary}",
                    law_id=law_id,
                    law_name=law_name,
                    category=category,
                    section="summary",
                    chunk_type="overview",
                )
            )

        # Chunk protected classes
        for pc in law.get("protected_classes", []):
            content = self._format_protected_class(pc, law_name)
            chunks.append(
                self._create_chunk(
                    content=content,
                    law_id=law_id,
                    law_name=law_name,
                    category=category,
                    section="protected_classes",
                    subsection=pc.get("id"),
                    chunk_type="protected_class",
                )
            )

        # Chunk prohibited practices
        for pp in law.get("prohibited_practices", []):
            content = self._format_prohibited_practice(pp, law_name)
            # This might need further chunking if too long
            for i, chunk_content in enumerate(self._split_if_needed(content)):
                chunks.append(
                    self._create_chunk(
                        content=chunk_content,
                        law_id=law_id,
                        law_name=law_name,
                        category=category,
                        section="prohibited_practices",
                        subsection=pp.get("id"),
                        chunk_type="prohibited_practice",
                        chunk_index=i,
                        severity=pp.get("severity"),
                    )
                )

        # Chunk safe harbors
        for sh in law.get("safe_harbors", []):
            content = self._format_safe_harbor(sh, law_name)
            chunks.append(
                self._create_chunk(
                    content=content,
                    law_id=law_id,
                    law_name=law_name,
                    category=category,
                    section="safe_harbors",
                    subsection=sh.get("id"),
                    chunk_type="safe_harbor",
                )
            )

        # Chunk AI/ML considerations
        for ai in law.get("ai_ml_considerations", []):
            content = self._format_ai_consideration(ai, law_name)
            chunks.append(
                self._create_chunk(
                    content=content,
                    law_id=law_id,
                    law_name=law_name,
                    category=category,
                    section="ai_ml_considerations",
                    subsection=ai.get("id"),
                    chunk_type="ai_consideration",
                )
            )

        # Chunk key cases
        for case in law.get("key_cases", []):
            content = self._format_key_case(case, law_name)
            chunks.append(
                self._create_chunk(
                    content=content,
                    law_id=law_id,
                    law_name=law_name,
                    category=category,
                    section="key_cases",
                    chunk_type="case_law",
                )
            )

        return chunks

    def _create_chunk(
        self,
        content: str,
        law_id: str,
        law_name: str,
        category: str,
        section: str,
        chunk_type: str,
        subsection: str | None = None,
        chunk_index: int = 0,
        **extra_metadata: Any,
    ) -> dict[str, Any]:
        """Create a chunk dictionary with standardized metadata."""
        chunk_id = f"{law_id}-{section}"
        if subsection:
            chunk_id += f"-{subsection}"
        if chunk_index > 0:
            chunk_id += f"-{chunk_index}"

        metadata = {
            "law_id": law_id,
            "law_name": law_name,
            "category": category,
            "section": section,
            "chunk_type": chunk_type,
            **extra_metadata,
        }
        if subsection:
            metadata["subsection"] = subsection

        return {
            "id": chunk_id,
            "content": content,
            "metadata": metadata,
        }

    def _split_if_needed(self, content: str) -> list[str]:
        """Split content if it exceeds chunk size."""
        if len(content) <= self.chunk_size:
            return [content]

        chunks = []
        start = 0
        while start < len(content):
            end = start + self.chunk_size

            # Try to break at a sentence boundary
            if end < len(content):
                # Look for sentence ending within last 20% of chunk
                search_start = int(end - self.chunk_size * 0.2)
                for i in range(end, search_start, -1):
                    if content[i] in ".!?\n":
                        end = i + 1
                        break

            chunks.append(content[start:end].strip())
            start = end - self.chunk_overlap

        return chunks

    def _format_protected_class(self, pc: dict, law_name: str) -> str:
        """Format a protected class for embedding."""
        lines = [
            f"Protected Class under {law_name}: {pc.get('name', 'Unknown')}",
            "",
            pc.get("description", ""),
        ]

        if examples := pc.get("examples"):
            lines.append("\nExamples:")
            for ex in examples:
                lines.append(f"  - {ex}")

        return "\n".join(lines)

    def _format_prohibited_practice(self, pp: dict, law_name: str) -> str:
        """Format a prohibited practice for embedding."""
        lines = [
            f"Prohibited Practice under {law_name}: {pp.get('name', 'Unknown')}",
            f"Severity: {pp.get('severity', 'unknown')}",
            "",
            pp.get("description", ""),
        ]

        if elements := pp.get("elements"):
            lines.append("\nElements Required:")
            for el in elements:
                if isinstance(el, dict):
                    lines.append(f"  - {el.get('element', '')}: {el.get('description', '')}")
                else:
                    lines.append(f"  - {el}")

        if examples := pp.get("examples"):
            lines.append("\nExamples:")
            for ex in examples:
                lines.append(f"  - {ex}")

        return "\n".join(lines)

    def _format_safe_harbor(self, sh: dict, law_name: str) -> str:
        """Format a safe harbor for embedding."""
        lines = [
            f"Safe Harbor under {law_name}: {sh.get('name', 'Unknown')}",
            "",
            sh.get("description", ""),
        ]

        if applies_to := sh.get("applies_to"):
            lines.append(f"\nApplies to: {', '.join(applies_to)}")

        if conditions := sh.get("conditions"):
            lines.append("\nConditions:")
            for cond in conditions:
                lines.append(f"  - {cond}")

        return "\n".join(lines)

    def _format_ai_consideration(self, ai: dict, law_name: str) -> str:
        """Format AI/ML consideration for embedding."""
        lines = [
            f"AI/ML Consideration under {law_name}: {ai.get('name', 'Unknown')}",
            "",
            ai.get("description", ""),
        ]

        if risks := ai.get("risks"):
            lines.append("\nRisks:")
            for risk in risks:
                lines.append(f"  - {risk}")

        if requirements := ai.get("requirements"):
            lines.append("\nRequirements:")
            for req in requirements:
                lines.append(f"  - {req}")

        return "\n".join(lines)

    def _format_key_case(self, case: dict, law_name: str) -> str:
        """Format a key case for embedding."""
        return (
            f"Key Case for {law_name}: {case.get('name', 'Unknown')}\n"
            f"Citation: {case.get('citation', 'Unknown')}\n\n"
            f"Holding: {case.get('holding', '')}\n\n"
            f"Relevance: {case.get('relevance', '')}"
        )


# Singleton instance
_embedding_generator: EmbeddingGenerator | None = None


def get_embedding_generator() -> EmbeddingGenerator:
    """Get the singleton embedding generator instance."""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator
