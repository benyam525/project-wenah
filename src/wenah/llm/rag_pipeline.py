"""
RAG (Retrieval-Augmented Generation) pipeline for compliance analysis.

Combines vector store retrieval with Claude analysis to provide
grounded, accurate compliance assessments.

Note: Requires optional heavy dependencies (chromadb, sentence-transformers).
"""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from wenah.config import settings
from wenah.core.types import (
    ProductFeatureInput,
    RAGResponse,
    RuleEvaluation,
)

# Optional heavy dependencies
try:
    from wenah.data.embeddings import EmbeddingGenerator, get_embedding_generator
    from wenah.data.vector_store import VectorStore, get_vector_store

    VECTOR_DEPS_AVAILABLE = True
except ImportError:
    VectorStore = None
    get_vector_store = None
    EmbeddingGenerator = None
    get_embedding_generator = None
    VECTOR_DEPS_AVAILABLE = False

try:
    from wenah.llm.claude_client import AnalysisType, ClaudeClient, get_claude_client

    CLAUDE_AVAILABLE = True
except ImportError:
    ClaudeClient = None
    get_claude_client = None
    AnalysisType = None
    CLAUDE_AVAILABLE = False

from wenah.llm.prompts import (
    SYSTEM_PROMPT_STRUCTURED_OUTPUT,
    build_proxy_variable_prompt,
    build_risk_analysis_prompt,
    format_context_documents,
)


class StructuredRAGResponse(BaseModel):
    """Structured response from RAG analysis."""

    analysis_summary: str = Field(..., description="Brief overall assessment")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in analysis")
    risk_level: str = Field(..., description="Overall risk level")
    violations: list[dict[str, Any]] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    mitigating_factors: list[str] = Field(default_factory=list)
    recommendations: list[dict[str, Any]] = Field(default_factory=list)
    cited_sources: list[str] = Field(default_factory=list)
    requires_human_review: bool = Field(default=False)
    human_review_reason: str | None = Field(default=None)


@dataclass
class RetrievalResult:
    """Result from document retrieval."""

    documents: list[dict[str, Any]] = field(default_factory=list)
    query: str = ""
    total_retrieved: int = 0
    filters_applied: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResult:
    """Complete result from RAG pipeline."""

    response: RAGResponse
    retrieval: RetrievalResult
    raw_llm_response: str = ""
    tokens_used: dict[str, int] = field(default_factory=dict)


class RAGPipeline:
    """
    RAG pipeline for civil rights compliance analysis.

    Combines:
    1. Query expansion and reformulation
    2. Vector store retrieval with filtering
    3. Context assembly and ranking
    4. Claude analysis with structured output
    5. Response validation via guardrails
    """

    def __init__(
        self,
        vector_store=None,
        embedding_generator=None,
        claude_client=None,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            vector_store: Vector store for document retrieval
            embedding_generator: Embedding generator for queries
            claude_client: Claude client for analysis

        Raises:
            ImportError: If required dependencies are not installed
        """
        if not VECTOR_DEPS_AVAILABLE:
            raise ImportError(
                "RAG pipeline requires chromadb and sentence-transformers. "
                "Install with: pip install chromadb sentence-transformers"
            )
        if not CLAUDE_AVAILABLE:
            raise ImportError(
                "RAG pipeline requires anthropic client. Install with: pip install anthropic"
            )

        self.vector_store = vector_store or get_vector_store()
        self.embedding_generator = embedding_generator or get_embedding_generator()
        self.claude_client = claude_client or get_claude_client()

        # Import guardrails lazily to avoid circular imports
        self._guardrails = None

    @property
    def guardrails(self):
        """Lazy load guardrails."""
        if self._guardrails is None:
            from wenah.llm.guardrails import HallucinationGuardrails

            self._guardrails = HallucinationGuardrails(self.vector_store)
        return self._guardrails

    def analyze(
        self,
        feature: ProductFeatureInput,
        rule_evaluations: list[RuleEvaluation] | None = None,
        top_k: int = 10,
        apply_guardrails: bool = True,
    ) -> RAGResult:
        """
        Perform RAG-based compliance analysis on a feature.

        Args:
            feature: The product feature to analyze
            rule_evaluations: Optional rule evaluations for context
            top_k: Number of documents to retrieve
            apply_guardrails: Whether to apply hallucination guardrails

        Returns:
            Complete RAG analysis result
        """
        # Step 1: Build retrieval queries
        queries = self._build_retrieval_queries(feature, rule_evaluations)

        # Step 2: Retrieve relevant documents
        retrieval_result = self._retrieve_documents(
            queries=queries,
            category=self._map_category(feature.category.value),
            top_k=top_k,
        )

        # Step 3: Build analysis prompt
        prompt = self._build_analysis_prompt(feature, retrieval_result.documents)

        # Step 4: Call Claude for analysis
        response = self.claude_client.analyze(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT_STRUCTURED_OUTPUT,
            max_tokens=4096,
            temperature=0.0,
        )

        # Step 5: Parse response into RAGResponse
        rag_response = self._parse_response(response.content)

        # Step 6: Apply guardrails if enabled
        if apply_guardrails and settings.enable_hallucination_guardrails:
            rag_response = self.guardrails.validate(rag_response, retrieval_result.documents)

        return RAGResult(
            response=rag_response,
            retrieval=retrieval_result,
            raw_llm_response=response.content,
            tokens_used=response.usage,
        )

    def analyze_escalated_rule(
        self,
        rule_evaluation: RuleEvaluation,
        feature: ProductFeatureInput,
        top_k: int = 5,
    ) -> RAGResult:
        """
        Perform focused analysis on a rule that was escalated to LLM.

        Args:
            rule_evaluation: The escalated rule evaluation
            feature: The feature being evaluated
            top_k: Number of documents to retrieve

        Returns:
            RAG analysis result for the escalated rule
        """
        # Build query from rule context
        llm_context = rule_evaluation.llm_context or {}
        question = llm_context.get(
            "question", f"Analyze compliance with {rule_evaluation.rule_name}"
        )
        required_analysis = llm_context.get("required_analysis", [])

        # Retrieve focused documents
        queries = [question] + [f"civil rights {topic}" for topic in required_analysis]
        retrieval_result = self._retrieve_documents(
            queries=queries,
            category=self._map_category(feature.category.value),
            top_k=top_k,
        )

        # Build focused prompt
        prompt = self._build_escalation_prompt(
            rule_evaluation=rule_evaluation,
            feature=feature,
            context_documents=retrieval_result.documents,
            question=question,
            required_analysis=required_analysis,
        )

        # Get analysis
        response = self.claude_client.analyze(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT_STRUCTURED_OUTPUT,
            max_tokens=2048,
        )

        rag_response = self._parse_response(response.content)

        if settings.enable_hallucination_guardrails:
            rag_response = self.guardrails.validate(rag_response, retrieval_result.documents)

        return RAGResult(
            response=rag_response,
            retrieval=retrieval_result,
            raw_llm_response=response.content,
            tokens_used=response.usage,
        )

    def analyze_proxy_variable(
        self,
        variable_name: str,
        usage_context: str,
        feature: ProductFeatureInput,
        top_k: int = 5,
    ) -> RAGResult:
        """
        Analyze a specific variable for proxy discrimination risk.

        Args:
            variable_name: Name of the variable
            usage_context: How the variable is used
            feature: Feature context
            top_k: Documents to retrieve

        Returns:
            Proxy variable analysis result
        """
        queries = [
            f"proxy variable {variable_name} discrimination",
            "disparate impact proxy variables",
            f"{variable_name} correlation protected class",
        ]

        retrieval_result = self._retrieve_documents(
            queries=queries,
            category=self._map_category(feature.category.value),
            top_k=top_k,
        )

        prompt = build_proxy_variable_prompt(
            variable_name=variable_name,
            usage_description=usage_context,
            decision_context=feature.decision_impact,
            context_documents=retrieval_result.documents,
        )

        response = self.claude_client.analyze(
            prompt=prompt,
            max_tokens=2048,
        )

        # Parse into RAGResponse format
        rag_response = RAGResponse(
            analysis=response.content,
            confidence_score=0.8,  # Default, will be adjusted by guardrails
            cited_sources=[],
            risk_factors=[f"Potential proxy: {variable_name}"],
            mitigating_factors=[],
            recommendation=f"Review use of {variable_name} for disparate impact",
            requires_human_review=True,
        )

        if settings.enable_hallucination_guardrails:
            rag_response = self.guardrails.validate(rag_response, retrieval_result.documents)

        return RAGResult(
            response=rag_response,
            retrieval=retrieval_result,
            raw_llm_response=response.content,
            tokens_used=response.usage,
        )

    def _build_retrieval_queries(
        self,
        feature: ProductFeatureInput,
        rule_evaluations: list[RuleEvaluation] | None,
    ) -> list[str]:
        """Build queries for document retrieval."""
        queries = []

        # Base query from feature
        queries.append(
            f"{feature.category.value} {feature.feature_type.value} civil rights compliance"
        )

        # Add queries based on data fields
        for df in feature.data_fields:
            if df.used_in_decisions:
                queries.append(f"{df.name} employment discrimination")
            if df.potential_proxy:
                queries.append(f"{df.name} proxy {df.potential_proxy} discrimination")

        # Add queries from rule evaluations
        if rule_evaluations:
            for eval in rule_evaluations:
                for ref in eval.law_references:
                    queries.append(ref)
                if eval.escalate_to_llm and eval.llm_context:
                    question = eval.llm_context.get("question", "")
                    if question:
                        queries.append(question)

        # Add algorithm-specific queries
        if feature.algorithm:
            if feature.algorithm.type in ["ml_model", "llm"]:
                queries.append("AI algorithm bias discrimination EEOC")
            for input_field in feature.algorithm.inputs:
                if any(term in input_field.lower() for term in ["video", "voice", "facial"]):
                    queries.append(f"AI {input_field} disability discrimination ADA")

        return queries[:10]  # Limit to 10 queries

    def _retrieve_documents(
        self,
        queries: list[str],
        category: str | None,
        top_k: int,
    ) -> RetrievalResult:
        """Retrieve and deduplicate documents."""
        all_docs = {}  # Use dict to deduplicate by ID

        for query in queries:
            results = self.vector_store.query_with_scores(
                query_text=query,
                n_results=min(top_k, 5),  # Limit per query
                category_filter=category,
            )

            for doc_text, metadata, score in results:
                doc_id = metadata.get("law_id", "") + metadata.get("section", "")
                if doc_id not in all_docs or score > all_docs[doc_id]["score"]:
                    all_docs[doc_id] = {
                        "content": doc_text,
                        "metadata": metadata,
                        "score": score,
                    }

        # Sort by score and take top_k
        sorted_docs = sorted(
            all_docs.values(),
            key=lambda x: x["score"],
            reverse=True,
        )[:top_k]

        return RetrievalResult(
            documents=[{"content": d["content"], "metadata": d["metadata"]} for d in sorted_docs],
            query="; ".join(queries[:3]),
            total_retrieved=len(sorted_docs),
            filters_applied={"category": category} if category else {},
        )

    def _build_analysis_prompt(
        self,
        feature: ProductFeatureInput,
        documents: list[dict[str, Any]],
    ) -> str:
        """Build the main analysis prompt."""
        feature_dict = {
            "name": feature.name,
            "category": feature.category.value,
            "description": feature.description,
            "data_fields": [
                {
                    "name": df.name,
                    "description": df.description,
                    "used_in_decisions": df.used_in_decisions,
                    "potential_proxy": df.potential_proxy,
                }
                for df in feature.data_fields
            ],
            "algorithm": {
                "name": feature.algorithm.name,
                "type": feature.algorithm.type,
                "inputs": feature.algorithm.inputs,
                "outputs": feature.algorithm.outputs,
                "bias_testing_done": feature.algorithm.bias_testing_done,
            }
            if feature.algorithm
            else None,
            "decision_impact": feature.decision_impact,
            "affected_population": feature.affected_population,
        }

        return build_risk_analysis_prompt(
            feature=feature_dict,
            context_documents=documents,
        )

    def _build_escalation_prompt(
        self,
        rule_evaluation: RuleEvaluation,
        feature: ProductFeatureInput,
        context_documents: list[dict[str, Any]],
        question: str,
        required_analysis: list[str],
    ) -> str:
        """Build prompt for escalated rule analysis."""
        analysis_items = "\n".join(f"- {item}" for item in required_analysis)

        return f"""## Escalated Compliance Analysis

### Context Documents
{format_context_documents(context_documents)}

### Rule Information
- **Rule:** {rule_evaluation.rule_name}
- **Initial Assessment:** {rule_evaluation.result.value}
- **Risk Score:** {rule_evaluation.risk_score}
- **Law Reference:** {", ".join(rule_evaluation.law_references)}

### Feature Being Evaluated
- **Name:** {feature.name}
- **Category:** {feature.category.value}
- **Description:** {feature.description}
- **Decision Impact:** {feature.decision_impact}

### Analysis Question
{question}

### Required Analysis Points
{analysis_items}

### Instructions
Provide a detailed analysis addressing the question and all required analysis points.
Base your analysis on the provided context documents.
Clearly indicate your confidence level and cite specific sources.

Respond with a JSON object containing:
- analysis_summary: Your assessment
- confidence_score: 0.0 to 1.0
- risk_level: critical/high/medium/low/minimal
- risk_factors: List of identified risks
- mitigating_factors: Factors that reduce risk
- recommendations: Prioritized action items
- cited_sources: Laws/provisions you relied on
- requires_human_review: true/false
"""

    def _parse_response(self, content: str) -> RAGResponse:
        """Parse LLM response into RAGResponse."""
        import json

        try:
            # Try to extract JSON
            content = content.strip()

            # Handle markdown code blocks
            if "```" in content:
                lines = content.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                content = "\n".join(json_lines)

            data = json.loads(content)

            return RAGResponse(
                analysis=data.get("analysis_summary", content),
                confidence_score=float(data.get("confidence_score", 0.7)),
                cited_sources=data.get("cited_sources", []),
                risk_factors=data.get("risk_factors", []),
                mitigating_factors=data.get("mitigating_factors", []),
                recommendation=self._format_recommendations(data.get("recommendations", [])),
                requires_human_review=data.get("requires_human_review", False),
            )

        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback for unstructured response
            return RAGResponse(
                analysis=content,
                confidence_score=0.5,  # Lower confidence for unstructured
                cited_sources=[],
                risk_factors=[],
                mitigating_factors=[],
                recommendation="Review the analysis above for specific recommendations.",
                requires_human_review=True,
            )

    def _format_recommendations(self, recommendations: list[dict[str, Any]]) -> str:
        """Format recommendations list into string."""
        if not recommendations:
            return "No specific recommendations identified."

        lines = []
        for rec in recommendations:
            priority = rec.get("priority", "")
            action = rec.get("action", rec.get("recommendation", ""))
            if priority:
                lines.append(f"{priority}. {action}")
            else:
                lines.append(f"- {action}")

        return "\n".join(lines)

    def _map_category(self, category: str) -> str | None:
        """Map feature category to law category."""
        mapping = {
            "hiring": "employment",
            "lending": "consumer",
            "housing": "housing",
            "insurance": "consumer",
            "general": None,  # No filter
        }
        return mapping.get(category)


# Singleton instance
_rag_pipeline: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    """Get singleton RAG pipeline instance."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
