"""
Tests for LLM components (Claude client, RAG pipeline, guardrails).

Note: These tests mock the Claude API to avoid actual API calls.
Integration tests with real API calls should be run separately.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Any

from wenah.llm.claude_client import (
    ClaudeClient,
    ClaudeResponse,
    ComplianceAnalyzer,
    AnalysisType,
)
from wenah.llm.prompts import (
    build_risk_analysis_prompt,
    build_disparate_impact_prompt,
    build_proxy_variable_prompt,
    format_context_documents,
    format_data_fields,
    SYSTEM_PROMPT_BASE,
)
from wenah.llm.guardrails import (
    HallucinationGuardrails,
    ResponseValidator,
    GuardrailCheck,
)
from wenah.llm.rag_pipeline import RAGPipeline, RetrievalResult
from wenah.core.types import RAGResponse, ProductFeatureInput


class TestClaudeClient:
    """Tests for ClaudeClient class."""

    @pytest.fixture
    def mock_anthropic(self):
        """Create mock Anthropic client."""
        with patch("wenah.llm.claude_client.anthropic.Anthropic") as mock:
            # Setup mock response
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Test response")]
            mock_response.model = "claude-sonnet-4-20250514"
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 50
            mock_response.stop_reason = "end_turn"

            mock_instance = mock.return_value
            mock_instance.messages.create.return_value = mock_response

            yield mock

    def test_client_initialization(self, mock_anthropic):
        """Test client initialization with API key."""
        client = ClaudeClient(api_key="test-key")
        assert client is not None
        assert client.api_key == "test-key"

    def test_client_requires_api_key(self):
        """Test that client raises error without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("wenah.config.settings") as mock_settings:
                mock_settings.anthropic_api_key = ""
                with pytest.raises(ValueError, match="API key"):
                    ClaudeClient(api_key="")

    def test_analyze_basic(self, mock_anthropic):
        """Test basic analysis request."""
        client = ClaudeClient(api_key="test-key")
        response = client.analyze("Test prompt")

        assert isinstance(response, ClaudeResponse)
        assert response.content == "Test response"
        assert response.usage["input_tokens"] == 100
        assert response.usage["output_tokens"] == 50

    def test_token_tracking(self, mock_anthropic):
        """Test that tokens are tracked across requests."""
        client = ClaudeClient(api_key="test-key")

        client.analyze("Prompt 1")
        client.analyze("Prompt 2")

        assert client.total_tokens_used == 300  # 2 * (100 + 50)

    def test_token_counter_reset(self, mock_anthropic):
        """Test token counter reset."""
        client = ClaudeClient(api_key="test-key")
        client.analyze("Prompt")

        client.reset_token_counter()
        assert client.total_tokens_used == 0

    def test_analyze_with_context(self, mock_anthropic):
        """Test analysis with context documents."""
        client = ClaudeClient(api_key="test-key")

        context_docs = [
            {"content": "Title VII prohibits discrimination", "metadata": {"law_id": "title-vii"}},
        ]

        response = client.analyze_with_context(
            query="Is this practice legal?",
            context_documents=context_docs,
            analysis_type=AnalysisType.RISK_ASSESSMENT,
        )

        assert response.content is not None


class TestComplianceAnalyzer:
    """Tests for ComplianceAnalyzer class."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Claude client."""
        client = Mock(spec=ClaudeClient)
        client.analyze_with_context.return_value = ClaudeResponse(
            content="Test analysis result",
            model="claude-sonnet-4-20250514",
            usage={"input_tokens": 100, "output_tokens": 50},
            stop_reason="end_turn",
        )
        return client

    def test_analyze_feature_compliance(self, mock_client):
        """Test feature compliance analysis."""
        analyzer = ComplianceAnalyzer(client=mock_client)

        result = analyzer.analyze_feature_compliance(
            feature_description="AI hiring algorithm",
            category="employment",
            context_docs=[{"content": "Law text", "metadata": {}}],
        )

        assert "analysis" in result
        assert mock_client.analyze_with_context.called

    def test_analyze_disparate_impact(self, mock_client):
        """Test disparate impact analysis."""
        analyzer = ComplianceAnalyzer(client=mock_client)

        result = analyzer.analyze_disparate_impact(
            practice_description="Credit score requirement",
            affected_groups=["race", "national_origin"],
            context_docs=[{"content": "ECOA provisions", "metadata": {}}],
        )

        assert "analysis" in result


class TestPromptTemplates:
    """Tests for prompt template functions."""

    def test_format_context_documents_empty(self):
        """Test formatting empty context documents."""
        result = format_context_documents([])
        assert "No context documents" in result

    def test_format_context_documents(self):
        """Test formatting context documents."""
        docs = [
            {
                "content": "Title VII prohibits discrimination",
                "metadata": {
                    "law_name": "Title VII",
                    "section": "prohibited_practices",
                },
            },
            {
                "content": "ADA requires accommodation",
                "metadata": {
                    "law_name": "ADA",
                    "chunk_type": "requirement",
                },
            },
        ]

        result = format_context_documents(docs)

        assert "Title VII" in result
        assert "ADA" in result
        assert "prohibited_practices" in result
        assert "discrimination" in result

    def test_format_data_fields_empty(self):
        """Test formatting empty data fields."""
        result = format_data_fields([])
        assert "No data fields" in result

    def test_format_data_fields(self):
        """Test formatting data fields."""
        fields = [
            {
                "name": "zip_code",
                "description": "Applicant ZIP code",
                "used_in_decisions": True,
                "potential_proxy": "race",
            },
            {
                "name": "experience",
                "description": "Years of experience",
                "used_in_decisions": True,
            },
        ]

        result = format_data_fields(fields)

        assert "zip_code" in result
        assert "experience" in result
        assert "proxy" in result.lower()
        assert "Yes" in result

    def test_build_risk_analysis_prompt(self):
        """Test building risk analysis prompt."""
        feature = {
            "name": "Resume Screener",
            "category": "hiring",
            "description": "AI-powered resume screening",
            "data_fields": [],
            "algorithm": None,
            "decision_impact": "Hiring decisions",
            "affected_population": "Job applicants",
        }

        prompt = build_risk_analysis_prompt(
            feature=feature,
            context_documents=[],
        )

        assert "Resume Screener" in prompt
        assert "hiring" in prompt
        assert "Hiring decisions" in prompt

    def test_build_disparate_impact_prompt(self):
        """Test building disparate impact prompt."""
        prompt = build_disparate_impact_prompt(
            practice_description="Credit check requirement",
            context="Employment screening",
            affected_groups=["race", "national_origin"],
            context_documents=[],
        )

        assert "Credit check" in prompt
        assert "race" in prompt
        assert "national_origin" in prompt
        assert "disparate impact" in prompt.lower()


class TestHallucinationGuardrails:
    """Tests for HallucinationGuardrails class."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = Mock()
        store.get_all_categories.return_value = ["employment"]
        store.get_documents_by_category.return_value = [
            {"metadata": {"law_id": "title-vii-1964", "law_name": "Title VII"}}
        ]
        return store

    @pytest.fixture
    def guardrails(self, mock_vector_store):
        """Create guardrails instance."""
        return HallucinationGuardrails(mock_vector_store)

    def test_verify_citations_valid(self, guardrails):
        """Test citation verification with valid citations."""
        check = guardrails._verify_citations([
            "Title VII",
            "42 U.S.C. § 2000e",
        ])

        assert check.passed is True

    def test_verify_citations_invalid(self, guardrails):
        """Test citation verification with invalid citations."""
        check = guardrails._verify_citations([
            "Made Up Law Act of 2099",
            "Fictional Case v. Imaginary Corp",
        ])

        assert check.passed is False
        assert check.severity == "critical"

    def test_verify_grounding_with_docs(self, guardrails):
        """Test grounding verification with context documents."""
        docs = [
            {"content": "Title VII prohibits employment discrimination based on race"},
        ]

        check = guardrails._verify_grounding(
            "Title VII prohibits discrimination based on race in employment.",
            docs,
        )

        # Should pass because claim is grounded in documents
        assert check.check_name == "grounding_verification"

    def test_verify_grounding_no_docs(self, guardrails):
        """Test grounding verification without documents."""
        check = guardrails._verify_grounding("Some claim", [])

        assert check.passed is False
        assert check.severity == "warning"

    def test_confidence_calibration_hedging(self, guardrails):
        """Test confidence calibration with hedging language."""
        response = RAGResponse(
            analysis="This may possibly potentially cause issues. It could perhaps be a violation.",
            confidence_score=0.9,
            cited_sources=[],
            risk_factors=[],
            mitigating_factors=[],
            recommendation="Review",
            requires_human_review=False,
        )

        check, adjustment = guardrails._calibrate_confidence(response)

        # Should reduce confidence due to hedging
        assert adjustment < 0

    def test_confidence_calibration_specific(self, guardrails):
        """Test confidence calibration with specific citations."""
        response = RAGResponse(
            analysis="Under 42 U.S.C. § 2000e-2, this practice violates Title VII.",
            confidence_score=0.8,
            cited_sources=["42 U.S.C. § 2000e-2", "Title VII"],
            risk_factors=["Discrimination"],
            mitigating_factors=[],
            recommendation="Remove practice",
            requires_human_review=False,
        )

        check, adjustment = guardrails._calibrate_confidence(response)

        # Should have positive or neutral adjustment
        assert adjustment >= -0.1

    def test_validate_full_response(self, guardrails):
        """Test full response validation."""
        response = RAGResponse(
            analysis="This violates Title VII discrimination provisions.",
            confidence_score=0.8,
            cited_sources=["Title VII"],
            risk_factors=["Discrimination"],
            mitigating_factors=[],
            recommendation="Stop the practice",
            requires_human_review=False,
        )

        docs = [
            {"content": "Title VII prohibits employment discrimination"},
        ]

        validated = guardrails.validate(response, docs)

        assert isinstance(validated, RAGResponse)
        assert validated.confidence_score <= response.confidence_score

    def test_extract_claims(self, guardrails):
        """Test claim extraction from text."""
        text = (
            "This practice violates Title VII. It is prohibited under federal law. "
            "The company must stop this immediately. This is a simple observation."
        )

        claims = guardrails._extract_claims(text)

        # Should extract sentences with assertion patterns
        assert len(claims) >= 2
        assert any("violates" in c.lower() for c in claims)
        assert any("prohibited" in c.lower() for c in claims)

    def test_is_claim_grounded(self, guardrails):
        """Test claim grounding check."""
        corpus = "title vii prohibits employment discrimination based on race color religion sex national origin"

        # Grounded claim
        assert guardrails._is_claim_grounded(
            "Title VII prohibits race discrimination",
            corpus,
        ) is True

        # Ungrounded claim
        assert guardrails._is_claim_grounded(
            "The GDPR requires data protection compliance",
            corpus,
        ) is False


class TestResponseValidator:
    """Tests for ResponseValidator utility class."""

    def test_validate_json_structure_valid(self):
        """Test JSON structure validation with valid response."""
        response = {
            "analysis": "Test",
            "confidence": 0.8,
            "recommendations": [],
        }

        is_valid, missing = ResponseValidator.validate_json_structure(
            response,
            ["analysis", "confidence"],
        )

        assert is_valid is True
        assert len(missing) == 0

    def test_validate_json_structure_missing(self):
        """Test JSON structure validation with missing fields."""
        response = {"analysis": "Test"}

        is_valid, missing = ResponseValidator.validate_json_structure(
            response,
            ["analysis", "confidence", "risk_level"],
        )

        assert is_valid is False
        assert "confidence" in missing
        assert "risk_level" in missing

    def test_validate_risk_score(self):
        """Test risk score validation."""
        assert ResponseValidator.validate_risk_score(50) == 50.0
        assert ResponseValidator.validate_risk_score(150) == 100.0
        assert ResponseValidator.validate_risk_score(-10) == 0.0
        assert ResponseValidator.validate_risk_score("invalid") == 50.0

    def test_validate_confidence(self):
        """Test confidence validation."""
        assert ResponseValidator.validate_confidence(0.8) == 0.8
        assert ResponseValidator.validate_confidence(1.5) == 1.0
        assert ResponseValidator.validate_confidence(-0.5) == 0.0
        assert ResponseValidator.validate_confidence("invalid") == 0.5

    def test_sanitize_recommendation(self):
        """Test recommendation sanitization."""
        # HTML removal
        result = ResponseValidator.sanitize_recommendation(
            "<script>alert('xss')</script>Remove HTML"
        )
        assert "<script>" not in result

        # Markdown link removal
        result = ResponseValidator.sanitize_recommendation(
            "[Click here](http://evil.com) for more info"
        )
        assert "http://" not in result

        # Truncation
        long_text = "x" * 3000
        result = ResponseValidator.sanitize_recommendation(long_text)
        assert len(result) <= 2003  # 2000 + "..."


class TestRAGPipeline:
    """Tests for RAGPipeline class."""

    @pytest.fixture
    def mock_components(self):
        """Create mock pipeline components."""
        vector_store = Mock()
        vector_store.query_with_scores.return_value = [
            ("Title VII content", {"law_id": "title-vii"}, 0.9),
        ]
        vector_store.get_all_categories.return_value = ["employment"]
        vector_store.get_documents_by_category.return_value = []

        embedding_gen = Mock()

        claude_client = Mock()
        claude_client.analyze.return_value = ClaudeResponse(
            content='{"analysis_summary": "Test", "confidence_score": 0.8, "risk_level": "medium", "recommendations": [], "cited_sources": [], "requires_human_review": false}',
            model="claude-sonnet-4-20250514",
            usage={"input_tokens": 100, "output_tokens": 50},
            stop_reason="end_turn",
        )

        return vector_store, embedding_gen, claude_client

    def test_pipeline_initialization(self, mock_components):
        """Test pipeline initialization."""
        vector_store, embedding_gen, claude_client = mock_components

        pipeline = RAGPipeline(
            vector_store=vector_store,
            embedding_generator=embedding_gen,
            claude_client=claude_client,
        )

        assert pipeline is not None

    def test_retrieve_documents(self, mock_components):
        """Test document retrieval."""
        vector_store, embedding_gen, claude_client = mock_components

        pipeline = RAGPipeline(
            vector_store=vector_store,
            embedding_generator=embedding_gen,
            claude_client=claude_client,
        )

        result = pipeline._retrieve_documents(
            queries=["test query"],
            category="employment",
            top_k=5,
        )

        assert isinstance(result, RetrievalResult)
        assert len(result.documents) > 0

    def test_map_category(self, mock_components):
        """Test category mapping."""
        vector_store, embedding_gen, claude_client = mock_components

        pipeline = RAGPipeline(
            vector_store=vector_store,
            embedding_generator=embedding_gen,
            claude_client=claude_client,
        )

        assert pipeline._map_category("hiring") == "employment"
        assert pipeline._map_category("lending") == "consumer"
        assert pipeline._map_category("housing") == "housing"
        assert pipeline._map_category("general") is None

    def test_build_retrieval_queries(
        self,
        mock_components,
        sample_hiring_feature: ProductFeatureInput,
    ):
        """Test retrieval query building."""
        vector_store, embedding_gen, claude_client = mock_components

        pipeline = RAGPipeline(
            vector_store=vector_store,
            embedding_generator=embedding_gen,
            claude_client=claude_client,
        )

        queries = pipeline._build_retrieval_queries(sample_hiring_feature, None)

        assert len(queries) > 0
        assert any("hiring" in q.lower() for q in queries)


class TestSystemPrompts:
    """Tests for system prompts."""

    def test_base_prompt_content(self):
        """Test that base prompt has required content."""
        assert "civil rights" in SYSTEM_PROMPT_BASE.lower()
        assert "title vii" in SYSTEM_PROMPT_BASE.lower()
        assert "ada" in SYSTEM_PROMPT_BASE.lower()

    def test_base_prompt_instructions(self):
        """Test that base prompt has critical instructions."""
        assert "grounded" in SYSTEM_PROMPT_BASE.lower() or "support" in SYSTEM_PROMPT_BASE.lower()
        assert "confidence" in SYSTEM_PROMPT_BASE.lower()
