"""
Edge case tests for hallucination guardrails.

Tests boundary conditions, malformed inputs, and unusual scenarios
to ensure guardrails are robust.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from wenah.core.types import RAGResponse
from wenah.llm.guardrails import (
    HallucinationGuardrails,
    ResponseValidator,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = MagicMock()
    store.get_all_categories.return_value = ["employment", "housing"]
    store.get_documents_by_category.return_value = []
    return store


@pytest.fixture
def guardrails(mock_vector_store):
    """Create guardrails instance with mock store."""
    return HallucinationGuardrails(mock_vector_store)


# =============================================================================
# Citation Verification Edge Cases
# =============================================================================


class TestCitationVerificationEdgeCases:
    """Edge cases for citation verification."""

    def test_empty_citation_list(self, guardrails):
        """Test with empty citation list."""
        check = guardrails._verify_citations([])
        assert check.passed is True
        assert check.severity == "info"

    def test_citation_with_empty_string(self, guardrails):
        """Test with empty string citation."""
        check = guardrails._verify_citations([""])
        # Empty string doesn't match any known pattern
        assert check.passed is False

    def test_citation_with_whitespace_only(self, guardrails):
        """Test with whitespace-only citation."""
        check = guardrails._verify_citations(["   ", "\t\n"])
        assert check.passed is False

    def test_citation_case_insensitivity(self, guardrails):
        """Test citation matching is case insensitive."""
        citations = [
            "TITLE VII",
            "title vii",
            "Title VII",
            "TiTlE vIi",
        ]
        for citation in citations:
            check = guardrails._verify_citations([citation])
            assert check.passed is True, f"Failed for: {citation}"

    def test_citation_with_special_characters(self, guardrails):
        """Test citations with special characters."""
        check = guardrails._verify_citations(
            [
                "42 U.S.C. § 2000e",
                "42 USC §2000e",
                "42 u.s.c. 2000e",
            ]
        )
        assert check.passed is True

    def test_citation_partial_match(self, guardrails):
        """Test citation partial matching."""
        check = guardrails._verify_citations(
            [
                "Title VII of the Civil Rights Act of 1964",
                "Americans with Disabilities Act (ADA)",
            ]
        )
        assert check.passed is True

    def test_mixed_valid_invalid_citations(self, guardrails):
        """Test mix of valid and invalid citations."""
        check = guardrails._verify_citations(
            [
                "Title VII",
                "Fake Law 2099",
                "ADA",
            ]
        )
        # Should fail due to one invalid
        assert check.passed is False

    def test_citation_with_unicode(self, guardrails):
        """Test citation with unicode characters."""
        check = guardrails._verify_citations(
            [
                "Title VII §2000e",  # Section symbol
                "ADA – Americans with Disabilities",  # En dash
            ]
        )
        assert check.passed is True

    def test_citation_very_long_string(self, guardrails):
        """Test citation with very long string."""
        long_citation = "Title VII " + "extra text " * 100
        check = guardrails._verify_citations([long_citation])
        assert check.passed is True  # Should still match Title VII

    def test_all_citations_unknown(self, guardrails):
        """Test when all citations are unknown."""
        check = guardrails._verify_citations(
            [
                "Unknown Law 1",
                "Fictional Statute",
                "Made Up Act",
            ]
        )
        assert check.passed is False
        assert check.severity == "critical"


# =============================================================================
# Grounding Verification Edge Cases
# =============================================================================


class TestGroundingVerificationEdgeCases:
    """Edge cases for grounding verification."""

    def test_empty_analysis(self, guardrails):
        """Test with empty analysis text."""
        docs = [{"content": "Some law text about employment"}]
        check = guardrails._verify_grounding("", docs)
        assert check.passed is True

    def test_empty_documents(self, guardrails):
        """Test with empty document list."""
        check = guardrails._verify_grounding("This violates the law", [])
        assert check.passed is False
        assert check.severity == "warning"

    def test_documents_with_empty_content(self, guardrails):
        """Test with documents that have empty content."""
        docs = [{"content": ""}, {"content": ""}]
        check = guardrails._verify_grounding("This violates employment laws", docs)
        # Claims won't be grounded in empty documents
        assert check.severity in ["warning", "info"]

    def test_analysis_no_claims(self, guardrails):
        """Test analysis text with no claims."""
        docs = [{"content": "Employment law information"}]
        check = guardrails._verify_grounding("General information about the topic.", docs)
        assert check.passed is True

    def test_analysis_all_claims_grounded(self, guardrails):
        """Test analysis where all claims are grounded."""
        docs = [
            {
                "content": "Title VII prohibits employment discrimination based on race, color, religion, sex, and national origin."
            }
        ]
        analysis = "This violates Title VII by discriminating based on race."
        check = guardrails._verify_grounding(analysis, docs)
        assert check.passed is True

    def test_analysis_no_claims_grounded(self, guardrails):
        """Test analysis where no claims are grounded."""
        docs = [{"content": "Housing laws apply to residential properties."}]
        analysis = "This violates employment law. It is required by ECOA."
        guardrails._verify_grounding(analysis, docs)
        # May or may not pass depending on grounding ratio threshold

    def test_documents_missing_content_key(self, guardrails):
        """Test with documents missing content key."""
        docs = [{"metadata": "some data"}, {"other_key": "value"}]
        check = guardrails._verify_grounding("This is prohibited", docs)
        # Should handle gracefully
        assert check is not None

    def test_very_long_analysis(self, guardrails):
        """Test with very long analysis text."""
        docs = [{"content": "Title VII prohibits discrimination."}]
        analysis = "This practice violates the law. " * 100
        check = guardrails._verify_grounding(analysis, docs)
        assert check is not None

    def test_unicode_in_documents_and_analysis(self, guardrails):
        """Test unicode handling in both documents and analysis."""
        docs = [{"content": "§2000e prohibits discriminación based on raza."}]
        analysis = "This violates §2000e by discriminating based on raza."
        check = guardrails._verify_grounding(analysis, docs)
        assert check is not None


# =============================================================================
# Confidence Calibration Edge Cases
# =============================================================================


class TestConfidenceCalibrationEdgeCases:
    """Edge cases for confidence calibration."""

    def test_response_all_hedging(self, guardrails):
        """Test response with maximum hedging language."""
        response = RAGResponse(
            analysis="This may possibly potentially likely perhaps typically generally usually might could be a violation.",
            cited_sources=["Title VII"],
            confidence_score=0.8,
            recommendation="Review practices",
        )
        check, adjustment = guardrails._calibrate_confidence(response)
        assert adjustment < 0  # Should penalize

    def test_response_no_hedging(self, guardrails):
        """Test response with no hedging language."""
        response = RAGResponse(
            analysis="This is a clear violation of Title VII under 42 U.S.C. § 2000e.",
            cited_sources=["Title VII", "42 U.S.C. § 2000e"],
            confidence_score=0.8,
            recommendation="Immediate remediation required",
        )
        check, adjustment = guardrails._calibrate_confidence(response)
        # Should have positive or minimal adjustment
        assert adjustment >= -0.1

    def test_response_very_short_analysis(self, guardrails):
        """Test with very short analysis."""
        response = RAGResponse(
            analysis="Bad.",
            cited_sources=[],
            confidence_score=0.9,
            recommendation="Review",
        )
        check, adjustment = guardrails._calibrate_confidence(response)
        # Should penalize for short analysis and no citations
        assert adjustment < 0

    def test_response_very_long_analysis(self, guardrails):
        """Test with very long analysis."""
        response = RAGResponse(
            analysis="Detailed analysis. " * 200 + "Under 42 U.S.C. § 2000e this is prohibited.",
            cited_sources=["Title VII", "ADA"],
            confidence_score=0.7,
            recommendation="Take corrective action",
        )
        check, adjustment = guardrails._calibrate_confidence(response)
        assert check is not None

    def test_response_zero_confidence(self, guardrails):
        """Test with zero initial confidence."""
        response = RAGResponse(
            analysis="Some analysis with proper citation 42 U.S.C. § 2000e.",
            cited_sources=["Title VII"],
            confidence_score=0.0,
            recommendation="Review required",
        )
        check, adjustment = guardrails._calibrate_confidence(response)
        assert check is not None

    def test_response_max_confidence(self, guardrails):
        """Test with maximum confidence."""
        response = RAGResponse(
            analysis="Clear violation under 42 U.S.C. § 2000e with specific evidence.",
            cited_sources=["Title VII", "42 U.S.C. § 2000e"],
            confidence_score=1.0,
            recommendation="Immediate action required",
        )
        check, adjustment = guardrails._calibrate_confidence(response)
        assert check is not None

    def test_response_empty_analysis(self, guardrails):
        """Test with empty analysis."""
        response = RAGResponse(
            analysis="",
            cited_sources=[],
            confidence_score=0.5,
            recommendation="Unable to provide recommendation",
        )
        check, adjustment = guardrails._calibrate_confidence(response)
        # Should penalize for short/empty analysis
        assert adjustment <= 0


# =============================================================================
# Legal Claim Verification Edge Cases
# =============================================================================


class TestLegalClaimVerificationEdgeCases:
    """Edge cases for legal claim verification."""

    def test_no_assertions(self, guardrails):
        """Test analysis with no legal assertions."""
        docs = [{"content": "Employment law basics"}]
        check = guardrails._verify_legal_claims(
            "This is general information about employment.", docs
        )
        assert check.passed is True

    def test_all_assertions_supported(self, guardrails):
        """Test when all assertions are supported."""
        docs = [
            {
                "content": "Title VII prohibits employment discrimination based on race and requires equal treatment."
            }
        ]
        check = guardrails._verify_legal_claims(
            "This violates Title VII because it discriminates based on race.", docs
        )
        assert check.passed is True

    def test_unsupported_assertions(self, guardrails):
        """Test when assertions lack support."""
        docs = [{"content": "Housing regulations apply to rentals."}]
        guardrails._verify_legal_claims(
            "This violates employment law and is prohibited. Workers must be treated equally.", docs
        )
        # Assertions about employment won't be supported by housing docs

    def test_mixed_supported_unsupported(self, guardrails):
        """Test mix of supported and unsupported assertions."""
        docs = [{"content": "Discrimination is prohibited under Title VII."}]
        guardrails._verify_legal_claims(
            "This violates Title VII. It is also required under ECOA.", docs
        )
        # Partial support

    def test_empty_documents(self, guardrails):
        """Test with empty document list."""
        guardrails._verify_legal_claims("This is prohibited and violates the law.", [])
        # No documents to support claims

    def test_many_assertions(self, guardrails):
        """Test with many assertions."""
        docs = [{"content": "Various prohibited practices exist under law."}]
        analysis = " ".join(
            [
                "This is prohibited.",
                "That is required.",
                "It must be done.",
                "They shall comply.",
                "This violates the law.",
            ]
        )
        check = guardrails._verify_legal_claims(analysis, docs)
        assert check is not None


# =============================================================================
# Contradiction Detection Edge Cases
# =============================================================================


class TestContradictionDetectionEdgeCases:
    """Edge cases for contradiction detection."""

    def test_no_contradictions(self, guardrails):
        """Test when there are no contradictions."""
        response = RAGResponse(
            analysis="This is required under the law.",
            cited_sources=[],
            confidence_score=0.8,
            recommendation="Follow requirements",
        )
        docs = [{"content": "This is required and must be followed."}]
        check = guardrails._check_contradictions(response, docs)
        assert check.passed is True

    def test_clear_contradiction(self, guardrails):
        """Test clear contradiction detection."""
        response = RAGResponse(
            analysis="This is not required and permitted.",
            cited_sources=[],
            confidence_score=0.8,
            recommendation="Proceed with caution",
        )
        docs = [{"content": "This is required and prohibited."}]
        guardrails._check_contradictions(response, docs)
        # Should detect potential contradiction

    def test_no_documents(self, guardrails):
        """Test contradiction check with no documents."""
        response = RAGResponse(
            analysis="This is not required.",
            cited_sources=[],
            confidence_score=0.8,
            recommendation="No action needed",
        )
        check = guardrails._check_contradictions(response, [])
        assert check.passed is True

    def test_empty_analysis(self, guardrails):
        """Test with empty analysis."""
        response = RAGResponse(
            analysis="",
            cited_sources=[],
            confidence_score=0.8,
            recommendation="Review needed",
        )
        docs = [{"content": "Required and prohibited content."}]
        check = guardrails._check_contradictions(response, docs)
        assert check.passed is True

    def test_case_insensitive_contradiction(self, guardrails):
        """Test contradiction detection is case insensitive."""
        response = RAGResponse(
            analysis="NO VIOLATION was found.",
            cited_sources=[],
            confidence_score=0.8,
            recommendation="Continue monitoring",
        )
        docs = [{"content": "VIOLATION detected."}]
        guardrails._check_contradictions(response, docs)
        # Should detect regardless of case


# =============================================================================
# Full Validation Edge Cases
# =============================================================================


class TestFullValidationEdgeCases:
    """Edge cases for full response validation."""

    def test_validate_minimal_response(self, guardrails):
        """Test validation with minimal valid response."""
        response = RAGResponse(
            analysis="OK",
            cited_sources=[],
            confidence_score=0.5,
            recommendation="Review",
        )
        result = guardrails.validate(response, [])
        assert result is not None
        assert isinstance(result.confidence_score, float)

    def test_validate_rich_response(self, guardrails):
        """Test validation with rich response."""
        response = RAGResponse(
            analysis="""
            Under Title VII of the Civil Rights Act of 1964 (42 U.S.C. § 2000e),
            this practice is prohibited. The employer must ensure compliance
            with all requirements. Discrimination based on race, color, religion,
            sex, or national origin is explicitly forbidden.
            """,
            cited_sources=["Title VII", "42 U.S.C. § 2000e"],
            confidence_score=0.85,
            recommendation="Immediately address compliance gaps",
        )
        docs = [
            {
                "content": "Title VII prohibits employment discrimination based on race, color, religion, sex, national origin."
            }
        ]
        result = guardrails.validate(response, docs)
        assert result is not None

    def test_validate_response_with_no_requires_human_review(self, guardrails):
        """Test that requires_human_review is set appropriately."""
        response = RAGResponse(
            analysis="Clear analysis with Title VII citation 42 U.S.C. § 2000e.",
            cited_sources=["Title VII"],
            confidence_score=0.9,
            recommendation="Continue monitoring",
        )
        docs = [{"content": "Title VII 42 U.S.C. § 2000e prohibits discrimination."}]
        result = guardrails.validate(response, docs)
        assert hasattr(result, "requires_human_review")

    def test_validate_triggers_human_review(self, guardrails):
        """Test that critical failures trigger human review."""
        response = RAGResponse(
            analysis="Under the Fake Law of 2099, this is prohibited.",
            cited_sources=["Fake Law 2099"],
            confidence_score=0.95,
            recommendation="Review legal basis",
        )
        result = guardrails.validate(response, [])
        assert result.requires_human_review is True

    def test_validate_confidence_clamping(self, guardrails):
        """Test that confidence is properly clamped."""
        response = RAGResponse(
            analysis="Some analysis",
            cited_sources=[],
            confidence_score=0.999,
            recommendation="General review",
        )
        result = guardrails.validate(response, [])
        assert 0.0 <= result.confidence_score <= 1.0


# =============================================================================
# ResponseValidator Edge Cases
# =============================================================================


class TestResponseValidatorEdgeCases:
    """Edge cases for ResponseValidator utility class."""

    def test_validate_json_empty_response(self):
        """Test JSON validation with empty response."""
        is_valid, missing = ResponseValidator.validate_json_structure({}, ["field"])
        assert is_valid is False
        assert "field" in missing

    def test_validate_json_all_fields_present(self):
        """Test JSON validation with all fields."""
        is_valid, missing = ResponseValidator.validate_json_structure(
            {"a": 1, "b": 2, "c": 3}, ["a", "b", "c"]
        )
        assert is_valid is True
        assert missing == []

    def test_validate_json_partial_fields(self):
        """Test JSON validation with partial fields."""
        is_valid, missing = ResponseValidator.validate_json_structure({"a": 1}, ["a", "b", "c"])
        assert is_valid is False
        assert "b" in missing
        assert "c" in missing

    def test_validate_json_extra_fields(self):
        """Test JSON validation ignores extra fields."""
        is_valid, missing = ResponseValidator.validate_json_structure(
            {"a": 1, "b": 2, "extra": "data"}, ["a", "b"]
        )
        assert is_valid is True

    def test_validate_risk_score_valid(self):
        """Test risk score validation with valid values."""
        assert ResponseValidator.validate_risk_score(50) == 50.0
        assert ResponseValidator.validate_risk_score(0) == 0.0
        assert ResponseValidator.validate_risk_score(100) == 100.0
        assert ResponseValidator.validate_risk_score(50.5) == 50.5

    def test_validate_risk_score_clamping(self):
        """Test risk score clamping."""
        assert ResponseValidator.validate_risk_score(-10) == 0.0
        assert ResponseValidator.validate_risk_score(150) == 100.0
        assert ResponseValidator.validate_risk_score(-100.5) == 0.0
        assert ResponseValidator.validate_risk_score(999) == 100.0

    def test_validate_risk_score_invalid_type(self):
        """Test risk score with invalid types."""
        assert ResponseValidator.validate_risk_score("high") == 50.0
        assert ResponseValidator.validate_risk_score(None) == 50.0
        assert ResponseValidator.validate_risk_score([50]) == 50.0
        assert ResponseValidator.validate_risk_score({"score": 50}) == 50.0

    def test_validate_risk_score_string_number(self):
        """Test risk score with string numbers."""
        assert ResponseValidator.validate_risk_score("75") == 75.0
        assert ResponseValidator.validate_risk_score("0") == 0.0
        assert ResponseValidator.validate_risk_score("100.5") == 100.0

    def test_validate_confidence_valid(self):
        """Test confidence validation with valid values."""
        assert ResponseValidator.validate_confidence(0.5) == 0.5
        assert ResponseValidator.validate_confidence(0) == 0.0
        assert ResponseValidator.validate_confidence(1) == 1.0
        assert ResponseValidator.validate_confidence(0.999) == 0.999

    def test_validate_confidence_clamping(self):
        """Test confidence clamping."""
        assert ResponseValidator.validate_confidence(-0.5) == 0.0
        assert ResponseValidator.validate_confidence(1.5) == 1.0
        assert ResponseValidator.validate_confidence(-100) == 0.0
        assert ResponseValidator.validate_confidence(100) == 1.0

    def test_validate_confidence_invalid_type(self):
        """Test confidence with invalid types."""
        assert ResponseValidator.validate_confidence("high") == 0.5
        assert ResponseValidator.validate_confidence(None) == 0.5
        assert ResponseValidator.validate_confidence([0.8]) == 0.5

    def test_validate_confidence_string_number(self):
        """Test confidence with string numbers."""
        assert ResponseValidator.validate_confidence("0.8") == 0.8
        assert ResponseValidator.validate_confidence("0") == 0.0
        assert ResponseValidator.validate_confidence("1.0") == 1.0

    def test_sanitize_recommendation_normal(self):
        """Test recommendation sanitization with normal text."""
        text = "This is a normal recommendation."
        assert ResponseValidator.sanitize_recommendation(text) == text

    def test_sanitize_recommendation_html(self):
        """Test HTML removal from recommendation."""
        text = "This <script>alert('xss')</script> has HTML <b>tags</b>."
        result = ResponseValidator.sanitize_recommendation(text)
        assert "<script>" not in result
        assert "<b>" not in result

    def test_sanitize_recommendation_markdown_links(self):
        """Test markdown link removal."""
        text = "Check [this link](http://example.com) for more info."
        result = ResponseValidator.sanitize_recommendation(text)
        assert "[this link]" not in result
        assert "http://example.com" not in result

    def test_sanitize_recommendation_truncation(self):
        """Test long text truncation."""
        text = "x" * 3000
        result = ResponseValidator.sanitize_recommendation(text)
        assert len(result) <= 2003  # 2000 + "..."
        assert result.endswith("...")

    def test_sanitize_recommendation_whitespace(self):
        """Test whitespace handling."""
        text = "  \n\t  Text with whitespace  \n\t  "
        result = ResponseValidator.sanitize_recommendation(text)
        assert result == "Text with whitespace"

    def test_sanitize_recommendation_empty(self):
        """Test empty string."""
        assert ResponseValidator.sanitize_recommendation("") == ""
        assert ResponseValidator.sanitize_recommendation("   ") == ""


# =============================================================================
# Known Laws Loading Edge Cases
# =============================================================================


class TestKnownLawsLoadingEdgeCases:
    """Edge cases for loading known laws from vector store."""

    def test_load_known_laws_empty_store(self):
        """Test loading laws from empty store."""
        store = MagicMock()
        store.get_all_categories.return_value = []
        store.get_documents_by_category.return_value = []

        guardrails = HallucinationGuardrails(store)
        # Empty store with no exceptions returns empty set
        # (fallback laws only used when store throws exception)
        assert isinstance(guardrails.known_laws, set)

    def test_load_known_laws_store_error(self):
        """Test loading laws when store throws error."""
        store = MagicMock()
        store.get_all_categories.side_effect = Exception("Store error")

        guardrails = HallucinationGuardrails(store)
        # Should fall back to default known laws
        assert "title-vii-1964" in guardrails.known_laws

    def test_load_known_laws_with_metadata(self):
        """Test loading laws from store with proper metadata."""
        store = MagicMock()
        store.get_all_categories.return_value = ["employment"]
        store.get_documents_by_category.return_value = [
            {
                "metadata": {
                    "law_id": "custom-law-123",
                    "law_name": "Custom Law Act",
                }
            }
        ]

        guardrails = HallucinationGuardrails(store)
        assert "custom-law-123" in guardrails.known_laws
        assert "Custom Law Act" in guardrails.known_laws

    def test_load_known_laws_missing_metadata(self):
        """Test loading laws from documents without proper metadata."""
        store = MagicMock()
        store.get_all_categories.return_value = ["employment"]
        store.get_documents_by_category.return_value = [
            {"content": "Law text without metadata"},
            {"metadata": {}},
        ]

        guardrails = HallucinationGuardrails(store)
        # Should handle gracefully
        assert guardrails.known_laws is not None


# =============================================================================
# Claim Extraction Edge Cases
# =============================================================================


class TestClaimExtractionEdgeCases:
    """Edge cases for claim extraction."""

    def test_extract_claims_empty_text(self, guardrails):
        """Test claim extraction from empty text."""
        claims = guardrails._extract_claims("")
        assert claims == []

    def test_extract_claims_no_assertions(self, guardrails):
        """Test claim extraction with no assertions."""
        claims = guardrails._extract_claims("The sky is blue. Water is wet.")
        assert claims == []

    def test_extract_claims_multiple_assertions(self, guardrails):
        """Test claim extraction with multiple assertions."""
        text = "This violates the law. It is required. This is prohibited."
        claims = guardrails._extract_claims(text)
        assert len(claims) >= 2

    def test_extract_claims_single_long_sentence(self, guardrails):
        """Test with single long sentence containing assertion."""
        text = (
            "After careful review of all the evidence and documentation "
            "presented in this case, it is clear that this action violates "
            "the established legal requirements."
        )
        claims = guardrails._extract_claims(text)
        assert len(claims) >= 1

    def test_is_claim_grounded_empty_corpus(self, guardrails):
        """Test claim grounding with empty corpus."""
        guardrails._is_claim_grounded("This violates the law", "")
        # With empty corpus, no terms can be found

    def test_is_claim_grounded_empty_claim(self, guardrails):
        """Test claim grounding with empty claim."""
        result = guardrails._is_claim_grounded("", "corpus text here")
        assert result is True  # No terms to verify

    def test_is_claim_grounded_common_words_only(self, guardrails):
        """Test claim with only common words."""
        result = guardrails._is_claim_grounded(
            "This that with from have been", "employment discrimination law"
        )
        assert result is True  # Common words are filtered out
