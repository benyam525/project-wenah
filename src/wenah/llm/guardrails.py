"""
Hallucination detection and guardrails for LLM outputs.

Implements multiple layers of validation to ensure LLM responses are
grounded in the provided legal context and don't contain fabricated
information.
"""

import re
from typing import Any
from dataclasses import dataclass, field
from enum import Enum

from wenah.config import settings
from wenah.core.types import RAGResponse, GuardrailCheck, ValidationStatus
from wenah.data.vector_store import VectorStore


class GuardrailSeverity(str, Enum):
    """Severity levels for guardrail checks."""

    CRITICAL = "critical"  # Response should be rejected/flagged
    WARNING = "warning"  # Response should be modified/flagged
    INFO = "info"  # Informational, no action needed


@dataclass
class GuardrailResult:
    """Result of all guardrail checks."""

    status: ValidationStatus
    checks: list[GuardrailCheck] = field(default_factory=list)
    modifications_made: list[str] = field(default_factory=list)
    original_confidence: float = 0.0
    adjusted_confidence: float = 0.0


class HallucinationGuardrails:
    """
    Multi-layer hallucination detection for civil rights compliance.

    Strategies:
    1. Citation verification - Check if cited laws/cases exist in database
    2. Cross-reference validation - Verify claims against retrieved documents
    3. Confidence calibration - Adjust confidence based on response patterns
    4. Factual grounding - Ensure claims can be traced to source material
    5. Legal claim verification - Validate legal assertions
    """

    # Known law identifiers in our database
    KNOWN_LAW_PATTERNS = {
        r"title[- ]?vii": "title-vii-1964",
        r"civil rights act.*1964": "title-vii-1964",
        r"42\s*u\.?s\.?c\.?\s*§?\s*2000e": "title-vii-1964",
        r"ada|americans with disabilities": "ada-1990",
        r"42\s*u\.?s\.?c\.?\s*§?\s*12101": "ada-1990",
        r"fair housing": "fha-1968",
        r"42\s*u\.?s\.?c\.?\s*§?\s*3601": "fha-1968",
        r"ecoa|equal credit opportunity": "ecoa-1974",
        r"15\s*u\.?s\.?c\.?\s*§?\s*1691": "ecoa-1974",
        r"fcra|fair credit reporting": "fcra-1970",
        r"15\s*u\.?s\.?c\.?\s*§?\s*1681": "fcra-1970",
    }

    # Hedging language patterns
    HEDGING_PATTERNS = [
        r"\bmay\b",
        r"\bmight\b",
        r"\bcould\b",
        r"\bpossibly\b",
        r"\bpotentially\b",
        r"\bgenerally\b",
        r"\btypically\b",
        r"\busually\b",
        r"\bperhaps\b",
        r"\blikely\b",
        r"\bunlikely\b",
        r"\bprobably\b",
        r"\bit depends\b",
        r"\bin some cases\b",
    ]

    # Assertion patterns that need grounding
    ASSERTION_PATTERNS = [
        r"violates?\b",
        r"prohibited\b",
        r"illegal\b",
        r"unlawful\b",
        r"discrimination\b",
        r"required\b",
        r"must\b",
        r"shall\b",
        r"mandatory\b",
    ]

    def __init__(self, vector_store: VectorStore):
        """
        Initialize guardrails.

        Args:
            vector_store: Vector store for verification queries
        """
        self.vector_store = vector_store
        self.known_laws = self._load_known_laws()

    def validate(
        self,
        response: RAGResponse,
        retrieved_docs: list[dict[str, Any]],
    ) -> RAGResponse:
        """
        Validate and potentially modify LLM response.

        Args:
            response: The RAG response to validate
            retrieved_docs: Documents that were retrieved for context

        Returns:
            Validated (potentially modified) response
        """
        checks = []
        original_confidence = response.confidence_score

        # Check 1: Citation verification
        citation_check = self._verify_citations(response.cited_sources)
        checks.append(citation_check)

        # Check 2: Cross-reference validation (grounding)
        grounding_check = self._verify_grounding(response.analysis, retrieved_docs)
        checks.append(grounding_check)

        # Check 3: Confidence calibration
        confidence_check, confidence_adjustment = self._calibrate_confidence(response)
        checks.append(confidence_check)

        # Check 4: Legal claim verification
        claim_check = self._verify_legal_claims(response.analysis, retrieved_docs)
        checks.append(claim_check)

        # Check 5: Contradiction detection
        contradiction_check = self._check_contradictions(response, retrieved_docs)
        checks.append(contradiction_check)

        # Apply modifications based on checks
        result = self._apply_guardrails(response, checks, confidence_adjustment)

        return result

    def _verify_citations(self, cited_sources: list[str]) -> GuardrailCheck:
        """Verify all cited sources exist in our database."""
        if not cited_sources:
            return GuardrailCheck(
                check_name="citation_verification",
                passed=True,
                details="No citations to verify",
                severity="info",
            )

        invalid_citations = []
        verified_citations = []

        for source in cited_sources:
            source_lower = source.lower()

            # Check against known patterns
            found = False
            for pattern, law_id in self.KNOWN_LAW_PATTERNS.items():
                if re.search(pattern, source_lower):
                    verified_citations.append(source)
                    found = True
                    break

            if not found:
                # Check if it's in our known laws set
                if not any(kl.lower() in source_lower for kl in self.known_laws):
                    invalid_citations.append(source)

        if invalid_citations:
            return GuardrailCheck(
                check_name="citation_verification",
                passed=False,
                details=f"Unverified citations: {invalid_citations[:3]}",
                severity="critical",
            )

        return GuardrailCheck(
            check_name="citation_verification",
            passed=True,
            details=f"Verified {len(verified_citations)} citations",
            severity="info",
        )

    def _verify_grounding(
        self,
        analysis: str,
        retrieved_docs: list[dict[str, Any]],
    ) -> GuardrailCheck:
        """Verify analysis claims are grounded in retrieved documents."""
        if not retrieved_docs:
            return GuardrailCheck(
                check_name="grounding_verification",
                passed=False,
                details="No context documents to verify against",
                severity="warning",
            )

        # Extract key claims from analysis
        claims = self._extract_claims(analysis)

        if not claims:
            return GuardrailCheck(
                check_name="grounding_verification",
                passed=True,
                details="No specific claims to verify",
                severity="info",
            )

        # Build corpus from retrieved documents
        doc_corpus = " ".join(
            doc.get("content", "").lower() for doc in retrieved_docs
        )

        # Check grounding for each claim
        grounded_claims = 0
        ungrounded_claims = []

        for claim in claims:
            if self._is_claim_grounded(claim, doc_corpus):
                grounded_claims += 1
            else:
                ungrounded_claims.append(claim[:50])

        grounding_ratio = grounded_claims / len(claims) if claims else 1.0

        if grounding_ratio < settings.min_grounding_ratio:
            return GuardrailCheck(
                check_name="grounding_verification",
                passed=False,
                details=f"Only {grounding_ratio:.0%} of claims grounded. Ungrounded: {ungrounded_claims[:2]}",
                severity="warning",
            )

        return GuardrailCheck(
            check_name="grounding_verification",
            passed=True,
            details=f"{grounding_ratio:.0%} of claims verified against sources",
            severity="info",
        )

    def _calibrate_confidence(
        self,
        response: RAGResponse,
    ) -> tuple[GuardrailCheck, float]:
        """
        Calibrate confidence score based on response patterns.

        Returns:
            Tuple of (check result, confidence adjustment)
        """
        analysis = response.analysis
        adjustment = 0.0

        # Count hedging language
        hedging_count = sum(
            len(re.findall(pattern, analysis, re.IGNORECASE))
            for pattern in self.HEDGING_PATTERNS
        )

        # Penalize for excessive hedging
        hedging_penalty = min(
            settings.max_hedging_penalty,
            hedging_count * 0.03
        )
        adjustment -= hedging_penalty

        # Penalize for few citations
        if len(response.cited_sources) < 2:
            adjustment -= 0.1

        # Penalize for short analysis (might be incomplete)
        if len(analysis) < 200:
            adjustment -= 0.1

        # Boost for specific legal citations
        citation_pattern = r"\d+\s*u\.?s\.?c\.?\s*§"
        if re.search(citation_pattern, analysis, re.IGNORECASE):
            adjustment += 0.05

        # Calculate if significant adjustment
        if adjustment < -0.2:
            return (
                GuardrailCheck(
                    check_name="confidence_calibration",
                    passed=False,
                    details=f"Confidence reduced by {abs(adjustment):.0%} due to hedging/lack of citations",
                    severity="warning",
                ),
                adjustment,
            )

        return (
            GuardrailCheck(
                check_name="confidence_calibration",
                passed=True,
                details=f"Confidence adjustment: {adjustment:+.0%}",
                severity="info",
            ),
            adjustment,
        )

    def _verify_legal_claims(
        self,
        analysis: str,
        retrieved_docs: list[dict[str, Any]],
    ) -> GuardrailCheck:
        """Verify that strong legal assertions are supported."""
        # Find strong assertions
        strong_assertions = []
        for pattern in self.ASSERTION_PATTERNS:
            matches = re.finditer(pattern, analysis, re.IGNORECASE)
            for match in matches:
                # Get surrounding context
                start = max(0, match.start() - 50)
                end = min(len(analysis), match.end() + 50)
                context = analysis[start:end]
                strong_assertions.append(context)

        if not strong_assertions:
            return GuardrailCheck(
                check_name="legal_claim_verification",
                passed=True,
                details="No strong legal assertions found",
                severity="info",
            )

        # Check if assertions have support in documents
        doc_text = " ".join(doc.get("content", "") for doc in retrieved_docs).lower()

        unsupported = []
        for assertion in strong_assertions[:5]:  # Check first 5
            # Look for key terms from assertion in documents
            key_terms = re.findall(r"\b[a-z]{4,}\b", assertion.lower())
            matches = sum(1 for term in key_terms if term in doc_text)

            if matches < len(key_terms) * 0.3:  # Less than 30% overlap
                unsupported.append(assertion[:30])

        if len(unsupported) > len(strong_assertions) * 0.5:
            return GuardrailCheck(
                check_name="legal_claim_verification",
                passed=False,
                details=f"Multiple unsupported assertions: {unsupported[:2]}",
                severity="warning",
            )

        return GuardrailCheck(
            check_name="legal_claim_verification",
            passed=True,
            details=f"Verified {len(strong_assertions)} legal assertions",
            severity="info",
        )

    def _check_contradictions(
        self,
        response: RAGResponse,
        retrieved_docs: list[dict[str, Any]],
    ) -> GuardrailCheck:
        """Check for contradictions between response and sources."""
        # This is a simplified check - full implementation would use
        # semantic comparison

        analysis_lower = response.analysis.lower()
        doc_text = " ".join(doc.get("content", "") for doc in retrieved_docs).lower()

        contradictions = []

        # Check for explicit contradiction patterns
        contradiction_patterns = [
            (r"not required", r"required|must|shall"),
            (r"permitted", r"prohibited|illegal"),
            (r"no violation", r"violation|violates"),
        ]

        for neg_pattern, pos_pattern in contradiction_patterns:
            if re.search(neg_pattern, analysis_lower):
                # Check if documents say the opposite
                if re.search(pos_pattern, doc_text):
                    # Potential contradiction - needs more context
                    contradictions.append(f"{neg_pattern} vs {pos_pattern}")

        if contradictions:
            return GuardrailCheck(
                check_name="contradiction_detection",
                passed=False,
                details=f"Potential contradictions detected: {contradictions[:2]}",
                severity="warning",
            )

        return GuardrailCheck(
            check_name="contradiction_detection",
            passed=True,
            details="No contradictions detected",
            severity="info",
        )

    def _extract_claims(self, text: str) -> list[str]:
        """Extract verifiable claims from text."""
        claims = []

        # Split into sentences
        sentences = re.split(r"[.!?]\s+", text)

        for sentence in sentences:
            # Look for sentences with assertion patterns
            if any(re.search(p, sentence, re.IGNORECASE) for p in self.ASSERTION_PATTERNS):
                claims.append(sentence.strip())

        return claims

    def _is_claim_grounded(self, claim: str, corpus: str) -> bool:
        """Check if a claim is grounded in the corpus."""
        # Extract key terms from claim
        claim_terms = set(re.findall(r"\b[a-z]{4,}\b", claim.lower()))

        # Remove common words
        common_words = {
            "that", "this", "with", "from", "have", "been",
            "would", "could", "should", "their", "there",
            "which", "where", "when", "what", "will",
        }
        claim_terms -= common_words

        if not claim_terms:
            return True  # No specific terms to verify

        # Check how many terms appear in corpus
        matches = sum(1 for term in claim_terms if term in corpus)
        match_ratio = matches / len(claim_terms)

        return match_ratio >= 0.4  # At least 40% of terms should appear

    def _apply_guardrails(
        self,
        response: RAGResponse,
        checks: list[GuardrailCheck],
        confidence_adjustment: float,
    ) -> RAGResponse:
        """Apply guardrail modifications to response."""
        # Count failures by severity
        critical_failures = [
            c for c in checks if not c.passed and c.severity == "critical"
        ]
        warnings = [
            c for c in checks if not c.passed and c.severity == "warning"
        ]

        # Calculate new confidence
        new_confidence = response.confidence_score + confidence_adjustment

        # Apply additional penalties for failures
        if critical_failures:
            new_confidence = min(new_confidence, 0.3)
            response.requires_human_review = True
        elif warnings:
            penalty = len(warnings) * 0.1
            new_confidence = max(0.1, new_confidence - penalty)

        # Clamp confidence
        new_confidence = max(0.0, min(1.0, new_confidence))
        response.confidence_score = new_confidence

        # Add guardrail alerts to analysis if critical
        if critical_failures:
            alert = "[GUARDRAIL ALERT] "
            alert += "; ".join(c.details for c in critical_failures)
            response.analysis = f"{alert}\n\n{response.analysis}"
            response.requires_human_review = True

        return response

    def _load_known_laws(self) -> set[str]:
        """Load known law identifiers from vector store."""
        try:
            categories = self.vector_store.get_all_categories()
            known = set()

            for category in categories:
                docs = self.vector_store.get_documents_by_category(category, limit=100)
                for doc in docs:
                    metadata = doc.get("metadata", {})
                    if law_id := metadata.get("law_id"):
                        known.add(law_id)
                    if law_name := metadata.get("law_name"):
                        known.add(law_name)

            return known
        except Exception:
            # Return basic known laws if vector store is empty
            return {
                "title-vii-1964",
                "ada-1990",
                "fha-1968",
                "ecoa-1974",
                "fcra-1970",
                "Title VII",
                "Americans with Disabilities Act",
                "Fair Housing Act",
                "Equal Credit Opportunity Act",
                "Fair Credit Reporting Act",
            }


class ResponseValidator:
    """
    Additional validation utilities for LLM responses.
    """

    @staticmethod
    def validate_json_structure(
        response: dict[str, Any],
        required_fields: list[str],
    ) -> tuple[bool, list[str]]:
        """
        Validate that response has required fields.

        Returns:
            Tuple of (is_valid, missing_fields)
        """
        missing = [f for f in required_fields if f not in response]
        return len(missing) == 0, missing

    @staticmethod
    def validate_risk_score(score: Any) -> float:
        """Validate and normalize a risk score."""
        try:
            score = float(score)
            return max(0.0, min(100.0, score))
        except (TypeError, ValueError):
            return 50.0  # Default to medium

    @staticmethod
    def validate_confidence(confidence: Any) -> float:
        """Validate and normalize a confidence score."""
        try:
            conf = float(confidence)
            return max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            return 0.5  # Default to medium confidence

    @staticmethod
    def sanitize_recommendation(text: str) -> str:
        """Sanitize recommendation text."""
        # Remove potential injection patterns
        text = re.sub(r"<[^>]+>", "", text)  # Remove HTML
        text = re.sub(r"\[.*?\]\(.*?\)", "", text)  # Remove markdown links
        text = text.strip()

        # Truncate if too long
        if len(text) > 2000:
            text = text[:2000] + "..."

        return text
