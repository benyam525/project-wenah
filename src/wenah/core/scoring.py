"""
Unified scoring system for civil rights compliance assessment.

Combines rule-based evaluations with LLM analysis to produce
integrated risk scores with explanations and confidence intervals.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from wenah.config import settings
from wenah.core.types import (
    CategoryBreakdown,
    ComponentScore,
    RAGResponse,
    RiskLevel,
    RuleEvaluation,
    RuleResult,
    UnifiedRiskScore,
)


class ScoreSource(str, Enum):
    """Sources of scoring components."""

    RULE_ENGINE = "rule_engine"
    LLM_ANALYSIS = "llm_analysis"
    CATEGORY_PROCESSOR = "category_processor"


@dataclass
class ScoringContext:
    """Context for scoring calculations."""

    feature_id: str
    feature_name: str
    category: str
    rule_evaluations: list[RuleEvaluation] = field(default_factory=list)
    rag_response: RAGResponse | None = None
    category_analysis: dict[str, Any] | None = None


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of how a score was calculated."""

    component_scores: list[ComponentScore]
    weight_adjustments: list[dict[str, Any]]
    confidence_factors: list[dict[str, Any]]
    explanation: str


class ScoringEngine:
    """
    Unified scoring engine combining rule-based and LLM assessments.

    Scoring Philosophy:
    - Rule violations are weighted higher (deterministic, clear law violations)
    - LLM analysis provides nuance and catches edge cases
    - Final score considers confidence levels from both sources
    - Critical violations can override normal scoring

    Score Calculation:
    1. Calculate weighted rule engine score
    2. Calculate weighted LLM score (if available)
    3. Apply dynamic weight adjustments based on context
    4. Generate unified score with confidence interval
    5. Produce human-readable explanations
    """

    def __init__(
        self,
        rule_base_weight: float | None = None,
        llm_base_weight: float | None = None,
        high_confidence_threshold: float | None = None,
        low_confidence_threshold: float | None = None,
    ):
        """
        Initialize the scoring engine.

        Args:
            rule_base_weight: Base weight for rule engine (default from config)
            llm_base_weight: Base weight for LLM analysis (default from config)
            high_confidence_threshold: Threshold for high confidence
            low_confidence_threshold: Threshold for low confidence
        """
        self.rule_base_weight = rule_base_weight or settings.rule_engine_base_weight
        self.llm_base_weight = llm_base_weight or settings.llm_analysis_base_weight
        self.high_confidence_threshold = (
            high_confidence_threshold or settings.high_confidence_threshold
        )
        self.low_confidence_threshold = (
            low_confidence_threshold or settings.low_confidence_threshold
        )

    def calculate_unified_score(
        self,
        rule_evaluations: list[RuleEvaluation],
        rag_response: RAGResponse | None = None,
        category_analysis: dict[str, Any] | None = None,
    ) -> UnifiedRiskScore:
        """
        Calculate unified risk score from all components.

        Args:
            rule_evaluations: Results from rule engine
            rag_response: Results from RAG pipeline (optional)
            category_analysis: Results from category processor (optional)

        Returns:
            Unified risk score with all breakdowns
        """
        component_scores = []

        # Calculate rule engine component
        rule_score = self._calculate_rule_score(rule_evaluations)
        component_scores.append(rule_score)

        # Calculate LLM component if available
        if rag_response:
            llm_score = self._calculate_llm_score(rag_response)
            component_scores.append(llm_score)

        # Check for critical violations that override normal scoring
        has_critical = self._has_critical_violation(rule_evaluations)

        # Adjust weights based on context
        adjusted_scores = self._adjust_weights(
            component_scores,
            has_critical=has_critical,
            rule_evaluations=rule_evaluations,
            rag_response=rag_response,
        )

        # Calculate final unified score
        overall_score = self._combine_scores(adjusted_scores)

        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(adjusted_scores, overall_score)

        # Determine risk level
        risk_level = self._score_to_risk_level(overall_score)

        # Check if human review is needed
        requires_human_review, review_reasons = self._check_human_review_needed(
            adjusted_scores, rule_evaluations, rag_response
        )

        # Generate category breakdown
        category_breakdown = self._calculate_category_breakdown(rule_evaluations)

        # Extract concerns and recommendations
        primary_concerns = self._extract_primary_concerns(rule_evaluations, rag_response)
        recommendations = self._generate_recommendations(
            rule_evaluations, rag_response, category_analysis
        )

        return UnifiedRiskScore(
            overall_score=overall_score,
            risk_level=risk_level,
            component_scores=adjusted_scores,
            category_breakdown=category_breakdown,
            confidence_interval=confidence_interval,
            primary_concerns=primary_concerns,
            recommendations=recommendations,
            requires_human_review=requires_human_review,
        )

    def _calculate_rule_score(
        self,
        evaluations: list[RuleEvaluation],
    ) -> ComponentScore:
        """Calculate weighted score from rule evaluations."""
        if not evaluations:
            return ComponentScore(
                source=ScoreSource.RULE_ENGINE.value,
                raw_score=0.0,
                confidence=1.0,
                weight=self.rule_base_weight,
                weighted_score=0.0,
                explanation="No applicable rules triggered",
            )

        # Separate by result type
        violations = [e for e in evaluations if e.result == RuleResult.VIOLATION]
        potential = [e for e in evaluations if e.result == RuleResult.POTENTIAL_VIOLATION]
        [e for e in evaluations if e.result == RuleResult.COMPLIANT]

        # Check for critical violations
        critical_violations = [e for e in violations if e.risk_score >= 80]

        if critical_violations:
            # Critical violation takes precedence
            max_eval = max(critical_violations, key=lambda e: e.risk_score)
            raw_score = float(max_eval.risk_score)
            confidence = float(max_eval.confidence)
            explanation = f"Critical violation: {max_eval.rule_name}"
        elif violations:
            # Multiple violations - use weighted average
            total_weight = sum(e.confidence for e in violations)
            if total_weight > 0:
                raw_score = sum(e.risk_score * e.confidence for e in violations) / total_weight
            else:
                # Fallback to simple average if all confidences are 0
                raw_score = sum(e.risk_score for e in violations) / len(violations)
            confidence = sum(e.confidence for e in violations) / len(violations)
            explanation = f"{len(violations)} violation(s) detected"
        elif potential:
            # Potential violations only
            total_weight = sum(e.confidence for e in potential)
            if total_weight > 0:
                raw_score = sum(e.risk_score * e.confidence for e in potential) / total_weight
            else:
                # Fallback to simple average if all confidences are 0
                raw_score = sum(e.risk_score for e in potential) / len(potential)
            # Lower confidence for potential violations
            confidence = sum(e.confidence for e in potential) / len(potential) * 0.8
            explanation = f"{len(potential)} potential violation(s) requiring review"
        else:
            # No violations - check for positive indicators
            positive = [e for e in evaluations if e.risk_score < 0]
            if positive:
                raw_score = max(0, 20 + sum(e.risk_score for e in positive))
                explanation = "Compliant with positive indicators"
            else:
                raw_score = 15.0  # Base low risk for no findings
                explanation = "No violations detected"
            confidence = 1.0

        return ComponentScore(
            source=ScoreSource.RULE_ENGINE.value,
            raw_score=raw_score,
            confidence=confidence,
            weight=self.rule_base_weight,
            weighted_score=raw_score * confidence * self.rule_base_weight,
            explanation=explanation,
        )

    def _calculate_llm_score(self, response: RAGResponse) -> ComponentScore:
        """Calculate weighted score from LLM analysis."""
        # Map LLM analysis to numeric score
        raw_score = self._llm_analysis_to_score(response)
        confidence = response.confidence_score

        # Build explanation
        if response.risk_factors:
            explanation = f"LLM identified {len(response.risk_factors)} risk factor(s)"
        else:
            explanation = "LLM analysis completed"

        if response.requires_human_review:
            explanation += " (human review recommended)"

        return ComponentScore(
            source=ScoreSource.LLM_ANALYSIS.value,
            raw_score=raw_score,
            confidence=confidence,
            weight=self.llm_base_weight,
            weighted_score=raw_score * confidence * self.llm_base_weight,
            explanation=explanation,
        )

    def _llm_analysis_to_score(self, response: RAGResponse) -> float:
        """Convert LLM analysis to numeric risk score."""
        base_score = 30.0  # Start at low-medium

        # Adjust based on risk factors
        risk_factor_weight = len(response.risk_factors) * 10
        base_score += min(risk_factor_weight, 40)

        # Adjust based on mitigating factors
        mitigating_weight = len(response.mitigating_factors) * 8
        base_score -= min(mitigating_weight, 25)

        # Adjust if human review required
        if response.requires_human_review:
            base_score += 15

        return max(0, min(100, base_score))

    def _has_critical_violation(
        self,
        evaluations: list[RuleEvaluation],
    ) -> bool:
        """Check if any evaluation represents a critical violation."""
        return any(e.result == RuleResult.VIOLATION and e.risk_score >= 80 for e in evaluations)

    def _adjust_weights(
        self,
        scores: list[ComponentScore],
        has_critical: bool,
        rule_evaluations: list[RuleEvaluation],
        rag_response: RAGResponse | None,
    ) -> list[ComponentScore]:
        """Adjust component weights based on context."""
        adjusted = []

        for score in scores:
            new_weight = score.weight

            if score.source == ScoreSource.RULE_ENGINE.value:
                # Increase rule weight for critical violations
                if has_critical:
                    new_weight = min(0.9, new_weight + 0.2)

            elif score.source == ScoreSource.LLM_ANALYSIS.value:
                # Decrease LLM weight for low confidence
                if score.confidence < self.low_confidence_threshold:
                    new_weight = new_weight * 0.5

                # Decrease LLM weight if critical rule violation exists
                if has_critical:
                    new_weight = max(0.1, new_weight - 0.2)

            # Recalculate weighted score with new weight
            new_weighted = score.raw_score * score.confidence * new_weight

            adjusted.append(
                ComponentScore(
                    source=score.source,
                    raw_score=score.raw_score,
                    confidence=score.confidence,
                    weight=new_weight,
                    weighted_score=new_weighted,
                    explanation=score.explanation,
                )
            )

        return adjusted

    def _combine_scores(self, scores: list[ComponentScore]) -> float:
        """Combine component scores into unified score."""
        if not scores:
            return 0.0

        # Normalize weights to sum to 1.0
        total_weight = sum(s.weight for s in scores)
        if total_weight == 0:
            return 0.0

        # Calculate weighted average
        weighted_sum = sum(s.weighted_score for s in scores)
        (weighted_sum / total_weight) * (
            100 / max(s.raw_score for s in scores) if any(s.raw_score > 0 for s in scores) else 1
        )

        # Simpler calculation: direct weighted average
        final_score = sum((s.raw_score * s.confidence * s.weight) / total_weight for s in scores)

        return max(0, min(100, final_score))

    def _calculate_confidence_interval(
        self,
        scores: list[ComponentScore],
        overall_score: float,
    ) -> tuple[float, float]:
        """Calculate confidence interval for the score."""
        if not scores:
            return (0, 100)

        # Calculate average confidence
        avg_confidence = sum(s.confidence for s in scores) / len(scores)

        # Wider margin for lower confidence
        margin = (1 - avg_confidence) * 20

        # Additional margin for mixed signals
        score_spread = max(s.raw_score for s in scores) - min(s.raw_score for s in scores)
        if score_spread > 30:
            margin += 5

        low = max(0, overall_score - margin)
        high = min(100, overall_score + margin)

        return (round(low, 1), round(high, 1))

    def _score_to_risk_level(self, score: float) -> RiskLevel:
        """Convert numeric score to risk level."""
        if score >= 80:
            return RiskLevel.CRITICAL
        elif score >= 60:
            return RiskLevel.HIGH
        elif score >= 40:
            return RiskLevel.MEDIUM
        elif score >= 20:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL

    def _check_human_review_needed(
        self,
        scores: list[ComponentScore],
        rule_evaluations: list[RuleEvaluation],
        rag_response: RAGResponse | None,
    ) -> tuple[bool, list[str]]:
        """Determine if human review is needed."""
        reasons = []

        # Check average confidence
        avg_confidence = sum(s.confidence for s in scores) / len(scores) if scores else 0
        if avg_confidence < self.low_confidence_threshold:
            reasons.append(f"Low confidence ({avg_confidence:.0%})")

        # Check for critical or high risk
        overall = self._combine_scores(scores)
        if overall >= 60:
            reasons.append(f"High risk score ({overall:.0f})")

        # Check for escalated rules (only if risk is at least medium)
        escalated = [e for e in rule_evaluations if e.escalate_to_llm]
        if escalated and overall >= 40:
            reasons.append(f"{len(escalated)} rule(s) require nuanced analysis")

        # Check LLM recommendation (only for medium+ risk to avoid false alarms)
        if rag_response and rag_response.requires_human_review and overall >= 40:
            reasons.append("LLM recommends human review")

        # Check for conflicting signals
        if scores and len(scores) > 1:
            score_values = [s.raw_score for s in scores]
            if max(score_values) - min(score_values) > 40:
                reasons.append("Conflicting signals between rule and LLM analysis")

        return len(reasons) > 0, reasons

    def _calculate_category_breakdown(
        self,
        evaluations: list[RuleEvaluation],
    ) -> CategoryBreakdown:
        """Calculate risk breakdown by law category."""
        # Group evaluations by law reference
        employment_scores = []
        housing_scores = []
        consumer_scores = []

        for eval in evaluations:
            for ref in eval.law_references:
                ref_lower = ref.lower()
                if "title" in ref_lower or "ada" in ref_lower or "employment" in ref_lower:
                    employment_scores.append(eval.risk_score * eval.confidence)
                elif "housing" in ref_lower or "fha" in ref_lower:
                    housing_scores.append(eval.risk_score * eval.confidence)
                elif "ecoa" in ref_lower or "fcra" in ref_lower or "consumer" in ref_lower:
                    consumer_scores.append(eval.risk_score * eval.confidence)

        def avg_score(scores: list[float]) -> float:
            return sum(scores) / len(scores) if scores else 0.0

        employment = avg_score(employment_scores)
        housing = avg_score(housing_scores)
        consumer = avg_score(consumer_scores)

        # Overall is max of categories (most concerning)
        overall = (
            max(employment, housing, consumer) if any([employment, housing, consumer]) else 0.0
        )

        return CategoryBreakdown(
            employment=round(employment, 1),
            housing=round(housing, 1),
            consumer=round(consumer, 1),
            overall=round(overall, 1),
        )

    def _extract_primary_concerns(
        self,
        evaluations: list[RuleEvaluation],
        rag_response: RAGResponse | None,
    ) -> list[str]:
        """Extract primary compliance concerns."""
        concerns = []

        # From rule evaluations
        violations = sorted(
            [
                e
                for e in evaluations
                if e.result in [RuleResult.VIOLATION, RuleResult.POTENTIAL_VIOLATION]
            ],
            key=lambda e: e.risk_score,
            reverse=True,
        )

        for v in violations[:3]:  # Top 3 violations
            concerns.append(
                f"{v.rule_name}: {v.recommendations[0] if v.recommendations else 'Review required'}"
            )

        # From LLM analysis
        if rag_response:
            concerns.extend(rag_response.risk_factors[:2])

        return concerns[:5]  # Limit to 5

    def _generate_recommendations(
        self,
        evaluations: list[RuleEvaluation],
        rag_response: RAGResponse | None,
        category_analysis: dict[str, Any] | None,
    ) -> list[str]:
        """Generate prioritized recommendations."""
        recommendations = []
        seen = set()

        # Priority 1: Critical violations
        critical = [e for e in evaluations if e.risk_score >= 80]
        for e in critical:
            for rec in e.recommendations:
                if rec not in seen:
                    recommendations.append(rec)
                    seen.add(rec)

        # Priority 2: High-risk violations
        high = [e for e in evaluations if 60 <= e.risk_score < 80]
        for e in high:
            for rec in e.recommendations:
                if rec not in seen:
                    recommendations.append(rec)
                    seen.add(rec)

        # Priority 3: LLM recommendations
        if rag_response and rag_response.recommendation:
            if rag_response.recommendation not in seen:
                recommendations.append(rag_response.recommendation)

        # Priority 4: Category processor recommendations
        if category_analysis:
            for rec in category_analysis.get("recommendations", []):
                rec_text = rec.get("recommendation", "") if isinstance(rec, dict) else str(rec)
                if rec_text and rec_text not in seen:
                    recommendations.append(rec_text)
                    seen.add(rec_text)

        return recommendations[:10]  # Limit to 10


class ScoreExplainer:
    """
    Generates human-readable explanations for compliance scores.
    """

    def explain_score(
        self,
        unified_score: UnifiedRiskScore,
        detail_level: str = "standard",
    ) -> str:
        """
        Generate explanation for a unified score.

        Args:
            unified_score: The score to explain
            detail_level: "brief", "standard", or "detailed"

        Returns:
            Human-readable explanation
        """
        parts = []

        # Overall summary
        parts.append(self._explain_overall(unified_score))

        if detail_level in ["standard", "detailed"]:
            # Component breakdown
            parts.append(self._explain_components(unified_score.component_scores))

            # Key concerns
            if unified_score.primary_concerns:
                parts.append(self._explain_concerns(unified_score.primary_concerns))

        if detail_level == "detailed":
            # Category breakdown
            parts.append(self._explain_categories(unified_score.category_breakdown))

            # Confidence interval
            parts.append(self._explain_confidence(unified_score))

        # Recommendations summary
        if unified_score.recommendations:
            parts.append(
                self._explain_recommendations(
                    unified_score.recommendations,
                    detail_level,
                )
            )

        return "\n\n".join(parts)

    def _explain_overall(self, score: UnifiedRiskScore) -> str:
        """Explain overall score."""
        level_descriptions = {
            RiskLevel.CRITICAL: "Critical compliance risk requiring immediate action",
            RiskLevel.HIGH: "Significant compliance risk requiring strong remediation",
            RiskLevel.MEDIUM: "Moderate compliance risk requiring review",
            RiskLevel.LOW: "Minor compliance concerns to monitor",
            RiskLevel.MINIMAL: "Low compliance risk - mostly compliant",
        }

        description = level_descriptions.get(score.risk_level, "Unknown risk level")

        text = f"**Overall Risk Score: {score.overall_score:.0f}/100 ({score.risk_level.value.upper()})**\n"
        text += f"{description}."

        if score.requires_human_review:
            text += "\n\n⚠️ **Human review recommended** for this assessment."

        return text

    def _explain_components(self, components: list[ComponentScore]) -> str:
        """Explain component scores."""
        lines = ["**Score Components:**"]

        for comp in components:
            source_name = comp.source.replace("_", " ").title()
            lines.append(
                f"- {source_name}: {comp.raw_score:.0f} "
                f"(confidence: {comp.confidence:.0%}, weight: {comp.weight:.0%})"
            )
            lines.append(f"  └─ {comp.explanation}")

        return "\n".join(lines)

    def _explain_concerns(self, concerns: list[str]) -> str:
        """Explain primary concerns."""
        lines = ["**Primary Concerns:**"]
        for i, concern in enumerate(concerns, 1):
            lines.append(f"{i}. {concern}")
        return "\n".join(lines)

    def _explain_categories(self, breakdown: CategoryBreakdown) -> str:
        """Explain category breakdown."""
        lines = ["**Risk by Category:**"]

        categories = [
            ("Employment (Title VII, ADA)", breakdown.employment),
            ("Housing (FHA)", breakdown.housing),
            ("Consumer (ECOA, FCRA)", breakdown.consumer),
        ]

        for name, score in categories:
            if score > 0:
                level = self._score_to_level_name(score)
                lines.append(f"- {name}: {score:.0f}/100 ({level})")

        return "\n".join(lines)

    def _explain_confidence(self, score: UnifiedRiskScore) -> str:
        """Explain confidence interval."""
        low, high = score.confidence_interval
        return (
            f"**Confidence Interval:** {low:.0f} - {high:.0f}\n"
            f"The actual risk score likely falls within this range based on "
            f"the confidence levels of the underlying analysis."
        )

    def _explain_recommendations(
        self,
        recommendations: list[str],
        detail_level: str,
    ) -> str:
        """Explain recommendations."""
        lines = ["**Recommended Actions:**"]

        limit = 3 if detail_level == "brief" else 5 if detail_level == "standard" else 10

        for i, rec in enumerate(recommendations[:limit], 1):
            # Truncate long recommendations
            if len(rec) > 200:
                rec = rec[:200] + "..."
            lines.append(f"{i}. {rec}")

        if len(recommendations) > limit:
            lines.append(f"... and {len(recommendations) - limit} more recommendations")

        return "\n".join(lines)

    def _score_to_level_name(self, score: float) -> str:
        """Convert score to risk level name."""
        if score >= 80:
            return "Critical"
        elif score >= 60:
            return "High"
        elif score >= 40:
            return "Medium"
        elif score >= 20:
            return "Low"
        else:
            return "Minimal"


# Singleton instances
_scoring_engine: ScoringEngine | None = None
_score_explainer: ScoreExplainer | None = None


def get_scoring_engine() -> ScoringEngine:
    """Get singleton scoring engine instance."""
    global _scoring_engine
    if _scoring_engine is None:
        _scoring_engine = ScoringEngine()
    return _scoring_engine


def get_score_explainer() -> ScoreExplainer:
    """Get singleton score explainer instance."""
    global _score_explainer
    if _score_explainer is None:
        _score_explainer = ScoreExplainer()
    return _score_explainer
