"""
Main compliance engine orchestrator.

Coordinates all components (rule engine, RAG pipeline, scoring) to
produce comprehensive compliance assessments.
"""

from typing import Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid

from wenah.core.types import (
    ProductFeatureInput,
    RuleEvaluation,
    RAGResponse,
    UnifiedRiskScore,
    RiskAssessmentResponse,
    FeatureAssessment,
    ViolationDetail,
    RecommendationItem,
    RiskLevel,
    RuleResult,
)
from wenah.core.scoring import (
    ScoringEngine,
    ScoreExplainer,
    get_scoring_engine,
    get_score_explainer,
)
from wenah.rules.rule_engine import RuleEngine, get_rule_engine
from wenah.rules.categories.employment import (
    EmploymentCategoryProcessor,
    get_employment_processor,
)
from wenah.config import settings

# Optional RAG pipeline - requires heavy ML dependencies
try:
    from wenah.llm.rag_pipeline import RAGPipeline, get_rag_pipeline, RAGResult
    RAG_AVAILABLE = True
except ImportError:
    RAGPipeline = None
    get_rag_pipeline = None
    RAGResult = None
    RAG_AVAILABLE = False


@dataclass
class AssessmentConfig:
    """Configuration for compliance assessment."""

    include_llm_analysis: bool = True
    include_category_analysis: bool = True
    top_k_documents: int = 10
    apply_guardrails: bool = True
    detail_level: str = "standard"  # brief, standard, detailed


@dataclass
class FeatureAnalysis:
    """Complete analysis results for a single feature."""

    feature: ProductFeatureInput
    rule_evaluations: list[RuleEvaluation] = field(default_factory=list)
    rag_result: RAGResult | None = None
    category_analysis: dict[str, Any] | None = None
    unified_score: UnifiedRiskScore | None = None
    explanation: str = ""


class ComplianceEngine:
    """
    Main orchestrator for civil rights compliance assessment.

    Coordinates:
    1. Rule engine evaluation
    2. Category-specific analysis
    3. RAG pipeline for nuanced analysis
    4. Unified scoring
    5. Report generation
    """

    def __init__(
        self,
        rule_engine: RuleEngine | None = None,
        rag_pipeline: RAGPipeline | None = None,
        scoring_engine: ScoringEngine | None = None,
        score_explainer: ScoreExplainer | None = None,
    ):
        """
        Initialize the compliance engine.

        Args:
            rule_engine: Rule engine instance
            rag_pipeline: RAG pipeline instance
            scoring_engine: Scoring engine instance
            score_explainer: Score explainer instance
        """
        self.rule_engine = rule_engine or get_rule_engine()
        self.rag_pipeline = rag_pipeline
        self.scoring_engine = scoring_engine or get_scoring_engine()
        self.score_explainer = score_explainer or get_score_explainer()

        # Category processors
        self.employment_processor = get_employment_processor()

        # Lazy load RAG pipeline (requires API key)
        self._rag_pipeline_initialized = False

    @property
    def rag(self) -> RAGPipeline:
        """Lazy-load RAG pipeline."""
        if not self._rag_pipeline_initialized:
            if self.rag_pipeline is None:
                self.rag_pipeline = get_rag_pipeline()
            self._rag_pipeline_initialized = True
        return self.rag_pipeline

    def assess_feature(
        self,
        feature: ProductFeatureInput,
        config: AssessmentConfig | None = None,
    ) -> FeatureAnalysis:
        """
        Perform comprehensive assessment of a single feature.

        Args:
            feature: The feature to assess
            config: Assessment configuration

        Returns:
            Complete feature analysis
        """
        config = config or AssessmentConfig()
        analysis = FeatureAnalysis(feature=feature)

        # Step 1: Rule engine evaluation
        analysis.rule_evaluations = self.rule_engine.evaluate(feature)

        # Step 2: Category-specific analysis
        if config.include_category_analysis:
            analysis.category_analysis = self._run_category_analysis(feature)

            # Convert category findings to rule evaluations for proper scoring
            category_evaluations = self._convert_category_findings_to_evaluations(
                feature, analysis.category_analysis
            )
            analysis.rule_evaluations.extend(category_evaluations)

        # Step 3: RAG analysis for escalated rules or high-risk items
        if config.include_llm_analysis:
            analysis.rag_result = self._run_rag_analysis(
                feature=feature,
                rule_evaluations=analysis.rule_evaluations,
                config=config,
            )

        # Step 4: Calculate unified score
        rag_response = analysis.rag_result.response if analysis.rag_result else None
        analysis.unified_score = self.scoring_engine.calculate_unified_score(
            rule_evaluations=analysis.rule_evaluations,
            rag_response=rag_response,
            category_analysis=analysis.category_analysis,
        )

        # Step 5: Generate explanation
        if analysis.unified_score:
            analysis.explanation = self.score_explainer.explain_score(
                analysis.unified_score,
                detail_level=config.detail_level,
            )

        return analysis

    def assess_product(
        self,
        product_name: str,
        features: list[ProductFeatureInput],
        config: AssessmentConfig | None = None,
    ) -> RiskAssessmentResponse:
        """
        Assess a complete product with multiple features.

        Args:
            product_name: Name of the product
            features: List of features to assess
            config: Assessment configuration

        Returns:
            Complete risk assessment response
        """
        config = config or AssessmentConfig()
        assessment_id = str(uuid.uuid4())[:8]

        # Assess each feature
        feature_analyses = []
        for feature in features:
            analysis = self.assess_feature(feature, config)
            feature_analyses.append(analysis)

        # Build feature assessments
        feature_assessments = [
            self._build_feature_assessment(analysis)
            for analysis in feature_analyses
        ]

        # Calculate overall scores
        overall_score, overall_level = self._calculate_overall_scores(feature_analyses)
        confidence_score = self._calculate_overall_confidence(feature_analyses)
        confidence_interval = self._calculate_overall_confidence_interval(feature_analyses)

        # Aggregate category breakdown
        category_breakdown = self._aggregate_category_breakdown(feature_analyses)

        # Collect all violations
        all_violations = []
        critical_violations = []
        for analysis in feature_analyses:
            violations = self._extract_violations(analysis)
            all_violations.extend(violations)
            critical_violations.extend([v for v in violations if v.severity == RiskLevel.CRITICAL])

        # Collect all recommendations
        all_recommendations = self._aggregate_recommendations(feature_analyses)

        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            product_name=product_name,
            overall_score=overall_score,
            overall_level=overall_level,
            feature_count=len(features),
            violation_count=len(all_violations),
        )

        # Determine human review needs
        requires_human_review, review_reasons = self._aggregate_human_review_needs(
            feature_analyses
        )

        # Extract key concerns and positive aspects
        key_concerns = self._extract_key_concerns(feature_analyses)
        positive_aspects = self._extract_positive_aspects(feature_analyses)

        return RiskAssessmentResponse(
            assessment_id=assessment_id,
            product_name=product_name,
            timestamp=datetime.now(timezone.utc),
            overall_risk_score=overall_score,
            overall_risk_level=overall_level,
            confidence_score=confidence_score,
            confidence_interval=confidence_interval,
            category_breakdown=category_breakdown,
            feature_assessments=feature_assessments,
            total_violations=len(all_violations),
            critical_violations=critical_violations,
            all_recommendations=all_recommendations,
            executive_summary=executive_summary,
            key_concerns=key_concerns,
            positive_aspects=positive_aspects,
            requires_human_review=requires_human_review,
            human_review_reasons=review_reasons,
            rules_version="1.0.0",
            laws_data_version="1.0.0",
        )

    def quick_assess(
        self,
        feature: ProductFeatureInput,
    ) -> dict[str, Any]:
        """
        Perform quick rule-only assessment (no LLM).

        Args:
            feature: The feature to assess

        Returns:
            Quick assessment results
        """
        config = AssessmentConfig(
            include_llm_analysis=False,
            include_category_analysis=True,
        )

        analysis = self.assess_feature(feature, config)

        return {
            "feature_id": feature.feature_id,
            "risk_score": analysis.unified_score.overall_score if analysis.unified_score else 0,
            "risk_level": analysis.unified_score.risk_level.value if analysis.unified_score else "unknown",
            "violations_count": len([
                e for e in analysis.rule_evaluations
                if e.result == RuleResult.VIOLATION
            ]),
            "requires_full_analysis": any(
                e.escalate_to_llm for e in analysis.rule_evaluations
            ),
            "primary_concerns": analysis.unified_score.primary_concerns[:3] if analysis.unified_score else [],
        }

    def _run_category_analysis(
        self,
        feature: ProductFeatureInput,
    ) -> dict[str, Any] | None:
        """Run category-specific analysis."""
        category = feature.category.value

        if category == "hiring":
            return self.employment_processor.analyze_feature(feature)

        # Other categories return stub analysis for now
        return {
            "status": "basic_analysis",
            "category": category,
            "findings": [],
            "recommendations": [],
        }

    def _convert_category_findings_to_evaluations(
        self,
        feature: ProductFeatureInput,
        category_analysis: dict[str, Any] | None,
    ) -> list[RuleEvaluation]:
        """
        Convert category analysis findings into RuleEvaluations for scoring.

        This ensures proxy variables and other findings are properly scored.
        """
        if not category_analysis:
            return []

        evaluations = []
        eval_id = 0

        # Convert proxy variable concerns to high-severity violations
        proxy_concerns = category_analysis.get("proxy_variable_concerns", [])
        for proxy in proxy_concerns:
            field_name = proxy.get("field", "unknown")
            proxy_for = proxy.get("proxy_for") or proxy.get("annotated_proxy_for")
            used_in_decisions = proxy.get("used_in_decisions", False)

            if proxy_for:
                proxy_list = proxy_for if isinstance(proxy_for, list) else [proxy_for]
                proxy_str = ", ".join(proxy_list)

                # Higher score if used in decisions
                risk_score = 75 if used_in_decisions else 55

                evaluations.append(RuleEvaluation(
                    rule_id=f"proxy-{eval_id}",
                    rule_name=f"Proxy Variable Detection: {field_name}",
                    result=RuleResult.VIOLATION if used_in_decisions else RuleResult.POTENTIAL_VIOLATION,
                    confidence=0.85,
                    risk_score=risk_score,
                    law_references=["Title VII", "ECOA"],
                    recommendations=[
                        f"Remove '{field_name}' from decision inputs or justify business necessity",
                        f"Conduct disparate impact analysis for {proxy_str} correlation",
                    ],
                    escalate_to_llm=True,
                    llm_context={"field": field_name, "proxy_for": proxy_str},
                ))
                eval_id += 1

        # Convert protected class exposure to critical violations
        protected_exposure = category_analysis.get("protected_class_exposure", [])
        for exposure in protected_exposure:
            field_name = exposure.get("field", "unknown")
            protected_class = exposure.get("protected_class", "unknown")
            used_in_decisions = exposure.get("used_in_decisions", False)

            risk_score = 90 if used_in_decisions else 70

            evaluations.append(RuleEvaluation(
                rule_id=f"protected-{eval_id}",
                rule_name=f"Protected Class Data: {protected_class}",
                result=RuleResult.VIOLATION,
                confidence=0.95,
                risk_score=risk_score,
                law_references=["Title VII", "ADA", "ADEA"],
                recommendations=[
                    f"Remove '{field_name}' from all decision-making processes",
                    "If required for EEO reporting, collect separately post-decision",
                ],
                escalate_to_llm=False,
                llm_context=None,
            ))
            eval_id += 1

        # Convert ADA concerns to critical violations
        ada_concerns = category_analysis.get("ada_concerns", [])
        for ada in ada_concerns:
            field_name = ada.get("field", "unknown")
            concern_type = ada.get("type", "ada_violation")

            evaluations.append(RuleEvaluation(
                rule_id=f"ada-{eval_id}",
                rule_name=f"ADA Violation: {concern_type}",
                result=RuleResult.VIOLATION,
                confidence=0.90,
                risk_score=85,
                law_references=["ADA § 102"],
                recommendations=[
                    f"Remove '{field_name}' from pre-offer inquiries",
                    "Medical inquiries only permitted post-conditional offer",
                ],
                escalate_to_llm=False,
                llm_context=None,
            ))
            eval_id += 1

        # Convert algorithm findings to violations
        findings = category_analysis.get("findings", [])
        for finding in findings:
            finding_type = finding.get("type", "unknown")
            severity = finding.get("severity", "medium")

            if finding_type == "missing_bias_testing":
                evaluations.append(RuleEvaluation(
                    rule_id=f"finding-{eval_id}",
                    rule_name="Missing Bias Testing",
                    result=RuleResult.VIOLATION,
                    confidence=0.85,
                    risk_score=65,
                    law_references=["Best Practice", "NYC Local Law 144"],
                    recommendations=[
                        "Conduct bias/fairness audit on algorithm",
                        "Document disparate impact analysis",
                    ],
                    escalate_to_llm=True,
                    llm_context={"concern": "Algorithm lacks documented bias testing"},
                ))
                eval_id += 1

            elif finding_type == "high_risk_hiring_algorithm":
                evaluations.append(RuleEvaluation(
                    rule_id=f"finding-{eval_id}",
                    rule_name="High-Risk Hiring Algorithm",
                    result=RuleResult.VIOLATION,
                    confidence=0.80,
                    risk_score=70,
                    law_references=["ADA", "Title VII"],
                    recommendations=[
                        "Evaluate algorithm for disability discrimination",
                        "Ensure accommodations available for candidates with disabilities",
                    ],
                    escalate_to_llm=True,
                    llm_context={"concern": finding.get("concern", "High-risk input detected")},
                ))
                eval_id += 1

        return evaluations

    def _run_rag_analysis(
        self,
        feature: ProductFeatureInput,
        rule_evaluations: list[RuleEvaluation],
        config: AssessmentConfig,
    ) -> RAGResult | None:
        """Run RAG analysis if needed."""
        # Check if RAG analysis is warranted
        needs_rag = (
            # Has escalated rules
            any(e.escalate_to_llm for e in rule_evaluations) or
            # Has high-risk evaluations
            any(e.risk_score >= 60 for e in rule_evaluations) or
            # Has potential violations needing clarification
            any(e.result == RuleResult.POTENTIAL_VIOLATION for e in rule_evaluations)
        )

        if not needs_rag:
            return None

        try:
            return self.rag.analyze(
                feature=feature,
                rule_evaluations=rule_evaluations,
                top_k=config.top_k_documents,
                apply_guardrails=config.apply_guardrails,
            )
        except Exception as e:
            # Log error but don't fail the assessment
            print(f"RAG analysis failed: {e}")
            return None

    def _build_feature_assessment(
        self,
        analysis: FeatureAnalysis,
    ) -> FeatureAssessment:
        """Build FeatureAssessment from analysis."""
        violations = self._extract_violations(analysis)

        recommendations = []
        if analysis.unified_score:
            for i, rec in enumerate(analysis.unified_score.recommendations[:5], 1):
                recommendations.append(RecommendationItem(
                    priority=i,
                    category=analysis.feature.category.value,
                    recommendation=rec,
                    rationale="Based on compliance analysis",
                    estimated_effort="medium",
                    law_references=[],
                ))

        compliant_aspects = []
        for eval in analysis.rule_evaluations:
            if eval.result == RuleResult.COMPLIANT:
                compliant_aspects.append(eval.rule_name)

        return FeatureAssessment(
            feature_id=analysis.feature.feature_id,
            feature_name=analysis.feature.name,
            risk_score=analysis.unified_score.overall_score if analysis.unified_score else 0,
            risk_level=analysis.unified_score.risk_level if analysis.unified_score else RiskLevel.MINIMAL,
            violations=violations,
            recommendations=recommendations,
            compliant_aspects=compliant_aspects[:5],
            requires_human_review=analysis.unified_score.requires_human_review if analysis.unified_score else False,
            llm_analysis_summary=analysis.rag_result.response.analysis[:500] if analysis.rag_result else None,
        )

    def _extract_violations(
        self,
        analysis: FeatureAnalysis,
    ) -> list[ViolationDetail]:
        """Extract violations from analysis."""
        violations = []

        for eval in analysis.rule_evaluations:
            if eval.result in [RuleResult.VIOLATION, RuleResult.POTENTIAL_VIOLATION]:
                severity = (
                    RiskLevel.CRITICAL if eval.risk_score >= 80
                    else RiskLevel.HIGH if eval.risk_score >= 60
                    else RiskLevel.MEDIUM if eval.risk_score >= 40
                    else RiskLevel.LOW
                )

                violations.append(ViolationDetail(
                    violation_id=f"v-{eval.rule_id}",
                    law_reference=eval.law_references[0] if eval.law_references else "",
                    law_name=self._law_ref_to_name(eval.law_references[0] if eval.law_references else ""),
                    section=eval.rule_name,
                    severity=severity,
                    description=eval.recommendations[0] if eval.recommendations else "Compliance issue detected",
                    affected_feature=analysis.feature.feature_id,
                    confidence=eval.confidence,
                    source="rule_engine",
                ))

        return violations

    def _law_ref_to_name(self, ref: str) -> str:
        """Convert law reference to human-readable name."""
        ref_lower = ref.lower()
        if "title-vii" in ref_lower or "2000e" in ref_lower:
            return "Title VII of the Civil Rights Act"
        elif "ada" in ref_lower or "12101" in ref_lower:
            return "Americans with Disabilities Act"
        elif "fha" in ref_lower or "3601" in ref_lower:
            return "Fair Housing Act"
        elif "ecoa" in ref_lower or "1691" in ref_lower:
            return "Equal Credit Opportunity Act"
        elif "fcra" in ref_lower or "1681" in ref_lower:
            return "Fair Credit Reporting Act"
        return ref

    def _calculate_overall_scores(
        self,
        analyses: list[FeatureAnalysis],
    ) -> tuple[float, RiskLevel]:
        """Calculate overall product scores."""
        if not analyses:
            return 0.0, RiskLevel.MINIMAL

        scores = [
            a.unified_score.overall_score
            for a in analyses
            if a.unified_score
        ]

        if not scores:
            return 0.0, RiskLevel.MINIMAL

        # Use max score (most risky feature determines product risk)
        overall = max(scores)

        # Determine level
        if overall >= 80:
            level = RiskLevel.CRITICAL
        elif overall >= 60:
            level = RiskLevel.HIGH
        elif overall >= 40:
            level = RiskLevel.MEDIUM
        elif overall >= 20:
            level = RiskLevel.LOW
        else:
            level = RiskLevel.MINIMAL

        return round(overall, 1), level

    def _calculate_overall_confidence(
        self,
        analyses: list[FeatureAnalysis],
    ) -> float:
        """Calculate overall confidence score."""
        confidences = []
        for a in analyses:
            if a.unified_score:
                for cs in a.unified_score.component_scores:
                    confidences.append(cs.confidence)

        return sum(confidences) / len(confidences) if confidences else 0.5

    def _calculate_overall_confidence_interval(
        self,
        analyses: list[FeatureAnalysis],
    ) -> tuple[float, float]:
        """Calculate overall confidence interval."""
        intervals = [
            a.unified_score.confidence_interval
            for a in analyses
            if a.unified_score
        ]

        if not intervals:
            return (0, 100)

        # Conservative interval - widest bounds
        low = min(i[0] for i in intervals)
        high = max(i[1] for i in intervals)

        return (low, high)

    def _aggregate_category_breakdown(
        self,
        analyses: list[FeatureAnalysis],
    ) -> Any:
        """Aggregate category breakdowns."""
        from wenah.core.types import CategoryBreakdown

        employment_scores = []
        housing_scores = []
        consumer_scores = []

        for a in analyses:
            if a.unified_score:
                bd = a.unified_score.category_breakdown
                if bd.employment > 0:
                    employment_scores.append(bd.employment)
                if bd.housing > 0:
                    housing_scores.append(bd.housing)
                if bd.consumer > 0:
                    consumer_scores.append(bd.consumer)

        def max_or_zero(scores: list[float]) -> float:
            return max(scores) if scores else 0.0

        return CategoryBreakdown(
            employment=max_or_zero(employment_scores),
            housing=max_or_zero(housing_scores),
            consumer=max_or_zero(consumer_scores),
            overall=max(
                max_or_zero(employment_scores),
                max_or_zero(housing_scores),
                max_or_zero(consumer_scores),
            ),
        )

    def _aggregate_recommendations(
        self,
        analyses: list[FeatureAnalysis],
    ) -> list[RecommendationItem]:
        """Aggregate and deduplicate recommendations."""
        all_recs = []
        seen = set()

        priority = 1
        for analysis in analyses:
            if analysis.unified_score:
                for rec in analysis.unified_score.recommendations:
                    if rec not in seen:
                        all_recs.append(RecommendationItem(
                            priority=priority,
                            category=analysis.feature.category.value,
                            recommendation=rec,
                            rationale="Based on compliance analysis",
                            estimated_effort="medium",
                            law_references=[],
                        ))
                        seen.add(rec)
                        priority += 1

        return all_recs[:15]  # Limit to 15

    def _generate_executive_summary(
        self,
        product_name: str,
        overall_score: float,
        overall_level: RiskLevel,
        feature_count: int,
        violation_count: int,
    ) -> str:
        """Generate executive summary."""
        level_text = {
            RiskLevel.CRITICAL: "critical compliance risks that require immediate attention",
            RiskLevel.HIGH: "significant compliance risks that should be addressed promptly",
            RiskLevel.MEDIUM: "moderate compliance concerns that warrant review",
            RiskLevel.LOW: "minor compliance considerations",
            RiskLevel.MINIMAL: "low compliance risk overall",
        }

        summary = f"Assessment of {product_name} analyzed {feature_count} feature(s) "
        summary += f"and identified {violation_count} potential compliance issue(s). "
        summary += f"The overall risk score is {overall_score:.0f}/100 ({overall_level.value}), "
        summary += f"indicating {level_text.get(overall_level, 'compliance status requiring review')}."

        return summary

    def _aggregate_human_review_needs(
        self,
        analyses: list[FeatureAnalysis],
    ) -> tuple[bool, list[str]]:
        """Aggregate human review requirements."""
        reasons = []

        for a in analyses:
            if a.unified_score and a.unified_score.requires_human_review:
                reasons.append(f"Feature '{a.feature.name}' flagged for review")

        return len(reasons) > 0, reasons[:5]

    def _extract_key_concerns(
        self,
        analyses: list[FeatureAnalysis],
    ) -> list[str]:
        """Extract key concerns across all features."""
        concerns = []
        for a in analyses:
            if a.unified_score:
                concerns.extend(a.unified_score.primary_concerns)
        return list(dict.fromkeys(concerns))[:5]  # Dedupe and limit

    def _extract_positive_aspects(
        self,
        analyses: list[FeatureAnalysis],
    ) -> list[str]:
        """Extract positive compliance aspects."""
        positives = []

        for a in analyses:
            # Check for bias testing
            if a.feature.algorithm and a.feature.algorithm.bias_testing_done:
                positives.append(f"Bias testing completed for {a.feature.name}")

            # Check for compliant rules
            compliant_count = sum(
                1 for e in a.rule_evaluations
                if e.result == RuleResult.COMPLIANT
            )
            if compliant_count > 0:
                positives.append(f"{a.feature.name} passed {compliant_count} compliance check(s)")

        return list(dict.fromkeys(positives))[:5]


# Singleton instance
_compliance_engine: ComplianceEngine | None = None


def get_compliance_engine() -> ComplianceEngine:
    """Get singleton compliance engine instance."""
    global _compliance_engine
    if _compliance_engine is None:
        _compliance_engine = ComplianceEngine()
    return _compliance_engine
