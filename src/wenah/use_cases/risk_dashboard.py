"""
Risk Assessment Dashboard use case.

Provides comprehensive compliance risk assessment with detailed scoring,
breakdowns, and actionable insights for product teams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from wenah.core.engine import (
    AssessmentConfig,
    ComplianceEngine,
    get_compliance_engine,
)
from wenah.core.scoring import ScoreExplainer, get_score_explainer
from wenah.core.types import (
    ProductFeatureInput,
    RecommendationItem,
    RiskAssessmentResponse,
    RiskLevel,
    ViolationDetail,
)


class DashboardViewType(str, Enum):
    """Types of dashboard views."""

    EXECUTIVE = "executive"  # High-level summary for executives
    DETAILED = "detailed"  # Full technical details
    COMPLIANCE_OFFICER = "compliance_officer"  # Focused on violations and remediation
    DEVELOPER = "developer"  # Technical implementation guidance


@dataclass
class RiskTrend:
    """Risk trend over time."""

    timestamp: datetime
    score: float
    risk_level: RiskLevel
    change_from_previous: float | None = None


@dataclass
class CategoryRiskDetail:
    """Detailed risk information for a category."""

    category: str
    display_name: str
    score: float
    risk_level: RiskLevel
    applicable_laws: list[str]
    violation_count: int
    top_concerns: list[str]
    recommendations: list[str]


@dataclass
class FeatureRiskSummary:
    """Risk summary for a single feature."""

    feature_id: str
    feature_name: str
    score: float
    risk_level: RiskLevel
    violation_count: int
    requires_attention: bool
    top_issue: str | None = None


@dataclass
class DashboardData:
    """Complete data for the risk dashboard."""

    # Header information
    product_name: str
    assessment_id: str
    generated_at: datetime
    view_type: DashboardViewType

    # Overall risk
    overall_score: float
    overall_risk_level: RiskLevel
    confidence_score: float
    confidence_interval: tuple[float, float]

    # Summary metrics
    total_features: int
    features_at_risk: int
    total_violations: int
    critical_violations: int

    # Breakdowns
    category_details: list[CategoryRiskDetail]
    feature_summaries: list[FeatureRiskSummary]

    # Key insights
    executive_summary: str
    key_concerns: list[str]
    immediate_actions: list[str]
    positive_aspects: list[str]

    # Detailed information
    all_violations: list[ViolationDetail] = field(default_factory=list)
    all_recommendations: list[RecommendationItem] = field(default_factory=list)

    # Compliance status
    requires_human_review: bool = False
    human_review_reasons: list[str] = field(default_factory=list)

    # Trend data (if historical data available)
    risk_trend: list[RiskTrend] = field(default_factory=list)


class RiskDashboard:
    """
    Risk Assessment Dashboard for comprehensive compliance analysis.

    Provides:
    - Overall risk scoring with confidence intervals
    - Category-level breakdowns (Employment, Housing, Consumer)
    - Feature-level risk summaries
    - Prioritized recommendations
    - Executive and detailed views
    - Trend analysis (when historical data available)
    """

    def __init__(
        self,
        compliance_engine: ComplianceEngine | None = None,
        score_explainer: ScoreExplainer | None = None,
    ):
        """
        Initialize the risk dashboard.

        Args:
            compliance_engine: Engine for compliance assessment
            score_explainer: Explainer for score narratives
        """
        self.engine = compliance_engine or get_compliance_engine()
        self.explainer = score_explainer or get_score_explainer()

    def assess_product(
        self,
        product_name: str,
        features: list[ProductFeatureInput],
        view_type: DashboardViewType = DashboardViewType.DETAILED,
        include_llm_analysis: bool = True,
    ) -> DashboardData:
        """
        Perform comprehensive risk assessment for a product.

        Args:
            product_name: Name of the product
            features: List of features to assess
            view_type: Type of dashboard view to generate
            include_llm_analysis: Whether to include LLM analysis

        Returns:
            Complete dashboard data
        """
        # Configure assessment based on view type
        config = self._get_config_for_view(view_type, include_llm_analysis)

        # Run assessment
        assessment = self.engine.assess_product(
            product_name=product_name,
            features=features,
            config=config,
        )

        # Transform into dashboard data
        return self._build_dashboard_data(
            assessment=assessment,
            view_type=view_type,
        )

    def assess_single_feature(
        self,
        feature: ProductFeatureInput,
        view_type: DashboardViewType = DashboardViewType.DETAILED,
    ) -> DashboardData:
        """
        Assess a single feature and generate dashboard data.

        Args:
            feature: Feature to assess
            view_type: Type of dashboard view

        Returns:
            Dashboard data for single feature
        """
        return self.assess_product(
            product_name=f"Feature: {feature.name}",
            features=[feature],
            view_type=view_type,
        )

    def get_quick_score(
        self,
        features: list[ProductFeatureInput],
    ) -> dict[str, Any]:
        """
        Get quick risk score without full analysis.

        Args:
            features: Features to assess

        Returns:
            Quick score summary
        """
        scores = []
        violations = 0
        features_at_risk = 0

        for feature in features:
            result = self.engine.quick_assess(feature)
            scores.append(result["risk_score"])
            violations += result["violations_count"]
            if result["risk_score"] >= 40:
                features_at_risk += 1

        overall = max(scores) if scores else 0

        return {
            "overall_score": overall,
            "risk_level": self._score_to_level(overall).value,
            "total_features": len(features),
            "features_at_risk": features_at_risk,
            "total_violations": violations,
            "requires_detailed_analysis": any(
                self.engine.quick_assess(f).get("requires_full_analysis") for f in features
            ),
        }

    def generate_report(
        self,
        dashboard_data: DashboardData,
        format: str = "markdown",
    ) -> str:
        """
        Generate a formatted report from dashboard data.

        Args:
            dashboard_data: Dashboard data to format
            format: Output format (markdown, text, html)

        Returns:
            Formatted report string
        """
        if format == "markdown":
            return self._generate_markdown_report(dashboard_data)
        elif format == "text":
            return self._generate_text_report(dashboard_data)
        else:
            return self._generate_markdown_report(dashboard_data)

    def _get_config_for_view(
        self,
        view_type: DashboardViewType,
        include_llm: bool,
    ) -> AssessmentConfig:
        """Get assessment config based on view type."""
        if view_type == DashboardViewType.EXECUTIVE:
            return AssessmentConfig(
                include_llm_analysis=include_llm,
                include_category_analysis=True,
                detail_level="brief",
            )
        elif view_type == DashboardViewType.COMPLIANCE_OFFICER:
            return AssessmentConfig(
                include_llm_analysis=True,  # Always need full analysis
                include_category_analysis=True,
                detail_level="detailed",
                apply_guardrails=True,
            )
        elif view_type == DashboardViewType.DEVELOPER:
            return AssessmentConfig(
                include_llm_analysis=include_llm,
                include_category_analysis=True,
                detail_level="detailed",
            )
        else:  # DETAILED
            return AssessmentConfig(
                include_llm_analysis=include_llm,
                include_category_analysis=True,
                detail_level="detailed",
                apply_guardrails=True,
            )

    def _build_dashboard_data(
        self,
        assessment: RiskAssessmentResponse,
        view_type: DashboardViewType,
    ) -> DashboardData:
        """Build dashboard data from assessment response."""
        # Build category details
        category_details = self._build_category_details(assessment)

        # Build feature summaries
        feature_summaries = [
            FeatureRiskSummary(
                feature_id=fa.feature_id,
                feature_name=fa.feature_name,
                score=fa.risk_score,
                risk_level=fa.risk_level,
                violation_count=len(fa.violations),
                requires_attention=fa.risk_score >= 40,
                top_issue=fa.violations[0].description if fa.violations else None,
            )
            for fa in assessment.feature_assessments
        ]

        # Count features at risk
        features_at_risk = sum(1 for fs in feature_summaries if fs.requires_attention)

        # Extract immediate actions (top 5 high-priority recommendations)
        immediate_actions = [rec.recommendation for rec in assessment.all_recommendations[:5]]

        return DashboardData(
            product_name=assessment.product_name,
            assessment_id=assessment.assessment_id,
            generated_at=datetime.now(UTC),
            view_type=view_type,
            overall_score=assessment.overall_risk_score,
            overall_risk_level=assessment.overall_risk_level,
            confidence_score=assessment.confidence_score,
            confidence_interval=assessment.confidence_interval,
            total_features=len(assessment.feature_assessments),
            features_at_risk=features_at_risk,
            total_violations=assessment.total_violations,
            critical_violations=len(assessment.critical_violations),
            category_details=category_details,
            feature_summaries=feature_summaries,
            executive_summary=assessment.executive_summary,
            key_concerns=assessment.key_concerns,
            immediate_actions=immediate_actions,
            positive_aspects=assessment.positive_aspects,
            all_violations=self._collect_all_violations(assessment),
            all_recommendations=assessment.all_recommendations,
            requires_human_review=assessment.requires_human_review,
            human_review_reasons=assessment.human_review_reasons,
        )

    def _build_category_details(
        self,
        assessment: RiskAssessmentResponse,
    ) -> list[CategoryRiskDetail]:
        """Build category-level risk details."""
        details = []
        breakdown = assessment.category_breakdown

        # Employment category
        if breakdown.employment > 0:
            emp_violations = [
                v
                for v in self._collect_all_violations(assessment)
                if "title" in v.law_reference.lower() or "ada" in v.law_reference.lower()
            ]
            details.append(
                CategoryRiskDetail(
                    category="employment",
                    display_name="Employment (Title VII, ADA)",
                    score=breakdown.employment,
                    risk_level=self._score_to_level(breakdown.employment),
                    applicable_laws=["Title VII", "ADA"],
                    violation_count=len(emp_violations),
                    top_concerns=[v.description for v in emp_violations[:3]],
                    recommendations=[
                        r.recommendation
                        for r in assessment.all_recommendations
                        if r.category == "hiring"
                    ][:3],
                )
            )

        # Housing category
        if breakdown.housing > 0:
            housing_violations = [
                v
                for v in self._collect_all_violations(assessment)
                if "fha" in v.law_reference.lower() or "housing" in v.law_reference.lower()
            ]
            details.append(
                CategoryRiskDetail(
                    category="housing",
                    display_name="Housing (FHA)",
                    score=breakdown.housing,
                    risk_level=self._score_to_level(breakdown.housing),
                    applicable_laws=["Fair Housing Act"],
                    violation_count=len(housing_violations),
                    top_concerns=[v.description for v in housing_violations[:3]],
                    recommendations=[
                        r.recommendation
                        for r in assessment.all_recommendations
                        if r.category == "housing"
                    ][:3],
                )
            )

        # Consumer category
        if breakdown.consumer > 0:
            consumer_violations = [
                v
                for v in self._collect_all_violations(assessment)
                if "ecoa" in v.law_reference.lower() or "fcra" in v.law_reference.lower()
            ]
            details.append(
                CategoryRiskDetail(
                    category="consumer",
                    display_name="Consumer (ECOA, FCRA)",
                    score=breakdown.consumer,
                    risk_level=self._score_to_level(breakdown.consumer),
                    applicable_laws=["ECOA", "FCRA"],
                    violation_count=len(consumer_violations),
                    top_concerns=[v.description for v in consumer_violations[:3]],
                    recommendations=[
                        r.recommendation
                        for r in assessment.all_recommendations
                        if r.category in ["lending", "insurance"]
                    ][:3],
                )
            )

        return details

    def _collect_all_violations(
        self,
        assessment: RiskAssessmentResponse,
    ) -> list[ViolationDetail]:
        """Collect all violations from assessment."""
        violations = []
        for fa in assessment.feature_assessments:
            violations.extend(fa.violations)
        return violations

    def _score_to_level(self, score: float) -> RiskLevel:
        """Convert score to risk level."""
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

    def _generate_markdown_report(self, data: DashboardData) -> str:
        """Generate markdown formatted report."""
        lines = [
            f"# Risk Assessment Report: {data.product_name}",
            "",
            f"**Assessment ID:** {data.assessment_id}",
            f"**Generated:** {data.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            data.executive_summary,
            "",
            "## Overall Risk Score",
            "",
            f"**Score:** {data.overall_score:.0f}/100 ({data.overall_risk_level.value.upper()})",
            f"**Confidence:** {data.confidence_score:.0%}",
            f"**Confidence Interval:** {data.confidence_interval[0]:.0f} - {data.confidence_interval[1]:.0f}",
            "",
        ]

        # Human review alert
        if data.requires_human_review:
            lines.extend(
                [
                    "> ⚠️ **Human Review Required**",
                    "> " + ", ".join(data.human_review_reasons[:3]),
                    "",
                ]
            )

        # Summary metrics
        lines.extend(
            [
                "## Summary Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Total Features | {data.total_features} |",
                f"| Features at Risk | {data.features_at_risk} |",
                f"| Total Violations | {data.total_violations} |",
                f"| Critical Violations | {data.critical_violations} |",
                "",
            ]
        )

        # Key concerns
        if data.key_concerns:
            lines.extend(
                [
                    "## Key Concerns",
                    "",
                ]
            )
            for i, concern in enumerate(data.key_concerns, 1):
                lines.append(f"{i}. {concern}")
            lines.append("")

        # Immediate actions
        if data.immediate_actions:
            lines.extend(
                [
                    "## Immediate Actions Required",
                    "",
                ]
            )
            for i, action in enumerate(data.immediate_actions, 1):
                lines.append(f"{i}. {action}")
            lines.append("")

        # Category breakdown
        if data.category_details:
            lines.extend(
                [
                    "## Risk by Category",
                    "",
                ]
            )
            for cat in data.category_details:
                lines.extend(
                    [
                        f"### {cat.display_name}",
                        "",
                        f"**Score:** {cat.score:.0f}/100 ({cat.risk_level.value})",
                        f"**Violations:** {cat.violation_count}",
                        "",
                    ]
                )
                if cat.top_concerns:
                    lines.append("**Top Concerns:**")
                    for concern in cat.top_concerns:
                        lines.append(f"- {concern}")
                    lines.append("")

        # Feature summaries
        if data.feature_summaries:
            lines.extend(
                [
                    "## Feature Risk Summary",
                    "",
                    "| Feature | Score | Risk Level | Violations | Action Needed |",
                    "|---------|-------|------------|------------|---------------|",
                ]
            )
            for fs in data.feature_summaries:
                action = "Yes" if fs.requires_attention else "No"
                lines.append(
                    f"| {fs.feature_name} | {fs.score:.0f} | {fs.risk_level.value} | {fs.violation_count} | {action} |"
                )
            lines.append("")

        # Positive aspects
        if data.positive_aspects:
            lines.extend(
                [
                    "## Positive Compliance Aspects",
                    "",
                ]
            )
            for aspect in data.positive_aspects:
                lines.append(f"- ✓ {aspect}")
            lines.append("")

        lines.extend(
            [
                "---",
                "",
                "*Report generated by Wenah Compliance Framework*",
            ]
        )

        return "\n".join(lines)

    def _generate_text_report(self, data: DashboardData) -> str:
        """Generate plain text report."""
        lines = [
            f"RISK ASSESSMENT REPORT: {data.product_name.upper()}",
            "=" * 60,
            "",
            f"Assessment ID: {data.assessment_id}",
            f"Generated: {data.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 40,
            data.executive_summary,
            "",
            "OVERALL RISK",
            "-" * 40,
            f"Score: {data.overall_score:.0f}/100 ({data.overall_risk_level.value.upper()})",
            f"Confidence: {data.confidence_score:.0%}",
            "",
            "SUMMARY METRICS",
            "-" * 40,
            f"Total Features: {data.total_features}",
            f"Features at Risk: {data.features_at_risk}",
            f"Total Violations: {data.total_violations}",
            f"Critical Violations: {data.critical_violations}",
            "",
        ]

        if data.key_concerns:
            lines.extend(
                [
                    "KEY CONCERNS",
                    "-" * 40,
                ]
            )
            for i, concern in enumerate(data.key_concerns, 1):
                lines.append(f"{i}. {concern}")
            lines.append("")

        if data.immediate_actions:
            lines.extend(
                [
                    "IMMEDIATE ACTIONS",
                    "-" * 40,
                ]
            )
            for i, action in enumerate(data.immediate_actions, 1):
                lines.append(f"{i}. {action}")
            lines.append("")

        return "\n".join(lines)


# Singleton instance
_risk_dashboard: RiskDashboard | None = None


def get_risk_dashboard() -> RiskDashboard:
    """Get singleton risk dashboard instance."""
    global _risk_dashboard
    if _risk_dashboard is None:
        _risk_dashboard = RiskDashboard()
    return _risk_dashboard
