"""
Core type definitions for the Wenah compliance framework.

This module contains all Pydantic models and enums used throughout the system.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# Enums
# =============================================================================


class ProductCategory(str, Enum):
    """Categories of products that can be evaluated."""

    HIRING = "hiring"
    LENDING = "lending"
    HOUSING = "housing"
    INSURANCE = "insurance"
    GENERAL = "general"


class FeatureType(str, Enum):
    """Types of product features."""

    DATA_COLLECTION = "data_collection"
    ALGORITHM = "algorithm"
    DECISION_MAKING = "decision_making"
    USER_INTERFACE = "user_interface"
    SCORING_MODEL = "scoring_model"
    AUTOMATED_DECISION = "automated_decision"
    HUMAN_ASSISTED = "human_assisted"


class RiskLevel(str, Enum):
    """Risk level classifications."""

    CRITICAL = "critical"  # 80-100: Clear violation, immediate action required
    HIGH = "high"  # 60-79: Significant risk, strong remediation needed
    MEDIUM = "medium"  # 40-59: Moderate risk, review recommended
    LOW = "low"  # 20-39: Minor concerns, monitor
    MINIMAL = "minimal"  # 0-19: Compliant or very low risk


class RuleResult(str, Enum):
    """Possible results from rule evaluation."""

    VIOLATION = "violation"
    POTENTIAL_VIOLATION = "potential_violation"
    COMPLIANT = "compliant"
    NEEDS_LLM_REVIEW = "needs_llm_review"


class LawCategory(str, Enum):
    """Categories of civil rights laws."""

    EMPLOYMENT = "employment"
    HOUSING = "housing"
    CONSUMER = "consumer"


class ValidationStatus(str, Enum):
    """Validation status for guardrails."""

    VALID = "valid"
    MODIFIED = "modified"
    FLAGGED = "flagged"
    REJECTED = "rejected"


# =============================================================================
# Input Models
# =============================================================================


class DataFieldSpec(BaseModel):
    """Specification for a data field being collected or used."""

    name: str = Field(..., description="Name of the data field")
    description: str = Field(..., description="Description of what this field contains")
    data_type: str = Field(..., description="Type of data: text, numeric, categorical, boolean")
    source: str = Field(..., description="Source: user_input, derived, third_party, inferred")
    required: bool = Field(default=False, description="Whether this field is required")
    used_in_decisions: bool = Field(
        default=False, description="Whether this field is used in decision-making"
    )
    potential_proxy: str | None = Field(
        default=None, description="Protected class this might proxy for"
    )


class AlgorithmSpec(BaseModel):
    """Specification for an algorithm or model."""

    name: str = Field(..., description="Name of the algorithm")
    type: str = Field(..., description="Type: ml_model, rule_based, heuristic, llm")
    inputs: list[str] = Field(default_factory=list, description="Input features")
    outputs: list[str] = Field(default_factory=list, description="Output predictions")
    description: str | None = Field(default=None, description="Description of the algorithm")
    training_data_description: str | None = Field(
        default=None, description="Description of training data used"
    )
    bias_testing_done: bool = Field(
        default=False, description="Whether bias testing has been performed"
    )


class ProductFeatureInput(BaseModel):
    """Input specification for a product feature to be evaluated."""

    feature_id: str = Field(..., description="Unique identifier for the feature")
    name: str = Field(..., description="Name of the feature")
    description: str = Field(..., description="Description of what the feature does")
    category: ProductCategory = Field(..., description="Product category")
    feature_type: FeatureType = Field(..., description="Type of feature")

    # Data handling
    data_fields: list[DataFieldSpec] = Field(
        default_factory=list, description="Data fields collected or used"
    )

    # Algorithm details
    algorithm: AlgorithmSpec | None = Field(
        default=None, description="Algorithm specification if applicable"
    )

    # Decision impact
    decision_impact: str = Field(..., description="What decisions does this feature influence?")
    affected_population: str = Field(..., description="Who is affected by this feature?")

    # Context
    company_size: int | None = Field(
        default=None, description="Company size for covered entity determination"
    )
    geographic_scope: list[str] = Field(
        default_factory=list, description="States/regions for state law applicability"
    )

    # Additional context
    additional_context: str | None = Field(
        default=None, description="Additional context for LLM analysis"
    )


# =============================================================================
# Rule Engine Models
# =============================================================================


class RuleCondition(BaseModel):
    """A single condition in a rule."""

    field: str = Field(..., description="Field path to evaluate")
    operator: str = Field(..., description="Comparison operator")
    value: Any = Field(default=None, description="Value for single comparison")
    values: list[Any] | None = Field(default=None, description="Values for multi-value comparison")


class RuleConditionGroup(BaseModel):
    """A group of conditions with AND/OR logic."""

    operator: str = Field(default="AND", description="Logical operator: AND or OR")
    items: list[RuleCondition | RuleConditionGroup] = Field(
        default_factory=list, description="Conditions or nested groups"
    )


# Rebuild model to resolve forward references
RuleConditionGroup.model_rebuild()


class RuleConsequence(BaseModel):
    """Consequence of a rule match."""

    violation: bool | str = Field(
        ..., description="True, False, or 'potential' for violation status"
    )
    risk_score: int = Field(
        ...,
        ge=-100,
        le=100,
        description="Risk score -100 to 100 (negative for positive indicators)",
    )
    law_reference: str = Field(..., description="Reference to applicable law")
    recommendation: str = Field(..., description="Recommended action")
    escalate_to_llm: bool = Field(
        default=False, description="Whether to escalate to LLM for nuanced analysis"
    )
    llm_context: dict[str, Any] | None = Field(
        default=None, description="Context to pass to LLM if escalated"
    )


class Rule(BaseModel):
    """A compliance rule definition."""

    id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Human-readable rule name")
    description: str = Field(..., description="Description of what the rule checks")
    severity: str = Field(..., description="Severity level: critical, high, medium, low")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in rule accuracy"
    )
    conditions: RuleConditionGroup = Field(..., description="Rule conditions")
    consequence: RuleConsequence = Field(..., description="Consequence if rule matches")


class RuleEvaluation(BaseModel):
    """Result of evaluating a single rule."""

    rule_id: str = Field(..., description="ID of the evaluated rule")
    rule_name: str = Field(..., description="Name of the evaluated rule")
    result: RuleResult = Field(..., description="Result of evaluation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in result")
    risk_score: int = Field(
        ...,
        ge=-100,
        le=100,
        description="Risk score -100 to 100 (negative for positive indicators)",
    )
    law_references: list[str] = Field(default_factory=list, description="Applicable law references")
    recommendations: list[str] = Field(default_factory=list, description="Recommended actions")
    escalate_to_llm: bool = Field(default=False, description="Whether to escalate to LLM")
    llm_context: dict[str, Any] | None = Field(default=None, description="Context for LLM analysis")


# =============================================================================
# LLM/RAG Models
# =============================================================================


class RAGResponse(BaseModel):
    """Structured output from RAG analysis."""

    analysis: str = Field(..., description="Detailed analysis text")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in analysis")
    cited_sources: list[str] = Field(default_factory=list, description="Law sources cited")
    risk_factors: list[str] = Field(default_factory=list, description="Identified risk factors")
    mitigating_factors: list[str] = Field(
        default_factory=list, description="Mitigating factors found"
    )
    recommendation: str = Field(..., description="Primary recommendation")
    requires_human_review: bool = Field(default=False, description="Whether human review is needed")


class GuardrailCheck(BaseModel):
    """Result of a single guardrail check."""

    check_name: str = Field(..., description="Name of the guardrail check")
    passed: bool = Field(..., description="Whether the check passed")
    details: str = Field(..., description="Details about the check result")
    severity: str = Field(..., description="Severity: critical, warning, info")


# =============================================================================
# Scoring Models
# =============================================================================


class ComponentScore(BaseModel):
    """Score from a single component (rule engine or LLM)."""

    source: str = Field(..., description="Source: rule_engine or llm_analysis")
    raw_score: float = Field(..., ge=0, le=100, description="Raw score 0-100")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level")
    weight: float = Field(..., ge=0.0, le=1.0, description="Contribution weight")
    weighted_score: float = Field(..., description="Score after weighting")
    explanation: str = Field(..., description="Explanation of the score")


class CategoryBreakdown(BaseModel):
    """Risk breakdown by law category."""

    employment: float = Field(default=0.0, ge=0, le=100)
    housing: float = Field(default=0.0, ge=0, le=100)
    consumer: float = Field(default=0.0, ge=0, le=100)
    overall: float = Field(default=0.0, ge=0, le=100)


class UnifiedRiskScore(BaseModel):
    """Unified risk score combining all components."""

    overall_score: float = Field(..., ge=0, le=100, description="Overall risk score")
    risk_level: RiskLevel = Field(..., description="Risk level classification")
    component_scores: list[ComponentScore] = Field(
        default_factory=list, description="Individual component scores"
    )
    category_breakdown: CategoryBreakdown = Field(
        default_factory=CategoryBreakdown, description="Breakdown by law category"
    )
    confidence_interval: tuple[float, float] = Field(
        ..., description="Confidence interval (low, high)"
    )
    primary_concerns: list[str] = Field(
        default_factory=list, description="Primary compliance concerns"
    )
    recommendations: list[str] = Field(default_factory=list, description="Recommended actions")
    requires_human_review: bool = Field(
        default=False, description="Whether human review is recommended"
    )


# =============================================================================
# API Response Models
# =============================================================================


class ViolationDetail(BaseModel):
    """Details of a potential violation."""

    violation_id: str = Field(..., description="Unique violation identifier")
    law_reference: str = Field(..., description="Law citation")
    law_name: str = Field(..., description="Human-readable law name")
    section: str = Field(..., description="Specific section violated")
    severity: RiskLevel = Field(..., description="Severity level")
    description: str = Field(..., description="Description of the violation")
    affected_feature: str = Field(..., description="Feature that triggered violation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level")
    source: str = Field(..., description="Source: rule_engine or llm_analysis")


class RecommendationItem(BaseModel):
    """Actionable recommendation."""

    priority: int = Field(..., ge=1, description="Priority (1 = highest)")
    category: str = Field(..., description="Recommendation category")
    recommendation: str = Field(..., description="The recommendation")
    rationale: str = Field(..., description="Why this is recommended")
    estimated_effort: str = Field(..., description="Effort: low, medium, high")
    law_references: list[str] = Field(default_factory=list, description="Related law references")


class FeatureAssessment(BaseModel):
    """Assessment results for a single feature."""

    feature_id: str = Field(..., description="Feature identifier")
    feature_name: str = Field(..., description="Feature name")
    risk_score: float = Field(..., ge=0, le=100, description="Feature risk score")
    risk_level: RiskLevel = Field(..., description="Risk level")
    violations: list[ViolationDetail] = Field(
        default_factory=list, description="Identified violations"
    )
    recommendations: list[RecommendationItem] = Field(
        default_factory=list, description="Recommendations"
    )
    compliant_aspects: list[str] = Field(
        default_factory=list, description="Aspects that are compliant"
    )
    requires_human_review: bool = Field(default=False)
    llm_analysis_summary: str | None = Field(default=None)


class RiskAssessmentResponse(BaseModel):
    """Complete risk assessment response."""

    assessment_id: str = Field(..., description="Unique assessment identifier")
    product_name: str = Field(..., description="Name of assessed product")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Assessment timestamp")

    # Overall scores
    overall_risk_score: float = Field(..., ge=0, le=100)
    overall_risk_level: RiskLevel
    confidence_score: float = Field(..., ge=0, le=1)
    confidence_interval: tuple[float, float]

    # Breakdowns
    category_breakdown: CategoryBreakdown
    feature_assessments: list[FeatureAssessment]

    # Violations and recommendations
    total_violations: int
    critical_violations: list[ViolationDetail]
    all_recommendations: list[RecommendationItem]

    # Executive summary
    executive_summary: str
    key_concerns: list[str]
    positive_aspects: list[str]

    # Review requirements
    requires_human_review: bool
    human_review_reasons: list[str]

    # Metadata
    rules_version: str
    laws_data_version: str


# =============================================================================
# Law Data Models
# =============================================================================


class ProhibitedPractice(BaseModel):
    """A prohibited practice under a law."""

    id: str
    name: str
    description: str
    severity: str
    elements: list[str] = Field(default_factory=list)


class SafeHarbor(BaseModel):
    """A safe harbor provision."""

    id: str
    name: str
    applies_to: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)


class LawDocument(BaseModel):
    """Structured representation of a civil rights law."""

    id: str = Field(..., description="Unique law identifier")
    name: str = Field(..., description="Official law name")
    citation: str = Field(..., description="Legal citation")
    category: LawCategory = Field(..., description="Law category")
    effective_date: str | None = Field(default=None)
    protected_classes: list[str] = Field(default_factory=list)
    covered_entities: list[str] = Field(default_factory=list)
    prohibited_practices: list[ProhibitedPractice] = Field(default_factory=list)
    safe_harbors: list[SafeHarbor] = Field(default_factory=list)
    remedies: list[str] = Field(default_factory=list)


# Enable forward references for nested models
RuleConditionGroup.model_rebuild()
