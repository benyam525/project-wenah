"""
API request and response schemas.

Defines Pydantic models for API validation and serialization.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# Enums for API
# =============================================================================


class APIProductCategory(str, Enum):
    """Product categories for API."""

    HIRING = "hiring"
    LENDING = "lending"
    HOUSING = "housing"
    INSURANCE = "insurance"
    GENERAL = "general"


class APIFeatureType(str, Enum):
    """Feature types for API."""

    DATA_COLLECTION = "data_collection"
    ALGORITHM = "algorithm"
    SCORING_MODEL = "scoring_model"
    AUTOMATED_DECISION = "automated_decision"
    HUMAN_ASSISTED = "human_assisted"


class APIRiskLevel(str, Enum):
    """Risk levels for API responses."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


class APILaunchDecision(str, Enum):
    """Launch decision for pre-launch check."""

    APPROVED = "approved"
    CONDITIONAL = "conditional"
    BLOCKED = "blocked"
    NEEDS_REVIEW = "needs_review"


class APIGuidanceLevel(str, Enum):
    """Guidance detail level."""

    QUICK = "quick"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class APIDashboardView(str, Enum):
    """Dashboard view types."""

    EXECUTIVE = "executive"
    DETAILED = "detailed"
    COMPLIANCE_OFFICER = "compliance_officer"
    DEVELOPER = "developer"


# =============================================================================
# Request Models - Data Field
# =============================================================================


class DataFieldRequest(BaseModel):
    """Data field specification in API request."""

    name: str = Field(..., description="Field name", min_length=1, max_length=100)
    description: str = Field(default="", description="Field description", max_length=500)
    data_type: str = Field(
        default="text", description="Data type (text, numeric, boolean, categorical)"
    )
    source: str = Field(default="user_input", description="Data source")
    required: bool = Field(default=False, description="Whether field is required")
    used_in_decisions: bool = Field(
        default=False, description="Whether used in automated decisions"
    )
    potential_proxy: str | None = Field(
        default=None, description="Protected class this may proxy for"
    )

    model_config = {"extra": "ignore"}


class AlgorithmRequest(BaseModel):
    """Algorithm specification in API request."""

    name: str = Field(..., description="Algorithm name", min_length=1, max_length=100)
    type: str = Field(..., description="Algorithm type (rule_based, ml_model, llm, etc.)")
    inputs: list[str] = Field(default_factory=list, description="Input field names")
    outputs: list[str] = Field(default_factory=list, description="Output field names")
    bias_testing_done: bool = Field(default=False, description="Whether bias testing completed")
    description: str | None = Field(default=None, description="Algorithm description")

    model_config = {"extra": "ignore"}


# =============================================================================
# Request Models - Feature
# =============================================================================


class FeatureRequest(BaseModel):
    """Feature specification in API request."""

    feature_id: str = Field(
        ..., description="Unique feature identifier", min_length=1, max_length=100
    )
    name: str = Field(..., description="Feature name", min_length=1, max_length=200)
    description: str = Field(..., description="Feature description", min_length=10, max_length=2000)
    category: APIProductCategory = Field(..., description="Product category")
    feature_type: APIFeatureType = Field(..., description="Type of feature")
    data_fields: list[DataFieldRequest] = Field(
        default_factory=list, description="Data fields used"
    )
    algorithm: AlgorithmRequest | None = Field(default=None, description="Algorithm specification")
    decision_impact: str = Field(
        ..., description="How this affects decisions", min_length=5, max_length=500
    )
    affected_population: str = Field(
        ..., description="Who is affected", min_length=3, max_length=200
    )
    company_size: int | None = Field(default=None, ge=1, description="Company size (employees)")
    additional_context: str | None = Field(
        default=None, description="Additional context", max_length=2000
    )

    model_config = {"extra": "ignore"}


# =============================================================================
# Request Models - Assessment
# =============================================================================


class RiskAssessmentRequest(BaseModel):
    """Request for risk assessment."""

    product_name: str = Field(..., description="Product name", min_length=1, max_length=200)
    features: list[FeatureRequest] = Field(
        ..., description="Features to assess", min_length=1, max_length=50
    )
    include_llm_analysis: bool = Field(default=True, description="Include LLM-powered analysis")
    view_type: APIDashboardView = Field(
        default=APIDashboardView.DETAILED, description="Dashboard view type"
    )

    model_config = {"extra": "ignore"}


class QuickAssessmentRequest(BaseModel):
    """Request for quick assessment (rules only)."""

    features: list[FeatureRequest] = Field(
        ..., description="Features to assess", min_length=1, max_length=50
    )

    model_config = {"extra": "ignore"}


class TextAssessmentRequest(BaseModel):
    """Request for text-based assessment with automatic field extraction."""

    name: str = Field(..., description="Feature name", min_length=1, max_length=200)
    category: APIProductCategory = Field(..., description="Product category")
    description: str = Field(
        ..., description="Free-text feature description", min_length=10, max_length=5000
    )
    include_llm_analysis: bool = Field(default=True, description="Include LLM-powered analysis")

    model_config = {"extra": "ignore"}


class FeatureAssessmentRequest(BaseModel):
    """Request for single feature assessment."""

    feature: FeatureRequest = Field(..., description="Feature to assess")
    include_llm_analysis: bool = Field(default=True, description="Include LLM-powered analysis")
    view_type: APIDashboardView = Field(
        default=APIDashboardView.DETAILED, description="Dashboard view type"
    )

    model_config = {"extra": "ignore"}


# =============================================================================
# Request Models - Design Guidance
# =============================================================================


class DesignGuidanceRequest(BaseModel):
    """Request for design guidance."""

    product_name: str = Field(..., description="Product name", min_length=1, max_length=200)
    features: list[FeatureRequest] = Field(
        ..., description="Features to get guidance for", min_length=1, max_length=50
    )
    guidance_level: APIGuidanceLevel = Field(
        default=APIGuidanceLevel.STANDARD, description="Level of detail"
    )

    model_config = {"extra": "ignore"}


class DataFieldCheckRequest(BaseModel):
    """Request to check a single data field."""

    field_name: str = Field(..., description="Field name to check", min_length=1, max_length=100)
    field_description: str = Field(default="", description="Field description", max_length=500)
    category: APIProductCategory = Field(
        default=APIProductCategory.HIRING, description="Product category"
    )
    used_in_decisions: bool = Field(default=True, description="Whether used in decisions")

    model_config = {"extra": "ignore"}


class AlgorithmCheckRequest(BaseModel):
    """Request to check an algorithm design."""

    algorithm: AlgorithmRequest = Field(..., description="Algorithm to check")
    category: APIProductCategory = Field(
        default=APIProductCategory.HIRING, description="Product category"
    )

    model_config = {"extra": "ignore"}


# =============================================================================
# Request Models - Pre-launch Check
# =============================================================================


class PrelaunchCheckRequest(BaseModel):
    """Request for pre-launch compliance check."""

    product_name: str = Field(..., description="Product name", min_length=1, max_length=200)
    features: list[FeatureRequest] = Field(
        ..., description="Features to check", min_length=1, max_length=50
    )
    documentation_status: dict[str, bool] = Field(
        default_factory=dict, description="Documentation requirement status"
    )
    evidence: dict[str, Any] = Field(
        default_factory=dict, description="Evidence for compliance checks"
    )

    model_config = {"extra": "ignore"}


class QuickPrelaunchRequest(BaseModel):
    """Request for quick pre-launch check."""

    features: list[FeatureRequest] = Field(
        ..., description="Features to check", min_length=1, max_length=50
    )

    model_config = {"extra": "ignore"}


# =============================================================================
# Response Models - Common
# =============================================================================


class ViolationResponse(BaseModel):
    """Violation detail in response."""

    violation_id: str
    law_reference: str
    law_name: str
    section: str
    severity: APIRiskLevel
    description: str
    affected_feature: str
    confidence: float
    source: str


class RecommendationResponse(BaseModel):
    """Recommendation in response."""

    priority: int
    category: str
    recommendation: str
    rationale: str
    estimated_effort: str
    law_references: list[str]


class CategoryBreakdownResponse(BaseModel):
    """Category risk breakdown."""

    employment: float = Field(ge=0, le=100)
    housing: float = Field(ge=0, le=100)
    consumer: float = Field(ge=0, le=100)
    overall: float = Field(ge=0, le=100)


class FeatureAssessmentResponse(BaseModel):
    """Feature assessment in response."""

    feature_id: str
    feature_name: str
    risk_score: float
    risk_level: APIRiskLevel
    violations: list[ViolationResponse]
    recommendations: list[RecommendationResponse]
    compliant_aspects: list[str]
    requires_human_review: bool
    llm_analysis_summary: str | None = None


# =============================================================================
# Response Models - Risk Assessment
# =============================================================================


class RiskAssessmentResponse(BaseModel):
    """Full risk assessment response."""

    assessment_id: str
    product_name: str
    timestamp: datetime
    overall_risk_score: float = Field(ge=0, le=100)
    overall_risk_level: APIRiskLevel
    confidence_score: float = Field(ge=0, le=1)
    confidence_interval: tuple[float, float]
    category_breakdown: CategoryBreakdownResponse
    feature_assessments: list[FeatureAssessmentResponse]
    total_violations: int
    critical_violations: list[ViolationResponse]
    all_recommendations: list[RecommendationResponse]
    executive_summary: str
    key_concerns: list[str]
    positive_aspects: list[str]
    requires_human_review: bool
    human_review_reasons: list[str]


class QuickAssessmentResponse(BaseModel):
    """Quick assessment response."""

    overall_score: float = Field(ge=0, le=100)
    risk_level: APIRiskLevel
    total_features: int
    features_at_risk: int
    total_violations: int
    requires_detailed_analysis: bool
    feature_scores: list[dict[str, Any]]


class FeatureQuickResponse(BaseModel):
    """Quick response for single feature."""

    feature_id: str
    risk_score: float
    risk_level: APIRiskLevel
    violations_count: int
    requires_full_analysis: bool
    primary_concerns: list[str]


class ExtractedFieldResponse(BaseModel):
    """Extracted data field in response."""

    name: str
    description: str
    used_in_decisions: bool
    potential_proxy: str | None


class TextAssessmentResponse(BaseModel):
    """Response for text-based assessment."""

    assessment_id: str
    feature_name: str
    timestamp: datetime
    extracted_fields: list[ExtractedFieldResponse]
    extraction_confidence: float = Field(ge=0, le=1)
    risk_score: float = Field(ge=0, le=100)
    risk_level: APIRiskLevel
    violations: list[ViolationResponse]
    recommendations: list[RecommendationResponse]
    executive_summary: str
    requires_human_review: bool


# =============================================================================
# Response Models - Design Guidance
# =============================================================================


class DataFieldGuidanceResponse(BaseModel):
    """Guidance for a data field."""

    field_name: str
    risk_level: APIRiskLevel
    design_choice: str  # recommended, caution, avoid, requires_review
    guidance: str
    alternatives: list[str]
    legal_references: list[str]
    is_protected_class: bool
    is_proxy_variable: bool
    proxy_for: str | None = None


class AlgorithmGuidanceResponse(BaseModel):
    """Guidance for an algorithm."""

    algorithm_type: str
    risk_level: APIRiskLevel
    design_choice: str
    guidance: str
    requirements: list[str]
    best_practices: list[str]
    testing_requirements: list[str]
    legal_references: list[str]


class FeatureGuidanceResponse(BaseModel):
    """Guidance for a feature."""

    feature_name: str
    category: str
    overall_risk: APIRiskLevel
    design_choice: str
    summary: str
    data_field_guidance: list[DataFieldGuidanceResponse]
    algorithm_guidance: AlgorithmGuidanceResponse | None = None
    general_recommendations: list[str]
    compliance_checklist: list[dict[str, Any]]
    applicable_laws: list[str]


class DesignGuidanceResponse(BaseModel):
    """Full design guidance response."""

    product_name: str
    generated_at: datetime
    guidance_level: APIGuidanceLevel
    feature_guidance: list[FeatureGuidanceResponse]
    overall_design_risk: APIRiskLevel
    critical_warnings: list[str]
    design_principles: list[str]
    next_steps: list[str]


# =============================================================================
# Response Models - Pre-launch Check
# =============================================================================


class ComplianceCheckItemResponse(BaseModel):
    """Individual compliance check result."""

    check_id: str
    category: str
    name: str
    description: str
    status: str  # passed, failed, warning, not_applicable, needs_review
    details: str
    remediation: str | None = None
    law_reference: str | None = None
    blocking: bool


class FeatureCheckResponse(BaseModel):
    """Feature check results."""

    feature_id: str
    feature_name: str
    check_items: list[ComplianceCheckItemResponse]
    passed_count: int
    failed_count: int
    warning_count: int
    overall_status: str
    blocking_issues: list[str]


class DocumentationRequirementResponse(BaseModel):
    """Documentation requirement status."""

    requirement_id: str
    name: str
    description: str
    required: bool
    provided: bool
    law_reference: str


class PrelaunchCheckResponse(BaseModel):
    """Full pre-launch check response."""

    product_name: str
    check_timestamp: datetime
    check_version: str
    launch_decision: APILaunchDecision
    decision_rationale: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    warning_checks: int
    feature_results: list[FeatureCheckResponse]
    blocking_issues: list[str]
    critical_violations: list[ViolationResponse]
    documentation_requirements: list[DocumentationRequirementResponse]
    documentation_complete: bool
    launch_conditions: list[str]
    pre_launch_actions: list[str]
    monitoring_requirements: list[str]
    sign_offs_required: list[str]
    sign_offs_obtained: list[str]


class QuickPrelaunchResponse(BaseModel):
    """Quick pre-launch check response."""

    can_launch: bool
    blocking_issues: list[str]
    warnings: list[str]
    recommendation: APILaunchDecision


# =============================================================================
# Response Models - Health & Info
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: datetime


class ComponentHealth(BaseModel):
    """Health status of an individual component."""

    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    latency_ms: float | None = None
    message: str | None = None


class DetailedHealthResponse(BaseModel):
    """Detailed health check response with component status."""

    status: str  # "healthy", "degraded", "unhealthy"
    version: str
    timestamp: datetime
    uptime_seconds: float
    components: list[ComponentHealth]
    checks_passed: int
    checks_failed: int


class ReadinessResponse(BaseModel):
    """Readiness probe response for Kubernetes."""

    ready: bool
    reason: str | None = None


class LivenessResponse(BaseModel):
    """Liveness probe response for Kubernetes."""

    alive: bool


class MetricsResponse(BaseModel):
    """Basic metrics response."""

    uptime_seconds: float
    requests_total: int
    requests_by_endpoint: dict[str, int]
    average_response_time_ms: float
    error_rate: float


class APIInfoResponse(BaseModel):
    """API information response."""

    name: str
    version: str
    description: str
    documentation_url: str
    supported_categories: list[str]
    supported_laws: list[str]


# =============================================================================
# Error Response
# =============================================================================


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str
    code: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now())


class ValidationErrorResponse(BaseModel):
    """Validation error response."""

    error: str = "Validation Error"
    detail: list[dict[str, Any]]
    code: str = "VALIDATION_ERROR"
