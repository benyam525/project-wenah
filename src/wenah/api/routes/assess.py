"""
Assessment API routes.

Provides endpoints for risk assessment of products and features.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from wenah.api.schemas import (
    APIDashboardView,
    APIRiskLevel,
    CategoryBreakdownResponse,
    ErrorResponse,
    ExtractedFieldResponse,
    FeatureAssessmentRequest,
    FeatureAssessmentResponse,
    FeatureQuickResponse,
    QuickAssessmentRequest,
    QuickAssessmentResponse,
    RecommendationResponse,
    RiskAssessmentRequest,
    RiskAssessmentResponse,
    TextAssessmentRequest,
    TextAssessmentResponse,
    ViolationResponse,
)
from wenah.core.types import (
    AlgorithmSpec,
    DataFieldSpec,
    FeatureType,
    ProductCategory,
    ProductFeatureInput,
    RiskLevel,
)
from wenah.use_cases.risk_dashboard import (
    DashboardViewType,
    get_risk_dashboard,
)

router = APIRouter(prefix="/assess", tags=["Risk Assessment"])


def _convert_risk_level(level: RiskLevel) -> APIRiskLevel:
    """Convert internal risk level to API risk level."""
    mapping = {
        RiskLevel.CRITICAL: APIRiskLevel.CRITICAL,
        RiskLevel.HIGH: APIRiskLevel.HIGH,
        RiskLevel.MEDIUM: APIRiskLevel.MEDIUM,
        RiskLevel.LOW: APIRiskLevel.LOW,
        RiskLevel.MINIMAL: APIRiskLevel.MINIMAL,
    }
    return mapping.get(level, APIRiskLevel.MEDIUM)


def _convert_view_type(view: APIDashboardView) -> DashboardViewType:
    """Convert API view type to internal view type."""
    mapping = {
        APIDashboardView.EXECUTIVE: DashboardViewType.EXECUTIVE,
        APIDashboardView.DETAILED: DashboardViewType.DETAILED,
        APIDashboardView.COMPLIANCE_OFFICER: DashboardViewType.COMPLIANCE_OFFICER,
        APIDashboardView.DEVELOPER: DashboardViewType.DEVELOPER,
    }
    return mapping.get(view, DashboardViewType.DETAILED)


def _convert_feature_request(feature_req: Any) -> ProductFeatureInput:
    """Convert API feature request to internal ProductFeatureInput."""
    # Convert category
    category_mapping = {
        "hiring": ProductCategory.HIRING,
        "lending": ProductCategory.LENDING,
        "housing": ProductCategory.HOUSING,
        "insurance": ProductCategory.INSURANCE,
        "general": ProductCategory.GENERAL,
    }
    category = category_mapping.get(feature_req.category.value, ProductCategory.GENERAL)

    # Convert feature type
    type_mapping = {
        "data_collection": FeatureType.DATA_COLLECTION,
        "algorithm": FeatureType.ALGORITHM,
        "scoring_model": FeatureType.SCORING_MODEL,
        "automated_decision": FeatureType.AUTOMATED_DECISION,
        "human_assisted": FeatureType.HUMAN_ASSISTED,
    }
    feature_type = type_mapping.get(feature_req.feature_type.value, FeatureType.ALGORITHM)

    # Convert data fields
    data_fields = [
        DataFieldSpec(
            name=df.name,
            description=df.description,
            data_type=df.data_type,
            source=df.source,
            required=df.required,
            used_in_decisions=df.used_in_decisions,
            potential_proxy=df.potential_proxy,
        )
        for df in feature_req.data_fields
    ]

    # Convert algorithm if present
    algorithm = None
    if feature_req.algorithm:
        algorithm = AlgorithmSpec(
            name=feature_req.algorithm.name,
            type=feature_req.algorithm.type,
            inputs=feature_req.algorithm.inputs,
            outputs=feature_req.algorithm.outputs,
            bias_testing_done=feature_req.algorithm.bias_testing_done,
            description=feature_req.algorithm.description,
        )

    return ProductFeatureInput(
        feature_id=feature_req.feature_id,
        name=feature_req.name,
        description=feature_req.description,
        category=category,
        feature_type=feature_type,
        data_fields=data_fields,
        algorithm=algorithm,
        decision_impact=feature_req.decision_impact,
        affected_population=feature_req.affected_population,
        company_size=feature_req.company_size,
        additional_context=feature_req.additional_context,
    )


def _build_violation_response(violation: Any) -> ViolationResponse:
    """Build violation response from internal violation."""
    return ViolationResponse(
        violation_id=violation.violation_id,
        law_reference=violation.law_reference,
        law_name=violation.law_name,
        section=violation.section,
        severity=_convert_risk_level(violation.severity),
        description=violation.description,
        affected_feature=violation.affected_feature,
        confidence=violation.confidence,
        source=violation.source,
    )


def _build_recommendation_response(rec: Any) -> RecommendationResponse:
    """Build recommendation response from internal recommendation."""
    return RecommendationResponse(
        priority=rec.priority,
        category=rec.category,
        recommendation=rec.recommendation,
        rationale=rec.rationale,
        estimated_effort=rec.estimated_effort,
        law_references=rec.law_references,
    )


@router.post(
    "/risk",
    response_model=RiskAssessmentResponse,
    summary="Full Risk Assessment",
    description="Perform comprehensive compliance risk assessment for a product with multiple features.",
    responses={
        200: {"description": "Assessment completed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def assess_risk(request: RiskAssessmentRequest) -> RiskAssessmentResponse:
    """
    Perform full risk assessment for a product.

    This endpoint analyzes all provided features against civil rights compliance
    requirements and returns a comprehensive risk assessment including:
    - Overall risk score and level
    - Category breakdowns (employment, housing, consumer)
    - Feature-level assessments
    - Violations and recommendations
    - Executive summary
    """
    try:
        dashboard = get_risk_dashboard()

        # Convert features
        features = [_convert_feature_request(f) for f in request.features]

        # Run assessment
        result = dashboard.assess_product(
            product_name=request.product_name,
            features=features,
            view_type=_convert_view_type(request.view_type),
            include_llm_analysis=request.include_llm_analysis,
        )

        # Build feature assessments
        feature_assessments = []
        for fa in result.feature_summaries:
            # Get violations for this feature
            feature_violations = [
                _build_violation_response(v)
                for v in result.all_violations
                if v.affected_feature == fa.feature_id
            ]

            # Get recommendations for this feature
            feature_recs = [
                _build_recommendation_response(r)
                for r in result.all_recommendations
                if r.category == fa.feature_name.lower() or True  # Include all for now
            ][:3]

            feature_assessments.append(
                FeatureAssessmentResponse(
                    feature_id=fa.feature_id,
                    feature_name=fa.feature_name,
                    risk_score=fa.score,
                    risk_level=_convert_risk_level(fa.risk_level),
                    violations=feature_violations,
                    recommendations=feature_recs,
                    compliant_aspects=[],
                    requires_human_review=fa.requires_attention,
                    llm_analysis_summary=None,
                )
            )

        # Build critical violations
        critical_violations = [
            _build_violation_response(v)
            for v in result.all_violations
            if v.severity == RiskLevel.CRITICAL
        ]

        # Build all recommendations
        all_recs = [_build_recommendation_response(r) for r in result.all_recommendations]

        return RiskAssessmentResponse(
            assessment_id=result.assessment_id,
            product_name=result.product_name,
            timestamp=result.generated_at,
            overall_risk_score=result.overall_score,
            overall_risk_level=_convert_risk_level(result.overall_risk_level),
            confidence_score=result.confidence_score,
            confidence_interval=result.confidence_interval,
            category_breakdown=CategoryBreakdownResponse(
                employment=result.category_details[0].score if result.category_details else 0,
                housing=0,
                consumer=0,
                overall=result.overall_score,
            ),
            feature_assessments=feature_assessments,
            total_violations=result.total_violations,
            critical_violations=critical_violations,
            all_recommendations=all_recs,
            executive_summary=result.executive_summary,
            key_concerns=result.key_concerns,
            positive_aspects=result.positive_aspects,
            requires_human_review=result.requires_human_review,
            human_review_reasons=result.human_review_reasons,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post(
    "/quick",
    response_model=QuickAssessmentResponse,
    summary="Quick Risk Assessment",
    description="Perform quick compliance check using rules only (no LLM analysis).",
    responses={
        200: {"description": "Quick assessment completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
    },
)
async def quick_assess(request: QuickAssessmentRequest) -> QuickAssessmentResponse:
    """
    Perform quick risk assessment without LLM analysis.

    This endpoint provides a fast compliance check using only rule-based
    evaluation. Useful for:
    - Initial screening during development
    - Batch processing of multiple products
    - Quick checks before detailed analysis
    """
    try:
        dashboard = get_risk_dashboard()

        # Convert features
        features = [_convert_feature_request(f) for f in request.features]

        # Run quick assessment
        result = dashboard.get_quick_score(features)

        # Build feature scores
        feature_scores = []
        for feature in features:
            quick_result = dashboard.engine.quick_assess(feature)
            feature_scores.append(
                {
                    "feature_id": feature.feature_id,
                    "risk_score": quick_result["risk_score"],
                    "risk_level": quick_result["risk_level"],
                    "violations_count": quick_result["violations_count"],
                }
            )

        return QuickAssessmentResponse(
            overall_score=result["overall_score"],
            risk_level=APIRiskLevel(result["risk_level"]),
            total_features=result["total_features"],
            features_at_risk=result["features_at_risk"],
            total_violations=result["total_violations"],
            requires_detailed_analysis=result["requires_detailed_analysis"],
            feature_scores=feature_scores,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post(
    "/text",
    response_model=TextAssessmentResponse,
    summary="Text-Based Assessment",
    description="Assess a feature from free-text description with automatic field extraction.",
    responses={
        200: {"description": "Text assessment completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def assess_text(request: TextAssessmentRequest) -> TextAssessmentResponse:
    """
    Assess a feature from a free-text description.

    This endpoint uses AI to extract structured compliance data from natural
    language descriptions, then runs full compliance analysis. Ideal for:
    - Quick assessments without detailed technical specs
    - Initial screening of product ideas
    - Non-technical stakeholder input

    The system will:
    1. Extract data fields mentioned in the description
    2. Identify potential proxy variables for protected classes
    3. Detect algorithm characteristics and automation level
    4. Run full compliance analysis with rule engine + LLM
    """
    try:
        import uuid

        from wenah.llm.text_extractor import get_text_extractor

        dashboard = get_risk_dashboard()

        # Extract structured data from text
        extractor = get_text_extractor(use_llm=request.include_llm_analysis)
        extraction = extractor.extract(
            description=request.description,
            name=request.name,
            category=request.category.value,
        )

        # Convert to ProductFeatureInput
        feature_id = f"text-{uuid.uuid4().hex[:8]}"
        feature = extractor.to_product_feature_input(
            extraction=extraction,
            feature_id=feature_id,
            name=request.name,
            description=request.description,
            category=request.category.value,
        )

        # Run full assessment
        result = dashboard.assess_single_feature(
            feature=feature,
            view_type=DashboardViewType.DETAILED,
        )

        # Build extracted fields response
        extracted_fields = [
            ExtractedFieldResponse(
                name=field.name,
                description=field.description,
                used_in_decisions=field.used_in_decisions,
                potential_proxy=field.potential_proxy,
            )
            for field in extraction.data_fields
        ]

        # Build violations
        violations = [_build_violation_response(v) for v in result.all_violations]

        # Build recommendations
        recommendations = [_build_recommendation_response(r) for r in result.all_recommendations][
            :5
        ]

        # Get feature summary
        if result.feature_summaries:
            fs = result.feature_summaries[0]
            risk_score = fs.score
            risk_level = _convert_risk_level(fs.risk_level)
        else:
            risk_score = result.overall_score
            risk_level = _convert_risk_level(result.overall_risk_level)

        return TextAssessmentResponse(
            assessment_id=result.assessment_id,
            feature_name=request.name,
            timestamp=result.generated_at,
            extracted_fields=extracted_fields,
            extraction_confidence=extraction.confidence,
            risk_score=risk_score,
            risk_level=risk_level,
            violations=violations,
            recommendations=recommendations,
            executive_summary=result.executive_summary or "Analysis complete.",
            requires_human_review=result.requires_human_review,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post(
    "/feature",
    response_model=FeatureAssessmentResponse,
    summary="Single Feature Assessment",
    description="Assess a single feature for compliance risks.",
    responses={
        200: {"description": "Feature assessment completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
    },
)
async def assess_feature(request: FeatureAssessmentRequest) -> FeatureAssessmentResponse:
    """
    Assess a single feature for compliance risks.

    Provides detailed analysis of one feature including:
    - Risk score and level
    - Specific violations
    - Targeted recommendations
    - LLM analysis summary (if enabled)
    """
    try:
        dashboard = get_risk_dashboard()

        # Convert feature
        feature = _convert_feature_request(request.feature)

        # Run assessment
        result = dashboard.assess_single_feature(
            feature=feature,
            view_type=_convert_view_type(request.view_type),
        )

        # Get first feature summary
        if result.feature_summaries:
            fs = result.feature_summaries[0]

            # Build violations
            violations = [_build_violation_response(v) for v in result.all_violations]

            # Build recommendations
            recommendations = [
                _build_recommendation_response(r) for r in result.all_recommendations
            ][:5]

            return FeatureAssessmentResponse(
                feature_id=fs.feature_id,
                feature_name=fs.feature_name,
                risk_score=fs.score,
                risk_level=_convert_risk_level(fs.risk_level),
                violations=violations,
                recommendations=recommendations,
                compliant_aspects=result.positive_aspects,
                requires_human_review=result.requires_human_review,
                llm_analysis_summary=result.executive_summary[:500]
                if result.executive_summary
                else None,
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Assessment produced no results",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post(
    "/feature/quick",
    response_model=FeatureQuickResponse,
    summary="Quick Feature Check",
    description="Quick compliance check for a single feature.",
)
async def quick_feature_check(request: FeatureAssessmentRequest) -> FeatureQuickResponse:
    """
    Quick compliance check for a single feature.

    Fast rule-based check without LLM analysis.
    """
    try:
        dashboard = get_risk_dashboard()

        # Convert feature
        feature = _convert_feature_request(request.feature)

        # Run quick assessment
        result = dashboard.engine.quick_assess(feature)

        return FeatureQuickResponse(
            feature_id=result["feature_id"],
            risk_score=result["risk_score"],
            risk_level=APIRiskLevel(result["risk_level"]),
            violations_count=result["violations_count"],
            requires_full_analysis=result["requires_full_analysis"],
            primary_concerns=result["primary_concerns"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
