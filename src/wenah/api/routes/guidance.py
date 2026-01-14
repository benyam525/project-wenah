"""
Design Guidance API routes.

Provides endpoints for proactive compliance guidance during product design.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from wenah.api.schemas import (
    AlgorithmCheckRequest,
    AlgorithmGuidanceResponse,
    APIGuidanceLevel,
    APIProductCategory,
    APIRiskLevel,
    DataFieldCheckRequest,
    DataFieldGuidanceResponse,
    DesignGuidanceRequest,
    DesignGuidanceResponse,
    ErrorResponse,
    FeatureGuidanceResponse,
)
from wenah.core.types import (
    AlgorithmSpec,
    DataFieldSpec,
    FeatureType,
    ProductCategory,
    ProductFeatureInput,
    RiskLevel,
)
from wenah.use_cases.design_guidance import (
    GuidanceLevel,
    get_design_guidance,
)

router = APIRouter(prefix="/guidance", tags=["Design Guidance"])


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


def _convert_guidance_level(level: APIGuidanceLevel) -> GuidanceLevel:
    """Convert API guidance level to internal guidance level."""
    mapping = {
        APIGuidanceLevel.QUICK: GuidanceLevel.QUICK,
        APIGuidanceLevel.STANDARD: GuidanceLevel.STANDARD,
        APIGuidanceLevel.COMPREHENSIVE: GuidanceLevel.COMPREHENSIVE,
    }
    return mapping.get(level, GuidanceLevel.STANDARD)


def _convert_category(cat: APIProductCategory) -> ProductCategory:
    """Convert API category to internal category."""
    mapping = {
        APIProductCategory.HIRING: ProductCategory.HIRING,
        APIProductCategory.LENDING: ProductCategory.LENDING,
        APIProductCategory.HOUSING: ProductCategory.HOUSING,
        APIProductCategory.INSURANCE: ProductCategory.INSURANCE,
        APIProductCategory.GENERAL: ProductCategory.GENERAL,
    }
    return mapping.get(cat, ProductCategory.GENERAL)


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


@router.post(
    "/design",
    response_model=DesignGuidanceResponse,
    summary="Get Design Guidance",
    description="Get comprehensive compliance guidance for product features during design phase.",
    responses={
        200: {"description": "Guidance generated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_design_guidance_endpoint(
    request: DesignGuidanceRequest,
) -> DesignGuidanceResponse:
    """
    Get design guidance for product features.

    This endpoint provides proactive compliance guidance during the design phase:
    - Data field recommendations (what to collect, what to avoid)
    - Algorithm design guidance
    - Proxy variable detection with alternatives
    - Compliance checklists
    - Design principles
    """
    try:
        guidance_engine = get_design_guidance()

        # Convert features
        features = [_convert_feature_request(f) for f in request.features]

        # Get guidance
        result = guidance_engine.get_guidance(
            product_name=request.product_name,
            features=features,
            level=_convert_guidance_level(request.guidance_level),
        )

        # Build feature guidance responses
        feature_guidance = []
        for fg in result.feature_guidance:
            # Build data field guidance
            data_field_guidance = [
                DataFieldGuidanceResponse(
                    field_name=dfg.field_name,
                    risk_level=_convert_risk_level(dfg.risk_level),
                    design_choice=dfg.design_choice.value,
                    guidance=dfg.guidance,
                    alternatives=dfg.alternatives,
                    legal_references=dfg.legal_references,
                    is_protected_class=dfg.is_protected_class,
                    is_proxy_variable=dfg.is_proxy_variable,
                    proxy_for=dfg.proxy_for,
                )
                for dfg in fg.data_field_guidance
            ]

            # Build algorithm guidance if present
            algo_guidance = None
            if fg.algorithm_guidance:
                ag = fg.algorithm_guidance
                algo_guidance = AlgorithmGuidanceResponse(
                    algorithm_type=ag.algorithm_type,
                    risk_level=_convert_risk_level(ag.risk_level),
                    design_choice=ag.design_choice.value,
                    guidance=ag.guidance,
                    requirements=ag.requirements,
                    best_practices=ag.best_practices,
                    testing_requirements=ag.testing_requirements,
                    legal_references=ag.legal_references,
                )

            feature_guidance.append(
                FeatureGuidanceResponse(
                    feature_name=fg.feature_name,
                    category=fg.category,
                    overall_risk=_convert_risk_level(fg.overall_risk),
                    design_choice=fg.design_choice.value,
                    summary=fg.summary,
                    data_field_guidance=data_field_guidance,
                    algorithm_guidance=algo_guidance,
                    general_recommendations=fg.general_recommendations,
                    compliance_checklist=fg.compliance_checklist,
                    applicable_laws=fg.applicable_laws,
                )
            )

        return DesignGuidanceResponse(
            product_name=result.product_name,
            generated_at=result.generated_at,
            guidance_level=APIGuidanceLevel(result.guidance_level.value),
            feature_guidance=feature_guidance,
            overall_design_risk=_convert_risk_level(result.overall_design_risk),
            critical_warnings=result.critical_warnings,
            design_principles=result.design_principles,
            next_steps=result.next_steps,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post(
    "/field",
    response_model=DataFieldGuidanceResponse,
    summary="Check Data Field",
    description="Check a single data field for compliance concerns.",
    responses={
        200: {"description": "Field guidance generated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
    },
)
async def check_data_field(request: DataFieldCheckRequest) -> DataFieldGuidanceResponse:
    """
    Check a single data field for compliance concerns.

    Quickly determine if a data field:
    - Is a protected class (direct discrimination risk)
    - Is a proxy variable (indirect discrimination risk)
    - Has specific compliance requirements
    - Has safer alternatives
    """
    try:
        guidance_engine = get_design_guidance()

        # Check the field
        result = guidance_engine.check_data_field(
            field_name=request.field_name,
            field_description=request.field_description,
            category=_convert_category(request.category),
            used_in_decisions=request.used_in_decisions,
        )

        return DataFieldGuidanceResponse(
            field_name=result.field_name,
            risk_level=_convert_risk_level(result.risk_level),
            design_choice=result.design_choice.value,
            guidance=result.guidance,
            alternatives=result.alternatives,
            legal_references=result.legal_references,
            is_protected_class=result.is_protected_class,
            is_proxy_variable=result.is_proxy_variable,
            proxy_for=result.proxy_for,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post(
    "/algorithm",
    response_model=AlgorithmGuidanceResponse,
    summary="Check Algorithm Design",
    description="Get guidance for an algorithm design.",
    responses={
        200: {"description": "Algorithm guidance generated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
    },
)
async def check_algorithm(request: AlgorithmCheckRequest) -> AlgorithmGuidanceResponse:
    """
    Get compliance guidance for an algorithm design.

    Evaluates algorithm for:
    - AI/ML bias risks
    - ADA accommodation requirements
    - Testing requirements
    - Best practices for compliant design
    """
    try:
        guidance_engine = get_design_guidance()

        # Convert algorithm
        algo = AlgorithmSpec(
            name=request.algorithm.name,
            type=request.algorithm.type,
            inputs=request.algorithm.inputs,
            outputs=request.algorithm.outputs,
            bias_testing_done=request.algorithm.bias_testing_done,
            description=request.algorithm.description,
        )

        # Check the algorithm
        result = guidance_engine.check_algorithm_design(
            algorithm=algo,
            category=_convert_category(request.category),
        )

        return AlgorithmGuidanceResponse(
            algorithm_type=result.algorithm_type,
            risk_level=_convert_risk_level(result.risk_level),
            design_choice=result.design_choice.value,
            guidance=result.guidance,
            requirements=result.requirements,
            best_practices=result.best_practices,
            testing_requirements=result.testing_requirements,
            legal_references=result.legal_references,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get(
    "/checklist/{category}",
    summary="Get Compliance Checklist",
    description="Get standard compliance checklist for a category.",
    responses={
        200: {"description": "Checklist returned"},
    },
)
async def get_checklist(category: APIProductCategory) -> dict[str, Any]:
    """
    Get standard compliance checklist for a product category.

    Returns a checklist of compliance requirements specific to the category
    (employment, lending, housing, etc.).
    """
    checklists = {
        APIProductCategory.HIRING: {
            "category": "employment",
            "applicable_laws": ["Title VII", "ADA", "ADEA"],
            "checklist": [
                {"item": "No protected class data in decision inputs", "required": True},
                {"item": "Proxy variables reviewed for disparate impact", "required": True},
                {"item": "Bias testing completed for AI algorithms", "required": True},
                {"item": "Job-relatedness documented for all criteria", "required": True},
                {"item": "ADA accommodation process documented", "required": True},
                {"item": "Human oversight for automated decisions", "required": False},
                {"item": "Appeals process available", "required": False},
            ],
        },
        APIProductCategory.LENDING: {
            "category": "consumer",
            "applicable_laws": ["ECOA", "FCRA"],
            "checklist": [
                {"item": "No prohibited bases in credit decisions", "required": True},
                {"item": "Adverse action notices provided", "required": True},
                {"item": "Credit model validated for fair lending", "required": True},
                {"item": "Disparate impact analysis documented", "required": True},
            ],
        },
        APIProductCategory.HOUSING: {
            "category": "housing",
            "applicable_laws": ["Fair Housing Act"],
            "checklist": [
                {"item": "No discrimination based on protected classes", "required": True},
                {"item": "Advertising compliant with FHA", "required": True},
                {"item": "Screening criteria uniformly applied", "required": True},
            ],
        },
    }

    return checklists.get(
        category,
        {
            "category": category.value,
            "applicable_laws": [],
            "checklist": [
                {"item": "Review for applicable civil rights laws", "required": True},
            ],
        },
    )


@router.get(
    "/protected-classes",
    summary="Get Protected Classes List",
    description="Get list of protected classes under federal civil rights laws.",
)
async def get_protected_classes() -> dict[str, Any]:
    """
    Get list of protected classes under federal civil rights laws.

    Returns protected classes organized by applicable law.
    """
    return {
        "title_vii": {
            "law": "Title VII of the Civil Rights Act",
            "protected_classes": [
                "Race",
                "Color",
                "Religion",
                "Sex (including pregnancy, sexual orientation, gender identity)",
                "National Origin",
            ],
        },
        "ada": {
            "law": "Americans with Disabilities Act",
            "protected_classes": [
                "Disability",
                "Perceived disability",
                "Association with person with disability",
            ],
        },
        "adea": {
            "law": "Age Discrimination in Employment Act",
            "protected_classes": [
                "Age (40 and older)",
            ],
        },
        "gina": {
            "law": "Genetic Information Nondiscrimination Act",
            "protected_classes": [
                "Genetic information",
                "Family medical history",
            ],
        },
        "fha": {
            "law": "Fair Housing Act",
            "protected_classes": [
                "Race",
                "Color",
                "Religion",
                "Sex",
                "National Origin",
                "Familial Status",
                "Disability",
            ],
        },
        "ecoa": {
            "law": "Equal Credit Opportunity Act",
            "protected_classes": [
                "Race",
                "Color",
                "Religion",
                "National Origin",
                "Sex",
                "Marital Status",
                "Age",
                "Receipt of public assistance",
            ],
        },
    }


@router.get(
    "/proxy-variables",
    summary="Get Known Proxy Variables",
    description="Get list of known proxy variables that may correlate with protected classes.",
)
async def get_proxy_variables() -> dict[str, Any]:
    """
    Get list of known proxy variables.

    These variables may correlate with protected classes and could result
    in disparate impact even without direct discrimination.
    """
    return {
        "high_risk": [
            {"variable": "ZIP code", "proxies_for": ["Race", "National Origin"]},
            {"variable": "Neighborhood", "proxies_for": ["Race", "National Origin"]},
            {"variable": "Name", "proxies_for": ["Race", "National Origin", "Gender"]},
            {"variable": "School name", "proxies_for": ["Race", "Socioeconomic Status"]},
            {"variable": "Criminal history", "proxies_for": ["Race"]},
            {"variable": "Credit score", "proxies_for": ["Race"]},
        ],
        "moderate_risk": [
            {"variable": "Graduation year", "proxies_for": ["Age"]},
            {"variable": "Years of experience", "proxies_for": ["Age"]},
            {"variable": "Language proficiency", "proxies_for": ["National Origin"]},
            {"variable": "Height/Weight", "proxies_for": ["Gender", "Disability"]},
        ],
        "requires_review": [
            {"variable": "Video analysis", "proxies_for": ["Race", "Gender", "Disability"]},
            {
                "variable": "Voice analysis",
                "proxies_for": ["Gender", "National Origin", "Disability"],
            },
            {"variable": "Facial analysis", "proxies_for": ["Race", "Gender", "Disability"]},
        ],
        "guidance": "Variables that correlate with protected classes may result in disparate impact liability even if not intentionally discriminatory. Always conduct disparate impact analysis when using these variables.",
    }
