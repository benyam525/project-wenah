"""
Pre-launch Check API routes.

Provides endpoints for pre-launch compliance verification.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from wenah.api.schemas import (
    APILaunchDecision,
    APIRiskLevel,
    ComplianceCheckItemResponse,
    DocumentationRequirementResponse,
    ErrorResponse,
    FeatureCheckResponse,
    PrelaunchCheckRequest,
    PrelaunchCheckResponse,
    QuickPrelaunchRequest,
    QuickPrelaunchResponse,
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
from wenah.use_cases.prelaunch_check import (
    LaunchDecision,
    get_prelaunch_checker,
)

router = APIRouter(prefix="/check", tags=["Pre-launch Check"])


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


def _convert_launch_decision(decision: LaunchDecision) -> APILaunchDecision:
    """Convert internal launch decision to API launch decision."""
    mapping = {
        LaunchDecision.APPROVED: APILaunchDecision.APPROVED,
        LaunchDecision.CONDITIONAL: APILaunchDecision.CONDITIONAL,
        LaunchDecision.BLOCKED: APILaunchDecision.BLOCKED,
        LaunchDecision.NEEDS_REVIEW: APILaunchDecision.NEEDS_REVIEW,
    }
    return mapping.get(decision, APILaunchDecision.NEEDS_REVIEW)


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


@router.post(
    "/prelaunch",
    response_model=PrelaunchCheckResponse,
    summary="Full Pre-launch Check",
    description="Comprehensive pre-launch compliance verification.",
    responses={
        200: {"description": "Pre-launch check completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def prelaunch_check(request: PrelaunchCheckRequest) -> PrelaunchCheckResponse:
    """
    Perform comprehensive pre-launch compliance check.

    This endpoint verifies all compliance requirements before product launch:
    - Comprehensive compliance checks (data, algorithm, process, documentation)
    - Launch decision recommendation
    - Blocking issue identification
    - Documentation completeness verification
    - Required conditions for launch
    - Post-launch monitoring requirements
    """
    try:
        checker = get_prelaunch_checker()

        # Convert features
        features = [_convert_feature_request(f) for f in request.features]

        # Run pre-launch check
        result = checker.run_prelaunch_check(
            product_name=request.product_name,
            features=features,
            documentation_status=request.documentation_status,
            evidence=request.evidence,
        )

        # Build feature check responses
        feature_results = []
        for fr in result.feature_results:
            check_items = [
                ComplianceCheckItemResponse(
                    check_id=ci.check_id,
                    category=ci.category,
                    name=ci.name,
                    description=ci.description,
                    status=ci.status.value,
                    details=ci.details,
                    remediation=ci.remediation,
                    law_reference=ci.law_reference,
                    blocking=ci.blocking,
                )
                for ci in fr.check_items
            ]

            feature_results.append(
                FeatureCheckResponse(
                    feature_id=fr.feature_id,
                    feature_name=fr.feature_name,
                    check_items=check_items,
                    passed_count=fr.passed_count,
                    failed_count=fr.failed_count,
                    warning_count=fr.warning_count,
                    overall_status=fr.overall_status.value,
                    blocking_issues=fr.blocking_issues,
                )
            )

        # Build documentation requirements
        doc_requirements = [
            DocumentationRequirementResponse(
                requirement_id=dr.requirement_id,
                name=dr.name,
                description=dr.description,
                required=dr.required,
                provided=dr.provided,
                law_reference=dr.law_reference,
            )
            for dr in result.documentation_requirements
        ]

        # Build critical violations
        critical_violations = [_build_violation_response(v) for v in result.critical_violations]

        return PrelaunchCheckResponse(
            product_name=result.product_name,
            check_timestamp=result.check_timestamp,
            check_version=result.check_version,
            launch_decision=_convert_launch_decision(result.launch_decision),
            decision_rationale=result.decision_rationale,
            total_checks=result.total_checks,
            passed_checks=result.passed_checks,
            failed_checks=result.failed_checks,
            warning_checks=result.warning_checks,
            feature_results=feature_results,
            blocking_issues=result.blocking_issues,
            critical_violations=critical_violations,
            documentation_requirements=doc_requirements,
            documentation_complete=result.documentation_complete,
            launch_conditions=result.launch_conditions,
            pre_launch_actions=result.pre_launch_actions,
            monitoring_requirements=result.monitoring_requirements,
            sign_offs_required=result.sign_offs_required,
            sign_offs_obtained=result.sign_offs_obtained,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post(
    "/quick",
    response_model=QuickPrelaunchResponse,
    summary="Quick Pre-launch Check",
    description="Quick pre-launch check without full analysis.",
    responses={
        200: {"description": "Quick check completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
    },
)
async def quick_prelaunch_check(request: QuickPrelaunchRequest) -> QuickPrelaunchResponse:
    """
    Quick pre-launch compliance check.

    Fast check for blocking issues without full analysis.
    Useful for CI/CD pipelines or quick validation.
    """
    try:
        checker = get_prelaunch_checker()

        # Convert features
        features = [_convert_feature_request(f) for f in request.features]

        # Run quick check
        result = checker.quick_check(features)

        # Convert recommendation to launch decision
        rec_mapping = {
            "APPROVED": APILaunchDecision.APPROVED,
            "BLOCKED": APILaunchDecision.BLOCKED,
        }
        recommendation = rec_mapping.get(
            result["recommendation"],
            APILaunchDecision.NEEDS_REVIEW,
        )

        return QuickPrelaunchResponse(
            can_launch=result["can_launch"],
            blocking_issues=result["blocking_issues"],
            warnings=result["warnings"],
            recommendation=recommendation,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get(
    "/requirements",
    summary="Get Documentation Requirements",
    description="Get list of documentation requirements for compliance.",
)
async def get_documentation_requirements() -> dict[str, Any]:
    """
    Get standard documentation requirements for compliance.

    Returns list of required documentation for pre-launch compliance.
    """
    return {
        "requirements": [
            {
                "requirement_id": "doc-req-001",
                "name": "Algorithm Specification",
                "description": "Complete documentation of algorithm inputs, logic, and outputs",
                "required": True,
                "law_reference": "Best Practice",
            },
            {
                "requirement_id": "doc-req-002",
                "name": "Bias Testing Report",
                "description": "Results of bias testing across protected groups",
                "required": True,
                "law_reference": "EEOC Guidance",
            },
            {
                "requirement_id": "doc-req-003",
                "name": "Job Analysis",
                "description": "Documentation of job requirements and criteria rationale",
                "required": True,
                "law_reference": "Uniform Guidelines",
            },
            {
                "requirement_id": "doc-req-004",
                "name": "Accommodation Procedures",
                "description": "Documented process for ADA accommodations",
                "required": True,
                "law_reference": "ADA",
            },
            {
                "requirement_id": "doc-req-005",
                "name": "Monitoring Plan",
                "description": "Plan for ongoing disparate impact monitoring",
                "required": True,
                "law_reference": "Best Practice",
            },
        ],
        "guidance": "All required documentation must be provided before launch. "
        "Documentation should be retained for at least 2 years per EEOC requirements.",
    }


@router.get(
    "/checklist",
    summary="Get Standard Compliance Checklist",
    description="Get standard compliance checklist for pre-launch verification.",
)
async def get_compliance_checklist() -> dict[str, Any]:
    """
    Get standard compliance checklist for pre-launch.

    Returns comprehensive checklist organized by category.
    """
    return {
        "data_collection": [
            {
                "check_id": "dc-001",
                "name": "No Protected Class in Decisions",
                "description": "Verify no protected class data is used in automated decisions",
                "law_reference": "Title VII, ADA",
                "blocking": True,
            },
            {
                "check_id": "dc-002",
                "name": "Proxy Variable Review",
                "description": "All proxy variables reviewed for disparate impact",
                "law_reference": "Title VII",
                "blocking": True,
            },
            {
                "check_id": "dc-003",
                "name": "Data Minimization",
                "description": "Only necessary data is collected",
                "law_reference": "Best Practice",
                "blocking": False,
            },
        ],
        "algorithm": [
            {
                "check_id": "al-001",
                "name": "Bias Testing Complete",
                "description": "Algorithm tested for bias across protected groups",
                "law_reference": "EEOC Guidance",
                "blocking": True,
            },
            {
                "check_id": "al-002",
                "name": "Disparate Impact Analysis",
                "description": "Disparate impact ratios documented and acceptable",
                "law_reference": "Title VII, Uniform Guidelines",
                "blocking": True,
            },
            {
                "check_id": "al-003",
                "name": "Model Documentation",
                "description": "Model inputs, outputs, and decision logic documented",
                "law_reference": "Best Practice",
                "blocking": False,
            },
            {
                "check_id": "al-004",
                "name": "ADA Accommodation Process",
                "description": "Process exists for candidates requiring accommodations",
                "law_reference": "ADA",
                "blocking": True,
            },
        ],
        "process": [
            {
                "check_id": "pr-001",
                "name": "Human Oversight",
                "description": "Human review process for automated decisions",
                "law_reference": "Best Practice",
                "blocking": False,
            },
            {
                "check_id": "pr-002",
                "name": "Appeals Process",
                "description": "Mechanism for candidates to appeal decisions",
                "law_reference": "Best Practice",
                "blocking": False,
            },
        ],
        "documentation": [
            {
                "check_id": "doc-001",
                "name": "Job-Relatedness Documentation",
                "description": "All criteria documented as job-related",
                "law_reference": "Title VII, Uniform Guidelines",
                "blocking": True,
            },
            {
                "check_id": "doc-002",
                "name": "Validation Study",
                "description": "Selection procedures validated per EEOC guidelines",
                "law_reference": "Uniform Guidelines",
                "blocking": False,
            },
            {
                "check_id": "doc-003",
                "name": "Impact Analysis Records",
                "description": "Disparate impact analysis documented and retained",
                "law_reference": "EEOC Guidance",
                "blocking": True,
            },
        ],
    }


@router.get(
    "/sign-offs",
    summary="Get Required Sign-offs",
    description="Get list of required sign-offs based on risk level.",
)
async def get_sign_off_requirements(risk_level: APIRiskLevel) -> dict[str, Any]:
    """
    Get required sign-offs based on risk level.

    Higher risk levels require more stakeholder sign-offs.
    """
    sign_offs = {
        APIRiskLevel.CRITICAL: [
            "Product Owner",
            "Legal Counsel",
            "Chief Compliance Officer",
            "HR/People Operations",
            "Executive Sponsor",
        ],
        APIRiskLevel.HIGH: [
            "Product Owner",
            "Legal Counsel",
            "Chief Compliance Officer",
        ],
        APIRiskLevel.MEDIUM: [
            "Product Owner",
            "Compliance Team",
        ],
        APIRiskLevel.LOW: [
            "Product Owner",
        ],
        APIRiskLevel.MINIMAL: [
            "Product Owner",
        ],
    }

    return {
        "risk_level": risk_level.value,
        "required_sign_offs": sign_offs.get(risk_level, ["Product Owner"]),
        "guidance": "All required sign-offs must be obtained before launch. "
        "Sign-offs should include acknowledgment of compliance review results.",
    }


@router.get(
    "/monitoring",
    summary="Get Monitoring Requirements",
    description="Get standard post-launch monitoring requirements.",
)
async def get_monitoring_requirements() -> dict[str, Any]:
    """
    Get standard post-launch monitoring requirements.

    Returns monitoring requirements for ongoing compliance.
    """
    return {
        "requirements": [
            {
                "frequency": "Monthly",
                "activity": "Monitor selection rates by protected group",
                "law_reference": "EEOC Guidelines",
            },
            {
                "frequency": "Quarterly",
                "activity": "Conduct disparate impact analysis",
                "law_reference": "Title VII",
            },
            {
                "frequency": "Ongoing",
                "activity": "Track and investigate discrimination complaints",
                "law_reference": "Title VII, ADA",
            },
            {
                "frequency": "Ongoing",
                "activity": "Monitor algorithm drift for AI systems",
                "law_reference": "Best Practice",
            },
            {
                "frequency": "Annually",
                "activity": "Full compliance audit",
                "law_reference": "Best Practice",
            },
        ],
        "retention": {
            "period": "Minimum 2 years",
            "law_reference": "EEOC Requirements",
            "note": "Some states may require longer retention periods",
        },
        "reporting": {
            "adverse_impact_threshold": 0.8,
            "note": "Adverse impact ratios below 0.8 (four-fifths rule) require investigation",
        },
    }
