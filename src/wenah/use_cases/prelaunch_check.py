"""
Pre-launch Compliance Check use case.

Provides final verification before product deployment, ensuring all
compliance requirements are met and documented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from enum import Enum

from wenah.core.types import (
    ProductFeatureInput,
    RiskLevel,
    ViolationDetail,
)
from wenah.core.engine import (
    ComplianceEngine,
    AssessmentConfig,
    get_compliance_engine,
)
from wenah.use_cases.design_guidance import (
    DesignGuidanceEngine,
    get_design_guidance,
)


class CheckStatus(str, Enum):
    """Status of a compliance check."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"
    NEEDS_REVIEW = "needs_review"


class LaunchDecision(str, Enum):
    """Launch decision recommendation."""

    APPROVED = "approved"  # Safe to launch
    CONDITIONAL = "conditional"  # Launch with conditions/monitoring
    BLOCKED = "blocked"  # Do not launch until issues resolved
    NEEDS_REVIEW = "needs_review"  # Requires human decision


@dataclass
class ComplianceCheckItem:
    """Individual compliance check item."""

    check_id: str
    category: str
    name: str
    description: str
    status: CheckStatus
    details: str
    remediation: str | None = None
    law_reference: str | None = None
    blocking: bool = False
    evidence_required: bool = False
    evidence_provided: bool = False


@dataclass
class FeatureCheckResult:
    """Compliance check results for a single feature."""

    feature_id: str
    feature_name: str
    check_items: list[ComplianceCheckItem]
    passed_count: int
    failed_count: int
    warning_count: int
    overall_status: CheckStatus
    blocking_issues: list[str] = field(default_factory=list)


@dataclass
class DocumentationRequirement:
    """Documentation requirement for compliance."""

    requirement_id: str
    name: str
    description: str
    required: bool
    provided: bool
    law_reference: str


@dataclass
class PrelaunchCheckResponse:
    """Complete pre-launch check response."""

    product_name: str
    check_timestamp: datetime
    check_version: str

    # Overall decision
    launch_decision: LaunchDecision
    decision_rationale: str

    # Summary metrics
    total_checks: int
    passed_checks: int
    failed_checks: int
    warning_checks: int

    # Feature results
    feature_results: list[FeatureCheckResult]

    # Blocking issues
    blocking_issues: list[str]
    critical_violations: list[ViolationDetail]

    # Documentation
    documentation_requirements: list[DocumentationRequirement]
    documentation_complete: bool

    # Conditions (if conditional approval)
    launch_conditions: list[str] = field(default_factory=list)

    # Required actions before launch
    pre_launch_actions: list[str] = field(default_factory=list)

    # Post-launch monitoring requirements
    monitoring_requirements: list[str] = field(default_factory=list)

    # Sign-off requirements
    sign_offs_required: list[str] = field(default_factory=list)
    sign_offs_obtained: list[str] = field(default_factory=list)


class PrelaunchChecker:
    """
    Pre-launch Compliance Checker for final deployment verification.

    Provides:
    - Comprehensive compliance checklist verification
    - Blocking issue identification
    - Documentation completeness check
    - Launch decision recommendation
    - Required conditions and monitoring for launch
    """

    # Standard compliance checks by category
    STANDARD_CHECKS = {
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
            {
                "check_id": "pr-003",
                "name": "Adverse Action Notices",
                "description": "Process for notifying candidates of adverse decisions",
                "law_reference": "FCRA (if applicable)",
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

    # Documentation requirements
    DOCUMENTATION_REQUIREMENTS = [
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
    ]

    def __init__(
        self,
        compliance_engine: ComplianceEngine | None = None,
        design_guidance: DesignGuidanceEngine | None = None,
    ):
        """
        Initialize the pre-launch checker.

        Args:
            compliance_engine: Engine for compliance assessment
            design_guidance: Engine for design guidance
        """
        self.engine = compliance_engine or get_compliance_engine()
        self.guidance = design_guidance or get_design_guidance()

    def run_prelaunch_check(
        self,
        product_name: str,
        features: list[ProductFeatureInput],
        documentation_status: dict[str, bool] | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> PrelaunchCheckResponse:
        """
        Run comprehensive pre-launch compliance check.

        Args:
            product_name: Name of the product
            features: List of features to check
            documentation_status: Status of documentation requirements
            evidence: Evidence provided for compliance checks

        Returns:
            Complete pre-launch check response
        """
        documentation_status = documentation_status or {}
        evidence = evidence or {}

        # Run compliance assessment
        config = AssessmentConfig(
            include_llm_analysis=True,
            include_category_analysis=True,
            detail_level="detailed",
            apply_guardrails=True,
        )

        assessment = self.engine.assess_product(
            product_name=product_name,
            features=features,
            config=config,
        )

        # Run checks for each feature
        feature_results = []
        all_blocking = []
        total_passed = 0
        total_failed = 0
        total_warning = 0

        for feature in features:
            result = self._check_feature(feature, evidence)
            feature_results.append(result)
            all_blocking.extend(result.blocking_issues)
            total_passed += result.passed_count
            total_failed += result.failed_count
            total_warning += result.warning_count

        # Check documentation
        doc_requirements = self._check_documentation(documentation_status)
        doc_complete = all(d.provided for d in doc_requirements if d.required)

        # Determine launch decision
        launch_decision, rationale = self._determine_launch_decision(
            feature_results=feature_results,
            assessment=assessment,
            doc_complete=doc_complete,
            all_blocking=all_blocking,
        )

        # Generate conditions and requirements
        launch_conditions = self._generate_launch_conditions(
            launch_decision, feature_results, assessment
        )
        pre_launch_actions = self._generate_pre_launch_actions(
            feature_results, doc_requirements
        )
        monitoring_requirements = self._generate_monitoring_requirements(features)

        # Sign-off requirements
        sign_offs = self._get_required_sign_offs(assessment)

        return PrelaunchCheckResponse(
            product_name=product_name,
            check_timestamp=datetime.now(timezone.utc),
            check_version="1.0.0",
            launch_decision=launch_decision,
            decision_rationale=rationale,
            total_checks=total_passed + total_failed + total_warning,
            passed_checks=total_passed,
            failed_checks=total_failed,
            warning_checks=total_warning,
            feature_results=feature_results,
            blocking_issues=all_blocking,
            critical_violations=assessment.critical_violations,
            documentation_requirements=doc_requirements,
            documentation_complete=doc_complete,
            launch_conditions=launch_conditions,
            pre_launch_actions=pre_launch_actions,
            monitoring_requirements=monitoring_requirements,
            sign_offs_required=sign_offs,
            sign_offs_obtained=[],
        )

    def quick_check(
        self,
        features: list[ProductFeatureInput],
    ) -> dict[str, Any]:
        """
        Run quick pre-launch check without full analysis.

        Args:
            features: Features to check

        Returns:
            Quick check summary
        """
        blocking_issues = []
        warnings = []

        for feature in features:
            # Check for protected class usage
            for df in feature.data_fields:
                if self._is_protected_class(df.name) and df.used_in_decisions:
                    blocking_issues.append(
                        f"Protected class '{df.name}' used in decisions for {feature.name}"
                    )

                if df.potential_proxy and df.used_in_decisions:
                    warnings.append(
                        f"Proxy variable '{df.name}' needs disparate impact review"
                    )

            # Check algorithm
            if feature.algorithm:
                if feature.algorithm.type in ["ml_model", "llm"] and not feature.algorithm.bias_testing_done:
                    blocking_issues.append(
                        f"AI algorithm in {feature.name} requires bias testing"
                    )

        can_launch = len(blocking_issues) == 0

        return {
            "can_launch": can_launch,
            "blocking_issues": blocking_issues,
            "warnings": warnings,
            "recommendation": "APPROVED" if can_launch else "BLOCKED",
        }

    def _check_feature(
        self,
        feature: ProductFeatureInput,
        evidence: dict[str, Any],
    ) -> FeatureCheckResult:
        """Run compliance checks for a single feature."""
        check_items = []
        blocking = []

        # Data collection checks
        for check_def in self.STANDARD_CHECKS["data_collection"]:
            status, details, remediation = self._evaluate_data_check(
                check_def, feature, evidence
            )
            item = ComplianceCheckItem(
                check_id=check_def["check_id"],
                category="Data Collection",
                name=check_def["name"],
                description=check_def["description"],
                status=status,
                details=details,
                remediation=remediation,
                law_reference=check_def.get("law_reference"),
                blocking=check_def.get("blocking", False),
            )
            check_items.append(item)

            if status == CheckStatus.FAILED and item.blocking:
                blocking.append(f"{item.name}: {details}")

        # Algorithm checks (if applicable)
        if feature.algorithm:
            for check_def in self.STANDARD_CHECKS["algorithm"]:
                status, details, remediation = self._evaluate_algorithm_check(
                    check_def, feature, evidence
                )
                item = ComplianceCheckItem(
                    check_id=check_def["check_id"],
                    category="Algorithm",
                    name=check_def["name"],
                    description=check_def["description"],
                    status=status,
                    details=details,
                    remediation=remediation,
                    law_reference=check_def.get("law_reference"),
                    blocking=check_def.get("blocking", False),
                )
                check_items.append(item)

                if status == CheckStatus.FAILED and item.blocking:
                    blocking.append(f"{item.name}: {details}")

        # Process checks
        for check_def in self.STANDARD_CHECKS["process"]:
            status, details, remediation = self._evaluate_process_check(
                check_def, feature, evidence
            )
            item = ComplianceCheckItem(
                check_id=check_def["check_id"],
                category="Process",
                name=check_def["name"],
                description=check_def["description"],
                status=status,
                details=details,
                remediation=remediation,
                law_reference=check_def.get("law_reference"),
                blocking=check_def.get("blocking", False),
            )
            check_items.append(item)

        # Documentation checks
        for check_def in self.STANDARD_CHECKS["documentation"]:
            status, details, remediation = self._evaluate_doc_check(
                check_def, feature, evidence
            )
            item = ComplianceCheckItem(
                check_id=check_def["check_id"],
                category="Documentation",
                name=check_def["name"],
                description=check_def["description"],
                status=status,
                details=details,
                remediation=remediation,
                law_reference=check_def.get("law_reference"),
                blocking=check_def.get("blocking", False),
            )
            check_items.append(item)

            if status == CheckStatus.FAILED and item.blocking:
                blocking.append(f"{item.name}: {details}")

        # Count results
        passed = sum(1 for c in check_items if c.status == CheckStatus.PASSED)
        failed = sum(1 for c in check_items if c.status == CheckStatus.FAILED)
        warnings = sum(1 for c in check_items if c.status == CheckStatus.WARNING)

        # Determine overall status
        if failed > 0:
            overall = CheckStatus.FAILED
        elif warnings > 0:
            overall = CheckStatus.WARNING
        else:
            overall = CheckStatus.PASSED

        return FeatureCheckResult(
            feature_id=feature.feature_id,
            feature_name=feature.name,
            check_items=check_items,
            passed_count=passed,
            failed_count=failed,
            warning_count=warnings,
            overall_status=overall,
            blocking_issues=blocking,
        )

    def _evaluate_data_check(
        self,
        check_def: dict[str, Any],
        feature: ProductFeatureInput,
        evidence: dict[str, Any],
    ) -> tuple[CheckStatus, str, str | None]:
        """Evaluate a data collection check."""
        check_id = check_def["check_id"]

        if check_id == "dc-001":
            # Check for protected class in decisions
            protected_in_decisions = [
                df.name for df in feature.data_fields
                if self._is_protected_class(df.name) and df.used_in_decisions
            ]

            if protected_in_decisions:
                return (
                    CheckStatus.FAILED,
                    f"Protected class fields used in decisions: {protected_in_decisions}",
                    "Remove protected class data from decision inputs",
                )
            return (
                CheckStatus.PASSED,
                "No protected class data used in decisions",
                None,
            )

        elif check_id == "dc-002":
            # Check proxy variable review
            proxies = [df for df in feature.data_fields if df.potential_proxy]

            if proxies:
                review_key = f"proxy_review_{feature.feature_id}"
                if evidence.get(review_key):
                    return (
                        CheckStatus.PASSED,
                        f"Proxy variables reviewed: {[p.name for p in proxies]}",
                        None,
                    )
                return (
                    CheckStatus.WARNING,
                    f"Proxy variables need review: {[p.name for p in proxies]}",
                    "Conduct disparate impact analysis for proxy variables",
                )
            return (
                CheckStatus.PASSED,
                "No proxy variables identified",
                None,
            )

        elif check_id == "dc-003":
            # Data minimization - informational
            field_count = len(feature.data_fields)
            if field_count > 20:
                return (
                    CheckStatus.WARNING,
                    f"Large number of data fields ({field_count})",
                    "Review if all fields are necessary",
                )
            return (
                CheckStatus.PASSED,
                f"{field_count} data fields - within reasonable range",
                None,
            )

        return (CheckStatus.NOT_APPLICABLE, "Check not applicable", None)

    def _evaluate_algorithm_check(
        self,
        check_def: dict[str, Any],
        feature: ProductFeatureInput,
        evidence: dict[str, Any],
    ) -> tuple[CheckStatus, str, str | None]:
        """Evaluate an algorithm check."""
        check_id = check_def["check_id"]
        algo = feature.algorithm

        if not algo:
            return (CheckStatus.NOT_APPLICABLE, "No algorithm", None)

        if check_id == "al-001":
            # Bias testing
            if algo.bias_testing_done:
                return (
                    CheckStatus.PASSED,
                    "Bias testing completed",
                    None,
                )
            if algo.type in ["ml_model", "llm", "neural_network"]:
                return (
                    CheckStatus.FAILED,
                    "AI algorithm requires bias testing",
                    "Complete bias testing across protected groups",
                )
            return (
                CheckStatus.WARNING,
                "Bias testing recommended",
                "Consider bias testing for non-ML algorithms",
            )

        elif check_id == "al-002":
            # Disparate impact analysis
            evidence_key = f"disparate_impact_{feature.feature_id}"
            if evidence.get(evidence_key):
                return (
                    CheckStatus.PASSED,
                    "Disparate impact analysis documented",
                    None,
                )
            if algo.type in ["ml_model", "llm"]:
                return (
                    CheckStatus.FAILED,
                    "Disparate impact analysis required",
                    "Document adverse impact ratios by protected group",
                )
            return (
                CheckStatus.WARNING,
                "Disparate impact analysis recommended",
                None,
            )

        elif check_id == "al-003":
            # Model documentation
            evidence_key = f"model_doc_{feature.feature_id}"
            if evidence.get(evidence_key):
                return (
                    CheckStatus.PASSED,
                    "Model documentation complete",
                    None,
                )
            return (
                CheckStatus.WARNING,
                "Model documentation needed",
                "Document inputs, outputs, and decision logic",
            )

        elif check_id == "al-004":
            # ADA accommodation
            risky_inputs = [
                inp for inp in algo.inputs
                if any(r in inp.lower() for r in ["video", "voice", "facial", "speech"])
            ]

            if risky_inputs:
                evidence_key = f"ada_accommodation_{feature.feature_id}"
                if evidence.get(evidence_key):
                    return (
                        CheckStatus.PASSED,
                        "ADA accommodation process documented",
                        None,
                    )
                return (
                    CheckStatus.FAILED,
                    f"ADA accommodation needed for: {risky_inputs}",
                    "Document accommodation process for candidates with disabilities",
                )
            return (
                CheckStatus.PASSED,
                "No high-risk inputs requiring ADA accommodation",
                None,
            )

        return (CheckStatus.NOT_APPLICABLE, "Check not applicable", None)

    def _evaluate_process_check(
        self,
        check_def: dict[str, Any],
        feature: ProductFeatureInput,
        evidence: dict[str, Any],
    ) -> tuple[CheckStatus, str, str | None]:
        """Evaluate a process check."""
        check_id = check_def["check_id"]

        if check_id == "pr-001":
            # Human oversight
            has_human = (
                feature.additional_context and
                "human" in feature.additional_context.lower()
            )
            evidence_key = f"human_oversight_{feature.feature_id}"

            if has_human or evidence.get(evidence_key):
                return (
                    CheckStatus.PASSED,
                    "Human oversight process documented",
                    None,
                )
            return (
                CheckStatus.WARNING,
                "Human oversight recommended",
                "Implement human review for automated decisions",
            )

        elif check_id == "pr-002":
            # Appeals process
            evidence_key = f"appeals_{feature.feature_id}"
            if evidence.get(evidence_key):
                return (
                    CheckStatus.PASSED,
                    "Appeals process documented",
                    None,
                )
            return (
                CheckStatus.WARNING,
                "Appeals process recommended",
                "Implement mechanism for appealing decisions",
            )

        elif check_id == "pr-003":
            # Adverse action notices
            return (
                CheckStatus.NEEDS_REVIEW,
                "Verify adverse action notice process",
                None,
            )

        return (CheckStatus.NOT_APPLICABLE, "Check not applicable", None)

    def _evaluate_doc_check(
        self,
        check_def: dict[str, Any],
        feature: ProductFeatureInput,
        evidence: dict[str, Any],
    ) -> tuple[CheckStatus, str, str | None]:
        """Evaluate a documentation check."""
        check_id = check_def["check_id"]

        if check_id == "doc-001":
            # Job-relatedness
            evidence_key = f"job_related_{feature.feature_id}"
            if evidence.get(evidence_key):
                return (
                    CheckStatus.PASSED,
                    "Job-relatedness documented",
                    None,
                )
            return (
                CheckStatus.FAILED,
                "Job-relatedness documentation required",
                "Document how each criterion relates to job requirements",
            )

        elif check_id == "doc-002":
            # Validation study
            evidence_key = f"validation_{feature.feature_id}"
            if evidence.get(evidence_key):
                return (
                    CheckStatus.PASSED,
                    "Validation study complete",
                    None,
                )
            return (
                CheckStatus.WARNING,
                "Validation study recommended",
                "Consider formal validation study per Uniform Guidelines",
            )

        elif check_id == "doc-003":
            # Impact analysis records
            evidence_key = f"impact_analysis_{feature.feature_id}"
            if evidence.get(evidence_key):
                return (
                    CheckStatus.PASSED,
                    "Impact analysis documented",
                    None,
                )
            if feature.algorithm and feature.algorithm.type in ["ml_model", "llm"]:
                return (
                    CheckStatus.FAILED,
                    "Impact analysis required for AI algorithms",
                    "Document disparate impact analysis results",
                )
            return (
                CheckStatus.WARNING,
                "Impact analysis recommended",
                None,
            )

        return (CheckStatus.NOT_APPLICABLE, "Check not applicable", None)

    def _check_documentation(
        self,
        documentation_status: dict[str, bool],
    ) -> list[DocumentationRequirement]:
        """Check documentation requirements."""
        requirements = []

        for req_def in self.DOCUMENTATION_REQUIREMENTS:
            provided = documentation_status.get(req_def["requirement_id"], False)
            requirements.append(DocumentationRequirement(
                requirement_id=req_def["requirement_id"],
                name=req_def["name"],
                description=req_def["description"],
                required=req_def["required"],
                provided=provided,
                law_reference=req_def["law_reference"],
            ))

        return requirements

    def _determine_launch_decision(
        self,
        feature_results: list[FeatureCheckResult],
        assessment: Any,
        doc_complete: bool,
        all_blocking: list[str],
    ) -> tuple[LaunchDecision, str]:
        """Determine launch decision based on check results."""
        # Count issues
        total_failed = sum(r.failed_count for r in feature_results)
        total_blocking = len(all_blocking)

        # Check for critical violations
        has_critical = len(assessment.critical_violations) > 0

        if total_blocking > 0 or has_critical:
            return (
                LaunchDecision.BLOCKED,
                f"Launch blocked due to {total_blocking} blocking issue(s). "
                f"Critical compliance violations must be resolved before launch."
            )

        if not doc_complete:
            return (
                LaunchDecision.BLOCKED,
                "Required documentation incomplete. "
                "All required documentation must be provided before launch."
            )

        if total_failed > 0:
            return (
                LaunchDecision.NEEDS_REVIEW,
                f"{total_failed} check(s) failed. "
                f"Human review required to determine if launch can proceed."
            )

        if assessment.overall_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            return (
                LaunchDecision.CONDITIONAL,
                f"Risk level is {assessment.overall_risk_level.value}. "
                f"Launch approved with enhanced monitoring and conditions."
            )

        if assessment.requires_human_review:
            return (
                LaunchDecision.CONDITIONAL,
                "Human review was recommended during assessment. "
                "Launch approved with conditions pending review confirmation."
            )

        return (
            LaunchDecision.APPROVED,
            "All compliance checks passed. Product approved for launch."
        )

    def _generate_launch_conditions(
        self,
        decision: LaunchDecision,
        feature_results: list[FeatureCheckResult],
        assessment: Any,
    ) -> list[str]:
        """Generate conditions for conditional launch."""
        conditions = []

        if decision != LaunchDecision.CONDITIONAL:
            return conditions

        # Add monitoring conditions
        conditions.append(
            "Monitor disparate impact ratios weekly for first 90 days"
        )
        conditions.append(
            "Report any adverse impact ratio > 0.8 to compliance team"
        )

        # Add feature-specific conditions
        for result in feature_results:
            warning_items = [
                c for c in result.check_items
                if c.status == CheckStatus.WARNING
            ]
            for item in warning_items[:2]:
                if item.remediation:
                    conditions.append(item.remediation)

        return conditions[:5]

    def _generate_pre_launch_actions(
        self,
        feature_results: list[FeatureCheckResult],
        doc_requirements: list[DocumentationRequirement],
    ) -> list[str]:
        """Generate required actions before launch."""
        actions = []

        # Missing documentation
        for doc in doc_requirements:
            if doc.required and not doc.provided:
                actions.append(f"Provide {doc.name}")

        # Failed checks
        for result in feature_results:
            for check in result.check_items:
                if check.status == CheckStatus.FAILED and check.remediation:
                    actions.append(check.remediation)

        return list(dict.fromkeys(actions))[:10]

    def _generate_monitoring_requirements(
        self,
        features: list[ProductFeatureInput],
    ) -> list[str]:
        """Generate post-launch monitoring requirements."""
        requirements = [
            "Monitor selection rates by protected group monthly",
            "Track and investigate discrimination complaints",
            "Conduct quarterly disparate impact analysis",
            "Maintain records per EEOC requirements (minimum 2 years)",
        ]

        # Add feature-specific monitoring
        for feature in features:
            if feature.algorithm and feature.algorithm.type in ["ml_model", "llm"]:
                requirements.append(
                    f"Monitor algorithm drift for {feature.name}"
                )

        return requirements

    def _get_required_sign_offs(self, assessment: Any) -> list[str]:
        """Get required sign-offs based on risk level."""
        sign_offs = ["Product Owner"]

        if assessment.overall_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            sign_offs.extend([
                "Legal Counsel",
                "Chief Compliance Officer",
            ])
        elif assessment.overall_risk_level == RiskLevel.MEDIUM:
            sign_offs.append("Compliance Team")

        if assessment.requires_human_review:
            sign_offs.append("HR/People Operations")

        return sign_offs

    def _is_protected_class(self, field_name: str) -> bool:
        """Check if field name indicates protected class data."""
        protected = {
            "race", "color", "religion", "sex", "gender", "national_origin",
            "age", "disability", "genetic", "pregnancy", "veteran",
        }
        field_lower = field_name.lower()
        return any(p in field_lower for p in protected)


# Singleton instance
_prelaunch_checker: PrelaunchChecker | None = None


def get_prelaunch_checker() -> PrelaunchChecker:
    """Get singleton pre-launch checker instance."""
    global _prelaunch_checker
    if _prelaunch_checker is None:
        _prelaunch_checker = PrelaunchChecker()
    return _prelaunch_checker
