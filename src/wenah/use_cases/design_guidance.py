"""
Design Guidance use case.

Provides proactive compliance guidance during product design phase,
helping teams make compliant choices before implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from wenah.core.engine import (
    ComplianceEngine,
    get_compliance_engine,
)
from wenah.core.types import (
    AlgorithmSpec,
    ProductCategory,
    ProductFeatureInput,
    RiskLevel,
)
from wenah.rules.categories.employment import (
    get_employment_processor,
)


class GuidanceLevel(str, Enum):
    """Level of guidance detail."""

    QUICK = "quick"  # Fast, high-level guidance
    STANDARD = "standard"  # Balanced guidance
    COMPREHENSIVE = "comprehensive"  # Full detailed guidance


class DesignChoice(str, Enum):
    """Types of design choices."""

    RECOMMENDED = "recommended"  # Safe, compliant choice
    CAUTION = "caution"  # Proceed with care
    AVOID = "avoid"  # High risk, should avoid
    REQUIRES_REVIEW = "requires_review"  # Needs expert review


@dataclass
class DataFieldGuidance:
    """Guidance for a specific data field."""

    field_name: str
    field_description: str
    risk_level: RiskLevel
    design_choice: DesignChoice
    guidance: str
    alternatives: list[str] = field(default_factory=list)
    legal_references: list[str] = field(default_factory=list)
    is_protected_class: bool = False
    is_proxy_variable: bool = False
    proxy_for: str | None = None


@dataclass
class AlgorithmGuidance:
    """Guidance for algorithm design."""

    algorithm_type: str
    risk_level: RiskLevel
    design_choice: DesignChoice
    guidance: str
    requirements: list[str] = field(default_factory=list)
    best_practices: list[str] = field(default_factory=list)
    testing_requirements: list[str] = field(default_factory=list)
    legal_references: list[str] = field(default_factory=list)


@dataclass
class FeatureDesignGuidance:
    """Complete design guidance for a feature."""

    feature_name: str
    category: str
    overall_risk: RiskLevel
    design_choice: DesignChoice
    summary: str
    data_field_guidance: list[DataFieldGuidance] = field(default_factory=list)
    algorithm_guidance: AlgorithmGuidance | None = None
    general_recommendations: list[str] = field(default_factory=list)
    compliance_checklist: list[dict[str, Any]] = field(default_factory=list)
    applicable_laws: list[str] = field(default_factory=list)


@dataclass
class DesignGuidanceResponse:
    """Complete design guidance response."""

    product_name: str
    generated_at: datetime
    guidance_level: GuidanceLevel
    feature_guidance: list[FeatureDesignGuidance]
    overall_design_risk: RiskLevel
    critical_warnings: list[str] = field(default_factory=list)
    design_principles: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)


class DesignGuidanceEngine:
    """
    Design Guidance Engine for proactive compliance during product design.

    Provides:
    - Data field selection guidance (what to collect, what to avoid)
    - Algorithm design recommendations
    - Proxy variable detection and alternatives
    - Compliance checklists for each feature
    - Design principles for civil rights compliance
    """

    # Protected class identifiers
    PROTECTED_CLASSES = {
        "race",
        "color",
        "religion",
        "sex",
        "gender",
        "national_origin",
        "age",
        "disability",
        "genetic_information",
        "pregnancy",
        "sexual_orientation",
        "gender_identity",
        "veteran_status",
        "citizenship",
        "marital_status",
        "familial_status",
    }

    # Known proxy variables and what they proxy for
    PROXY_VARIABLES = {
        "zip_code": ["race", "national_origin"],
        "zipcode": ["race", "national_origin"],
        "postal_code": ["race", "national_origin"],
        "neighborhood": ["race", "national_origin"],
        "school_name": ["race", "socioeconomic_status"],
        "university": ["race", "socioeconomic_status"],
        "college": ["race", "socioeconomic_status"],
        "name": ["race", "national_origin", "gender"],
        "first_name": ["gender", "national_origin"],
        "last_name": ["race", "national_origin"],
        "surname": ["race", "national_origin"],
        "language": ["national_origin"],
        "accent": ["national_origin"],
        "facial_features": ["race", "gender"],
        "voice": ["gender", "disability"],
        "height": ["gender", "disability"],
        "weight": ["gender", "disability"],
        "credit_score": ["race"],
        "arrest_record": ["race"],
        "criminal_history": ["race"],
        "birth_year": ["age"],
        "graduation_year": ["age"],
        "years_experience": ["age"],
    }

    # Risky algorithm inputs
    RISKY_ALGORITHM_INPUTS = {
        "video",
        "facial",
        "voice",
        "speech",
        "image",
        "photo",
        "personality",
        "emotion",
        "sentiment",
        "body_language",
    }

    def __init__(
        self,
        compliance_engine: ComplianceEngine | None = None,
    ):
        """
        Initialize the design guidance engine.

        Args:
            compliance_engine: Engine for compliance assessment
        """
        self.engine = compliance_engine or get_compliance_engine()
        self.employment_processor = get_employment_processor()

    def get_guidance(
        self,
        product_name: str,
        features: list[ProductFeatureInput],
        level: GuidanceLevel = GuidanceLevel.STANDARD,
    ) -> DesignGuidanceResponse:
        """
        Get design guidance for product features.

        Args:
            product_name: Name of the product
            features: List of features being designed
            level: Level of guidance detail

        Returns:
            Complete design guidance response
        """
        feature_guidance = []
        critical_warnings = []
        overall_max_risk = RiskLevel.MINIMAL

        for feature in features:
            guidance = self._analyze_feature_design(feature, level)
            feature_guidance.append(guidance)

            # Track highest risk
            if self._risk_level_to_score(guidance.overall_risk) > self._risk_level_to_score(
                overall_max_risk
            ):
                overall_max_risk = guidance.overall_risk

            # Collect critical warnings
            if guidance.design_choice == DesignChoice.AVOID:
                critical_warnings.append(f"Feature '{feature.name}': {guidance.summary}")

        return DesignGuidanceResponse(
            product_name=product_name,
            generated_at=datetime.now(UTC),
            guidance_level=level,
            feature_guidance=feature_guidance,
            overall_design_risk=overall_max_risk,
            critical_warnings=critical_warnings,
            design_principles=self._get_design_principles(features),
            next_steps=self._get_next_steps(overall_max_risk, feature_guidance),
        )

    def check_data_field(
        self,
        field_name: str,
        field_description: str = "",
        category: ProductCategory = ProductCategory.HIRING,
        used_in_decisions: bool = True,
    ) -> DataFieldGuidance:
        """
        Check a single data field for compliance concerns.

        Args:
            field_name: Name of the field
            field_description: Description of the field
            category: Product category
            used_in_decisions: Whether field is used in decisions

        Returns:
            Guidance for the data field
        """
        field_lower = field_name.lower()
        desc_lower = field_description.lower()

        # Check if protected class
        is_protected = any(pc in field_lower or pc in desc_lower for pc in self.PROTECTED_CLASSES)

        # Check if proxy variable
        is_proxy = False
        proxy_for = None
        for proxy_name, proxied_classes in self.PROXY_VARIABLES.items():
            if proxy_name in field_lower:
                is_proxy = True
                proxy_for = proxied_classes[0]
                break

        # Determine risk and guidance
        if is_protected and used_in_decisions:
            return DataFieldGuidance(
                field_name=field_name,
                field_description=field_description,
                risk_level=RiskLevel.CRITICAL,
                design_choice=DesignChoice.AVOID,
                guidance=f"Direct use of protected class '{field_name}' in decisions is prohibited under federal civil rights laws.",
                alternatives=[
                    "Remove from decision inputs",
                    "Use only for EEO reporting (collected separately)",
                ],
                legal_references=["Title VII", "ADA", "ADEA"],
                is_protected_class=True,
                is_proxy_variable=False,
            )
        elif is_protected and not used_in_decisions:
            return DataFieldGuidance(
                field_name=field_name,
                field_description=field_description,
                risk_level=RiskLevel.MEDIUM,
                design_choice=DesignChoice.CAUTION,
                guidance=f"Collecting '{field_name}' requires careful handling. Ensure strict separation from decision-making processes.",
                alternatives=["Collect separately for voluntary EEO reporting only"],
                legal_references=["Title VII", "EEOC Guidance"],
                is_protected_class=True,
                is_proxy_variable=False,
            )
        elif is_proxy and used_in_decisions:
            return DataFieldGuidance(
                field_name=field_name,
                field_description=field_description,
                risk_level=RiskLevel.HIGH,
                design_choice=DesignChoice.REQUIRES_REVIEW,
                guidance=f"'{field_name}' may serve as a proxy for '{proxy_for}'. Use requires disparate impact analysis.",
                alternatives=self._get_proxy_alternatives(field_name),
                legal_references=["Title VII - Disparate Impact", "Griggs v. Duke Power"],
                is_protected_class=False,
                is_proxy_variable=True,
                proxy_for=proxy_for,
            )
        else:
            return DataFieldGuidance(
                field_name=field_name,
                field_description=field_description,
                risk_level=RiskLevel.LOW,
                design_choice=DesignChoice.RECOMMENDED,
                guidance=f"'{field_name}' appears to be a neutral data field. Ensure it's job-related and consistently applied.",
                alternatives=[],
                legal_references=[],
                is_protected_class=False,
                is_proxy_variable=False,
            )

    def check_algorithm_design(
        self,
        algorithm: AlgorithmSpec,
        category: ProductCategory,
    ) -> AlgorithmGuidance:
        """
        Check algorithm design for compliance concerns.

        Args:
            algorithm: Algorithm specification
            category: Product category

        Returns:
            Guidance for algorithm design
        """
        # Check for risky inputs
        risky_inputs = []
        for input_field in algorithm.inputs:
            input_lower = input_field.lower()
            for risky in self.RISKY_ALGORITHM_INPUTS:
                if risky in input_lower:
                    risky_inputs.append(input_field)
                    break

        # Determine risk level
        if algorithm.type in ["ml_model", "llm", "neural_network"]:
            if risky_inputs:
                risk_level = RiskLevel.CRITICAL
                design_choice = DesignChoice.REQUIRES_REVIEW
                guidance = (
                    f"AI algorithm using {', '.join(risky_inputs)} inputs poses significant "
                    f"disability discrimination risk under ADA. Facial/voice analysis may "
                    f"disadvantage candidates with disabilities."
                )
            elif not algorithm.bias_testing_done:
                risk_level = RiskLevel.HIGH
                design_choice = DesignChoice.CAUTION
                guidance = (
                    "ML/AI algorithms require bias testing before deployment. "
                    "EEOC has indicated AI hiring tools are subject to Title VII."
                )
            else:
                risk_level = RiskLevel.MEDIUM
                design_choice = DesignChoice.RECOMMENDED
                guidance = (
                    "Algorithm has undergone bias testing. Continue monitoring for disparate impact "
                    "and maintain documentation of testing methodology."
                )
        elif algorithm.type == "rule_based":
            risk_level = RiskLevel.LOW
            design_choice = DesignChoice.RECOMMENDED
            guidance = (
                "Rule-based algorithms are more transparent and easier to audit for bias. "
                "Ensure rules don't encode discriminatory criteria."
            )
        else:
            risk_level = RiskLevel.MEDIUM
            design_choice = DesignChoice.CAUTION
            guidance = "Algorithm type should be reviewed for potential bias impacts."

        # Build requirements
        requirements = []
        if algorithm.type in ["ml_model", "llm", "neural_network"]:
            requirements.extend(
                [
                    "Conduct disparate impact analysis before deployment",
                    "Document training data composition and potential biases",
                    "Implement ongoing bias monitoring",
                    "Provide accommodation mechanism for disabled candidates",
                ]
            )
        if risky_inputs:
            requirements.extend(
                [
                    "Ensure alternative assessment methods for candidates who cannot participate",
                    "Document ADA accommodation procedures",
                ]
            )

        # Best practices
        best_practices = [
            "Maintain human oversight for automated decisions",
            "Document the relationship between inputs and job requirements",
            "Regularly audit for disparate impact by protected class",
            "Create appeals process for adverse decisions",
        ]

        # Testing requirements
        testing_requirements = [
            "Test for disparate impact across race, gender, age, disability",
            "Validate algorithm outputs against human expert decisions",
            "Document adverse impact ratios by protected group",
        ]

        return AlgorithmGuidance(
            algorithm_type=algorithm.type,
            risk_level=risk_level,
            design_choice=design_choice,
            guidance=guidance,
            requirements=requirements,
            best_practices=best_practices,
            testing_requirements=testing_requirements,
            legal_references=["Title VII", "ADA", "EEOC AI Guidance"],
        )

    def get_compliance_checklist(
        self,
        feature: ProductFeatureInput,
    ) -> list[dict[str, Any]]:
        """
        Generate compliance checklist for a feature.

        Args:
            feature: Feature to generate checklist for

        Returns:
            List of checklist items
        """
        checklist = []

        # Data collection items
        checklist.append(
            {
                "category": "Data Collection",
                "item": "No protected class data used in decision-making",
                "required": True,
                "law": "Title VII, ADA",
            }
        )

        if any(df.potential_proxy for df in feature.data_fields):
            checklist.append(
                {
                    "category": "Data Collection",
                    "item": "Proxy variables analyzed for disparate impact",
                    "required": True,
                    "law": "Title VII",
                }
            )

        # Algorithm items
        if feature.algorithm:
            checklist.append(
                {
                    "category": "Algorithm",
                    "item": "Bias testing completed",
                    "required": True,
                    "law": "Title VII, EEOC Guidance",
                }
            )

            if feature.algorithm.type in ["ml_model", "llm"]:
                checklist.append(
                    {
                        "category": "Algorithm",
                        "item": "Training data reviewed for historical bias",
                        "required": True,
                        "law": "Title VII",
                    }
                )

                checklist.append(
                    {
                        "category": "Algorithm",
                        "item": "Disparate impact ratios documented",
                        "required": True,
                        "law": "EEOC Guidelines",
                    }
                )

            for input_field in feature.algorithm.inputs:
                if any(r in input_field.lower() for r in self.RISKY_ALGORITHM_INPUTS):
                    checklist.append(
                        {
                            "category": "Algorithm",
                            "item": f"ADA accommodation process for {input_field} analysis",
                            "required": True,
                            "law": "ADA",
                        }
                    )

        # Process items
        checklist.append(
            {
                "category": "Process",
                "item": "Human review process for adverse decisions",
                "required": feature.category == ProductCategory.HIRING,
                "law": "Best Practice",
            }
        )

        checklist.append(
            {
                "category": "Process",
                "item": "Appeals mechanism documented",
                "required": False,
                "law": "Best Practice",
            }
        )

        checklist.append(
            {
                "category": "Documentation",
                "item": "Job-relatedness documented for all criteria",
                "required": True,
                "law": "Title VII, Uniform Guidelines",
            }
        )

        return checklist

    def _analyze_feature_design(
        self,
        feature: ProductFeatureInput,
        level: GuidanceLevel,
    ) -> FeatureDesignGuidance:
        """Analyze feature design and generate guidance."""
        # Analyze each data field
        field_guidance = [
            self.check_data_field(
                field_name=df.name,
                field_description=df.description,
                category=feature.category,
                used_in_decisions=df.used_in_decisions,
            )
            for df in feature.data_fields
        ]

        # Analyze algorithm if present
        algo_guidance = None
        if feature.algorithm:
            algo_guidance = self.check_algorithm_design(
                feature.algorithm,
                feature.category,
            )

        # Determine overall risk
        max_field_risk = max(
            (self._risk_level_to_score(fg.risk_level) for fg in field_guidance), default=0
        )
        algo_risk = self._risk_level_to_score(algo_guidance.risk_level) if algo_guidance else 0
        overall_risk_score = max(max_field_risk, algo_risk)
        overall_risk = self._score_to_risk_level(overall_risk_score)

        # Determine design choice
        if overall_risk == RiskLevel.CRITICAL:
            design_choice = DesignChoice.AVOID
            summary = (
                "Critical compliance issues detected. Design changes required before proceeding."
            )
        elif overall_risk == RiskLevel.HIGH:
            design_choice = DesignChoice.REQUIRES_REVIEW
            summary = (
                "Significant compliance concerns. Expert review recommended before proceeding."
            )
        elif overall_risk == RiskLevel.MEDIUM:
            design_choice = DesignChoice.CAUTION
            summary = "Moderate compliance considerations. Proceed with documented safeguards."
        else:
            design_choice = DesignChoice.RECOMMENDED
            summary = "Design appears compliant. Maintain documentation and monitoring."

        # Generate recommendations
        recommendations = []
        for fg in field_guidance:
            if fg.design_choice in [DesignChoice.AVOID, DesignChoice.REQUIRES_REVIEW]:
                recommendations.append(fg.guidance)
        if algo_guidance and algo_guidance.design_choice != DesignChoice.RECOMMENDED:
            recommendations.extend(algo_guidance.requirements)

        # Get compliance checklist
        checklist = self.get_compliance_checklist(feature) if level != GuidanceLevel.QUICK else []

        # Determine applicable laws
        applicable_laws = self._get_applicable_laws(feature)

        return FeatureDesignGuidance(
            feature_name=feature.name,
            category=feature.category.value,
            overall_risk=overall_risk,
            design_choice=design_choice,
            summary=summary,
            data_field_guidance=field_guidance,
            algorithm_guidance=algo_guidance,
            general_recommendations=recommendations[:5],
            compliance_checklist=checklist,
            applicable_laws=applicable_laws,
        )

    def _get_proxy_alternatives(self, field_name: str) -> list[str]:
        """Get alternatives for proxy variables."""
        field_lower = field_name.lower()

        if "zip" in field_lower or "postal" in field_lower:
            return [
                "Use commute time/distance to workplace instead",
                "Remove location-based criteria if not job-related",
            ]
        elif "school" in field_lower or "university" in field_lower:
            return [
                "Focus on specific skills or certifications",
                "Use job-related assessments instead of school name",
            ]
        elif "name" in field_lower:
            return [
                "Use anonymized identifiers during screening",
                "Implement blind resume review",
            ]
        elif "credit" in field_lower:
            return [
                "Verify job-relatedness before using",
                "Consider specific financial factors only if job-related",
            ]
        else:
            return ["Review for job-relatedness", "Consider removing from decision criteria"]

    def _get_applicable_laws(self, feature: ProductFeatureInput) -> list[str]:
        """Get list of applicable laws for a feature."""
        laws = []

        if feature.category == ProductCategory.HIRING:
            laws.extend(
                ["Title VII of the Civil Rights Act", "Americans with Disabilities Act (ADA)"]
            )
            if feature.company_size and feature.company_size >= 20:
                laws.append("Age Discrimination in Employment Act (ADEA)")
        elif feature.category == ProductCategory.LENDING:
            laws.extend(["Equal Credit Opportunity Act (ECOA)", "Fair Credit Reporting Act (FCRA)"])
        elif feature.category == ProductCategory.HOUSING:
            laws.append("Fair Housing Act (FHA)")

        return laws

    def _get_design_principles(self, features: list[ProductFeatureInput]) -> list[str]:
        """Get general design principles for compliance."""
        principles = [
            "Ensure all criteria are job-related and consistent with business necessity",
            "Maintain separation between data collection and decision-making for protected class data",
            "Document the relationship between selection criteria and job requirements",
            "Implement human oversight for automated decision systems",
            "Establish processes for monitoring disparate impact",
            "Create accessible alternatives for candidates with disabilities",
        ]

        # Add category-specific principles
        categories = {f.category for f in features}
        if ProductCategory.HIRING in categories:
            principles.append("Follow EEOC Uniform Guidelines on Employee Selection Procedures")

        return principles

    def _get_next_steps(
        self,
        overall_risk: RiskLevel,
        guidance: list[FeatureDesignGuidance],
    ) -> list[str]:
        """Get recommended next steps."""
        steps = []

        if overall_risk == RiskLevel.CRITICAL:
            steps.extend(
                [
                    "Address critical design issues before proceeding",
                    "Consult with legal counsel on compliance requirements",
                    "Remove or redesign high-risk features",
                ]
            )
        elif overall_risk == RiskLevel.HIGH:
            steps.extend(
                [
                    "Conduct detailed compliance review with legal team",
                    "Document business necessity for all criteria",
                    "Plan for bias testing before deployment",
                ]
            )
        else:
            steps.extend(
                [
                    "Complete compliance checklist items",
                    "Document design decisions and rationale",
                    "Schedule pre-launch compliance review",
                ]
            )

        # Add feature-specific steps
        for fg in guidance:
            if fg.algorithm_guidance and not fg.algorithm_guidance.risk_level == RiskLevel.LOW:
                steps.append(f"Complete bias testing for {fg.feature_name}")

        return list(dict.fromkeys(steps))[:6]  # Dedupe and limit

    def _risk_level_to_score(self, level: RiskLevel) -> int:
        """Convert risk level to numeric score."""
        mapping = {
            RiskLevel.CRITICAL: 90,
            RiskLevel.HIGH: 70,
            RiskLevel.MEDIUM: 50,
            RiskLevel.LOW: 30,
            RiskLevel.MINIMAL: 10,
        }
        return mapping.get(level, 50)

    def _score_to_risk_level(self, score: int) -> RiskLevel:
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


# Singleton instance
_design_guidance: DesignGuidanceEngine | None = None


def get_design_guidance() -> DesignGuidanceEngine:
    """Get singleton design guidance instance."""
    global _design_guidance
    if _design_guidance is None:
        _design_guidance = DesignGuidanceEngine()
    return _design_guidance
