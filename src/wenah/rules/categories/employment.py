"""
Employment category processor for civil rights compliance.

Specialized processing for Title VII and ADA compliance rules,
including employment-specific logic and enhancements.
"""

from typing import Any

from wenah.core.types import (
    DataFieldSpec,
    ProductFeatureInput,
)

# Protected class indicators that trigger scrutiny
PROTECTED_CLASS_FIELDS = {
    # Race/Color
    "race",
    "ethnicity",
    "ethnic_background",
    "racial_identity",
    "skin_color",
    "complexion",
    # Religion
    "religion",
    "religious_affiliation",
    "faith",
    "religious_belief",
    "church",
    "mosque",
    "synagogue",
    "temple",
    # Sex/Gender
    "sex",
    "gender",
    "gender_identity",
    "sexual_orientation",
    "pregnancy",
    "pregnant",
    "maternity",
    # National Origin
    "national_origin",
    "country_of_origin",
    "nationality",
    "citizenship",
    "birthplace",
    "native_language",
    "immigration_status",
    # Disability (ADA)
    "disability",
    "disabled",
    "handicap",
    "impairment",
    "medical_condition",
    "health_condition",
    "mental_health",
    "physical_limitation",
    "accommodation_needed",
    # Age (ADEA - not Title VII but often evaluated together)
    "age",
    "date_of_birth",
    "birth_date",
    "graduation_year",
}

# Proxy variables that may correlate with protected classes
PROXY_INDICATORS = {
    # Geographic (race/national origin proxy)
    "zip_code": ["race", "national_origin"],
    "postal_code": ["race", "national_origin"],
    "neighborhood": ["race", "national_origin"],
    "census_tract": ["race", "national_origin"],
    "school_district": ["race", "national_origin"],
    # Name-based (race/national origin/sex proxy)
    "first_name": ["race", "national_origin", "sex"],
    "last_name": ["race", "national_origin"],
    "surname": ["race", "national_origin"],
    # Education (race/socioeconomic proxy)
    "school_attended": ["race", "socioeconomic"],
    "college_attended": ["race", "socioeconomic"],
    "university": ["race", "socioeconomic"],
    # Social (various proxies)
    "social_media": ["religion", "national_origin", "disability"],
    "profile_photo": ["race", "sex", "disability", "religion"],
    "voice_recording": ["sex", "national_origin", "disability"],
}

# Medical inquiry indicators (ADA pre-offer prohibition)
MEDICAL_INQUIRY_FIELDS = {
    "disability",
    "medical_history",
    "health_condition",
    "prescription",
    "medication",
    "workers_compensation",
    "sick_leave_history",
    "mental_health",
    "physical_exam",
    "drug_test_result",
    "genetic_information",
    "family_medical_history",
}


class EmploymentCategoryProcessor:
    """
    Specialized processor for employment discrimination compliance.

    Provides enhanced analysis beyond basic rule matching for
    employment-specific civil rights concerns.
    """

    def __init__(self) -> None:
        """Initialize the employment processor."""
        self.protected_class_fields = PROTECTED_CLASS_FIELDS
        self.proxy_indicators = PROXY_INDICATORS
        self.medical_inquiry_fields = MEDICAL_INQUIRY_FIELDS

    def analyze_feature(
        self,
        feature: ProductFeatureInput,
    ) -> dict[str, Any]:
        """
        Perform comprehensive employment compliance analysis.

        Args:
            feature: The product feature to analyze

        Returns:
            Analysis results with findings and recommendations
        """
        analysis: dict[str, Any] = {
            "feature_id": feature.feature_id,
            "category": "employment",
            "findings": [],
            "risk_factors": [],
            "recommendations": [],
            "protected_class_exposure": [],
            "proxy_variable_concerns": [],
            "ada_concerns": [],
        }

        # Analyze data fields
        for data_field in feature.data_fields:
            field_analysis = self._analyze_data_field(data_field, feature)
            if field_analysis:
                analysis["findings"].extend(field_analysis.get("findings", []))
                analysis["risk_factors"].extend(field_analysis.get("risk_factors", []))
                analysis["protected_class_exposure"].extend(
                    field_analysis.get("protected_class_exposure", [])
                )
                analysis["proxy_variable_concerns"].extend(field_analysis.get("proxy_concerns", []))
                analysis["ada_concerns"].extend(field_analysis.get("ada_concerns", []))

        # Analyze algorithm if present
        if feature.algorithm:
            algo_analysis = self._analyze_algorithm(feature)
            analysis["findings"].extend(algo_analysis.get("findings", []))
            analysis["risk_factors"].extend(algo_analysis.get("risk_factors", []))

        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)

        # Calculate overall risk level
        analysis["risk_level"] = self._calculate_risk_level(analysis)

        return analysis

    def _analyze_data_field(
        self,
        data_field: DataFieldSpec,
        feature: ProductFeatureInput,
    ) -> dict[str, Any]:
        """Analyze a single data field for compliance concerns."""
        findings = []
        risk_factors = []
        protected_class_exposure = []
        proxy_concerns = []
        ada_concerns = []

        field_name_lower = data_field.name.lower()

        # Check for direct protected class data collection
        for protected_field in self.protected_class_fields:
            if protected_field in field_name_lower:
                findings.append(
                    {
                        "type": "protected_class_data",
                        "field": data_field.name,
                        "matched": protected_field,
                        "severity": "critical" if data_field.used_in_decisions else "high",
                    }
                )
                protected_class_exposure.append(
                    {
                        "field": data_field.name,
                        "protected_class": protected_field,
                        "used_in_decisions": data_field.used_in_decisions,
                    }
                )

        # Check for proxy variables
        for proxy_field, related_classes in self.proxy_indicators.items():
            if proxy_field in field_name_lower:
                proxy_concerns.append(
                    {
                        "field": data_field.name,
                        "proxy_for": related_classes,
                        "used_in_decisions": data_field.used_in_decisions,
                    }
                )
                if data_field.used_in_decisions:
                    risk_factors.append(
                        f"Field '{data_field.name}' may proxy for {', '.join(related_classes)}"
                    )

        # Check for medical inquiries (ADA)
        if feature.category.value == "hiring":
            for medical_field in self.medical_inquiry_fields:
                if medical_field in field_name_lower:
                    ada_concerns.append(
                        {
                            "type": "pre_offer_medical_inquiry",
                            "field": data_field.name,
                            "severity": "critical",
                        }
                    )
                    findings.append(
                        {
                            "type": "ada_violation",
                            "field": data_field.name,
                            "issue": "Pre-offer medical inquiry prohibited under ADA",
                        }
                    )

        # Check for potential proxy annotation
        if data_field.potential_proxy:
            proxy_concerns.append(
                {
                    "field": data_field.name,
                    "annotated_proxy_for": data_field.potential_proxy,
                    "used_in_decisions": data_field.used_in_decisions,
                }
            )

        return {
            "findings": findings,
            "risk_factors": risk_factors,
            "protected_class_exposure": protected_class_exposure,
            "proxy_concerns": proxy_concerns,
            "ada_concerns": ada_concerns,
        }

    def _analyze_algorithm(
        self,
        feature: ProductFeatureInput,
    ) -> dict[str, Any]:
        """Analyze algorithm for compliance concerns."""
        findings: list[dict[str, Any]] = []
        risk_factors: list[str] = []

        algo = feature.algorithm
        if not algo:
            return {"findings": findings, "risk_factors": risk_factors}

        # Check algorithm inputs for protected class data
        for input_field in algo.inputs:
            input_lower = input_field.lower()
            for protected_field in self.protected_class_fields:
                if protected_field in input_lower:
                    findings.append(
                        {
                            "type": "algorithm_protected_class_input",
                            "input": input_field,
                            "matched": protected_field,
                            "severity": "critical",
                        }
                    )

        # Check for bias testing
        if not algo.bias_testing_done:
            if algo.type in ["ml_model", "llm"]:
                risk_factors.append("ML/AI algorithm has not undergone bias testing")
                findings.append(
                    {
                        "type": "missing_bias_testing",
                        "algorithm": algo.name,
                        "severity": "high",
                    }
                )

        # Check for high-risk algorithm types in hiring
        if feature.category.value == "hiring":
            risky_inputs = {"video", "facial", "voice", "speech", "emotion"}
            for input_field in algo.inputs:
                if any(r in input_field.lower() for r in risky_inputs):
                    findings.append(
                        {
                            "type": "high_risk_hiring_algorithm",
                            "input": input_field,
                            "severity": "high",
                            "concern": "May disadvantage individuals with disabilities",
                        }
                    )

        return {"findings": findings, "risk_factors": risk_factors}

    def _generate_recommendations(
        self,
        analysis: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate recommendations based on analysis findings."""
        recommendations = []
        priority = 1

        # Critical: Protected class data in decisions
        protected_in_decisions = [
            p for p in analysis.get("protected_class_exposure", []) if p.get("used_in_decisions")
        ]
        if protected_in_decisions:
            recommendations.append(
                {
                    "priority": priority,
                    "category": "data_collection",
                    "recommendation": (
                        "Remove protected class data from decision-making inputs. "
                        "If required for EEO reporting, collect separately after "
                        "hiring decision and ensure strict separation."
                    ),
                    "affected_fields": [p["field"] for p in protected_in_decisions],
                }
            )
            priority += 1

        # Critical: ADA concerns
        if analysis.get("ada_concerns"):
            recommendations.append(
                {
                    "priority": priority,
                    "category": "ada_compliance",
                    "recommendation": (
                        "Remove all medical and disability-related inquiries from "
                        "pre-offer stage. Medical examinations are only permitted "
                        "after conditional offer of employment."
                    ),
                    "affected_fields": [c["field"] for c in analysis["ada_concerns"]],
                }
            )
            priority += 1

        # High: Proxy variables
        proxy_in_decisions = [
            p for p in analysis.get("proxy_variable_concerns", []) if p.get("used_in_decisions")
        ]
        if proxy_in_decisions:
            recommendations.append(
                {
                    "priority": priority,
                    "category": "disparate_impact",
                    "recommendation": (
                        "Conduct disparate impact analysis on outcomes by protected class. "
                        "If impact exists, demonstrate business necessity or remove "
                        "proxy variables from decision inputs."
                    ),
                    "affected_fields": [p["field"] for p in proxy_in_decisions],
                }
            )
            priority += 1

        # Medium: Missing bias testing
        bias_findings = [
            f for f in analysis.get("findings", []) if f.get("type") == "missing_bias_testing"
        ]
        if bias_findings:
            recommendations.append(
                {
                    "priority": priority,
                    "category": "bias_testing",
                    "recommendation": (
                        "Conduct bias audit on AI/ML algorithms before deployment. "
                        "Test for disparate impact across all protected classes. "
                        "Document results and remediation steps."
                    ),
                }
            )
            priority += 1

        return recommendations

    def _calculate_risk_level(self, analysis: dict[str, Any]) -> str:
        """Calculate overall risk level from analysis."""
        findings = analysis.get("findings", [])

        # Count by severity
        critical_count = sum(1 for f in findings if f.get("severity") == "critical")
        high_count = sum(1 for f in findings if f.get("severity") == "high")

        if critical_count > 0:
            return "critical"
        elif high_count > 0:
            return "high"
        elif len(findings) > 0:
            return "medium"
        else:
            return "low"

    def check_covered_entity(
        self,
        company_size: int | None,
    ) -> dict[str, Any]:
        """
        Check if employer is covered under employment laws.

        Args:
            company_size: Number of employees

        Returns:
            Coverage status for various laws
        """
        if company_size is None:
            return {
                "title_vii_covered": "unknown",
                "ada_covered": "unknown",
                "note": "Company size not provided",
            }

        return {
            "title_vii_covered": company_size >= 15,
            "ada_covered": company_size >= 15,
            "adea_covered": company_size >= 20,  # Age discrimination
            "fmla_covered": company_size >= 50,  # Family medical leave
            "notes": self._get_coverage_notes(company_size),
        }

    def _get_coverage_notes(self, company_size: int) -> list[str]:
        """Get notes about coverage thresholds."""
        notes = []

        if company_size < 15:
            notes.append(
                "Employer may not be covered by federal employment discrimination "
                "laws, but state laws may still apply."
            )
        elif company_size < 20:
            notes.append(
                "Employer is covered by Title VII and ADA but not ADEA (age discrimination)."
            )
        elif company_size < 50:
            notes.append("Employer is covered by Title VII, ADA, and ADEA but not FMLA.")
        else:
            notes.append("Employer is covered by all major federal employment laws.")

        return notes

    def get_protected_classes_for_context(
        self,
        feature: ProductFeatureInput,
    ) -> list[dict[str, Any]]:
        """
        Get relevant protected classes based on feature context.

        Args:
            feature: The feature being evaluated

        Returns:
            List of relevant protected class information
        """
        protected_classes = [
            {
                "id": "race",
                "name": "Race",
                "law": "Title VII",
                "citation": "42 U.S.C. § 2000e-2",
                "relevant": True,
            },
            {
                "id": "color",
                "name": "Color",
                "law": "Title VII",
                "citation": "42 U.S.C. § 2000e-2",
                "relevant": True,
            },
            {
                "id": "religion",
                "name": "Religion",
                "law": "Title VII",
                "citation": "42 U.S.C. § 2000e-2",
                "relevant": True,
                "note": "Reasonable accommodation required",
            },
            {
                "id": "sex",
                "name": "Sex (including pregnancy, sexual orientation, gender identity)",
                "law": "Title VII",
                "citation": "42 U.S.C. § 2000e-2; Bostock v. Clayton County (2020)",
                "relevant": True,
            },
            {
                "id": "national_origin",
                "name": "National Origin",
                "law": "Title VII",
                "citation": "42 U.S.C. § 2000e-2",
                "relevant": True,
            },
            {
                "id": "disability",
                "name": "Disability",
                "law": "ADA",
                "citation": "42 U.S.C. § 12112",
                "relevant": True,
                "note": "Reasonable accommodation required; pre-offer medical inquiries prohibited",
            },
        ]

        return protected_classes


# Singleton instance
_employment_processor: EmploymentCategoryProcessor | None = None


def get_employment_processor() -> EmploymentCategoryProcessor:
    """Get the singleton employment processor instance."""
    global _employment_processor
    if _employment_processor is None:
        _employment_processor = EmploymentCategoryProcessor()
    return _employment_processor
