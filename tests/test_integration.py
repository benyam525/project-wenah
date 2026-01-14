"""
Integration tests for end-to-end compliance workflows.

Tests the complete flow from feature input to compliance assessment results.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from wenah.core.engine import AssessmentConfig, ComplianceEngine
from wenah.core.scoring import get_scoring_engine
from wenah.core.types import (
    AlgorithmSpec,
    DataFieldSpec,
    FeatureType,
    ProductCategory,
    ProductFeatureInput,
    RiskLevel,
    RuleResult,
)
from wenah.rules.rule_engine import get_rule_engine
from wenah.use_cases.design_guidance import get_design_guidance
from wenah.use_cases.prelaunch_check import PrelaunchChecker, get_prelaunch_checker

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def compliant_hiring_feature() -> ProductFeatureInput:
    """A compliant hiring feature with no protected class data."""
    return ProductFeatureInput(
        feature_id="compliant-001",
        name="Skills Assessment",
        description="Technical skills assessment based on job-related criteria",
        category=ProductCategory.HIRING,
        feature_type=FeatureType.ALGORITHM,
        data_fields=[
            DataFieldSpec(
                name="skills_score",
                description="Score from technical assessment",
                data_type="numeric",
                source="assessment",
                required=True,
                used_in_decisions=True,
            ),
            DataFieldSpec(
                name="experience_years",
                description="Years of relevant experience",
                data_type="numeric",
                source="application",
                required=True,
                used_in_decisions=True,
            ),
        ],
        algorithm=AlgorithmSpec(
            name="SkillsMatcher",
            type="rule_based",
            inputs=["skills_score", "job_requirements"],
            outputs=["match_score"],
            bias_testing_done=True,
            description="Rule-based skills matching algorithm",
        ),
        decision_impact="Helps rank candidates for interviews",
        affected_population="Job applicants",
        company_size=100,
    )


@pytest.fixture
def risky_hiring_feature() -> ProductFeatureInput:
    """A risky hiring feature with protected class data."""
    return ProductFeatureInput(
        feature_id="risky-001",
        name="Candidate Evaluator",
        description="AI-powered candidate evaluation system",
        category=ProductCategory.HIRING,
        feature_type=FeatureType.SCORING_MODEL,
        data_fields=[
            DataFieldSpec(
                name="age",
                description="Candidate age",
                data_type="numeric",
                source="application",
                required=True,
                used_in_decisions=True,
            ),
            DataFieldSpec(
                name="zip_code",
                description="Candidate ZIP code",
                data_type="text",
                source="application",
                required=True,
                used_in_decisions=True,
                potential_proxy="race",
            ),
            DataFieldSpec(
                name="name",
                description="Candidate full name",
                data_type="text",
                source="application",
                required=True,
                used_in_decisions=True,
            ),
        ],
        algorithm=AlgorithmSpec(
            name="CandidateAI",
            type="ml_model",
            inputs=["age", "zip_code", "name"],
            outputs=["score"],
            bias_testing_done=False,
            description="ML model for scoring candidates",
        ),
        decision_impact="Automatically rejects candidates below threshold",
        affected_population="Job applicants",
        company_size=500,
    )


@pytest.fixture
def lending_feature() -> ProductFeatureInput:
    """A lending feature for consumer category testing."""
    return ProductFeatureInput(
        feature_id="lending-001",
        name="Credit Decision Engine",
        description="Automated credit decision system",
        category=ProductCategory.LENDING,
        feature_type=FeatureType.AUTOMATED_DECISION,
        data_fields=[
            DataFieldSpec(
                name="credit_score",
                description="Applicant credit score",
                data_type="numeric",
                source="credit_bureau",
                required=True,
                used_in_decisions=True,
            ),
            DataFieldSpec(
                name="income",
                description="Annual income",
                data_type="numeric",
                source="application",
                required=True,
                used_in_decisions=True,
            ),
        ],
        decision_impact="Determines credit approval and terms",
        affected_population="Credit applicants",
    )


# =============================================================================
# Rule Engine Integration Tests
# =============================================================================


class TestRuleEngineIntegration:
    """Integration tests for rule engine workflow."""

    def test_evaluate_compliant_feature(self, compliant_hiring_feature: ProductFeatureInput):
        """Test rule engine with compliant feature."""
        engine = get_rule_engine()
        results = engine.evaluate(compliant_hiring_feature)

        # Should have some results
        assert results is not None
        assert isinstance(results, list)

        # If there are violations, check they have expected attributes
        for r in results:
            assert hasattr(r, "rule_id")
            assert hasattr(r, "risk_score")
            assert hasattr(r, "result")

    def test_evaluate_risky_feature(self, risky_hiring_feature: ProductFeatureInput):
        """Test rule engine with risky feature."""
        engine = get_rule_engine()
        results = engine.evaluate(risky_hiring_feature)

        # Should detect violations
        assert len(results) > 0

        # Check for high risk scores (violations)
        high_risk_results = [r for r in results if r.risk_score > 50]
        # Risky feature should have some high risk evaluations
        assert len(high_risk_results) >= 1

    def test_rule_results_have_required_fields(self, compliant_hiring_feature: ProductFeatureInput):
        """Test that rule results have all required fields."""
        engine = get_rule_engine()
        results = engine.evaluate(compliant_hiring_feature)

        for result in results:
            assert result.rule_id is not None
            assert result.rule_name is not None
            assert result.result in [
                RuleResult.VIOLATION,
                RuleResult.POTENTIAL_VIOLATION,
                RuleResult.COMPLIANT,
                RuleResult.NEEDS_LLM_REVIEW,
            ]
            assert 0 <= result.confidence <= 1
            # risk_score can be negative for compliant/bonus rules
            assert -100 <= result.risk_score <= 100


# =============================================================================
# Scoring Engine Integration Tests
# =============================================================================


class TestScoringEngineIntegration:
    """Integration tests for scoring engine."""

    def test_calculate_unified_score_rule_only(self, compliant_hiring_feature: ProductFeatureInput):
        """Test unified score calculation with rules only."""
        scoring_engine = get_scoring_engine()
        rule_engine = get_rule_engine()

        # Get rule evaluations
        evaluations = rule_engine.evaluate(compliant_hiring_feature)

        # Calculate score
        score = scoring_engine.calculate_unified_score(
            rule_evaluations=evaluations,
            rag_response=None,
        )

        assert score.overall_score >= 0
        assert score.overall_score <= 100
        assert len(score.component_scores) >= 0
        # Check confidence interval exists
        assert score.confidence_interval is not None

    def test_score_comparison_compliant_vs_risky(
        self,
        compliant_hiring_feature: ProductFeatureInput,
        risky_hiring_feature: ProductFeatureInput,
    ):
        """Test that risky features score higher than compliant ones."""
        scoring_engine = get_scoring_engine()
        rule_engine = get_rule_engine()

        # Score compliant feature
        compliant_evaluations = rule_engine.evaluate(compliant_hiring_feature)
        compliant_score = scoring_engine.calculate_unified_score(
            rule_evaluations=compliant_evaluations,
            rag_response=None,
        )

        # Score risky feature
        risky_evaluations = rule_engine.evaluate(risky_hiring_feature)
        risky_score = scoring_engine.calculate_unified_score(
            rule_evaluations=risky_evaluations,
            rag_response=None,
        )

        # Risky feature should have higher (worse) risk score
        assert risky_score.overall_score >= compliant_score.overall_score

    def test_risk_level_classification(self, risky_hiring_feature: ProductFeatureInput):
        """Test that risk levels are properly classified."""
        scoring_engine = get_scoring_engine()
        rule_engine = get_rule_engine()

        evaluations = rule_engine.evaluate(risky_hiring_feature)
        score = scoring_engine.calculate_unified_score(
            rule_evaluations=evaluations,
            rag_response=None,
        )

        # Risk level should be a valid enum value
        assert score.risk_level in [
            RiskLevel.CRITICAL,
            RiskLevel.HIGH,
            RiskLevel.MEDIUM,
            RiskLevel.LOW,
            RiskLevel.MINIMAL,
        ]


# =============================================================================
# Compliance Engine Integration Tests
# =============================================================================


class TestComplianceEngineIntegration:
    """Integration tests for compliance engine."""

    @patch("wenah.core.engine.get_rag_pipeline")
    def test_assess_feature_without_llm(
        self,
        mock_get_rag,
        compliant_hiring_feature: ProductFeatureInput,
    ):
        """Test feature assessment without LLM."""
        mock_rag = MagicMock()
        mock_get_rag.return_value = mock_rag

        engine = ComplianceEngine()
        config = AssessmentConfig(include_llm_analysis=False)

        result = engine.assess_feature(compliant_hiring_feature, config)

        assert result is not None
        assert result.feature.feature_id == "compliant-001"
        assert hasattr(result, "rule_evaluations")
        assert hasattr(result, "unified_score")

    @patch("wenah.core.engine.get_rag_pipeline")
    def test_assess_multiple_features(
        self,
        mock_get_rag,
        compliant_hiring_feature: ProductFeatureInput,
        risky_hiring_feature: ProductFeatureInput,
    ):
        """Test assessing multiple features."""
        mock_rag = MagicMock()
        mock_get_rag.return_value = mock_rag

        engine = ComplianceEngine()
        config = AssessmentConfig(include_llm_analysis=False)

        # Assess each feature individually
        results = [
            engine.assess_feature(compliant_hiring_feature, config),
            engine.assess_feature(risky_hiring_feature, config),
        ]

        assert len(results) == 2
        assert results[0].feature.feature_id == "compliant-001"
        assert results[1].feature.feature_id == "risky-001"


# =============================================================================
# Use Case Integration Tests
# =============================================================================


class TestDesignGuidanceIntegration:
    """Integration tests for design guidance use case."""

    def test_check_protected_class_field(self):
        """Test checking a protected class field."""
        engine = get_design_guidance()
        result = engine.check_data_field(
            field_name="race",
            field_description="Candidate race",
            category=ProductCategory.HIRING,
            used_in_decisions=True,
        )

        assert result is not None
        assert result.is_protected_class is True
        assert result.risk_level == RiskLevel.CRITICAL

    def test_check_proxy_variable(self):
        """Test checking a proxy variable."""
        engine = get_design_guidance()
        result = engine.check_data_field(
            field_name="zip_code",
            field_description="Applicant ZIP code",
            category=ProductCategory.HIRING,
            used_in_decisions=True,
        )

        assert result is not None
        assert result.is_proxy_variable is True
        assert result.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]

    def test_check_safe_field(self):
        """Test checking a safe field."""
        engine = get_design_guidance()
        result = engine.check_data_field(
            field_name="skills_assessment_score",
            field_description="Technical skills test score",
            category=ProductCategory.HIRING,
            used_in_decisions=True,
        )

        assert result is not None
        assert result.is_protected_class is False
        assert result.risk_level in [RiskLevel.LOW, RiskLevel.MINIMAL]


class TestPrelaunchCheckIntegration:
    """Integration tests for pre-launch check use case."""

    @patch("wenah.use_cases.prelaunch_check.get_compliance_engine")
    def test_run_prelaunch_check(
        self,
        mock_get_engine,
        compliant_hiring_feature: ProductFeatureInput,
    ):
        """Test full pre-launch check workflow."""
        # Setup mock
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.feature = compliant_hiring_feature
        mock_result.unified_score = MagicMock()
        mock_result.unified_score.overall_score = 25.0
        mock_result.unified_score.risk_level = RiskLevel.LOW
        mock_result.rule_evaluations = []
        mock_result.rag_result = None

        mock_engine.assess_features.return_value = [mock_result]
        mock_get_engine.return_value = mock_engine

        checker = PrelaunchChecker()
        result = checker.run_prelaunch_check(
            product_name="Test Product",
            features=[compliant_hiring_feature],
            documentation_status={
                "bias_testing_report": True,
                "algorithm_specification": True,
            },
        )

        assert result is not None
        assert result.product_name == "Test Product"
        assert hasattr(result, "launch_decision")
        assert hasattr(result, "blocking_issues")

    def test_quick_check_workflow(self, compliant_hiring_feature: ProductFeatureInput):
        """Test quick pre-launch check."""
        checker = get_prelaunch_checker()
        result = checker.quick_check([compliant_hiring_feature])

        assert "can_launch" in result
        assert "blocking_issues" in result
        assert "recommendation" in result


# =============================================================================
# Full Pipeline Integration Tests
# =============================================================================


class TestFullPipelineIntegration:
    """Tests for complete compliance pipeline."""

    def test_design_to_launch_pipeline(
        self,
        compliant_hiring_feature: ProductFeatureInput,
    ):
        """Test pipeline from design guidance to launch check."""
        # Step 1: Design guidance
        guidance_engine = get_design_guidance()

        for field in compliant_hiring_feature.data_fields:
            guidance = guidance_engine.check_data_field(
                field_name=field.name,
                field_description=field.description,
                category=compliant_hiring_feature.category,
                used_in_decisions=field.used_in_decisions,
            )
            # Verify guidance is returned
            assert guidance is not None

        # Step 2: Pre-launch quick check
        checker = get_prelaunch_checker()
        quick_result = checker.quick_check([compliant_hiring_feature])

        assert "can_launch" in quick_result
        assert "recommendation" in quick_result

    def test_risk_detection_consistency(
        self,
        risky_hiring_feature: ProductFeatureInput,
    ):
        """Test that risk detection is consistent across components."""
        # Rule engine should detect risk
        rule_engine = get_rule_engine()
        evaluations = rule_engine.evaluate(risky_hiring_feature)
        assert len(evaluations) > 0

        # Design guidance should flag risky fields
        guidance_engine = get_design_guidance()
        for field in risky_hiring_feature.data_fields:
            guidance = guidance_engine.check_data_field(
                field_name=field.name,
                field_description=field.description,
                category=risky_hiring_feature.category,
                used_in_decisions=field.used_in_decisions,
            )
            if field.name in ["age", "zip_code", "name"]:
                # These should be flagged as risky
                assert guidance.risk_level != RiskLevel.MINIMAL

        # Pre-launch check should have concerns
        checker = get_prelaunch_checker()
        quick_result = checker.quick_check([risky_hiring_feature])
        # Should either have blocking issues or warnings
        has_concerns = len(quick_result["blocking_issues"]) > 0 or len(quick_result["warnings"]) > 0
        assert has_concerns or quick_result["recommendation"] != "APPROVED"


# =============================================================================
# Cross-Category Tests
# =============================================================================


class TestCrossCategoryIntegration:
    """Tests for cross-category functionality."""

    def test_employment_vs_lending_categories(
        self,
        risky_hiring_feature: ProductFeatureInput,
        lending_feature: ProductFeatureInput,
    ):
        """Test different categories are handled appropriately."""
        rule_engine = get_rule_engine()

        # Evaluate hiring feature
        hiring_results = rule_engine.evaluate(risky_hiring_feature)

        # Evaluate lending feature
        lending_results = rule_engine.evaluate(lending_feature)

        # Both should return results
        assert hiring_results is not None
        assert lending_results is not None

        # Categories should be different
        assert risky_hiring_feature.category == ProductCategory.HIRING
        assert lending_feature.category == ProductCategory.LENDING

    def test_guidance_for_different_categories(self):
        """Test design guidance works across categories."""
        guidance_engine = get_design_guidance()

        # Test hiring category
        hiring_guidance = guidance_engine.check_data_field(
            field_name="skills_score",
            field_description="Technical skills assessment",
            category=ProductCategory.HIRING,
            used_in_decisions=True,
        )
        assert hiring_guidance is not None

        # Test lending category
        lending_guidance = guidance_engine.check_data_field(
            field_name="credit_score",
            field_description="Applicant credit score",
            category=ProductCategory.LENDING,
            used_in_decisions=True,
        )
        assert lending_guidance is not None

        # Test housing category
        housing_guidance = guidance_engine.check_data_field(
            field_name="rental_history",
            field_description="Previous rental history",
            category=ProductCategory.HOUSING,
            used_in_decisions=True,
        )
        assert housing_guidance is not None


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case integration tests."""

    def test_empty_data_fields(self):
        """Test feature with no data fields."""
        feature = ProductFeatureInput(
            feature_id="empty-001",
            name="Simple Feature",
            description="A feature with no data fields",
            category=ProductCategory.GENERAL,
            feature_type=FeatureType.USER_INTERFACE,
            data_fields=[],
            decision_impact="Display only",
            affected_population="All users",
        )

        rule_engine = get_rule_engine()
        results = rule_engine.evaluate(feature)

        # Should handle empty data fields gracefully
        assert results is not None

    def test_feature_with_all_proxies(self):
        """Test feature where all fields are proxies."""
        feature = ProductFeatureInput(
            feature_id="proxy-heavy-001",
            name="Proxy Feature",
            description="A feature with many proxy variables",
            category=ProductCategory.HIRING,
            feature_type=FeatureType.SCORING_MODEL,
            data_fields=[
                DataFieldSpec(
                    name="zip_code",
                    description="ZIP code",
                    data_type="text",
                    source="application",
                    used_in_decisions=True,
                    potential_proxy="race",
                ),
                DataFieldSpec(
                    name="school_name",
                    description="School attended",
                    data_type="text",
                    source="application",
                    used_in_decisions=True,
                    potential_proxy="socioeconomic_status",
                ),
            ],
            decision_impact="Scoring",
            affected_population="Applicants",
        )

        guidance_engine = get_design_guidance()
        for field in feature.data_fields:
            guidance = guidance_engine.check_data_field(
                field_name=field.name,
                field_description=field.description,
                category=feature.category,
                used_in_decisions=True,
            )
            # All proxy fields should be flagged
            assert guidance.is_proxy_variable is True or guidance.risk_level != RiskLevel.MINIMAL
