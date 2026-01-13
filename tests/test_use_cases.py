"""
Tests for use case implementations.

Tests cover:
- RiskDashboard: Product and feature assessment, report generation
- DesignGuidanceEngine: Data field and algorithm guidance
- PrelaunchChecker: Compliance checks and launch decisions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from wenah.core.types import (
    ProductFeatureInput,
    ProductCategory,
    FeatureType,
    DataFieldSpec,
    AlgorithmSpec,
    RiskLevel,
    RuleEvaluation,
    RuleResult,
    CategoryBreakdown,
    ViolationDetail,
    RecommendationItem,
    RiskAssessmentResponse,
    FeatureAssessment,
)
from wenah.use_cases.risk_dashboard import (
    RiskDashboard,
    DashboardData,
    DashboardViewType,
    FeatureRiskSummary,
    CategoryRiskDetail,
    get_risk_dashboard,
)
from wenah.use_cases.design_guidance import (
    DesignGuidanceEngine,
    DesignGuidanceResponse,
    GuidanceLevel,
    DesignChoice,
    DataFieldGuidance,
    AlgorithmGuidance,
    get_design_guidance,
)
from wenah.use_cases.prelaunch_check import (
    PrelaunchChecker,
    PrelaunchCheckResponse,
    LaunchDecision,
    CheckStatus,
    ComplianceCheckItem,
    get_prelaunch_checker,
)


class TestRiskDashboard:
    """Tests for RiskDashboard class."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock compliance engine."""
        engine = Mock()

        # Mock assess_product response
        engine.assess_product.return_value = RiskAssessmentResponse(
            assessment_id="test-001",
            product_name="Test Product",
            timestamp=datetime.now(timezone.utc),
            overall_risk_score=55.0,
            overall_risk_level=RiskLevel.MEDIUM,
            confidence_score=0.8,
            confidence_interval=(50.0, 60.0),
            category_breakdown=CategoryBreakdown(
                employment=55.0,
                housing=0.0,
                consumer=0.0,
                overall=55.0,
            ),
            feature_assessments=[
                FeatureAssessment(
                    feature_id="f1",
                    feature_name="Test Feature",
                    risk_score=55.0,
                    risk_level=RiskLevel.MEDIUM,
                    violations=[],
                    recommendations=[],
                    compliant_aspects=["Human oversight"],
                    requires_human_review=False,
                ),
            ],
            total_violations=1,
            critical_violations=[],
            all_recommendations=[
                RecommendationItem(
                    priority=1,
                    category="hiring",
                    recommendation="Review for compliance",
                    rationale="Best practice",
                    estimated_effort="low",
                    law_references=[],
                ),
            ],
            executive_summary="Test assessment completed.",
            key_concerns=["Minor concern"],
            positive_aspects=["Human oversight in place"],
            requires_human_review=False,
            human_review_reasons=[],
            rules_version="1.0.0",
            laws_data_version="1.0.0",
        )

        # Mock quick_assess
        engine.quick_assess.return_value = {
            "feature_id": "f1",
            "risk_score": 45.0,
            "risk_level": "medium",
            "violations_count": 1,
            "requires_full_analysis": False,
            "primary_concerns": [],
        }

        return engine

    @pytest.fixture
    def dashboard(self, mock_engine):
        """Create dashboard with mock engine."""
        return RiskDashboard(compliance_engine=mock_engine)

    def test_assess_product_basic(
        self,
        dashboard,
        sample_hiring_feature,
    ):
        """Test basic product assessment."""
        result = dashboard.assess_product(
            product_name="Test Product",
            features=[sample_hiring_feature],
            view_type=DashboardViewType.DETAILED,
        )

        assert isinstance(result, DashboardData)
        assert result.product_name == "Test Product"
        assert result.overall_score == 55.0
        assert result.overall_risk_level == RiskLevel.MEDIUM

    def test_assess_product_executive_view(
        self,
        dashboard,
        sample_hiring_feature,
    ):
        """Test executive view assessment."""
        result = dashboard.assess_product(
            product_name="Test Product",
            features=[sample_hiring_feature],
            view_type=DashboardViewType.EXECUTIVE,
        )

        assert result.view_type == DashboardViewType.EXECUTIVE

    def test_assess_single_feature(
        self,
        dashboard,
        sample_hiring_feature,
    ):
        """Test single feature assessment."""
        result = dashboard.assess_single_feature(
            feature=sample_hiring_feature,
            view_type=DashboardViewType.DETAILED,
        )

        assert isinstance(result, DashboardData)
        # Mock returns "Test Product" as product name
        assert result.product_name is not None

    def test_get_quick_score(
        self,
        dashboard,
        sample_hiring_feature,
    ):
        """Test quick score calculation."""
        result = dashboard.get_quick_score([sample_hiring_feature])

        assert "overall_score" in result
        assert "risk_level" in result
        assert "total_features" in result
        assert result["total_features"] == 1

    def test_generate_markdown_report(self, dashboard):
        """Test markdown report generation."""
        data = DashboardData(
            product_name="Test Product",
            assessment_id="test-001",
            generated_at=datetime.now(timezone.utc),
            view_type=DashboardViewType.DETAILED,
            overall_score=60.0,
            overall_risk_level=RiskLevel.HIGH,
            confidence_score=0.8,
            confidence_interval=(55.0, 65.0),
            total_features=2,
            features_at_risk=1,
            total_violations=2,
            critical_violations=0,
            category_details=[],
            feature_summaries=[
                FeatureRiskSummary(
                    feature_id="f1",
                    feature_name="Feature 1",
                    score=60.0,
                    risk_level=RiskLevel.HIGH,
                    violation_count=2,
                    requires_attention=True,
                    top_issue="Test issue",
                ),
            ],
            executive_summary="Test summary.",
            key_concerns=["Concern 1"],
            immediate_actions=["Action 1"],
            positive_aspects=["Positive 1"],
        )

        report = dashboard.generate_report(data, format="markdown")

        assert "# Risk Assessment Report" in report
        assert "Test Product" in report
        assert "60/100" in report
        assert "HIGH" in report

    def test_generate_text_report(self, dashboard):
        """Test plain text report generation."""
        data = DashboardData(
            product_name="Test Product",
            assessment_id="test-001",
            generated_at=datetime.now(timezone.utc),
            view_type=DashboardViewType.DETAILED,
            overall_score=30.0,
            overall_risk_level=RiskLevel.LOW,
            confidence_score=0.9,
            confidence_interval=(25.0, 35.0),
            total_features=1,
            features_at_risk=0,
            total_violations=0,
            critical_violations=0,
            category_details=[],
            feature_summaries=[],
            executive_summary="All clear.",
            key_concerns=[],
            immediate_actions=[],
            positive_aspects=[],
        )

        report = dashboard.generate_report(data, format="text")

        assert "RISK ASSESSMENT REPORT" in report
        assert "TEST PRODUCT" in report.upper()


class TestDesignGuidanceEngine:
    """Tests for DesignGuidanceEngine class."""

    @pytest.fixture
    def guidance_engine(self):
        """Create design guidance engine with mocks."""
        with patch("wenah.use_cases.design_guidance.get_compliance_engine"):
            with patch("wenah.use_cases.design_guidance.get_employment_processor"):
                return DesignGuidanceEngine()

    def test_check_protected_class_field(self, guidance_engine):
        """Test detection of protected class field."""
        result = guidance_engine.check_data_field(
            field_name="race",
            field_description="Applicant's race",
            category=ProductCategory.HIRING,
            used_in_decisions=True,
        )

        assert isinstance(result, DataFieldGuidance)
        assert result.is_protected_class is True
        assert result.risk_level == RiskLevel.CRITICAL
        assert result.design_choice == DesignChoice.AVOID

    def test_check_protected_class_not_in_decisions(self, guidance_engine):
        """Test protected class collected but not used in decisions."""
        result = guidance_engine.check_data_field(
            field_name="disability_status",
            field_description="For accommodation purposes",
            category=ProductCategory.HIRING,
            used_in_decisions=False,
        )

        assert result.is_protected_class is True
        assert result.risk_level == RiskLevel.MEDIUM
        assert result.design_choice == DesignChoice.CAUTION

    def test_check_proxy_variable(self, guidance_engine):
        """Test detection of proxy variable."""
        result = guidance_engine.check_data_field(
            field_name="zip_code",
            field_description="Applicant's ZIP code",
            category=ProductCategory.HIRING,
            used_in_decisions=True,
        )

        assert result.is_proxy_variable is True
        assert result.proxy_for is not None
        assert result.risk_level == RiskLevel.HIGH
        assert result.design_choice == DesignChoice.REQUIRES_REVIEW
        assert len(result.alternatives) > 0

    def test_check_neutral_field(self, guidance_engine):
        """Test neutral data field."""
        result = guidance_engine.check_data_field(
            field_name="years_of_python_experience",
            field_description="Programming experience",
            category=ProductCategory.HIRING,
            used_in_decisions=True,
        )

        assert result.is_protected_class is False
        assert result.is_proxy_variable is False
        assert result.risk_level == RiskLevel.LOW
        assert result.design_choice == DesignChoice.RECOMMENDED

    def test_check_algorithm_ml_no_testing(self, guidance_engine):
        """Test ML algorithm without bias testing."""
        algo = AlgorithmSpec(
            name="ResumeRanker",
            type="ml_model",
            inputs=["resume_text"],
            outputs=["score"],
            bias_testing_done=False,
        )

        result = guidance_engine.check_algorithm_design(
            algo, ProductCategory.HIRING
        )

        assert isinstance(result, AlgorithmGuidance)
        assert result.risk_level == RiskLevel.HIGH
        assert result.design_choice == DesignChoice.CAUTION
        assert len(result.requirements) > 0

    def test_check_algorithm_with_video_input(self, guidance_engine):
        """Test algorithm with video/facial analysis input."""
        algo = AlgorithmSpec(
            name="VideoInterviewer",
            type="ml_model",
            inputs=["video", "facial_expression"],
            outputs=["confidence_score"],
            bias_testing_done=True,
        )

        result = guidance_engine.check_algorithm_design(
            algo, ProductCategory.HIRING
        )

        assert result.risk_level == RiskLevel.CRITICAL
        assert result.design_choice == DesignChoice.REQUIRES_REVIEW
        assert "ADA" in " ".join(result.legal_references)

    def test_check_algorithm_rule_based(self, guidance_engine):
        """Test rule-based algorithm."""
        algo = AlgorithmSpec(
            name="SimpleScorer",
            type="rule_based",
            inputs=["test_score"],
            outputs=["pass_fail"],
            bias_testing_done=False,
        )

        result = guidance_engine.check_algorithm_design(
            algo, ProductCategory.HIRING
        )

        assert result.risk_level == RiskLevel.LOW
        assert result.design_choice == DesignChoice.RECOMMENDED

    def test_get_guidance_full(
        self,
        guidance_engine,
        sample_hiring_feature,
    ):
        """Test full guidance generation."""
        result = guidance_engine.get_guidance(
            product_name="Test Product",
            features=[sample_hiring_feature],
            level=GuidanceLevel.COMPREHENSIVE,
        )

        assert isinstance(result, DesignGuidanceResponse)
        assert result.product_name == "Test Product"
        assert len(result.feature_guidance) == 1
        assert len(result.design_principles) > 0

    def test_get_compliance_checklist(
        self,
        guidance_engine,
        sample_hiring_feature,
    ):
        """Test compliance checklist generation."""
        checklist = guidance_engine.get_compliance_checklist(sample_hiring_feature)

        assert isinstance(checklist, list)
        assert len(checklist) > 0
        assert all("category" in item for item in checklist)
        assert all("item" in item for item in checklist)

    def test_guidance_with_proxy_feature(
        self,
        guidance_engine,
        sample_hiring_feature_with_proxy,
    ):
        """Test guidance for feature with proxy variables."""
        result = guidance_engine.get_guidance(
            product_name="Test Product",
            features=[sample_hiring_feature_with_proxy],
            level=GuidanceLevel.STANDARD,
        )

        # Should have warnings about proxy variables
        feature_guidance = result.feature_guidance[0]
        proxy_fields = [
            fg for fg in feature_guidance.data_field_guidance
            if fg.is_proxy_variable
        ]
        assert len(proxy_fields) > 0


class TestPrelaunchChecker:
    """Tests for PrelaunchChecker class."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock compliance engine."""
        engine = Mock()
        engine.assess_product.return_value = RiskAssessmentResponse(
            assessment_id="test-001",
            product_name="Test Product",
            timestamp=datetime.now(timezone.utc),
            overall_risk_score=45.0,
            overall_risk_level=RiskLevel.MEDIUM,
            confidence_score=0.8,
            confidence_interval=(40.0, 50.0),
            category_breakdown=CategoryBreakdown(
                employment=45.0,
                housing=0.0,
                consumer=0.0,
                overall=45.0,
            ),
            feature_assessments=[],
            total_violations=0,
            critical_violations=[],
            all_recommendations=[],
            executive_summary="Test",
            key_concerns=[],
            positive_aspects=[],
            requires_human_review=False,
            human_review_reasons=[],
            rules_version="1.0.0",
            laws_data_version="1.0.0",
        )
        return engine

    @pytest.fixture
    def checker(self, mock_engine):
        """Create pre-launch checker with mocks."""
        with patch("wenah.use_cases.prelaunch_check.get_design_guidance"):
            return PrelaunchChecker(compliance_engine=mock_engine)

    def test_quick_check_compliant(
        self,
        checker,
        sample_compliant_feature,
    ):
        """Test quick check for compliant feature."""
        result = checker.quick_check([sample_compliant_feature])

        assert result["can_launch"] is True
        assert len(result["blocking_issues"]) == 0
        assert result["recommendation"] == "APPROVED"

    def test_quick_check_protected_class(
        self,
        checker,
        sample_hiring_feature_with_protected_class,
    ):
        """Test quick check detects protected class usage."""
        result = checker.quick_check([sample_hiring_feature_with_protected_class])

        assert result["can_launch"] is False
        assert len(result["blocking_issues"]) > 0
        assert result["recommendation"] == "BLOCKED"

    def test_quick_check_untested_algorithm(self, checker):
        """Test quick check detects untested AI algorithm."""
        feature = ProductFeatureInput(
            feature_id="test-001",
            name="AI Screener",
            description="AI-powered screening",
            category=ProductCategory.HIRING,
            feature_type=FeatureType.ALGORITHM,
            data_fields=[],
            algorithm=AlgorithmSpec(
                name="AIModel",
                type="ml_model",
                inputs=["resume"],
                outputs=["score"],
                bias_testing_done=False,
            ),
            decision_impact="Hiring decisions",
            affected_population="Applicants",
        )

        result = checker.quick_check([feature])

        assert result["can_launch"] is False
        assert any("bias testing" in issue.lower() for issue in result["blocking_issues"])

    def test_run_prelaunch_check_basic(
        self,
        checker,
        sample_hiring_feature,
    ):
        """Test full pre-launch check."""
        result = checker.run_prelaunch_check(
            product_name="Test Product",
            features=[sample_hiring_feature],
        )

        assert isinstance(result, PrelaunchCheckResponse)
        assert result.product_name == "Test Product"
        assert result.total_checks > 0
        assert result.launch_decision in LaunchDecision

    def test_run_prelaunch_check_with_evidence(
        self,
        checker,
        sample_hiring_feature,
    ):
        """Test pre-launch check with evidence provided."""
        evidence = {
            f"job_related_{sample_hiring_feature.feature_id}": True,
            f"human_oversight_{sample_hiring_feature.feature_id}": True,
        }

        result = checker.run_prelaunch_check(
            product_name="Test Product",
            features=[sample_hiring_feature],
            evidence=evidence,
        )

        assert result.passed_checks > 0

    def test_run_prelaunch_check_with_documentation(
        self,
        checker,
        sample_hiring_feature,
    ):
        """Test pre-launch check with documentation status."""
        doc_status = {
            "doc-req-001": True,
            "doc-req-002": True,
            "doc-req-003": True,
            "doc-req-004": True,
            "doc-req-005": True,
        }

        result = checker.run_prelaunch_check(
            product_name="Test Product",
            features=[sample_hiring_feature],
            documentation_status=doc_status,
        )

        assert result.documentation_complete is True

    def test_launch_decision_blocked(self, checker):
        """Test blocked launch decision."""
        # Feature with protected class in decisions
        feature = ProductFeatureInput(
            feature_id="test-001",
            name="Bad Feature",
            description="Feature with violations",
            category=ProductCategory.HIRING,
            feature_type=FeatureType.DATA_COLLECTION,
            data_fields=[
                DataFieldSpec(
                    name="race",
                    description="Applicant race",
                    data_type="text",
                    source="user_input",
                    required=True,
                    used_in_decisions=True,
                ),
            ],
            decision_impact="Hiring",
            affected_population="Applicants",
        )

        result = checker.run_prelaunch_check(
            product_name="Test Product",
            features=[feature],
        )

        assert result.launch_decision == LaunchDecision.BLOCKED
        assert len(result.blocking_issues) > 0

    def test_launch_decision_conditional(self, checker, mock_engine):
        """Test conditional launch decision."""
        # Modify mock to return high risk
        mock_engine.assess_product.return_value.overall_risk_level = RiskLevel.HIGH
        mock_engine.assess_product.return_value.overall_risk_score = 70.0

        feature = ProductFeatureInput(
            feature_id="test-001",
            name="Test Feature",
            description="Compliant feature",
            category=ProductCategory.HIRING,
            feature_type=FeatureType.ALGORITHM,
            data_fields=[],
            decision_impact="Hiring",
            affected_population="Applicants",
        )

        # Provide all required documentation
        doc_status = {
            "doc-req-001": True,
            "doc-req-002": True,
            "doc-req-003": True,
            "doc-req-004": True,
            "doc-req-005": True,
        }

        evidence = {
            "job_related_test-001": True,
        }

        result = checker.run_prelaunch_check(
            product_name="Test Product",
            features=[feature],
            documentation_status=doc_status,
            evidence=evidence,
        )

        assert result.launch_decision == LaunchDecision.CONDITIONAL
        assert len(result.launch_conditions) > 0

    def test_monitoring_requirements_generated(
        self,
        checker,
        sample_hiring_feature,
    ):
        """Test monitoring requirements generation."""
        result = checker.run_prelaunch_check(
            product_name="Test Product",
            features=[sample_hiring_feature],
        )

        assert len(result.monitoring_requirements) > 0
        assert any("monitor" in req.lower() for req in result.monitoring_requirements)

    def test_sign_offs_required(self, checker, mock_engine):
        """Test sign-off requirements based on risk level."""
        mock_engine.assess_product.return_value.overall_risk_level = RiskLevel.CRITICAL

        feature = ProductFeatureInput(
            feature_id="test-001",
            name="Test Feature",
            description="High risk feature",
            category=ProductCategory.HIRING,
            feature_type=FeatureType.ALGORITHM,
            data_fields=[],
            decision_impact="Hiring",
            affected_population="Applicants",
        )

        result = checker.run_prelaunch_check(
            product_name="Test Product",
            features=[feature],
        )

        assert "Legal Counsel" in result.sign_offs_required


class TestSingletonGetters:
    """Tests for singleton getter functions."""

    def test_get_risk_dashboard(self):
        """Test singleton risk dashboard getter."""
        with patch("wenah.use_cases.risk_dashboard.get_compliance_engine"):
            dashboard1 = get_risk_dashboard()
            dashboard2 = get_risk_dashboard()

            assert dashboard1 is dashboard2

    def test_get_design_guidance(self):
        """Test singleton design guidance getter."""
        with patch("wenah.use_cases.design_guidance.get_compliance_engine"):
            with patch("wenah.use_cases.design_guidance.get_employment_processor"):
                guidance1 = get_design_guidance()
                guidance2 = get_design_guidance()

                assert guidance1 is guidance2

    def test_get_prelaunch_checker(self):
        """Test singleton pre-launch checker getter."""
        with patch("wenah.use_cases.prelaunch_check.get_compliance_engine"):
            with patch("wenah.use_cases.prelaunch_check.get_design_guidance"):
                checker1 = get_prelaunch_checker()
                checker2 = get_prelaunch_checker()

                assert checker1 is checker2


class TestIntegrationScenarios:
    """Integration tests for use case workflows."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock compliance engine for integration tests."""
        engine = Mock()
        engine.assess_product.return_value = RiskAssessmentResponse(
            assessment_id="int-001",
            product_name="Integration Test",
            timestamp=datetime.now(timezone.utc),
            overall_risk_score=35.0,
            overall_risk_level=RiskLevel.LOW,
            confidence_score=0.9,
            confidence_interval=(30.0, 40.0),
            category_breakdown=CategoryBreakdown(
                employment=35.0,
                housing=0.0,
                consumer=0.0,
                overall=35.0,
            ),
            feature_assessments=[
                FeatureAssessment(
                    feature_id="f1",
                    feature_name="Test Feature",
                    risk_score=35.0,
                    risk_level=RiskLevel.LOW,
                    violations=[],
                    recommendations=[],
                    compliant_aspects=["Bias testing done"],
                    requires_human_review=False,
                ),
            ],
            total_violations=0,
            critical_violations=[],
            all_recommendations=[],
            executive_summary="Product is compliant.",
            key_concerns=[],
            positive_aspects=["Bias testing completed"],
            requires_human_review=False,
            human_review_reasons=[],
            rules_version="1.0.0",
            laws_data_version="1.0.0",
        )
        engine.quick_assess.return_value = {
            "feature_id": "f1",
            "risk_score": 35.0,
            "risk_level": "low",
            "violations_count": 0,
            "requires_full_analysis": False,
            "primary_concerns": [],
        }
        return engine

    def test_design_to_launch_workflow(
        self,
        mock_engine,
        sample_compliant_feature,
    ):
        """Test workflow from design guidance to launch check."""
        # Step 1: Get design guidance
        with patch("wenah.use_cases.design_guidance.get_compliance_engine"):
            with patch("wenah.use_cases.design_guidance.get_employment_processor"):
                guidance_engine = DesignGuidanceEngine()
                design_result = guidance_engine.get_guidance(
                    product_name="Test Product",
                    features=[sample_compliant_feature],
                    level=GuidanceLevel.STANDARD,
                )

        assert design_result.overall_design_risk in [RiskLevel.LOW, RiskLevel.MINIMAL, RiskLevel.MEDIUM]

        # Step 2: Run risk dashboard assessment
        dashboard = RiskDashboard(compliance_engine=mock_engine)
        dashboard_result = dashboard.assess_product(
            product_name="Test Product",
            features=[sample_compliant_feature],
        )

        assert dashboard_result.overall_score < 50

        # Step 3: Pre-launch check
        with patch("wenah.use_cases.prelaunch_check.get_design_guidance"):
            checker = PrelaunchChecker(compliance_engine=mock_engine)

            # Provide all documentation
            doc_status = {
                "doc-req-001": True,
                "doc-req-002": True,
                "doc-req-003": True,
                "doc-req-004": True,
                "doc-req-005": True,
            }

            evidence = {
                f"job_related_{sample_compliant_feature.feature_id}": True,
            }

            launch_result = checker.run_prelaunch_check(
                product_name="Test Product",
                features=[sample_compliant_feature],
                documentation_status=doc_status,
                evidence=evidence,
            )

        # Compliant feature should be approved
        assert launch_result.launch_decision in [
            LaunchDecision.APPROVED,
            LaunchDecision.CONDITIONAL,
        ]
