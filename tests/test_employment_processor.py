"""
Tests for the employment category processor.
"""

import pytest

from wenah.rules.categories.employment import (
    EmploymentCategoryProcessor,
    get_employment_processor,
    PROTECTED_CLASS_FIELDS,
    PROXY_INDICATORS,
)
from wenah.core.types import (
    ProductFeatureInput,
    ProductCategory,
    FeatureType,
    DataFieldSpec,
    AlgorithmSpec,
)


class TestEmploymentCategoryProcessor:
    """Tests for the EmploymentCategoryProcessor class."""

    @pytest.fixture
    def processor(self) -> EmploymentCategoryProcessor:
        """Create a processor instance."""
        return EmploymentCategoryProcessor()

    def test_initialization(self, processor: EmploymentCategoryProcessor):
        """Test processor initialization."""
        assert processor is not None
        assert len(processor.protected_class_fields) > 0
        assert len(processor.proxy_indicators) > 0
        assert len(processor.medical_inquiry_fields) > 0

    def test_analyze_protected_class_collection(
        self,
        processor: EmploymentCategoryProcessor,
        sample_hiring_feature_with_protected_class: ProductFeatureInput,
    ):
        """Test detection of protected class data collection."""
        analysis = processor.analyze_feature(sample_hiring_feature_with_protected_class)

        assert analysis["feature_id"] == "test-hiring-002"
        assert analysis["category"] == "employment"

        # Should have findings about protected class data
        assert len(analysis["findings"]) > 0
        assert len(analysis["protected_class_exposure"]) > 0

        # Should find race and disability
        protected_fields = [
            p["field"] for p in analysis["protected_class_exposure"]
        ]
        assert "race" in protected_fields or any("race" in f.lower() for f in protected_fields)

    def test_analyze_proxy_variables(
        self,
        processor: EmploymentCategoryProcessor,
        sample_hiring_feature_with_proxy: ProductFeatureInput,
    ):
        """Test detection of proxy variables."""
        analysis = processor.analyze_feature(sample_hiring_feature_with_proxy)

        # Should have proxy concerns
        assert len(analysis["proxy_variable_concerns"]) > 0

        # zip_code should be flagged
        proxy_fields = [p["field"] for p in analysis["proxy_variable_concerns"]]
        assert "zip_code" in proxy_fields

    def test_analyze_video_interview(
        self,
        processor: EmploymentCategoryProcessor,
        sample_video_interview_feature: ProductFeatureInput,
    ):
        """Test detection of high-risk video interview features."""
        analysis = processor.analyze_feature(sample_video_interview_feature)

        # Should have findings about missing bias testing
        missing_bias = [
            f for f in analysis["findings"]
            if f.get("type") == "missing_bias_testing"
        ]
        assert len(missing_bias) > 0

        # Should have findings about high-risk algorithm
        high_risk = [
            f for f in analysis["findings"]
            if f.get("type") == "high_risk_hiring_algorithm"
        ]
        assert len(high_risk) > 0

    def test_analyze_compliant_feature(
        self,
        processor: EmploymentCategoryProcessor,
        sample_compliant_feature: ProductFeatureInput,
    ):
        """Test analysis of a compliant feature."""
        analysis = processor.analyze_feature(sample_compliant_feature)

        # Risk level should be low
        assert analysis["risk_level"] in ["low", "medium"]

        # Should have no critical findings
        critical_findings = [
            f for f in analysis["findings"]
            if f.get("severity") == "critical"
        ]
        assert len(critical_findings) == 0

    def test_generate_recommendations(
        self,
        processor: EmploymentCategoryProcessor,
        sample_hiring_feature_with_protected_class: ProductFeatureInput,
    ):
        """Test recommendation generation."""
        analysis = processor.analyze_feature(sample_hiring_feature_with_protected_class)

        # Should have recommendations
        assert len(analysis["recommendations"]) > 0

        # Recommendations should be prioritized
        priorities = [r["priority"] for r in analysis["recommendations"]]
        assert priorities == sorted(priorities)  # Should be in order

    def test_check_covered_entity(self, processor: EmploymentCategoryProcessor):
        """Test covered entity checking."""
        # Small employer
        result = processor.check_covered_entity(10)
        assert result["title_vii_covered"] is False
        assert result["ada_covered"] is False

        # Medium employer (15+ employees)
        result = processor.check_covered_entity(20)
        assert result["title_vii_covered"] is True
        assert result["ada_covered"] is True
        assert result["adea_covered"] is True

        # Large employer (50+ employees)
        result = processor.check_covered_entity(100)
        assert result["fmla_covered"] is True

        # Unknown size
        result = processor.check_covered_entity(None)
        assert result["title_vii_covered"] == "unknown"

    def test_get_protected_classes_for_context(
        self,
        processor: EmploymentCategoryProcessor,
        sample_hiring_feature: ProductFeatureInput,
    ):
        """Test getting protected classes for feature context."""
        classes = processor.get_protected_classes_for_context(sample_hiring_feature)

        assert len(classes) > 0

        # Should include standard Title VII classes
        class_ids = [c["id"] for c in classes]
        assert "race" in class_ids
        assert "sex" in class_ids
        assert "religion" in class_ids
        assert "national_origin" in class_ids
        assert "disability" in class_ids  # ADA


class TestProtectedClassFields:
    """Tests for protected class field detection."""

    def test_race_fields_detected(self):
        """Test that race-related fields are in the set."""
        race_fields = ["race", "ethnicity", "ethnic_background"]
        for field in race_fields:
            assert field in PROTECTED_CLASS_FIELDS

    def test_religion_fields_detected(self):
        """Test that religion-related fields are in the set."""
        religion_fields = ["religion", "religious_affiliation", "faith"]
        for field in religion_fields:
            assert field in PROTECTED_CLASS_FIELDS

    def test_disability_fields_detected(self):
        """Test that disability-related fields are in the set."""
        disability_fields = ["disability", "medical_condition", "physical_limitation"]
        for field in disability_fields:
            assert field in PROTECTED_CLASS_FIELDS


class TestProxyIndicators:
    """Tests for proxy variable indicators."""

    def test_geographic_proxies(self):
        """Test geographic proxy detection."""
        assert "zip_code" in PROXY_INDICATORS
        assert "neighborhood" in PROXY_INDICATORS

        # Should indicate race as potential proxy
        assert "race" in PROXY_INDICATORS["zip_code"]

    def test_name_proxies(self):
        """Test name-based proxy detection."""
        assert "first_name" in PROXY_INDICATORS
        assert "last_name" in PROXY_INDICATORS

        # Should indicate multiple protected classes
        assert len(PROXY_INDICATORS["first_name"]) > 1


class TestSingletonPattern:
    """Tests for singleton pattern."""

    def test_get_employment_processor_singleton(self):
        """Test that singleton works."""
        processor1 = get_employment_processor()
        processor2 = get_employment_processor()

        assert processor1 is processor2


class TestRiskLevelCalculation:
    """Tests for risk level calculation."""

    @pytest.fixture
    def processor(self) -> EmploymentCategoryProcessor:
        """Create a processor instance."""
        return EmploymentCategoryProcessor()

    def test_critical_risk_level(self, processor: EmploymentCategoryProcessor):
        """Test critical risk level assignment."""
        analysis = {
            "findings": [
                {"type": "test", "severity": "critical"},
                {"type": "test", "severity": "medium"},
            ]
        }

        level = processor._calculate_risk_level(analysis)
        assert level == "critical"

    def test_high_risk_level(self, processor: EmploymentCategoryProcessor):
        """Test high risk level assignment."""
        analysis = {
            "findings": [
                {"type": "test", "severity": "high"},
                {"type": "test", "severity": "medium"},
            ]
        }

        level = processor._calculate_risk_level(analysis)
        assert level == "high"

    def test_low_risk_level(self, processor: EmploymentCategoryProcessor):
        """Test low risk level assignment."""
        analysis = {
            "findings": []
        }

        level = processor._calculate_risk_level(analysis)
        assert level == "low"
