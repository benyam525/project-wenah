"""
Tests for category processors (housing and consumer).

Tests the stub implementations for non-MVP categories.
"""

from __future__ import annotations

import pytest

from wenah.core.types import (
    DataFieldSpec,
    FeatureType,
    ProductCategory,
    ProductFeatureInput,
)
from wenah.rules.categories.consumer import (
    CONSUMER_PROXY_INDICATORS,
    ECOA_PROTECTED_CLASSES,
    ConsumerCategoryProcessor,
    get_consumer_processor,
)
from wenah.rules.categories.housing import (
    FHA_PROTECTED_CLASSES,
    HOUSING_PROXY_INDICATORS,
    HousingCategoryProcessor,
    get_housing_processor,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def housing_feature() -> ProductFeatureInput:
    """Create a sample housing feature."""
    return ProductFeatureInput(
        feature_id="housing-001",
        name="Tenant Screening",
        description="Automated tenant screening system",
        category=ProductCategory.HOUSING,
        feature_type=FeatureType.ALGORITHM,
        data_fields=[
            DataFieldSpec(
                name="credit_score",
                description="Applicant credit score",
                data_type="numeric",
                source="credit_bureau",
                used_in_decisions=True,
            ),
            DataFieldSpec(
                name="income",
                description="Monthly income",
                data_type="numeric",
                source="application",
                used_in_decisions=True,
            ),
        ],
        decision_impact="Determines tenant approval",
        affected_population="Rental applicants",
    )


@pytest.fixture
def consumer_feature() -> ProductFeatureInput:
    """Create a sample consumer/lending feature."""
    return ProductFeatureInput(
        feature_id="consumer-001",
        name="Credit Decision",
        description="Automated credit approval system",
        category=ProductCategory.LENDING,
        feature_type=FeatureType.SCORING_MODEL,
        data_fields=[
            DataFieldSpec(
                name="credit_history",
                description="Credit history length",
                data_type="numeric",
                source="credit_bureau",
                used_in_decisions=True,
            ),
            DataFieldSpec(
                name="debt_to_income",
                description="Debt to income ratio",
                data_type="numeric",
                source="calculated",
                used_in_decisions=True,
            ),
        ],
        decision_impact="Determines credit approval",
        affected_population="Credit applicants",
    )


# =============================================================================
# Housing Processor Tests
# =============================================================================


class TestHousingCategoryProcessor:
    """Tests for the housing category processor."""

    def test_init(self):
        """Test housing processor initialization."""
        processor = HousingCategoryProcessor()
        assert processor.protected_classes == FHA_PROTECTED_CLASSES
        assert processor.proxy_indicators == HOUSING_PROXY_INDICATORS

    def test_analyze_feature(self, housing_feature: ProductFeatureInput):
        """Test housing feature analysis."""
        processor = HousingCategoryProcessor()
        result = processor.analyze_feature(housing_feature)

        assert result["feature_id"] == "housing-001"
        assert result["category"] == "housing"
        assert result["status"] == "stub_implementation"
        assert "Fair Housing Act" in result["applicable_law"]
        assert isinstance(result["protected_classes"], list)
        assert len(result["protected_classes"]) > 0

    def test_get_protected_classes(self):
        """Test getting FHA protected classes."""
        processor = HousingCategoryProcessor()
        classes = processor.get_protected_classes()

        assert len(classes) == 7
        class_ids = [c["id"] for c in classes]
        assert "race" in class_ids
        assert "color" in class_ids
        assert "religion" in class_ids
        assert "national_origin" in class_ids
        assert "sex" in class_ids
        assert "familial_status" in class_ids
        assert "disability" in class_ids

    def test_protected_classes_have_law_reference(self):
        """Test that protected classes reference FHA."""
        processor = HousingCategoryProcessor()
        classes = processor.get_protected_classes()

        for pc in classes:
            assert pc["law"] == "Fair Housing Act"
            assert "name" in pc

    def test_familial_status_note(self):
        """Test familial status has explanatory note."""
        processor = HousingCategoryProcessor()
        classes = processor.get_protected_classes()

        familial = next(c for c in classes if c["id"] == "familial_status")
        assert "note" in familial
        assert "children" in familial["note"].lower()

    def test_disability_note(self):
        """Test disability has accommodation note."""
        processor = HousingCategoryProcessor()
        classes = processor.get_protected_classes()

        disability = next(c for c in classes if c["id"] == "disability")
        assert "note" in disability
        assert "accommodation" in disability["note"].lower()


class TestHousingProxyIndicators:
    """Tests for housing proxy indicators."""

    def test_zip_code_proxy(self):
        """Test zip code is a known proxy."""
        assert "zip_code" in HOUSING_PROXY_INDICATORS
        proxies = HOUSING_PROXY_INDICATORS["zip_code"]
        assert "race" in proxies
        assert "national_origin" in proxies

    def test_school_district_proxy(self):
        """Test school district is a known proxy."""
        assert "school_district" in HOUSING_PROXY_INDICATORS
        proxies = HOUSING_PROXY_INDICATORS["school_district"]
        assert "race" in proxies
        assert "familial_status" in proxies

    def test_crime_statistics_proxy(self):
        """Test crime statistics is a known proxy."""
        assert "crime_statistics" in HOUSING_PROXY_INDICATORS
        assert "race" in HOUSING_PROXY_INDICATORS["crime_statistics"]


class TestHousingProcessorSingleton:
    """Tests for housing processor singleton."""

    def test_get_housing_processor(self):
        """Test singleton returns same instance."""
        processor1 = get_housing_processor()
        processor2 = get_housing_processor()
        assert processor1 is processor2

    def test_singleton_is_housing_processor(self):
        """Test singleton is correct type."""
        processor = get_housing_processor()
        assert isinstance(processor, HousingCategoryProcessor)


# =============================================================================
# Consumer Processor Tests
# =============================================================================


class TestConsumerCategoryProcessor:
    """Tests for the consumer category processor."""

    def test_init(self):
        """Test consumer processor initialization."""
        processor = ConsumerCategoryProcessor()
        assert processor.protected_classes == ECOA_PROTECTED_CLASSES
        assert processor.proxy_indicators == CONSUMER_PROXY_INDICATORS

    def test_analyze_feature(self, consumer_feature: ProductFeatureInput):
        """Test consumer feature analysis."""
        processor = ConsumerCategoryProcessor()
        result = processor.analyze_feature(consumer_feature)

        assert result["feature_id"] == "consumer-001"
        assert result["category"] == "consumer"
        assert result["status"] == "stub_implementation"
        assert "applicable_laws" in result
        assert len(result["applicable_laws"]) == 2
        # Check for full law names in applicable_laws
        assert any("Equal Credit Opportunity" in law for law in result["applicable_laws"])
        assert any("Fair Credit Reporting" in law for law in result["applicable_laws"])

    def test_get_protected_classes(self):
        """Test getting ECOA protected classes."""
        processor = ConsumerCategoryProcessor()
        classes = processor.get_protected_classes()

        assert len(classes) == 8
        class_ids = [c["id"] for c in classes]
        assert "race" in class_ids
        assert "marital_status" in class_ids
        assert "age" in class_ids
        assert "public_assistance" in class_ids

    def test_protected_classes_have_law_reference(self):
        """Test that protected classes reference ECOA."""
        processor = ConsumerCategoryProcessor()
        classes = processor.get_protected_classes()

        for pc in classes:
            assert pc["law"] == "ECOA"

    def test_age_has_capacity_note(self):
        """Test age has capacity to contract note."""
        processor = ConsumerCategoryProcessor()
        classes = processor.get_protected_classes()

        age_class = next(c for c in classes if c["id"] == "age")
        assert "note" in age_class
        assert "capacity" in age_class["note"].lower()

    def test_check_fcra_requirements(self, consumer_feature: ProductFeatureInput):
        """Test FCRA requirements check."""
        processor = ConsumerCategoryProcessor()
        result = processor.check_fcra_requirements(consumer_feature)

        assert result["status"] == "stub_implementation"
        assert "requirements" in result
        assert len(result["requirements"]) > 0
        assert any("permissible purpose" in req.lower() for req in result["requirements"])
        assert any("adverse action" in req.lower() for req in result["requirements"])


class TestConsumerProxyIndicators:
    """Tests for consumer proxy indicators."""

    def test_zip_code_proxy(self):
        """Test zip code is a known proxy."""
        assert "zip_code" in CONSUMER_PROXY_INDICATORS
        proxies = CONSUMER_PROXY_INDICATORS["zip_code"]
        assert "race" in proxies
        assert "national_origin" in proxies

    def test_income_source_proxy(self):
        """Test income source is a known proxy."""
        assert "income_source" in CONSUMER_PROXY_INDICATORS
        proxies = CONSUMER_PROXY_INDICATORS["income_source"]
        assert "public_assistance" in proxies
        assert "age" in proxies

    def test_employment_type_proxy(self):
        """Test employment type is a known proxy."""
        assert "employment_type" in CONSUMER_PROXY_INDICATORS
        proxies = CONSUMER_PROXY_INDICATORS["employment_type"]
        assert "race" in proxies
        assert "sex" in proxies


class TestConsumerProcessorSingleton:
    """Tests for consumer processor singleton."""

    def test_get_consumer_processor(self):
        """Test singleton returns same instance."""
        processor1 = get_consumer_processor()
        processor2 = get_consumer_processor()
        assert processor1 is processor2

    def test_singleton_is_consumer_processor(self):
        """Test singleton is correct type."""
        processor = get_consumer_processor()
        assert isinstance(processor, ConsumerCategoryProcessor)


# =============================================================================
# Cross-Category Tests
# =============================================================================


class TestCrossCategoryComparison:
    """Tests comparing housing and consumer processors."""

    def test_different_protected_classes(self):
        """Test that housing and consumer have different protected classes."""
        # Housing has familial_status, consumer doesn't
        assert "familial_status" in FHA_PROTECTED_CLASSES
        assert "familial_status" not in ECOA_PROTECTED_CLASSES

        # Consumer has marital_status and public_assistance, housing doesn't
        assert "marital_status" in ECOA_PROTECTED_CLASSES
        assert "marital_status" not in FHA_PROTECTED_CLASSES

        assert "public_assistance" in ECOA_PROTECTED_CLASSES
        assert "public_assistance" not in FHA_PROTECTED_CLASSES

    def test_common_protected_classes(self):
        """Test protected classes common to both."""
        common = FHA_PROTECTED_CLASSES & ECOA_PROTECTED_CLASSES
        assert "race" in common
        assert "color" in common
        assert "religion" in common
        assert "national_origin" in common
        assert "sex" in common

    def test_both_have_zip_code_proxy(self):
        """Test both categories recognize zip code as proxy."""
        assert "zip_code" in HOUSING_PROXY_INDICATORS
        assert "zip_code" in CONSUMER_PROXY_INDICATORS

    def test_analysis_returns_different_categories(
        self, housing_feature: ProductFeatureInput, consumer_feature: ProductFeatureInput
    ):
        """Test analysis returns correct category for each."""
        housing_processor = HousingCategoryProcessor()
        consumer_processor = ConsumerCategoryProcessor()

        housing_result = housing_processor.analyze_feature(housing_feature)
        consumer_result = consumer_processor.analyze_feature(consumer_feature)

        assert housing_result["category"] == "housing"
        assert consumer_result["category"] == "consumer"
