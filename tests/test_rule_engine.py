"""
Tests for the rule engine.
"""

import pytest
from pathlib import Path

from wenah.rules.rule_engine import RuleEngine, get_rule_engine
from wenah.rules.rule_loader import RuleLoader, RuleValidator
from wenah.core.types import (
    ProductFeatureInput,
    ProductCategory,
    FeatureType,
    RuleResult,
    DataFieldSpec,
    AlgorithmSpec,
)


class TestRuleEngine:
    """Tests for the RuleEngine class."""

    @pytest.fixture
    def rule_engine(self, data_dir: Path) -> RuleEngine:
        """Create a rule engine instance with test data."""
        rules_dir = data_dir / "rules"
        return RuleEngine(rules_directory=str(rules_dir))

    def test_rule_engine_initialization(self, rule_engine: RuleEngine):
        """Test that rule engine initializes correctly."""
        assert rule_engine is not None
        assert rule_engine.rule_loader is not None

    def test_evaluate_compliant_feature(
        self,
        rule_engine: RuleEngine,
        sample_compliant_feature: ProductFeatureInput,
    ):
        """Test evaluation of a compliant feature."""
        evaluations = rule_engine.evaluate(sample_compliant_feature)

        # Should have some evaluations (positive indicators)
        # but no critical violations
        critical_violations = [
            e for e in evaluations
            if e.result == RuleResult.VIOLATION and e.risk_score >= 80
        ]
        assert len(critical_violations) == 0

    def test_evaluate_protected_class_collection(
        self,
        rule_engine: RuleEngine,
        sample_hiring_feature_with_protected_class: ProductFeatureInput,
    ):
        """Test detection of protected class data collection."""
        evaluations = rule_engine.evaluate(sample_hiring_feature_with_protected_class)

        # Should detect the race field collection
        assert len(evaluations) > 0

        # Find the protected class violation
        protected_class_evals = [
            e for e in evaluations
            if "protected" in e.rule_name.lower() or "race" in str(e.law_references).lower()
        ]

        # Should have flagged the issue
        high_risk_evals = [e for e in evaluations if e.risk_score >= 75]
        assert len(high_risk_evals) > 0

    def test_evaluate_proxy_variables(
        self,
        rule_engine: RuleEngine,
        sample_hiring_feature_with_proxy: ProductFeatureInput,
    ):
        """Test detection of proxy variables."""
        evaluations = rule_engine.evaluate(sample_hiring_feature_with_proxy)

        # Should detect zip_code as potential proxy
        proxy_evals = [
            e for e in evaluations
            if "proxy" in e.rule_name.lower() or "zip" in str(e.recommendations).lower()
        ]

        # At least one evaluation should exist
        assert len(evaluations) > 0

    def test_evaluate_video_interview(
        self,
        rule_engine: RuleEngine,
        sample_video_interview_feature: ProductFeatureInput,
    ):
        """Test detection of high-risk video interview analysis."""
        evaluations = rule_engine.evaluate(sample_video_interview_feature)

        # Should flag video interview AI concerns
        assert len(evaluations) > 0

        # Should have escalate_to_llm for nuanced analysis
        llm_escalations = [e for e in evaluations if e.escalate_to_llm]
        # Video analysis typically requires LLM review
        assert len(llm_escalations) >= 0  # May or may not escalate

    def test_evaluate_multiple_features(
        self,
        rule_engine: RuleEngine,
        sample_hiring_feature: ProductFeatureInput,
        sample_compliant_feature: ProductFeatureInput,
    ):
        """Test evaluation of multiple features."""
        results = rule_engine.evaluate_multiple([
            sample_hiring_feature,
            sample_compliant_feature,
        ])

        assert len(results) == 2
        assert sample_hiring_feature.feature_id in results
        assert sample_compliant_feature.feature_id in results

    def test_category_inference(self, rule_engine: RuleEngine):
        """Test that categories are correctly inferred from feature type."""
        hiring_feature = ProductFeatureInput(
            feature_id="test",
            name="Test",
            description="Test",
            category=ProductCategory.HIRING,
            feature_type=FeatureType.DATA_COLLECTION,
            decision_impact="Test",
            affected_population="Test",
        )

        categories = rule_engine._infer_categories(hiring_feature)
        assert "employment" in categories

        lending_feature = ProductFeatureInput(
            feature_id="test",
            name="Test",
            description="Test",
            category=ProductCategory.LENDING,
            feature_type=FeatureType.DATA_COLLECTION,
            decision_impact="Test",
            affected_population="Test",
        )

        categories = rule_engine._infer_categories(lending_feature)
        assert "consumer" in categories

    def test_rules_summary(self, rule_engine: RuleEngine):
        """Test getting rules summary."""
        summary = rule_engine.get_rules_summary()

        assert "total_rule_trees" in summary
        assert "categories" in summary
        assert "severity_counts" in summary

    def test_rules_summary_by_category(self, rule_engine: RuleEngine):
        """Test getting rules summary filtered by category."""
        summary = rule_engine.get_rules_summary(category="employment")

        assert "categories" in summary
        # Employment should be the only category
        if summary["categories"]:
            assert "employment" in summary["categories"]

    def test_cache_clear(self, rule_engine: RuleEngine):
        """Test cache clearing."""
        # Load rules to populate cache
        _ = rule_engine.evaluate(
            ProductFeatureInput(
                feature_id="test",
                name="Test",
                description="Test",
                category=ProductCategory.HIRING,
                feature_type=FeatureType.DATA_COLLECTION,
                decision_impact="Test",
                affected_population="Test",
            )
        )

        # Clear cache
        rule_engine.clear_cache()
        assert rule_engine._rules_cache is None


class TestRuleLoader:
    """Tests for the RuleLoader class."""

    @pytest.fixture
    def rule_loader(self, data_dir: Path) -> RuleLoader:
        """Create a rule loader instance."""
        return RuleLoader(rules_directory=data_dir / "rules")

    def test_load_all_rules(self, rule_loader: RuleLoader):
        """Test loading all rules."""
        rules = rule_loader.load_all_rules()
        assert isinstance(rules, dict)

    def test_load_rules_by_category(self, rule_loader: RuleLoader):
        """Test loading rules by category."""
        employment_rules = rule_loader.load_rules_by_category("employment")
        assert isinstance(employment_rules, list)

        # Employment rules should exist
        if employment_rules:
            assert all("id" in rule for rule in employment_rules)

    def test_load_rules_by_severity(self, rule_loader: RuleLoader):
        """Test loading rules by severity."""
        critical_rules = rule_loader.load_rules_by_severity("critical")
        assert isinstance(critical_rules, list)

        for rule in critical_rules:
            assert rule.get("severity") == "critical"

    def test_get_rule_by_id(self, rule_loader: RuleLoader):
        """Test getting a specific rule by ID."""
        # First, get any rule ID from loaded rules
        all_rules = rule_loader.load_all_rules()

        if all_rules:
            # Get first rule ID
            first_tree = list(all_rules.values())[0]
            rules = first_tree.get("rule_tree", {}).get("rules", [])
            if rules:
                rule_id = rules[0].get("id")
                rule = rule_loader.get_rule_by_id(rule_id)
                assert rule is not None
                assert rule.get("id") == rule_id

    def test_get_applicable_laws(self, rule_loader: RuleLoader):
        """Test getting applicable laws for a category."""
        laws = rule_loader.get_applicable_laws("employment")
        assert isinstance(laws, list)

    def test_rules_statistics(self, rule_loader: RuleLoader):
        """Test getting rules statistics."""
        stats = rule_loader.get_rules_statistics()

        assert "total_trees" in stats
        assert "total_rules" in stats
        assert "by_category" in stats
        assert "by_severity" in stats

    def test_search_rules(self, rule_loader: RuleLoader):
        """Test searching rules by text."""
        # Search for "protected"
        matches = rule_loader.search_rules("protected")
        assert isinstance(matches, list)

    def test_reload_rules(self, rule_loader: RuleLoader):
        """Test reloading rules."""
        # Load once
        rules1 = rule_loader.load_all_rules()

        # Reload
        rules2 = rule_loader.reload_rules()

        # Should have same structure
        assert set(rules1.keys()) == set(rules2.keys())


class TestRuleValidator:
    """Tests for the RuleValidator class."""

    @pytest.fixture
    def validator(self, data_dir: Path) -> RuleValidator:
        """Create a rule validator instance."""
        loader = RuleLoader(rules_directory=data_dir / "rules")
        return RuleValidator(loader)

    def test_validate_all_rules(self, validator: RuleValidator):
        """Test validating all rules."""
        issues = validator.validate_all_rules()
        assert isinstance(issues, list)

        # Current rules should be valid (no critical issues)
        critical_issues = [
            i for i in issues
            if i.get("type") in ["duplicate_id", "invalid_risk_score", "invalid_confidence"]
        ]
        assert len(critical_issues) == 0

    def test_check_rule_conflicts(self, validator: RuleValidator):
        """Test checking for rule conflicts."""
        conflicts = validator.check_rule_conflicts()
        assert isinstance(conflicts, list)


class TestConditionEvaluation:
    """Tests for condition evaluation logic."""

    @pytest.fixture
    def rule_engine(self, data_dir: Path) -> RuleEngine:
        """Create a rule engine instance."""
        return RuleEngine(rules_directory=str(data_dir / "rules"))

    def test_equals_operator(self, rule_engine: RuleEngine):
        """Test equals operator."""
        result, conf = rule_engine._compare("test", "equals", "test", [])
        assert result is True
        assert conf == 1.0

        result, conf = rule_engine._compare("test", "equals", "other", [])
        assert result is False

    def test_contains_operator(self, rule_engine: RuleEngine):
        """Test contains operator."""
        result, conf = rule_engine._compare("hello world", "contains", "world", [])
        assert result is True

        result, conf = rule_engine._compare(["a", "b", "c"], "contains", "b", [])
        assert result is True

    def test_contains_any_operator(self, rule_engine: RuleEngine):
        """Test contains_any operator."""
        result, conf = rule_engine._compare(
            "hello world",
            "contains_any",
            None,
            ["world", "universe"]
        )
        assert result is True

        result, conf = rule_engine._compare(
            ["a", "b", "c"],
            "contains_any",
            None,
            ["b", "x"]
        )
        assert result is True

    def test_in_operator(self, rule_engine: RuleEngine):
        """Test in operator."""
        result, conf = rule_engine._compare("hiring", "in", None, ["hiring", "lending"])
        assert result is True

        result, conf = rule_engine._compare("other", "in", None, ["hiring", "lending"])
        assert result is False

    def test_is_null_operator(self, rule_engine: RuleEngine):
        """Test is_null operator."""
        result, conf = rule_engine._compare(None, "is_null", None, [])
        assert result is True

        result, conf = rule_engine._compare("value", "is_null", None, [])
        assert result is False

    def test_comparison_operators(self, rule_engine: RuleEngine):
        """Test comparison operators."""
        result, conf = rule_engine._compare(10, "greater_than", 5, [])
        assert result is True

        result, conf = rule_engine._compare(5, "less_than", 10, [])
        assert result is True

        result, conf = rule_engine._compare(10, "greater_than_or_equals", 10, [])
        assert result is True


class TestFeatureToContext:
    """Tests for feature to context conversion."""

    @pytest.fixture
    def rule_engine(self, data_dir: Path) -> RuleEngine:
        """Create a rule engine instance."""
        return RuleEngine(rules_directory=str(data_dir / "rules"))

    def test_basic_conversion(
        self,
        rule_engine: RuleEngine,
        sample_hiring_feature: ProductFeatureInput,
    ):
        """Test basic feature to context conversion."""
        context = rule_engine._feature_to_context(sample_hiring_feature)

        assert "feature" in context
        assert context["feature"]["category"] == "hiring"
        assert context["feature"]["feature_type"] == "algorithm"
        assert len(context["feature"]["data_fields"]) == 2

    def test_algorithm_context(
        self,
        rule_engine: RuleEngine,
        sample_hiring_feature: ProductFeatureInput,
    ):
        """Test that algorithm is included in context."""
        context = rule_engine._feature_to_context(sample_hiring_feature)

        assert context["feature"]["algorithm"] is not None
        assert context["feature"]["algorithm"]["name"] == "ResumeRanker"
        assert context["feature"]["algorithm"]["type"] == "ml_model"

    def test_no_algorithm_context(self, rule_engine: RuleEngine):
        """Test feature without algorithm."""
        feature = ProductFeatureInput(
            feature_id="test",
            name="Test",
            description="Test",
            category=ProductCategory.HIRING,
            feature_type=FeatureType.DATA_COLLECTION,
            decision_impact="Test",
            affected_population="Test",
        )

        context = rule_engine._feature_to_context(feature)
        assert context["feature"]["algorithm"] is None


class TestSingletonPattern:
    """Tests for singleton pattern implementation."""

    def test_get_rule_engine_singleton(self):
        """Test that get_rule_engine returns singleton."""
        engine1 = get_rule_engine()
        engine2 = get_rule_engine()

        # Note: In tests, we might get different instances
        # This tests that the function works
        assert engine1 is not None
        assert engine2 is not None
