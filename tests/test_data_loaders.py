"""
Tests for data loaders (laws, rules, precedents).

Tests the YAML loading and parsing functionality.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from wenah.data.loaders import (
    LawLoader,
    RuleLoader,
    PrecedentLoader,
    load_employment_laws,
    load_all_laws,
    load_all_rules,
)
from wenah.core.types import LawCategory


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_laws_dir():
    """Create a temporary directory with sample law files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        laws_dir = Path(tmpdir)

        # Create employment subdirectory
        employment_dir = laws_dir / "employment"
        employment_dir.mkdir()

        # Create a sample law file
        sample_law = {
            "law": {
                "id": "test-law-001",
                "name": "Test Employment Law",
                "citation": "Test Citation",
                "category": "employment",
                "effective_date": "1964-07-02",
                "protected_classes": [
                    {"id": "race", "name": "Race"},
                    {"id": "gender", "name": "Gender"},
                ],
                "covered_entities": [
                    {"id": "employers", "name": "Employers"},
                ],
                "prohibited_practices": [
                    {
                        "id": "pp-001",
                        "name": "Discrimination",
                        "severity": "critical",
                    },
                ],
                "ai_ml_considerations": [
                    {
                        "id": "ai-001",
                        "concern": "Algorithmic bias",
                    },
                ],
                "remedies": ["Compensatory damages", "Injunctive relief"],
            }
        }

        with open(employment_dir / "test_law.yaml", "w") as f:
            yaml.dump(sample_law, f)

        yield laws_dir


@pytest.fixture
def temp_rules_dir():
    """Create a temporary directory with sample rule files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rules_dir = Path(tmpdir)

        # Create decision_trees subdirectory
        trees_dir = rules_dir / "decision_trees"
        trees_dir.mkdir()

        # Create a sample rule file
        sample_rules = {
            "rule_tree": {
                "id": "test-tree-001",
                "name": "Test Rule Tree",
                "category": "employment",
                "version": "1.0.0",
                "rules": [
                    {
                        "id": "rule-001",
                        "name": "Test Rule",
                        "description": "A test rule",
                        "severity": "high",
                        "conditions": {
                            "operator": "AND",
                            "items": [
                                {
                                    "field": "data_fields",
                                    "operator": "contains",
                                    "value": "race",
                                }
                            ],
                        },
                    },
                    {
                        "id": "rule-002",
                        "name": "Another Rule",
                        "severity": "medium",
                    },
                ],
            }
        }

        with open(trees_dir / "test_rules.yaml", "w") as f:
            yaml.dump(sample_rules, f)

        yield rules_dir


@pytest.fixture
def temp_precedents_dir():
    """Create a temporary directory with sample precedent files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        precedents_dir = Path(tmpdir)

        # Create a sample precedent file
        sample_precedents = {
            "precedents": [
                {
                    "id": "prec-001",
                    "case_name": "Test v. Company",
                    "citation": "123 F.3d 456",
                    "category": "employment",
                    "year": 2020,
                    "summary": "Important test case",
                    "holding": "Discrimination found",
                },
                {
                    "id": "prec-002",
                    "case_name": "Another v. Corp",
                    "citation": "789 F.3d 012",
                    "category": "housing",
                    "year": 2021,
                },
            ]
        }

        with open(precedents_dir / "test_precedents.yaml", "w") as f:
            yaml.dump(sample_precedents, f)

        yield precedents_dir


# =============================================================================
# LawLoader Tests
# =============================================================================


class TestLawLoader:
    """Tests for the LawLoader class."""

    def test_init_with_directory(self, temp_laws_dir: Path):
        """Test initialization with custom directory."""
        loader = LawLoader(temp_laws_dir)
        assert loader.laws_directory == temp_laws_dir

    def test_init_with_default(self):
        """Test initialization with default directory."""
        loader = LawLoader()
        assert loader.laws_directory is not None

    def test_load_all_laws(self, temp_laws_dir: Path):
        """Test loading all laws."""
        loader = LawLoader(temp_laws_dir)
        laws = loader.load_all_laws()

        assert "test-law-001" in laws
        assert laws["test-law-001"]["law"]["name"] == "Test Employment Law"

    def test_load_category(self, temp_laws_dir: Path):
        """Test loading laws by category."""
        loader = LawLoader(temp_laws_dir)
        laws = loader.load_category("employment")

        assert "test-law-001" in laws

    def test_load_nonexistent_category(self, temp_laws_dir: Path):
        """Test loading non-existent category returns empty dict."""
        loader = LawLoader(temp_laws_dir)
        laws = loader.load_category("nonexistent")

        assert laws == {}

    def test_load_law_file(self, temp_laws_dir: Path):
        """Test loading a single law file."""
        loader = LawLoader(temp_laws_dir)
        law_file = temp_laws_dir / "employment" / "test_law.yaml"
        law_data = loader.load_law_file(law_file)

        assert law_data is not None
        assert "law" in law_data
        assert law_data["law"]["id"] == "test-law-001"

    def test_load_law_file_not_found(self, temp_laws_dir: Path):
        """Test loading non-existent file returns None."""
        loader = LawLoader(temp_laws_dir)
        result = loader.load_law_file(temp_laws_dir / "nonexistent.yaml")

        assert result is None

    def test_load_law_by_id(self, temp_laws_dir: Path):
        """Test loading law by ID."""
        loader = LawLoader(temp_laws_dir)
        law = loader.load_law_by_id("test-law-001")

        assert law is not None
        assert law["law"]["name"] == "Test Employment Law"

    def test_load_law_by_id_not_found(self, temp_laws_dir: Path):
        """Test loading non-existent law ID returns None."""
        loader = LawLoader(temp_laws_dir)
        law = loader.load_law_by_id("nonexistent-law")

        assert law is None

    def test_get_protected_classes(self, temp_laws_dir: Path):
        """Test getting protected classes for a law."""
        loader = LawLoader(temp_laws_dir)
        classes = loader.get_protected_classes("test-law-001")

        assert len(classes) == 2
        class_ids = [c["id"] for c in classes]
        assert "race" in class_ids
        assert "gender" in class_ids

    def test_get_protected_classes_not_found(self, temp_laws_dir: Path):
        """Test getting protected classes for non-existent law."""
        loader = LawLoader(temp_laws_dir)
        classes = loader.get_protected_classes("nonexistent")

        assert classes == []

    def test_get_prohibited_practices(self, temp_laws_dir: Path):
        """Test getting prohibited practices for a law."""
        loader = LawLoader(temp_laws_dir)
        practices = loader.get_prohibited_practices("test-law-001")

        assert len(practices) == 1
        assert practices[0]["id"] == "pp-001"
        assert practices[0]["severity"] == "critical"

    def test_get_ai_considerations(self, temp_laws_dir: Path):
        """Test getting AI considerations for a law."""
        loader = LawLoader(temp_laws_dir)
        considerations = loader.get_ai_considerations("test-law-001")

        assert len(considerations) == 1
        assert "algorithmic bias" in considerations[0]["concern"].lower()

    def test_to_pydantic(self, temp_laws_dir: Path):
        """Test converting law data to Pydantic model."""
        loader = LawLoader(temp_laws_dir)
        law_data = loader.load_law_by_id("test-law-001")
        law_doc = loader.to_pydantic(law_data)

        assert law_doc is not None
        assert law_doc.id == "test-law-001"
        assert law_doc.name == "Test Employment Law"
        assert law_doc.category == LawCategory.EMPLOYMENT
        assert "race" in law_doc.protected_classes

    def test_to_pydantic_invalid_data(self, temp_laws_dir: Path):
        """Test converting invalid data returns None."""
        loader = LawLoader(temp_laws_dir)

        assert loader.to_pydantic(None) is None
        assert loader.to_pydantic({}) is None
        assert loader.to_pydantic({"other": "data"}) is None


class TestLawLoaderYAMLErrors:
    """Tests for YAML error handling."""

    def test_load_invalid_yaml(self, temp_laws_dir: Path):
        """Test loading invalid YAML file."""
        # Create an invalid YAML file
        invalid_file = temp_laws_dir / "employment" / "invalid.yaml"
        with open(invalid_file, "w") as f:
            f.write("invalid: yaml: content: [")

        loader = LawLoader(temp_laws_dir)
        result = loader.load_law_file(invalid_file)

        assert result is None


# =============================================================================
# RuleLoader Tests
# =============================================================================


class TestRuleLoader:
    """Tests for the RuleLoader class."""

    def test_init_with_directory(self, temp_rules_dir: Path):
        """Test initialization with custom directory."""
        loader = RuleLoader(temp_rules_dir)
        assert loader.rules_directory == temp_rules_dir

    def test_load_all_rules(self, temp_rules_dir: Path):
        """Test loading all rules."""
        loader = RuleLoader(temp_rules_dir)
        rules = loader.load_all_rules()

        assert "test-tree-001" in rules
        assert rules["test-tree-001"]["rule_tree"]["name"] == "Test Rule Tree"

    def test_load_rule_file(self, temp_rules_dir: Path):
        """Test loading a single rule file."""
        loader = RuleLoader(temp_rules_dir)
        rule_file = temp_rules_dir / "decision_trees" / "test_rules.yaml"
        rule_data = loader.load_rule_file(rule_file)

        assert rule_data is not None
        assert "rule_tree" in rule_data

    def test_load_rules_by_category(self, temp_rules_dir: Path):
        """Test loading rules by category."""
        loader = RuleLoader(temp_rules_dir)
        rules = loader.load_rules_by_category("employment")

        assert len(rules) == 2
        rule_ids = [r["id"] for r in rules]
        assert "rule-001" in rule_ids
        assert "rule-002" in rule_ids

    def test_load_rules_by_nonexistent_category(self, temp_rules_dir: Path):
        """Test loading rules for non-existent category."""
        loader = RuleLoader(temp_rules_dir)
        rules = loader.load_rules_by_category("nonexistent")

        assert rules == []

    def test_get_rule_by_id(self, temp_rules_dir: Path):
        """Test getting rule by ID."""
        loader = RuleLoader(temp_rules_dir)
        rule = loader.get_rule_by_id("rule-001")

        assert rule is not None
        assert rule["name"] == "Test Rule"
        assert rule["severity"] == "high"

    def test_get_rule_by_id_not_found(self, temp_rules_dir: Path):
        """Test getting non-existent rule."""
        loader = RuleLoader(temp_rules_dir)
        rule = loader.get_rule_by_id("nonexistent-rule")

        assert rule is None

    def test_load_rules_no_decision_trees_dir(self):
        """Test loading rules when decision_trees dir doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = RuleLoader(tmpdir)
            rules = loader.load_all_rules()

            assert rules == {}


# =============================================================================
# PrecedentLoader Tests
# =============================================================================


class TestPrecedentLoader:
    """Tests for the PrecedentLoader class."""

    def test_init_with_directory(self, temp_precedents_dir: Path):
        """Test initialization with custom directory."""
        loader = PrecedentLoader(temp_precedents_dir)
        assert loader.precedents_directory == temp_precedents_dir

    def test_load_all_precedents(self, temp_precedents_dir: Path):
        """Test loading all precedents."""
        loader = PrecedentLoader(temp_precedents_dir)
        precedents = loader.load_all_precedents()

        assert len(precedents) == 2
        ids = [p["id"] for p in precedents]
        assert "prec-001" in ids
        assert "prec-002" in ids

    def test_load_precedent_file(self, temp_precedents_dir: Path):
        """Test loading a single precedent file."""
        loader = PrecedentLoader(temp_precedents_dir)
        prec_file = temp_precedents_dir / "test_precedents.yaml"
        prec_data = loader.load_precedent_file(prec_file)

        assert prec_data is not None
        assert "precedents" in prec_data

    def test_load_precedents_by_category(self, temp_precedents_dir: Path):
        """Test loading precedents by category."""
        loader = PrecedentLoader(temp_precedents_dir)

        employment_precs = loader.load_precedents_by_category("employment")
        assert len(employment_precs) == 1
        assert employment_precs[0]["id"] == "prec-001"

        housing_precs = loader.load_precedents_by_category("housing")
        assert len(housing_precs) == 1
        assert housing_precs[0]["id"] == "prec-002"

    def test_load_precedents_nonexistent_category(self, temp_precedents_dir: Path):
        """Test loading precedents for non-existent category."""
        loader = PrecedentLoader(temp_precedents_dir)
        precs = loader.load_precedents_by_category("nonexistent")

        assert precs == []

    def test_load_precedents_nonexistent_dir(self):
        """Test loading precedents from non-existent directory."""
        loader = PrecedentLoader("/nonexistent/path")
        precs = loader.load_all_precedents()

        assert precs == []


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience loading functions."""

    @patch("wenah.data.loaders.LawLoader")
    def test_load_employment_laws(self, mock_loader_class):
        """Test load_employment_laws convenience function."""
        mock_loader = MagicMock()
        mock_loader.load_category.return_value = {"law-1": {"law": {"id": "law-1"}}}
        mock_loader_class.return_value = mock_loader

        result = load_employment_laws()

        mock_loader.load_category.assert_called_once_with("employment")
        assert "law-1" in result

    @patch("wenah.data.loaders.LawLoader")
    def test_load_all_laws_func(self, mock_loader_class):
        """Test load_all_laws convenience function."""
        mock_loader = MagicMock()
        mock_loader.load_all_laws.return_value = {"law-1": {}, "law-2": {}}
        mock_loader_class.return_value = mock_loader

        result = load_all_laws()

        mock_loader.load_all_laws.assert_called_once()
        assert len(result) == 2

    @patch("wenah.data.loaders.RuleLoader")
    def test_load_all_rules_func(self, mock_loader_class):
        """Test load_all_rules convenience function."""
        mock_loader = MagicMock()
        mock_loader.load_all_rules.return_value = {"tree-1": {}}
        mock_loader_class.return_value = mock_loader

        result = load_all_rules()

        mock_loader.load_all_rules.assert_called_once()
        assert "tree-1" in result


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_law_file(self, temp_laws_dir: Path):
        """Test handling empty law file."""
        empty_file = temp_laws_dir / "employment" / "empty.yaml"
        with open(empty_file, "w") as f:
            f.write("")

        loader = LawLoader(temp_laws_dir)
        result = loader.load_law_file(empty_file)

        assert result is None

    def test_law_without_id(self, temp_laws_dir: Path):
        """Test handling law without ID."""
        no_id_law = {"law": {"name": "No ID Law"}}
        no_id_file = temp_laws_dir / "employment" / "no_id.yaml"
        with open(no_id_file, "w") as f:
            yaml.dump(no_id_law, f)

        loader = LawLoader(temp_laws_dir)
        laws = loader.load_category("employment")

        # Should have the test law but not the no-id law
        assert "test-law-001" in laws

    def test_rule_tree_without_rules(self, temp_rules_dir: Path):
        """Test handling rule tree without rules list."""
        no_rules = {
            "rule_tree": {
                "id": "no-rules-tree",
                "name": "Tree Without Rules",
                "category": "employment",
            }
        }
        no_rules_file = temp_rules_dir / "decision_trees" / "no_rules.yaml"
        with open(no_rules_file, "w") as f:
            yaml.dump(no_rules, f)

        loader = RuleLoader(temp_rules_dir)
        rules = loader.load_rules_by_category("employment")

        # Should still get rules from the original file
        assert len(rules) >= 2

    def test_precedent_file_without_precedents_key(self, temp_precedents_dir: Path):
        """Test handling precedent file without precedents key."""
        no_key = {"other_data": "value"}
        no_key_file = temp_precedents_dir / "no_key.yaml"
        with open(no_key_file, "w") as f:
            yaml.dump(no_key, f)

        loader = PrecedentLoader(temp_precedents_dir)
        precs = loader.load_all_precedents()

        # Should still have precedents from original file
        assert len(precs) >= 2
