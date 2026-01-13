"""
Data loaders for law documents and rules.

Handles loading and parsing of YAML files containing civil rights laws,
decision tree rules, and case precedents.
"""

from pathlib import Path
from typing import Any

import yaml

from wenah.config import settings
from wenah.core.types import LawDocument, LawCategory


class LawLoader:
    """
    Loads civil rights law documents from YAML files.

    Parses structured law data and provides access to law documents
    for the compliance framework.
    """

    def __init__(self, laws_directory: str | Path | None = None):
        """
        Initialize the law loader.

        Args:
            laws_directory: Directory containing law YAML files.
                           Defaults to config setting.
        """
        self.laws_directory = Path(laws_directory or settings.laws_directory)

    def load_all_laws(self) -> dict[str, dict[str, Any]]:
        """
        Load all law documents from all categories.

        Returns:
            Dictionary mapping law IDs to law data
        """
        all_laws = {}

        for category_dir in self.laws_directory.iterdir():
            if category_dir.is_dir():
                category_laws = self.load_category(category_dir.name)
                all_laws.update(category_laws)

        return all_laws

    def load_category(self, category: str) -> dict[str, dict[str, Any]]:
        """
        Load all laws in a specific category.

        Args:
            category: Category name (employment, housing, consumer)

        Returns:
            Dictionary mapping law IDs to law data
        """
        category_path = self.laws_directory / category
        if not category_path.exists():
            return {}

        laws = {}
        for yaml_file in category_path.glob("*.yaml"):
            law_data = self.load_law_file(yaml_file)
            if law_data and "law" in law_data:
                law_id = law_data["law"].get("id")
                if law_id:
                    laws[law_id] = law_data

        return laws

    def load_law_file(self, file_path: Path) -> dict[str, Any] | None:
        """
        Load a single law YAML file.

        Args:
            file_path: Path to the YAML file

        Returns:
            Parsed law data or None if error
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except (yaml.YAMLError, OSError) as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def load_law_by_id(self, law_id: str) -> dict[str, Any] | None:
        """
        Load a specific law by its ID.

        Args:
            law_id: The law identifier

        Returns:
            Law data or None if not found
        """
        all_laws = self.load_all_laws()
        return all_laws.get(law_id)

    def get_protected_classes(self, law_id: str) -> list[dict[str, Any]]:
        """
        Get protected classes for a specific law.

        Args:
            law_id: The law identifier

        Returns:
            List of protected class definitions
        """
        law_data = self.load_law_by_id(law_id)
        if law_data and "law" in law_data:
            return law_data["law"].get("protected_classes", [])
        return []

    def get_prohibited_practices(self, law_id: str) -> list[dict[str, Any]]:
        """
        Get prohibited practices for a specific law.

        Args:
            law_id: The law identifier

        Returns:
            List of prohibited practice definitions
        """
        law_data = self.load_law_by_id(law_id)
        if law_data and "law" in law_data:
            return law_data["law"].get("prohibited_practices", [])
        return []

    def get_ai_considerations(self, law_id: str) -> list[dict[str, Any]]:
        """
        Get AI/ML considerations for a specific law.

        Args:
            law_id: The law identifier

        Returns:
            List of AI/ML consideration definitions
        """
        law_data = self.load_law_by_id(law_id)
        if law_data and "law" in law_data:
            return law_data["law"].get("ai_ml_considerations", [])
        return []

    def to_pydantic(self, law_data: dict[str, Any]) -> LawDocument | None:
        """
        Convert raw law data to Pydantic model.

        Args:
            law_data: Raw law data dictionary

        Returns:
            LawDocument model or None if invalid
        """
        if not law_data or "law" not in law_data:
            return None

        law = law_data["law"]
        try:
            return LawDocument(
                id=law.get("id", ""),
                name=law.get("name", ""),
                citation=law.get("citation", ""),
                category=LawCategory(law.get("category", "employment")),
                effective_date=law.get("effective_date"),
                protected_classes=[
                    pc.get("id", "") for pc in law.get("protected_classes", [])
                ],
                covered_entities=[
                    ce.get("id", "") for ce in law.get("covered_entities", [])
                ],
                remedies=law.get("remedies", []),
            )
        except Exception as e:
            print(f"Error converting to Pydantic: {e}")
            return None


class RuleLoader:
    """
    Loads decision tree rules from YAML files.

    Parses rule definitions for the rule engine.
    """

    def __init__(self, rules_directory: str | Path | None = None):
        """
        Initialize the rule loader.

        Args:
            rules_directory: Directory containing rule YAML files.
                            Defaults to config setting.
        """
        self.rules_directory = Path(rules_directory or settings.rules_directory)

    def load_all_rules(self) -> dict[str, dict[str, Any]]:
        """
        Load all rule definitions.

        Returns:
            Dictionary mapping rule tree IDs to rule data
        """
        all_rules = {}

        # Load from decision_trees subdirectory
        trees_dir = self.rules_directory / "decision_trees"
        if trees_dir.exists():
            for yaml_file in trees_dir.glob("*.yaml"):
                rule_data = self.load_rule_file(yaml_file)
                if rule_data and "rule_tree" in rule_data:
                    tree_id = rule_data["rule_tree"].get("id")
                    if tree_id:
                        all_rules[tree_id] = rule_data

        return all_rules

    def load_rule_file(self, file_path: Path) -> dict[str, Any] | None:
        """
        Load a single rule YAML file.

        Args:
            file_path: Path to the YAML file

        Returns:
            Parsed rule data or None if error
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except (yaml.YAMLError, OSError) as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def load_rules_by_category(self, category: str) -> list[dict[str, Any]]:
        """
        Load rules for a specific category.

        Args:
            category: Category name (employment, housing, consumer)

        Returns:
            List of rule definitions for the category
        """
        all_rules = self.load_all_rules()
        category_rules = []

        for rule_data in all_rules.values():
            tree = rule_data.get("rule_tree", {})
            if tree.get("category") == category:
                category_rules.extend(tree.get("rules", []))

        return category_rules

    def get_rule_by_id(self, rule_id: str) -> dict[str, Any] | None:
        """
        Get a specific rule by ID.

        Args:
            rule_id: The rule identifier

        Returns:
            Rule definition or None if not found
        """
        all_rules = self.load_all_rules()

        for rule_data in all_rules.values():
            tree = rule_data.get("rule_tree", {})
            for rule in tree.get("rules", []):
                if rule.get("id") == rule_id:
                    return rule

        return None


class PrecedentLoader:
    """
    Loads case precedents from YAML files.

    Parses case law examples for the RAG pipeline.
    """

    def __init__(self, precedents_directory: str | Path | None = None):
        """
        Initialize the precedent loader.

        Args:
            precedents_directory: Directory containing precedent YAML files.
                                 Defaults to config setting.
        """
        self.precedents_directory = Path(
            precedents_directory or settings.precedents_directory
        )

    def load_all_precedents(self) -> list[dict[str, Any]]:
        """
        Load all precedents from all files.

        Returns:
            List of precedent definitions
        """
        all_precedents = []

        if not self.precedents_directory.exists():
            return []

        for yaml_file in self.precedents_directory.glob("*.yaml"):
            file_data = self.load_precedent_file(yaml_file)
            if file_data:
                precedents = file_data.get("precedents", [])
                all_precedents.extend(precedents)

        return all_precedents

    def load_precedent_file(self, file_path: Path) -> dict[str, Any] | None:
        """
        Load a single precedent YAML file.

        Args:
            file_path: Path to the YAML file

        Returns:
            Parsed precedent data or None if error
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except (yaml.YAMLError, OSError) as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def load_precedents_by_category(self, category: str) -> list[dict[str, Any]]:
        """
        Load precedents for a specific category.

        Args:
            category: Category name

        Returns:
            List of precedents for the category
        """
        all_precedents = self.load_all_precedents()
        return [p for p in all_precedents if p.get("category") == category]


# Convenience functions
def load_employment_laws() -> dict[str, dict[str, Any]]:
    """Load all employment laws."""
    loader = LawLoader()
    return loader.load_category("employment")


def load_all_laws() -> dict[str, dict[str, Any]]:
    """Load all laws from all categories."""
    loader = LawLoader()
    return loader.load_all_laws()


def load_all_rules() -> dict[str, dict[str, Any]]:
    """Load all decision tree rules."""
    loader = RuleLoader()
    return loader.load_all_rules()
