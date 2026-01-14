"""
Rule loader for decision tree rules.

Handles loading, validation, and management of compliance rules from YAML files.
"""

from pathlib import Path
from typing import Any

import yaml

from wenah.config import settings


class RuleLoader:
    """
    Loads and manages decision tree rules from YAML files.

    Provides functionality for loading rules by category, validating
    rule structure, and hot-reloading rules.
    """

    def __init__(self, rules_directory: str | Path | None = None):
        """
        Initialize the rule loader.

        Args:
            rules_directory: Directory containing rule YAML files.
                            Defaults to config setting.
        """
        self.rules_directory = Path(rules_directory or settings.rules_directory)
        self._cache: dict[str, dict[str, Any]] | None = None
        self._last_load_time: float = 0

    def load_all_rules(self, use_cache: bool = True) -> dict[str, dict[str, Any]]:
        """
        Load all rule definitions from all files.

        Args:
            use_cache: Whether to use cached rules if available

        Returns:
            Dictionary mapping rule tree IDs to rule data
        """
        if use_cache and self._cache is not None:
            return self._cache

        all_rules = {}
        trees_dir = self.rules_directory / "decision_trees"

        if trees_dir.exists():
            for yaml_file in trees_dir.glob("*.yaml"):
                rule_data = self._load_yaml_file(yaml_file)
                if rule_data and "rule_tree" in rule_data:
                    tree_id = rule_data["rule_tree"].get("id")
                    if tree_id:
                        # Validate and normalize the rule tree
                        validated = self._validate_rule_tree(rule_data)
                        if validated:
                            all_rules[tree_id] = validated

        self._cache = all_rules
        return all_rules

    def load_rules_by_category(self, category: str) -> list[dict[str, Any]]:
        """
        Load all rules for a specific category.

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
                rules = tree.get("rules", [])
                category_rules.extend(rules)

        return category_rules

    def load_rules_by_severity(
        self,
        severity: str,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Load rules filtered by severity level.

        Args:
            severity: Severity level (critical, high, medium, low, positive)
            category: Optional category filter

        Returns:
            List of matching rules
        """
        if category:
            all_rules = self.load_rules_by_category(category)
        else:
            all_rules = []
            for tree_data in self.load_all_rules().values():
                tree = tree_data.get("rule_tree", {})
                all_rules.extend(tree.get("rules", []))

        return [r for r in all_rules if r.get("severity") == severity]

    def get_rule_by_id(self, rule_id: str) -> dict[str, Any] | None:
        """
        Get a specific rule by its ID.

        Args:
            rule_id: The rule identifier

        Returns:
            Rule definition or None if not found
        """
        all_rules = self.load_all_rules()

        for tree_data in all_rules.values():
            tree = tree_data.get("rule_tree", {})
            for rule in tree.get("rules", []):
                if rule.get("id") == rule_id:
                    return rule

        return None

    def get_applicable_laws(self, category: str) -> list[str]:
        """
        Get list of applicable law IDs for a category.

        Args:
            category: The category name

        Returns:
            List of law IDs that apply to this category
        """
        all_rules = self.load_all_rules()
        laws = set()

        for tree_data in all_rules.values():
            tree = tree_data.get("rule_tree", {})
            if tree.get("category") == category:
                applicable = tree.get("applicable_laws", [])
                laws.update(applicable)

        return list(laws)

    def reload_rules(self) -> dict[str, dict[str, Any]]:
        """
        Force reload of all rules from disk.

        Returns:
            Freshly loaded rules
        """
        self._cache = None
        return self.load_all_rules(use_cache=False)

    def _load_yaml_file(self, file_path: Path) -> dict[str, Any] | None:
        """
        Load a single YAML file.

        Args:
            file_path: Path to the YAML file

        Returns:
            Parsed YAML data or None if error
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except (yaml.YAMLError, OSError) as e:
            print(f"Error loading rule file {file_path}: {e}")
            return None

    def _validate_rule_tree(
        self,
        rule_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Validate and normalize a rule tree structure.

        Args:
            rule_data: Raw rule tree data

        Returns:
            Validated rule tree or None if invalid
        """
        tree = rule_data.get("rule_tree", {})

        # Required fields
        required = ["id", "category", "rules"]
        for field in required:
            if field not in tree:
                print(f"Rule tree missing required field: {field}")
                return None

        # Validate each rule
        validated_rules = []
        for rule in tree.get("rules", []):
            validated_rule = self._validate_rule(rule)
            if validated_rule:
                validated_rules.append(validated_rule)

        tree["rules"] = validated_rules
        return rule_data

    def _validate_rule(self, rule: dict[str, Any]) -> dict[str, Any] | None:
        """
        Validate a single rule structure.

        Args:
            rule: Raw rule dictionary

        Returns:
            Validated rule or None if invalid
        """
        # Required fields
        required = ["id", "name", "conditions", "consequence"]
        for field in required:
            if field not in rule:
                print(f"Rule {rule.get('id', 'unknown')} missing field: {field}")
                return None

        # Set defaults
        rule.setdefault("severity", "medium")
        rule.setdefault("confidence", 1.0)
        rule.setdefault("description", "")

        # Validate consequence
        consequence = rule.get("consequence", {})
        consequence.setdefault("violation", False)
        consequence.setdefault("risk_score", 0)
        consequence.setdefault("law_reference", "")
        consequence.setdefault("recommendation", "")
        consequence.setdefault("escalate_to_llm", False)
        rule["consequence"] = consequence

        # Validate conditions
        conditions = rule.get("conditions", {})
        conditions.setdefault("operator", "AND")
        conditions.setdefault("items", [])
        rule["conditions"] = conditions

        return rule

    def get_rules_statistics(self) -> dict[str, Any]:
        """
        Get statistics about loaded rules.

        Returns:
            Statistics dictionary
        """
        all_rules = self.load_all_rules()

        stats = {
            "total_trees": len(all_rules),
            "total_rules": 0,
            "by_category": {},
            "by_severity": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "positive": 0,
            },
            "escalate_to_llm_count": 0,
        }

        for tree_data in all_rules.values():
            tree = tree_data.get("rule_tree", {})
            category = tree.get("category", "unknown")
            rules = tree.get("rules", [])

            if category not in stats["by_category"]:
                stats["by_category"][category] = 0

            stats["by_category"][category] += len(rules)
            stats["total_rules"] += len(rules)

            for rule in rules:
                severity = rule.get("severity", "medium")
                if severity in stats["by_severity"]:
                    stats["by_severity"][severity] += 1

                if rule.get("consequence", {}).get("escalate_to_llm"):
                    stats["escalate_to_llm_count"] += 1

        return stats

    def export_rules_to_json(self, output_path: str | Path) -> None:
        """
        Export all rules to a JSON file.

        Args:
            output_path: Path for the output JSON file
        """
        import json

        all_rules = self.load_all_rules()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_rules, f, indent=2)

    def search_rules(
        self,
        query: str,
        fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search rules by text query.

        Args:
            query: Search query string
            fields: Fields to search in (default: name, description)

        Returns:
            List of matching rules
        """
        if fields is None:
            fields = ["name", "description"]

        query_lower = query.lower()
        matches = []

        all_rules = self.load_all_rules()
        for tree_data in all_rules.values():
            tree = tree_data.get("rule_tree", {})
            for rule in tree.get("rules", []):
                for field in fields:
                    value = rule.get(field, "")
                    if isinstance(value, str) and query_lower in value.lower():
                        matches.append(rule)
                        break

        return matches


class RuleValidator:
    """
    Validates rule definitions and checks for conflicts.
    """

    def __init__(self, rule_loader: RuleLoader):
        """
        Initialize the validator.

        Args:
            rule_loader: RuleLoader instance to use
        """
        self.rule_loader = rule_loader

    def validate_all_rules(self) -> list[dict[str, Any]]:
        """
        Validate all rules and return issues found.

        Returns:
            List of validation issues
        """
        issues = []
        all_rules = self.rule_loader.load_all_rules()

        for tree_id, tree_data in all_rules.items():
            tree = tree_data.get("rule_tree", {})
            rules = tree.get("rules", [])

            # Check for duplicate IDs
            rule_ids = [r.get("id") for r in rules]
            duplicates = [rid for rid in rule_ids if rule_ids.count(rid) > 1]
            if duplicates:
                issues.append(
                    {
                        "type": "duplicate_id",
                        "tree_id": tree_id,
                        "rule_ids": list(set(duplicates)),
                    }
                )

            # Validate each rule
            for rule in rules:
                rule_issues = self._validate_rule_content(rule, tree_id)
                issues.extend(rule_issues)

        return issues

    def _validate_rule_content(
        self,
        rule: dict[str, Any],
        tree_id: str,
    ) -> list[dict[str, Any]]:
        """Validate content of a single rule."""
        issues = []
        rule_id = rule.get("id", "unknown")

        # Check risk score range
        risk_score = rule.get("consequence", {}).get("risk_score", 0)
        if not -100 <= risk_score <= 100:
            issues.append(
                {
                    "type": "invalid_risk_score",
                    "tree_id": tree_id,
                    "rule_id": rule_id,
                    "value": risk_score,
                }
            )

        # Check confidence range
        confidence = rule.get("confidence", 1.0)
        if not 0 <= confidence <= 1:
            issues.append(
                {
                    "type": "invalid_confidence",
                    "tree_id": tree_id,
                    "rule_id": rule_id,
                    "value": confidence,
                }
            )

        # Check for empty conditions
        conditions = rule.get("conditions", {})
        if not conditions.get("items"):
            issues.append(
                {
                    "type": "empty_conditions",
                    "tree_id": tree_id,
                    "rule_id": rule_id,
                }
            )

        # Check for missing recommendation
        recommendation = rule.get("consequence", {}).get("recommendation", "")
        if not recommendation.strip():
            issues.append(
                {
                    "type": "missing_recommendation",
                    "tree_id": tree_id,
                    "rule_id": rule_id,
                }
            )

        return issues

    def check_rule_conflicts(self) -> list[dict[str, Any]]:
        """
        Check for potentially conflicting rules.

        Returns:
            List of potential conflicts
        """
        conflicts = []
        all_rules = self.rule_loader.load_all_rules()

        # Collect all rules with their conditions
        rules_by_category: dict[str, list[dict]] = {}
        for tree_data in all_rules.values():
            tree = tree_data.get("rule_tree", {})
            category = tree.get("category", "unknown")
            if category not in rules_by_category:
                rules_by_category[category] = []
            rules_by_category[category].extend(tree.get("rules", []))

        # Check for rules with similar conditions but different outcomes
        for category, rules in rules_by_category.items():
            for i, rule1 in enumerate(rules):
                for rule2 in rules[i + 1 :]:
                    if self._conditions_overlap(rule1, rule2):
                        v1 = rule1.get("consequence", {}).get("violation")
                        v2 = rule2.get("consequence", {}).get("violation")
                        if v1 != v2:
                            conflicts.append(
                                {
                                    "type": "conflicting_outcomes",
                                    "category": category,
                                    "rule1_id": rule1.get("id"),
                                    "rule2_id": rule2.get("id"),
                                    "rule1_violation": v1,
                                    "rule2_violation": v2,
                                }
                            )

        return conflicts

    def _conditions_overlap(
        self,
        rule1: dict[str, Any],
        rule2: dict[str, Any],
    ) -> bool:
        """
        Check if two rules have overlapping conditions.

        This is a simplified check - full overlap detection would
        require more sophisticated analysis.
        """
        # Extract field paths from both rules
        fields1 = self._extract_field_paths(rule1.get("conditions", {}))
        fields2 = self._extract_field_paths(rule2.get("conditions", {}))

        # Check for significant overlap
        common = fields1.intersection(fields2)
        if len(common) >= 2:
            return True

        return False

    def _extract_field_paths(
        self,
        conditions: dict[str, Any],
    ) -> set[str]:
        """Extract all field paths from conditions recursively."""
        fields = set()

        items = conditions.get("items", [])
        for item in items:
            if "field" in item:
                fields.add(item["field"])
            elif "items" in item:
                # Nested condition group
                fields.update(self._extract_field_paths(item))

        return fields
