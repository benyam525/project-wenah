"""
Decision tree rule engine for compliance evaluation.

Evaluates product features against civil rights compliance rules using
a configurable decision tree approach.
"""

from typing import Any

from wenah.core.types import (
    ProductFeatureInput,
    RuleEvaluation,
    RuleResult,
    Rule,
    RuleConditionGroup,
    RuleConsequence,
)
from wenah.rules.rule_loader import RuleLoader


class RuleEngine:
    """
    Decision tree-based rule evaluation engine.

    Evaluates product features against compliance rules and returns
    structured evaluations with risk scores and recommendations.
    """

    def __init__(self, rules_directory: str | None = None):
        """
        Initialize the rule engine.

        Args:
            rules_directory: Directory containing rule YAML files.
                            Defaults to config setting.
        """
        self.rule_loader = RuleLoader(rules_directory)
        self._rules_cache: dict[str, list[Rule]] | None = None

    def evaluate(
        self,
        feature: ProductFeatureInput,
        categories: list[str] | None = None,
    ) -> list[RuleEvaluation]:
        """
        Evaluate a product feature against all applicable rules.

        Args:
            feature: The product feature to evaluate
            categories: Optional list of categories to evaluate against.
                       Defaults to inferring from feature category.

        Returns:
            List of rule evaluations with scores and recommendations
        """
        # Determine which categories to evaluate
        if categories is None:
            categories = self._infer_categories(feature)

        evaluations = []

        for category in categories:
            rules = self._get_rules_for_category(category)
            for rule in rules:
                evaluation = self._evaluate_rule(rule, feature)
                if evaluation is not None:
                    evaluations.append(evaluation)

        # Sort by risk score (highest first)
        evaluations.sort(key=lambda e: e.risk_score, reverse=True)

        return evaluations

    def evaluate_multiple(
        self,
        features: list[ProductFeatureInput],
        categories: list[str] | None = None,
    ) -> dict[str, list[RuleEvaluation]]:
        """
        Evaluate multiple features against rules.

        Args:
            features: List of product features to evaluate
            categories: Optional list of categories to evaluate against

        Returns:
            Dictionary mapping feature IDs to their evaluations
        """
        results = {}
        for feature in features:
            results[feature.feature_id] = self.evaluate(feature, categories)
        return results

    def _infer_categories(self, feature: ProductFeatureInput) -> list[str]:
        """Infer applicable rule categories from feature."""
        category_map = {
            "hiring": ["employment"],
            "lending": ["consumer"],
            "housing": ["housing"],
            "insurance": ["consumer"],
            "general": ["employment", "housing", "consumer"],
        }
        return category_map.get(feature.category.value, ["employment"])

    def _get_rules_for_category(self, category: str) -> list[Rule]:
        """Get all rules for a specific category."""
        if self._rules_cache is None:
            self._rules_cache = {}

        if category not in self._rules_cache:
            rule_dicts = self.rule_loader.load_rules_by_category(category)
            self._rules_cache[category] = [
                self._dict_to_rule(r) for r in rule_dicts
            ]

        return self._rules_cache[category]

    def _dict_to_rule(self, rule_dict: dict[str, Any]) -> Rule:
        """Convert rule dictionary to Rule model."""
        return Rule(
            id=rule_dict.get("id", ""),
            name=rule_dict.get("name", ""),
            description=rule_dict.get("description", ""),
            severity=rule_dict.get("severity", "medium"),
            confidence=rule_dict.get("confidence", 1.0),
            conditions=RuleConditionGroup(**rule_dict.get("conditions", {})),
            consequence=RuleConsequence(**rule_dict.get("consequence", {})),
        )

    def _evaluate_rule(
        self,
        rule: Rule,
        feature: ProductFeatureInput,
    ) -> RuleEvaluation | None:
        """
        Evaluate a single rule against a feature.

        Returns None if the rule doesn't match (compliant).
        Returns RuleEvaluation if the rule matches or partially matches.
        """
        # Convert feature to evaluation context
        context = self._feature_to_context(feature)

        # Evaluate conditions
        matches, match_confidence = self._evaluate_conditions(
            rule.conditions, context
        )

        if not matches:
            return None

        # Determine result based on consequence
        consequence = rule.consequence
        if consequence.violation is True:
            result = RuleResult.VIOLATION
        elif consequence.violation == "potential":
            result = RuleResult.POTENTIAL_VIOLATION
        elif consequence.escalate_to_llm:
            result = RuleResult.NEEDS_LLM_REVIEW
        else:
            result = RuleResult.COMPLIANT

        # Calculate effective confidence
        effective_confidence = rule.confidence * match_confidence

        return RuleEvaluation(
            rule_id=rule.id,
            rule_name=rule.name,
            result=result,
            confidence=effective_confidence,
            risk_score=consequence.risk_score,
            law_references=[consequence.law_reference],
            recommendations=[consequence.recommendation],
            escalate_to_llm=consequence.escalate_to_llm,
            llm_context=consequence.llm_context,
        )

    def _feature_to_context(self, feature: ProductFeatureInput) -> dict[str, Any]:
        """
        Convert a ProductFeatureInput to an evaluation context dictionary.

        This flattens the feature into a format suitable for condition evaluation.
        """
        context = {
            "feature": {
                "feature_id": feature.feature_id,
                "name": feature.name,
                "description": feature.description,
                "category": feature.category.value,
                "feature_type": feature.feature_type.value,
                "decision_impact": feature.decision_impact,
                "affected_population": feature.affected_population,
                "company_size": feature.company_size,
                "geographic_scope": feature.geographic_scope,
                "additional_context": feature.additional_context or "",
                "data_fields": [
                    {
                        "name": df.name,
                        "description": df.description,
                        "data_type": df.data_type,
                        "source": df.source,
                        "required": df.required,
                        "used_in_decisions": df.used_in_decisions,
                        "potential_proxy": df.potential_proxy,
                    }
                    for df in feature.data_fields
                ],
            }
        }

        # Add algorithm context if present
        if feature.algorithm:
            context["feature"]["algorithm"] = {
                "name": feature.algorithm.name,
                "type": feature.algorithm.type,
                "inputs": feature.algorithm.inputs,
                "outputs": feature.algorithm.outputs,
                "training_data_description": feature.algorithm.training_data_description,
                "bias_testing_done": feature.algorithm.bias_testing_done,
            }
        else:
            context["feature"]["algorithm"] = None

        return context

    def _evaluate_conditions(
        self,
        conditions: RuleConditionGroup,
        context: dict[str, Any],
    ) -> tuple[bool, float]:
        """
        Recursively evaluate condition groups.

        Returns:
            Tuple of (matches, confidence) where confidence represents
            how strongly the conditions matched.
        """
        operator = conditions.operator.upper()
        items = conditions.items

        if not items:
            return False, 0.0

        results: list[tuple[bool, float]] = []

        for item in items:
            if hasattr(item, "operator") and hasattr(item, "items"):
                # Nested condition group
                match, conf = self._evaluate_conditions(
                    RuleConditionGroup(**item.model_dump()), context
                )
                results.append((match, conf))
            else:
                # Single condition
                match, conf = self._evaluate_single_condition(item, context)
                results.append((match, conf))

        if operator == "AND":
            # All must match
            all_match = all(r[0] for r in results)
            avg_confidence = sum(r[1] for r in results) / len(results) if results else 0
            return all_match, avg_confidence if all_match else 0.0

        elif operator == "OR":
            # At least one must match
            matching = [r for r in results if r[0]]
            if matching:
                # Return highest confidence among matches
                max_confidence = max(r[1] for r in matching)
                return True, max_confidence
            return False, 0.0

        elif operator == "NOT":
            # Invert the result
            if results:
                match, conf = results[0]
                return not match, conf
            return True, 1.0

        return False, 0.0

    def _evaluate_single_condition(
        self,
        condition: Any,
        context: dict[str, Any],
    ) -> tuple[bool, float]:
        """
        Evaluate a single condition against context.

        Returns:
            Tuple of (matches, confidence)
        """
        # Extract condition parameters
        if hasattr(condition, "model_dump"):
            cond_dict = condition.model_dump()
        elif isinstance(condition, dict):
            cond_dict = condition
        else:
            return False, 0.0

        field_path = cond_dict.get("field", "")
        operator = cond_dict.get("operator", "equals")
        value = cond_dict.get("value")
        values = cond_dict.get("values", [])

        # Get field value from context
        field_value = self._get_field_value(field_path, context)

        # Handle array field paths (e.g., "feature.data_fields[*].name")
        if "[*]" in field_path:
            return self._evaluate_array_condition(
                field_path, operator, value, values, context
            )

        # Evaluate based on operator
        return self._compare(field_value, operator, value, values)

    def _get_field_value(
        self,
        field_path: str,
        context: dict[str, Any],
    ) -> Any:
        """
        Get a value from context using dot notation path.

        Supports paths like "feature.category" or "feature.algorithm.type"
        """
        parts = field_path.split(".")
        current = context

        for part in parts:
            if "[*]" in part:
                # Array wildcard - handled separately
                return None
            if isinstance(current, dict):
                current = current.get(part)
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return None

            if current is None:
                return None

        return current

    def _evaluate_array_condition(
        self,
        field_path: str,
        operator: str,
        value: Any,
        values: list[Any],
        context: dict[str, Any],
    ) -> tuple[bool, float]:
        """
        Evaluate conditions on array fields.

        Handles paths like "feature.data_fields[*].name" which should
        check if ANY item in the array matches.
        """
        # Split path at [*]
        parts = field_path.split("[*]")
        if len(parts) != 2:
            return False, 0.0

        array_path = parts[0].rstrip(".")
        item_path = parts[1].lstrip(".")

        # Get the array
        array = self._get_field_value(array_path, context)
        if not isinstance(array, list):
            return False, 0.0

        # Check each item in the array
        matches = 0
        for item in array:
            item_value = self._get_nested_value(item, item_path)
            match, _ = self._compare(item_value, operator, value, values)
            if match:
                matches += 1

        if matches > 0:
            # Confidence based on how many items matched
            confidence = min(1.0, matches / max(len(array), 1))
            return True, confidence

        return False, 0.0

    def _get_nested_value(self, obj: Any, path: str) -> Any:
        """Get nested value from object using dot notation."""
        if not path:
            return obj

        parts = path.split(".")
        current = obj

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return None

            if current is None:
                return None

        return current

    def _compare(
        self,
        field_value: Any,
        operator: str,
        value: Any,
        values: list[Any],
    ) -> tuple[bool, float]:
        """
        Compare field value against condition value(s).

        Returns:
            Tuple of (matches, confidence)
        """
        if field_value is None and operator != "is_null" and operator != "is_not_null":
            return False, 0.0

        operator = operator.lower()

        if operator == "equals":
            match = field_value == value
            return match, 1.0 if match else 0.0

        elif operator == "not_equals":
            match = field_value != value
            return match, 1.0 if match else 0.0

        elif operator == "in":
            match = field_value in values
            return match, 1.0 if match else 0.0

        elif operator == "not_in":
            match = field_value not in values
            return match, 1.0 if match else 0.0

        elif operator == "contains":
            if isinstance(field_value, str):
                match = value.lower() in field_value.lower()
            elif isinstance(field_value, list):
                match = value in field_value
            else:
                match = False
            return match, 1.0 if match else 0.0

        elif operator == "contains_any":
            if isinstance(field_value, str):
                field_lower = field_value.lower()
                match = any(v.lower() in field_lower for v in values)
            elif isinstance(field_value, list):
                match = any(v in field_value for v in values)
            else:
                match = False
            return match, 1.0 if match else 0.0

        elif operator == "contains_all":
            if isinstance(field_value, str):
                field_lower = field_value.lower()
                match = all(v.lower() in field_lower for v in values)
            elif isinstance(field_value, list):
                match = all(v in field_value for v in values)
            else:
                match = False
            return match, 1.0 if match else 0.0

        elif operator == "is_null":
            match = field_value is None
            return match, 1.0 if match else 0.0

        elif operator == "is_not_null":
            match = field_value is not None
            return match, 1.0 if match else 0.0

        elif operator == "greater_than":
            try:
                match = field_value > value
                return match, 1.0 if match else 0.0
            except (TypeError, ValueError):
                return False, 0.0

        elif operator == "less_than":
            try:
                match = field_value < value
                return match, 1.0 if match else 0.0
            except (TypeError, ValueError):
                return False, 0.0

        elif operator == "greater_than_or_equals":
            try:
                match = field_value >= value
                return match, 1.0 if match else 0.0
            except (TypeError, ValueError):
                return False, 0.0

        elif operator == "less_than_or_equals":
            try:
                match = field_value <= value
                return match, 1.0 if match else 0.0
            except (TypeError, ValueError):
                return False, 0.0

        elif operator == "regex":
            import re
            try:
                match = bool(re.search(value, str(field_value)))
                return match, 1.0 if match else 0.0
            except re.error:
                return False, 0.0

        return False, 0.0

    def get_rules_summary(self, category: str | None = None) -> dict[str, Any]:
        """
        Get a summary of loaded rules.

        Args:
            category: Optional category to filter by

        Returns:
            Summary dictionary with rule counts and categories
        """
        all_rules = self.rule_loader.load_all_rules()

        summary = {
            "total_rule_trees": len(all_rules),
            "categories": {},
            "severity_counts": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "positive": 0,
            },
        }

        for tree_id, tree_data in all_rules.items():
            tree = tree_data.get("rule_tree", {})
            cat = tree.get("category", "unknown")

            if category and cat != category:
                continue

            if cat not in summary["categories"]:
                summary["categories"][cat] = {"rule_count": 0, "tree_ids": []}

            rules = tree.get("rules", [])
            summary["categories"][cat]["rule_count"] += len(rules)
            summary["categories"][cat]["tree_ids"].append(tree_id)

            for rule in rules:
                severity = rule.get("severity", "medium")
                if severity in summary["severity_counts"]:
                    summary["severity_counts"][severity] += 1

        return summary

    def clear_cache(self) -> None:
        """Clear the rules cache to force reload."""
        self._rules_cache = None


# Singleton instance
_rule_engine: RuleEngine | None = None


def get_rule_engine() -> RuleEngine:
    """Get the singleton rule engine instance."""
    global _rule_engine
    if _rule_engine is None:
        _rule_engine = RuleEngine()
    return _rule_engine
