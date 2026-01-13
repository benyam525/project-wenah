"""
Wenah: Civil Rights Compliance Framework

A Python-based ML framework that helps companies build responsible products
by evaluating them against federal and state civil rights laws.
"""

__version__ = "0.1.0"
__author__ = "Project Wenah Team"

from wenah.core.types import (
    ProductCategory,
    FeatureType,
    RiskLevel,
    RuleResult,
)

__all__ = [
    "__version__",
    "ProductCategory",
    "FeatureType",
    "RiskLevel",
    "RuleResult",
]
