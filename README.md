# Project Wenah

Civil Rights Compliance Framework for Responsible Product Development.

A Python-based ML framework that helps companies build responsible products by evaluating them against federal and state civil rights laws. Uses a hybrid approach combining deterministic decision tree rules with Claude-powered RAG analysis.

## Features

- **Risk Assessment Dashboard** - Comprehensive compliance risk scoring
- **Design Guidance** - Product design compliance recommendations
- **Pre-launch Compliance Check** - Final verification before deployment

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from wenah.core.engine import get_compliance_engine
from wenah.core.types import ProductFeatureInput, ProductCategory, FeatureType

engine = get_compliance_engine()

# Assess a feature
analysis = engine.assess_feature(feature)
print(analysis.explanation)
```

## Architecture

- **Rule Engine**: Deterministic compliance rules based on civil rights law
- **RAG Pipeline**: Claude-powered nuanced analysis with hallucination guardrails
- **Unified Scoring**: Combines rule and LLM scores with confidence intervals

## Law Coverage

- Employment: Title VII, ADA
- Housing: FHA (coming soon)
- Consumer: ECOA, FCRA (coming soon)

## License

Proprietary
