"""
Text extraction module for parsing free-text descriptions into structured data.

Uses Claude to extract data fields, algorithm details, and compliance-relevant
information from product feature descriptions.
"""

import json
import re
from typing import Any

from pydantic import BaseModel

from wenah.config import settings
from wenah.core.types import (
    ProductFeatureInput,
    ProductCategory,
    FeatureType,
    DataFieldSpec,
    AlgorithmSpec,
)


# Proxy variable mappings for civil rights compliance
PROXY_MAPPINGS = {
    "zip_code": ["race", "national_origin", "ethnicity"],
    "zip code": ["race", "national_origin", "ethnicity"],
    "zipcode": ["race", "national_origin", "ethnicity"],
    "address": ["race", "national_origin"],
    "neighborhood": ["race", "national_origin"],
    "name": ["national_origin", "sex", "race"],
    "first_name": ["sex", "national_origin"],
    "last_name": ["national_origin", "race"],
    "age": ["age"],
    "birth_date": ["age"],
    "birthdate": ["age"],
    "date_of_birth": ["age"],
    "graduation_year": ["age"],
    "years_of_experience": ["age"],
    "facial_recognition": ["race", "sex", "disability", "religion"],
    "face_analysis": ["race", "sex", "disability"],
    "photo": ["race", "sex", "disability", "religion"],
    "profile_photo": ["race", "sex", "disability", "religion"],
    "video_interview": ["race", "sex", "disability", "national_origin"],
    "voice_analysis": ["sex", "national_origin", "disability"],
    "accent": ["national_origin"],
    "employment_gap": ["sex", "disability"],
    "employment_gaps": ["sex", "disability"],
    "career_gap": ["sex", "disability"],
    "school": ["race", "socioeconomic"],
    "university": ["race", "socioeconomic"],
    "college": ["race", "socioeconomic"],
    "credit_score": ["race", "national_origin"],
    "credit_history": ["race", "national_origin"],
    "criminal_history": ["race"],
    "criminal_record": ["race"],
    "arrest_record": ["race"],
    "marital_status": ["sex"],
    "family_status": ["sex", "familial_status"],
    "children": ["sex", "familial_status"],
    "pregnancy": ["sex"],
    "disability": ["disability"],
    "medical_history": ["disability"],
    "health_condition": ["disability"],
    "religion": ["religion"],
    "religious": ["religion"],
    "gender": ["sex"],
    "sex": ["sex"],
    "race": ["race"],
    "ethnicity": ["race", "national_origin"],
    "national_origin": ["national_origin"],
    "citizenship": ["national_origin"],
    "social_media": ["religion", "national_origin", "disability", "political"],
    "cultural_fit": ["race", "national_origin", "religion"],
}


class ExtractedDataField(BaseModel):
    """Extracted data field from text."""
    name: str
    description: str = ""
    used_in_decisions: bool = True
    potential_proxy: str | None = None


class ExtractedAlgorithm(BaseModel):
    """Extracted algorithm details from text."""
    type: str = "unknown"
    has_human_review: bool = True
    bias_testing_mentioned: bool = False
    automated_decisions: bool = False


class ExtractionResult(BaseModel):
    """Result of text extraction."""
    data_fields: list[ExtractedDataField]
    algorithm: ExtractedAlgorithm
    decision_impact: str
    affected_population: str
    has_appeals_process: bool = True
    confidence: float = 0.8


class TextExtractor:
    """
    Extracts structured compliance data from free-text descriptions.

    Uses a combination of keyword matching and Claude API for intelligent extraction.
    """

    def __init__(self, use_llm: bool = True):
        """
        Initialize the text extractor.

        Args:
            use_llm: Whether to use Claude for extraction. If False, uses keyword matching only.
        """
        self.use_llm = use_llm
        self._client = None

    def _get_client(self):
        """Lazy-load the Claude client."""
        if self._client is None and self.use_llm:
            try:
                import anthropic
                api_key = settings.anthropic_api_key
                if api_key:
                    self._client = anthropic.Anthropic(api_key=api_key)
            except Exception:
                pass
        return self._client

    def extract(
        self,
        description: str,
        name: str = "",
        category: str = "general",
    ) -> ExtractionResult:
        """
        Extract structured data from a feature description.

        Args:
            description: The free-text description to parse
            name: Feature name for context
            category: Product category (hiring, lending, housing, etc.)

        Returns:
            ExtractionResult with extracted fields and algorithm details
        """
        # Try LLM extraction first if available
        client = self._get_client()
        if client:
            try:
                return self._extract_with_llm(description, name, category)
            except Exception as e:
                print(f"LLM extraction failed, falling back to keyword: {e}")

        # Fall back to keyword-based extraction
        return self._extract_with_keywords(description, name, category)

    def _extract_with_llm(
        self,
        description: str,
        name: str,
        category: str,
    ) -> ExtractionResult:
        """Extract using Claude API."""
        client = self._get_client()

        prompt = f"""Analyze this product feature description for civil rights compliance. Extract all data fields and characteristics.

FEATURE NAME: {name}
CATEGORY: {category}
DESCRIPTION: {description}

Extract and return a JSON object with this exact structure:
{{
    "data_fields": [
        {{
            "name": "field_name_in_snake_case",
            "description": "brief description of what this field captures",
            "used_in_decisions": true/false,
            "potential_proxy": "protected_class or null"
        }}
    ],
    "algorithm": {{
        "type": "rule_based|ml_model|ai_llm|unknown",
        "has_human_review": true/false,
        "bias_testing_mentioned": true/false,
        "automated_decisions": true/false
    }},
    "decision_impact": "description of what decisions are made",
    "affected_population": "who is affected by this feature",
    "has_appeals_process": true/false,
    "confidence": 0.0-1.0
}}

IMPORTANT RULES:
1. Look for EXPLICIT data fields mentioned (zip code, age, name, etc.)
2. Look for IMPLICIT data collection (facial recognition implies collecting facial data)
3. For potential_proxy, use these mappings:
   - zip_code, address, neighborhood → "race" (geographic proxies)
   - name, first_name, last_name → "national_origin"
   - age, birth_date, graduation_year → "age"
   - facial_recognition, photo → "race" (visual proxies)
   - employment_gaps → "sex" (affects women disproportionately)
   - credit_score, criminal_history → "race"
4. Set automated_decisions=true if decisions happen "automatically" or "without human review"
5. Set has_appeals_process=false if "no appeals" or similar is mentioned
6. Set confidence based on how clear the description is (0.5-1.0)

Return ONLY the JSON object, no other text."""

        response = client.messages.create(
            model=settings.claude_model or "claude-sonnet-4-20250514",
            max_tokens=2000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse the response
        content = response.content[0].text

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()
            data = json.loads(json_str)

            return ExtractionResult(
                data_fields=[
                    ExtractedDataField(**field)
                    for field in data.get("data_fields", [])
                ],
                algorithm=ExtractedAlgorithm(**data.get("algorithm", {})),
                decision_impact=data.get("decision_impact", "Unknown"),
                affected_population=data.get("affected_population", "General users"),
                has_appeals_process=data.get("has_appeals_process", True),
                confidence=data.get("confidence", 0.8),
            )

        # If JSON parsing fails, fall back to keywords
        return self._extract_with_keywords(description, name, category)

    def _extract_with_keywords(
        self,
        description: str,
        name: str,
        category: str,
    ) -> ExtractionResult:
        """Extract using keyword matching."""
        description_lower = description.lower()
        name_lower = name.lower()
        combined_text = f"{name_lower} {description_lower}"

        data_fields = []
        found_fields = set()

        # Check for proxy variables in text
        for keyword, proxies in PROXY_MAPPINGS.items():
            if keyword in combined_text and keyword not in found_fields:
                # Normalize the field name
                field_name = keyword.replace(" ", "_").replace("-", "_")

                data_fields.append(ExtractedDataField(
                    name=field_name,
                    description=f"Uses {keyword} data",
                    used_in_decisions=True,
                    potential_proxy=proxies[0] if proxies else None,
                ))
                found_fields.add(keyword)

        # Detect algorithm characteristics
        has_human_review = not any(phrase in combined_text for phrase in [
            "without human review",
            "no human review",
            "automated",
            "automatically",
            "no manual",
            "fully automated",
        ])

        automated_decisions = any(phrase in combined_text for phrase in [
            "automat",
            "ai system",
            "ml model",
            "machine learning",
            "algorithm",
            "scoring",
        ])

        bias_testing = any(phrase in combined_text for phrase in [
            "bias test",
            "fairness audit",
            "bias audit",
            "disparate impact analysis",
        ])

        has_appeals = not any(phrase in combined_text for phrase in [
            "no appeal",
            "no recourse",
            "final decision",
            "cannot be contested",
        ])

        # Determine algorithm type
        alg_type = "unknown"
        if any(term in combined_text for term in ["ai", "machine learning", "ml", "neural", "deep learning"]):
            alg_type = "ml_model"
        elif any(term in combined_text for term in ["llm", "gpt", "language model", "chatbot"]):
            alg_type = "ai_llm"
        elif any(term in combined_text for term in ["rule", "criteria", "threshold", "score"]):
            alg_type = "rule_based"

        # Determine affected population based on category
        affected_pop = "General users"
        if category == "hiring":
            affected_pop = "Job applicants and candidates"
        elif category == "lending":
            affected_pop = "Loan applicants"
        elif category == "housing":
            affected_pop = "Housing applicants and tenants"
        elif category == "insurance":
            affected_pop = "Insurance applicants"

        return ExtractionResult(
            data_fields=data_fields,
            algorithm=ExtractedAlgorithm(
                type=alg_type,
                has_human_review=has_human_review,
                bias_testing_mentioned=bias_testing,
                automated_decisions=automated_decisions,
            ),
            decision_impact=f"Automated {category} decisions" if automated_decisions else f"{category.title()} decisions",
            affected_population=affected_pop,
            has_appeals_process=has_appeals,
            confidence=0.7 if data_fields else 0.5,
        )

    def to_product_feature_input(
        self,
        extraction: ExtractionResult,
        feature_id: str,
        name: str,
        description: str,
        category: str,
    ) -> ProductFeatureInput:
        """
        Convert extraction result to ProductFeatureInput for the rule engine.

        Args:
            extraction: The extraction result
            feature_id: Unique feature ID
            name: Feature name
            description: Original description
            category: Product category

        Returns:
            ProductFeatureInput ready for rule engine evaluation
        """
        # Map category string to enum
        category_map = {
            "hiring": ProductCategory.HIRING,
            "lending": ProductCategory.LENDING,
            "housing": ProductCategory.HOUSING,
            "insurance": ProductCategory.INSURANCE,
            "general": ProductCategory.GENERAL,
        }
        product_category = category_map.get(category.lower(), ProductCategory.GENERAL)

        # Convert extracted fields to DataFieldSpec
        data_fields = [
            DataFieldSpec(
                name=field.name,
                description=field.description,
                data_type="text",
                source="user_input",
                required=False,
                used_in_decisions=field.used_in_decisions,
                potential_proxy=field.potential_proxy,
            )
            for field in extraction.data_fields
        ]

        # Convert algorithm details
        algorithm = None
        if extraction.algorithm.automated_decisions:
            alg_type_map = {
                "ml_model": "ml_model",
                "ai_llm": "llm",
                "rule_based": "rule_based",
                "unknown": "unknown",
            }
            algorithm = AlgorithmSpec(
                name=f"{name} Algorithm",
                type=alg_type_map.get(extraction.algorithm.type, "unknown"),
                inputs=[f.name for f in extraction.data_fields],
                outputs=["decision", "score"],
                bias_testing_done=extraction.algorithm.bias_testing_mentioned,
                description=f"Algorithm for {name}",
            )

        # Determine feature type
        feature_type = FeatureType.ALGORITHM
        if extraction.algorithm.automated_decisions and not extraction.algorithm.has_human_review:
            feature_type = FeatureType.AUTOMATED_DECISION
        elif extraction.algorithm.has_human_review:
            feature_type = FeatureType.HUMAN_ASSISTED
        elif data_fields:
            feature_type = FeatureType.DATA_COLLECTION

        return ProductFeatureInput(
            feature_id=feature_id,
            name=name,
            description=description,
            category=product_category,
            feature_type=feature_type,
            data_fields=data_fields,
            algorithm=algorithm,
            decision_impact=extraction.decision_impact,
            affected_population=extraction.affected_population,
            company_size=None,
            additional_context=f"Extracted confidence: {extraction.confidence}",
        )


# Singleton instance
_extractor: TextExtractor | None = None


def get_text_extractor(use_llm: bool = True) -> TextExtractor:
    """Get the singleton text extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = TextExtractor(use_llm=use_llm)
    return _extractor


def extract_and_convert(
    feature_id: str,
    name: str,
    description: str,
    category: str,
    use_llm: bool = True,
) -> ProductFeatureInput:
    """
    Convenience function to extract and convert in one step.

    Args:
        feature_id: Unique feature ID
        name: Feature name
        description: Free-text description
        category: Product category
        use_llm: Whether to use Claude API

    Returns:
        ProductFeatureInput ready for compliance analysis
    """
    extractor = get_text_extractor(use_llm=use_llm)
    extraction = extractor.extract(description, name, category)
    return extractor.to_product_feature_input(
        extraction, feature_id, name, description, category
    )
