"""
Claude API client wrapper for compliance analysis.

Provides structured output support and retry logic for the Anthropic Claude API.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel

from wenah.config import settings


class AnalysisType(str, Enum):
    """Types of compliance analysis."""

    RISK_ASSESSMENT = "risk_assessment"
    DESIGN_GUIDANCE = "design_guidance"
    LEGAL_RESEARCH = "legal_research"
    VIOLATION_ANALYSIS = "violation_analysis"
    ACCOMMODATION_REVIEW = "accommodation_review"


@dataclass
class ClaudeResponse:
    """Wrapper for Claude API responses."""

    content: str
    model: str
    usage: dict[str, int]
    stop_reason: str | None
    raw_response: Any = None


T = TypeVar("T", bound=BaseModel)


class ClaudeClient:
    """
    Client for interacting with Claude API for compliance analysis.

    Features:
    - Structured output parsing with Pydantic models
    - Retry logic with exponential backoff
    - Token usage tracking
    - Response caching (optional)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
    ):
        """
        Initialize the Claude client.

        Args:
            api_key: Anthropic API key. Defaults to config setting.
            model: Model to use. Defaults to config setting.
            max_retries: Maximum retry attempts for failed requests.
        """
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.claude_model
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError(
                "Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable."
            )

        self._client = anthropic.Anthropic(api_key=self.api_key)
        self._total_tokens_used = 0

    def analyze(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> ClaudeResponse:
        """
        Send an analysis request to Claude.

        Args:
            prompt: The user prompt to send
            system_prompt: Optional system prompt for context
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0 = deterministic)

        Returns:
            ClaudeResponse with the analysis result
        """
        messages = [{"role": "user", "content": prompt}]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        if temperature > 0:
            kwargs["temperature"] = temperature

        response = self._call_with_retry(**kwargs)

        # Track usage
        self._total_tokens_used += response.usage.input_tokens + response.usage.output_tokens

        # Extract text from first text block in response
        content_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                content_text = block.text
                break

        return ClaudeResponse(
            content=content_text,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            stop_reason=response.stop_reason,
            raw_response=response,
        )

    def analyze_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
        max_tokens: int = 4096,
    ) -> T:
        """
        Send an analysis request and parse response into a Pydantic model.

        Args:
            prompt: The user prompt to send
            response_model: Pydantic model class for response parsing
            system_prompt: Optional system prompt for context
            max_tokens: Maximum tokens in response

        Returns:
            Parsed response as the specified Pydantic model
        """
        # Build prompt with JSON schema instructions
        schema = response_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)

        structured_prompt = f"""{prompt}

Please respond with a JSON object that conforms to this schema:

```json
{schema_str}
```

Respond ONLY with the JSON object, no additional text or markdown formatting."""

        # Add system prompt about JSON output
        json_system = (
            "You are a legal compliance analyst. Always respond with valid JSON "
            "that matches the requested schema. Do not include markdown code blocks "
            "or any text outside the JSON object."
        )

        if system_prompt:
            full_system = f"{system_prompt}\n\n{json_system}"
        else:
            full_system = json_system

        response = self.analyze(
            prompt=structured_prompt,
            system_prompt=full_system,
            max_tokens=max_tokens,
            temperature=0.0,  # Deterministic for structured output
        )

        # Parse response
        try:
            # Try to extract JSON from response
            content = response.content.strip()

            # Handle potential markdown code blocks
            if content.startswith("```"):
                # Extract content between code blocks
                lines = content.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                content = "\n".join(json_lines)

            data = json.loads(content)
            return response_model.model_validate(data)

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nContent: {response.content}")
        except Exception as e:
            raise ValueError(f"Failed to validate response: {e}\nContent: {response.content}")

    def analyze_with_context(
        self,
        query: str,
        context_documents: list[dict[str, Any]],
        analysis_type: AnalysisType,
        additional_instructions: str | None = None,
    ) -> ClaudeResponse:
        """
        Analyze with retrieved context documents (for RAG).

        Args:
            query: The analysis query
            context_documents: Retrieved documents with content and metadata
            analysis_type: Type of analysis to perform
            additional_instructions: Extra instructions for the analysis

        Returns:
            Analysis response
        """
        # Build context section
        context_parts = []
        for i, doc in enumerate(context_documents, 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            source = metadata.get("law_name", metadata.get("law_id", f"Document {i}"))
            section = metadata.get("section", "")

            context_parts.append(
                f"### Source {i}: {source}" + (f" - {section}" if section else "") + f"\n{content}"
            )

        context_text = "\n\n".join(context_parts)

        # Build system prompt based on analysis type
        system_prompt = self._get_system_prompt_for_type(analysis_type)

        # Build user prompt
        prompt = f"""## Context: Relevant Civil Rights Law Information

{context_text}

## Analysis Request

{query}"""

        if additional_instructions:
            prompt += f"\n\n## Additional Instructions\n\n{additional_instructions}"

        return self.analyze(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=4096,
        )

    def _get_system_prompt_for_type(self, analysis_type: AnalysisType) -> str:
        """Get appropriate system prompt for analysis type."""
        base_prompt = (
            "You are an expert legal compliance analyst specializing in civil rights law, "
            "including Title VII, the ADA, Fair Housing Act, ECOA, and FCRA. "
            "Your analysis must be grounded in the provided legal context. "
            "Only make claims that can be supported by the provided sources. "
            "If you are uncertain, clearly indicate your level of confidence."
        )

        type_prompts = {
            AnalysisType.RISK_ASSESSMENT: (
                f"{base_prompt}\n\n"
                "Your task is to assess compliance risks. Identify potential violations, "
                "rate their severity, and provide actionable recommendations. "
                "Consider both disparate treatment and disparate impact theories."
            ),
            AnalysisType.DESIGN_GUIDANCE: (
                f"{base_prompt}\n\n"
                "Your task is to provide guidance for designing compliant products. "
                "Focus on proactive measures to prevent discrimination and ensure "
                "accessibility. Suggest best practices and safe harbors where applicable."
            ),
            AnalysisType.LEGAL_RESEARCH: (
                f"{base_prompt}\n\n"
                "Your task is to research and explain relevant legal requirements. "
                "Cite specific statutes, regulations, and case law. Explain how "
                "legal principles apply to the specific situation described."
            ),
            AnalysisType.VIOLATION_ANALYSIS: (
                f"{base_prompt}\n\n"
                "Your task is to analyze potential violations in detail. "
                "Identify the specific legal elements that may be violated, "
                "the protected classes affected, and the strength of potential claims."
            ),
            AnalysisType.ACCOMMODATION_REVIEW: (
                f"{base_prompt}\n\n"
                "Your task is to review accommodation requirements under the ADA "
                "and religious accommodation requirements under Title VII. "
                "Assess whether proposed accommodations are reasonable and whether "
                "undue hardship defenses may apply."
            ),
        }

        return type_prompts.get(analysis_type, base_prompt)

    def _call_with_retry(self, **kwargs: Any) -> anthropic.types.Message:
        """Call the API with retry logic."""
        import time

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                return self._client.messages.create(**kwargs)  # type: ignore[no-any-return]
            except anthropic.RateLimitError as e:
                last_error = e
                wait_time = 2**attempt  # Exponential backoff
                time.sleep(wait_time)
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    last_error = e
                    wait_time = 2**attempt
                    time.sleep(wait_time)
                else:
                    raise

        raise last_error or Exception("Max retries exceeded")

    @property
    def total_tokens_used(self) -> int:
        """Get total tokens used across all requests."""
        return self._total_tokens_used

    def reset_token_counter(self) -> None:
        """Reset the token usage counter."""
        self._total_tokens_used = 0


class ComplianceAnalyzer:
    """
    High-level compliance analyzer using Claude.

    Provides domain-specific analysis methods for civil rights compliance.
    """

    def __init__(self, client: ClaudeClient | None = None):
        """
        Initialize the analyzer.

        Args:
            client: ClaudeClient instance. Creates new one if not provided.
        """
        self.client = client or ClaudeClient()

    def analyze_feature_compliance(
        self,
        feature_description: str,
        category: str,
        context_docs: list[dict[str, Any]],
        specific_concerns: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze a product feature for compliance issues.

        Args:
            feature_description: Description of the feature
            category: Category (employment, housing, consumer)
            context_docs: Retrieved legal context documents
            specific_concerns: Specific areas to focus on

        Returns:
            Analysis results dictionary
        """
        concerns_text = ""
        if specific_concerns:
            concerns_text = "\n\nSpecific concerns to address:\n" + "\n".join(
                f"- {c}" for c in specific_concerns
            )

        query = f"""Analyze the following product feature for civil rights compliance issues:

**Category:** {category}

**Feature Description:**
{feature_description}
{concerns_text}

Provide:
1. Identified compliance risks (with severity: critical, high, medium, low)
2. Applicable laws and specific provisions
3. Recommendations for remediation
4. Confidence level in your analysis (0-100%)"""

        response = self.client.analyze_with_context(
            query=query,
            context_documents=context_docs,
            analysis_type=AnalysisType.RISK_ASSESSMENT,
        )

        return {
            "analysis": response.content,
            "model": response.model,
            "tokens_used": response.usage,
        }

    def analyze_disparate_impact(
        self,
        practice_description: str,
        affected_groups: list[str],
        context_docs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Analyze a practice for potential disparate impact.

        Args:
            practice_description: Description of the practice
            affected_groups: Groups that may be affected
            context_docs: Retrieved legal context

        Returns:
            Disparate impact analysis
        """
        query = f"""Analyze the following practice for potential disparate impact under civil rights laws:

**Practice:**
{practice_description}

**Potentially Affected Groups:**
{", ".join(affected_groups)}

Analyze:
1. Whether the practice is facially neutral
2. Potential for disproportionate impact on protected classes
3. Whether business necessity defense may apply
4. Less discriminatory alternatives that might exist
5. Relevant case law and EEOC guidance"""

        response = self.client.analyze_with_context(
            query=query,
            context_documents=context_docs,
            analysis_type=AnalysisType.VIOLATION_ANALYSIS,
        )

        return {
            "analysis": response.content,
            "model": response.model,
            "tokens_used": response.usage,
        }

    def get_accommodation_guidance(
        self,
        accommodation_request: str,
        job_requirements: str,
        context_docs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Get guidance on accommodation requests.

        Args:
            accommodation_request: The requested accommodation
            job_requirements: Essential job functions
            context_docs: Retrieved legal context

        Returns:
            Accommodation guidance
        """
        query = f"""Provide guidance on the following accommodation situation:

**Accommodation Requested:**
{accommodation_request}

**Job Requirements/Essential Functions:**
{job_requirements}

Analyze:
1. Whether this is a reasonable accommodation under the ADA
2. What the interactive process should involve
3. When undue hardship might apply
4. Alternative accommodations to consider
5. Documentation requirements"""

        response = self.client.analyze_with_context(
            query=query,
            context_documents=context_docs,
            analysis_type=AnalysisType.ACCOMMODATION_REVIEW,
        )

        return {
            "analysis": response.content,
            "model": response.model,
            "tokens_used": response.usage,
        }


# Singleton instances
_claude_client: ClaudeClient | None = None
_compliance_analyzer: ComplianceAnalyzer | None = None


def get_claude_client() -> ClaudeClient:
    """Get singleton Claude client instance."""
    global _claude_client
    if _claude_client is None:
        _claude_client = ClaudeClient()
    return _claude_client


def get_compliance_analyzer() -> ComplianceAnalyzer:
    """Get singleton compliance analyzer instance."""
    global _compliance_analyzer
    if _compliance_analyzer is None:
        _compliance_analyzer = ComplianceAnalyzer()
    return _compliance_analyzer
