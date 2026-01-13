"""
Prompt templates for civil rights compliance analysis.

Contains structured prompts for various compliance analysis scenarios,
designed to elicit accurate, well-grounded responses from Claude.
"""

from typing import Any
from string import Template


# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPT_BASE = """You are an expert legal compliance analyst specializing in U.S. civil rights law. Your expertise covers:

- Title VII of the Civil Rights Act of 1964 (employment discrimination)
- Americans with Disabilities Act (ADA) - Title I (employment)
- Fair Housing Act (FHA)
- Equal Credit Opportunity Act (ECOA)
- Fair Credit Reporting Act (FCRA)

CRITICAL INSTRUCTIONS:
1. Only make claims that can be supported by the provided context documents
2. When citing laws, use accurate citations (e.g., "42 U.S.C. § 2000e-2")
3. Clearly distinguish between definite violations and potential concerns
4. If uncertain, explicitly state your confidence level
5. Do not invent or hallucinate legal provisions, cases, or statistics
6. Focus on actionable, practical recommendations

Your analysis should be thorough but concise, prioritizing the most significant issues."""


SYSTEM_PROMPT_RISK_ASSESSMENT = f"""{SYSTEM_PROMPT_BASE}

For this risk assessment:
- Identify specific compliance risks with severity ratings
- Map risks to applicable legal provisions
- Consider both disparate treatment and disparate impact theories
- Provide prioritized recommendations for remediation
- Note any areas requiring further investigation"""


SYSTEM_PROMPT_DESIGN_GUIDANCE = f"""{SYSTEM_PROMPT_BASE}

For design guidance:
- Focus on proactive compliance measures
- Suggest design patterns that minimize legal risk
- Identify safe harbors and best practices
- Consider accessibility requirements (ADA, WCAG)
- Recommend testing and validation approaches"""


SYSTEM_PROMPT_STRUCTURED_OUTPUT = f"""{SYSTEM_PROMPT_BASE}

IMPORTANT: You must respond with a valid JSON object matching the provided schema.
- Do not include any text before or after the JSON
- Do not use markdown code blocks
- Ensure all required fields are present
- Use null for optional fields you cannot determine"""


# =============================================================================
# Analysis Prompt Templates
# =============================================================================

RISK_ANALYSIS_PROMPT = Template("""## Civil Rights Compliance Risk Analysis

### Context Documents
$context_documents

### Feature to Analyze

**Feature Name:** $feature_name
**Category:** $category
**Description:** $description

**Data Fields Collected/Used:**
$data_fields

**Algorithm Details:**
$algorithm_details

**Decision Impact:** $decision_impact
**Affected Population:** $affected_population

### Analysis Request

Analyze this feature for civil rights compliance risks. For each risk identified:

1. **Risk Description**: What is the potential compliance issue?
2. **Severity**: Critical / High / Medium / Low
3. **Applicable Law**: Which law(s) and provision(s) apply?
4. **Legal Theory**: Disparate treatment, disparate impact, failure to accommodate, etc.
5. **Evidence Indicators**: What would indicate a violation exists?
6. **Recommendation**: Specific steps to mitigate the risk
7. **Confidence**: How confident are you in this assessment? (0-100%)

$additional_instructions""")


DISPARATE_IMPACT_PROMPT = Template("""## Disparate Impact Analysis

### Context Documents
$context_documents

### Practice Under Review

**Practice Description:** $practice_description
**Context:** $context
**Affected Groups:** $affected_groups

### Analysis Request

Conduct a disparate impact analysis under civil rights law:

1. **Facial Neutrality**: Is this practice facially neutral?

2. **Impact Assessment**:
   - Which protected classes might be disproportionately affected?
   - What data would be needed to measure impact?
   - Reference the four-fifths rule where applicable

3. **Business Necessity Defense**:
   - Could the employer demonstrate business necessity?
   - Is the practice job-related and consistent with business necessity?
   - What evidence would support or undermine this defense?

4. **Less Discriminatory Alternatives**:
   - Are there alternative practices that would achieve the same goal?
   - Would alternatives have less discriminatory impact?

5. **Relevant Precedents**:
   - What case law applies to this situation?
   - How have courts ruled on similar practices?

6. **Recommendations**:
   - What changes should be considered?
   - What documentation is needed?

$additional_instructions""")


PROXY_VARIABLE_PROMPT = Template("""## Proxy Variable Analysis

### Context Documents
$context_documents

### Variable Under Review

**Variable Name:** $variable_name
**How It's Used:** $usage_description
**Decision Context:** $decision_context

### Analysis Request

Analyze whether this variable may serve as a proxy for protected class characteristics:

1. **Potential Proxy Relationships**:
   - Which protected classes might this variable correlate with?
   - What is the likely strength of correlation?
   - Is this a well-documented proxy in legal precedent?

2. **Legal Risk Assessment**:
   - Could use of this variable support a disparate impact claim?
   - Are there relevant EEOC guidelines or court decisions?
   - What is the risk level for using this variable?

3. **Business Justification**:
   - What legitimate business purpose does this variable serve?
   - Is there a less discriminatory alternative?
   - How essential is this variable to the decision?

4. **Mitigation Strategies**:
   - Should this variable be removed from the model?
   - Can its influence be reduced or monitored?
   - What testing should be conducted?

5. **Confidence Assessment**:
   - How confident are you in this proxy relationship analysis?
   - What additional information would increase confidence?

$additional_instructions""")


ADA_COMPLIANCE_PROMPT = Template("""## ADA Compliance Analysis

### Context Documents
$context_documents

### Feature/Practice Under Review

**Description:** $description
**Stage:** $stage (pre-offer / post-offer / employment)
**Type:** $feature_type

### Analysis Request

Analyze this feature for ADA Title I compliance:

1. **Medical Inquiry Analysis** (if applicable):
   - Does this constitute a medical inquiry or examination?
   - Is it permitted at this stage of employment?
   - What are the confidentiality requirements?

2. **Accessibility Analysis**:
   - Is this feature accessible to individuals with disabilities?
   - What barriers might exist?
   - What accommodations might be needed?

3. **Discrimination Risk**:
   - Could this screen out individuals with disabilities?
   - Is there a direct threat analysis needed?
   - Does this assess actual ability to perform essential functions?

4. **Reasonable Accommodation**:
   - What accommodations should be available?
   - Is there an interactive process in place?
   - What would constitute undue hardship?

5. **Recommendations**:
   - Specific changes to ensure ADA compliance
   - Documentation requirements
   - Training needs

$additional_instructions""")


AI_ALGORITHM_PROMPT = Template("""## AI/Algorithm Compliance Analysis

### Context Documents
$context_documents

### Algorithm Details

**Name:** $algorithm_name
**Type:** $algorithm_type
**Inputs:** $inputs
**Outputs:** $outputs
**Training Data:** $training_data
**Bias Testing:** $bias_testing_status

**Decision Context:** $decision_context
**Affected Population:** $affected_population

### Analysis Request

Analyze this AI/algorithm for civil rights compliance:

1. **Input Analysis**:
   - Do any inputs directly encode protected class information?
   - Are there proxy variables in the inputs?
   - What data sources raise concerns?

2. **Output Analysis**:
   - What decisions does this algorithm influence?
   - Who is affected by these decisions?
   - Are outcomes measurable for disparate impact analysis?

3. **Training Data Concerns**:
   - Could historical bias be embedded in training data?
   - What groups might be underrepresented?
   - Has the training data been audited?

4. **Transparency and Explainability**:
   - Can decisions be explained to affected individuals?
   - Is there adequate human oversight?
   - Can individuals appeal algorithmic decisions?

5. **Testing and Validation**:
   - What bias testing has been or should be conducted?
   - What metrics should be monitored?
   - How often should the algorithm be audited?

6. **EEOC Guidance Compliance**:
   - Does this comply with EEOC guidance on algorithmic decision-making?
   - What documentation is required?

7. **Recommendations**:
   - Specific technical or process changes needed
   - Monitoring and audit requirements
   - Human oversight requirements

$additional_instructions""")


# =============================================================================
# Structured Output Prompts
# =============================================================================

STRUCTURED_RISK_ANALYSIS_PROMPT = Template("""## Compliance Risk Analysis

### Context Documents
$context_documents

### Feature Information
$feature_info

### Analysis Request

Analyze this feature and provide a structured risk assessment.

Respond with a JSON object matching this structure:
```json
{
  "analysis_summary": "Brief overall assessment",
  "confidence_score": 0.0 to 1.0,
  "risk_level": "critical|high|medium|low|minimal",
  "violations": [
    {
      "id": "unique identifier",
      "law_reference": "specific citation",
      "description": "what the violation is",
      "severity": "critical|high|medium|low",
      "confidence": 0.0 to 1.0,
      "affected_protected_classes": ["list of affected classes"]
    }
  ],
  "risk_factors": ["list of risk factors identified"],
  "mitigating_factors": ["list of factors that reduce risk"],
  "recommendations": [
    {
      "priority": 1,
      "action": "specific action to take",
      "rationale": "why this is recommended",
      "effort": "low|medium|high"
    }
  ],
  "cited_sources": ["list of law provisions cited"],
  "requires_human_review": true/false,
  "human_review_reason": "why human review is needed (if applicable)"
}
```

IMPORTANT: Respond ONLY with the JSON object. No additional text.""")


STRUCTURED_GUIDANCE_PROMPT = Template("""## Design Guidance Request

### Context Documents
$context_documents

### Design Question
$design_question

### Product Context
$product_context

Provide design guidance for civil rights compliance.

Respond with a JSON object matching this structure:
```json
{
  "guidance_summary": "Brief summary of key guidance",
  "applicable_laws": [
    {
      "law_id": "identifier",
      "law_name": "full name",
      "relevance": "why this law applies"
    }
  ],
  "design_recommendations": [
    {
      "area": "what aspect of design",
      "recommendation": "specific recommendation",
      "rationale": "legal/practical basis",
      "priority": "critical|high|medium|low"
    }
  ],
  "things_to_avoid": [
    {
      "practice": "what to avoid",
      "reason": "why it's problematic",
      "legal_risk": "what law it might violate"
    }
  ],
  "safe_harbors": [
    {
      "name": "safe harbor name",
      "description": "how to qualify",
      "benefits": "protection it provides"
    }
  ],
  "testing_recommendations": [
    "what testing should be done"
  ],
  "confidence_score": 0.0 to 1.0
}
```

IMPORTANT: Respond ONLY with the JSON object. No additional text.""")


# =============================================================================
# Helper Functions
# =============================================================================

def format_context_documents(documents: list[dict[str, Any]]) -> str:
    """Format context documents for inclusion in prompts."""
    if not documents:
        return "*No context documents provided*"

    parts = []
    for i, doc in enumerate(documents, 1):
        content = doc.get("content", "")
        metadata = doc.get("metadata", {})

        source = metadata.get("law_name", metadata.get("law_id", f"Document {i}"))
        section = metadata.get("section", "")
        chunk_type = metadata.get("chunk_type", "")

        header = f"### [{i}] {source}"
        if section:
            header += f" - {section}"
        if chunk_type:
            header += f" ({chunk_type})"

        parts.append(f"{header}\n{content}")

    return "\n\n".join(parts)


def format_data_fields(data_fields: list[dict[str, Any]]) -> str:
    """Format data fields for inclusion in prompts."""
    if not data_fields:
        return "*No data fields specified*"

    lines = []
    for field in data_fields:
        name = field.get("name", "Unknown")
        desc = field.get("description", "")
        used = "Yes" if field.get("used_in_decisions") else "No"
        proxy = field.get("potential_proxy", "")

        line = f"- **{name}**: {desc}"
        line += f" (Used in decisions: {used})"
        if proxy:
            line += f" [Potential proxy for: {proxy}]"
        lines.append(line)

    return "\n".join(lines)


def format_algorithm_details(algorithm: dict[str, Any] | None) -> str:
    """Format algorithm details for inclusion in prompts."""
    if not algorithm:
        return "*No algorithm specified*"

    lines = [
        f"- **Name:** {algorithm.get('name', 'Unknown')}",
        f"- **Type:** {algorithm.get('type', 'Unknown')}",
        f"- **Inputs:** {', '.join(algorithm.get('inputs', []))}",
        f"- **Outputs:** {', '.join(algorithm.get('outputs', []))}",
        f"- **Bias Testing Done:** {'Yes' if algorithm.get('bias_testing_done') else 'No'}",
    ]

    if training_desc := algorithm.get("training_data_description"):
        lines.append(f"- **Training Data:** {training_desc}")

    return "\n".join(lines)


def build_risk_analysis_prompt(
    feature: dict[str, Any],
    context_documents: list[dict[str, Any]],
    additional_instructions: str = "",
) -> str:
    """Build a complete risk analysis prompt."""
    return RISK_ANALYSIS_PROMPT.substitute(
        context_documents=format_context_documents(context_documents),
        feature_name=feature.get("name", "Unknown"),
        category=feature.get("category", "Unknown"),
        description=feature.get("description", ""),
        data_fields=format_data_fields(feature.get("data_fields", [])),
        algorithm_details=format_algorithm_details(feature.get("algorithm")),
        decision_impact=feature.get("decision_impact", "Not specified"),
        affected_population=feature.get("affected_population", "Not specified"),
        additional_instructions=additional_instructions,
    )


def build_disparate_impact_prompt(
    practice_description: str,
    context: str,
    affected_groups: list[str],
    context_documents: list[dict[str, Any]],
    additional_instructions: str = "",
) -> str:
    """Build a disparate impact analysis prompt."""
    return DISPARATE_IMPACT_PROMPT.substitute(
        context_documents=format_context_documents(context_documents),
        practice_description=practice_description,
        context=context,
        affected_groups=", ".join(affected_groups),
        additional_instructions=additional_instructions,
    )


def build_proxy_variable_prompt(
    variable_name: str,
    usage_description: str,
    decision_context: str,
    context_documents: list[dict[str, Any]],
    additional_instructions: str = "",
) -> str:
    """Build a proxy variable analysis prompt."""
    return PROXY_VARIABLE_PROMPT.substitute(
        context_documents=format_context_documents(context_documents),
        variable_name=variable_name,
        usage_description=usage_description,
        decision_context=decision_context,
        additional_instructions=additional_instructions,
    )


def build_ada_compliance_prompt(
    description: str,
    stage: str,
    feature_type: str,
    context_documents: list[dict[str, Any]],
    additional_instructions: str = "",
) -> str:
    """Build an ADA compliance analysis prompt."""
    return ADA_COMPLIANCE_PROMPT.substitute(
        context_documents=format_context_documents(context_documents),
        description=description,
        stage=stage,
        feature_type=feature_type,
        additional_instructions=additional_instructions,
    )


def build_ai_algorithm_prompt(
    algorithm: dict[str, Any],
    decision_context: str,
    affected_population: str,
    context_documents: list[dict[str, Any]],
    additional_instructions: str = "",
) -> str:
    """Build an AI/algorithm compliance analysis prompt."""
    return AI_ALGORITHM_PROMPT.substitute(
        context_documents=format_context_documents(context_documents),
        algorithm_name=algorithm.get("name", "Unknown"),
        algorithm_type=algorithm.get("type", "Unknown"),
        inputs=", ".join(algorithm.get("inputs", [])),
        outputs=", ".join(algorithm.get("outputs", [])),
        training_data=algorithm.get("training_data_description", "Not specified"),
        bias_testing_status="Completed" if algorithm.get("bias_testing_done") else "Not completed",
        decision_context=decision_context,
        affected_population=affected_population,
        additional_instructions=additional_instructions,
    )
