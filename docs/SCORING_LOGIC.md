# Equitas Scoring Logic

## Overview

Equitas uses a hybrid scoring system that combines:
1. **Rule Engine** (60% weight) - Deterministic rules for known violations
2. **LLM Analysis** (40% weight) - AI analysis for nuanced cases

---

## Risk Levels

| Score Range | Level | Description |
|-------------|-------|-------------|
| 80-100 | **CRITICAL** | Immediate legal action required |
| 60-79 | **HIGH** | Significant compliance risk |
| 40-59 | **MEDIUM** | Moderate concerns, review needed |
| 20-39 | **LOW** | Minor considerations |
| 0-19 | **MINIMAL** | Mostly compliant |

---

## Decision Tree

```
START
  │
  ▼
┌─────────────────────────────────────┐
│  EXTRACT DATA FIELDS FROM INPUT     │
│  - Protected classes (race, age...) │
│  - Proxy variables (zip, name...)   │
│  - Decision type (automated/human)  │
│  - Appeals process (yes/no)         │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│  EVALUATE EACH DATA FIELD           │
└─────────────────────────────────────┘
  │
  ├──► Protected Class Data?
  │    ├── YES + Used in Decisions → VIOLATION (90 pts)
  │    └── YES + Not in Decisions → VIOLATION (70 pts)
  │
  ├──► Proxy Variable?
  │    ├── YES + Used in Decisions → VIOLATION (75 pts)
  │    └── YES + Not in Decisions → POTENTIAL (55 pts)
  │
  ├──► ADA/Medical Inquiry?
  │    └── YES → VIOLATION (85 pts)
  │
  ├──► Missing Bias Testing?
  │    └── YES → VIOLATION (65 pts)
  │
  └──► High-Risk Algorithm?
       └── YES → VIOLATION (70 pts)
  │
  ▼
┌─────────────────────────────────────┐
│  CALCULATE RULE ENGINE SCORE        │
└─────────────────────────────────────┘
  │
  ├── Critical Violations (≥80)?
  │   └── YES → Use MAX score (Critical takes precedence)
  │
  ├── Multiple Violations?
  │   └── YES → Weighted average by confidence
  │
  ├── Potential Violations Only?
  │   └── YES → Weighted avg × 0.8 (lower confidence)
  │
  └── No Violations?
      └── Base score = 15 (minimal risk)
  │
  ▼
┌─────────────────────────────────────┐
│  LLM ANALYSIS (if warranted)        │
│  Triggered when:                    │
│  - Any rule has escalate_to_llm     │
│  - Any score ≥ 60                   │
│  - Potential violations exist       │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│  COMBINE SCORES                     │
│                                     │
│  Final = (Rule × 0.6) + (LLM × 0.4) │
│                                     │
│  * Weights adjust dynamically:      │
│  - Critical violation → Rule = 0.9  │
│  - Low LLM confidence → LLM = 0.2   │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│  DETERMINE RISK LEVEL               │
│  ≥80 = CRITICAL                     │
│  ≥60 = HIGH                         │
│  ≥40 = MEDIUM                       │
│  ≥20 = LOW                          │
│  <20 = MINIMAL                      │
└─────────────────────────────────────┘
  │
  ▼
END
```

---

## Violation Scores

### Protected Class Data (Direct Collection)

| Field | Used in Decisions | Score | Result |
|-------|-------------------|-------|--------|
| Race/Ethnicity | Yes | **90** | CRITICAL |
| Race/Ethnicity | No | 70 | HIGH |
| Sex/Gender | Yes | **90** | CRITICAL |
| Sex/Gender | No | 70 | HIGH |
| Age/DOB | Yes | **90** | CRITICAL |
| Age/DOB | No | 70 | HIGH |
| Religion | Yes | **90** | CRITICAL |
| Religion | No | 70 | HIGH |
| Disability | Yes | **90** | CRITICAL |
| Disability | No | 70 | HIGH |
| National Origin | Yes | **90** | CRITICAL |
| National Origin | No | 70 | HIGH |

### Proxy Variables

| Field | Proxy For | Used in Decisions | Score |
|-------|-----------|-------------------|-------|
| Zip Code | Race, National Origin | Yes | **75** |
| Zip Code | Race, National Origin | No | 55 |
| Name | National Origin, Sex | Yes | **75** |
| Name | National Origin, Sex | No | 55 |
| Credit Score | Race (disputed) | Yes | **75** |
| Credit Score | Race (disputed) | No | 55 |
| Criminal History | Race | Yes | **75** |
| Criminal History | Race | No | 55 |
| Photo/Video | Race, Age, Disability | Yes | **75** |
| Photo/Video | Race, Age, Disability | No | 55 |
| School/University | Race, Socioeconomic | Yes | **75** |
| School/University | Race, Socioeconomic | No | 55 |
| Employment Gaps | Sex (caregiving), Disability | Yes | **75** |
| Employment Gaps | Sex (caregiving), Disability | No | 55 |

### Process Violations

| Violation | Score | Notes |
|-----------|-------|-------|
| ADA/Medical Inquiry (pre-offer) | **85** | Medical questions before job offer |
| Missing Bias Testing | 65 | No documented fairness audit |
| High-Risk Algorithm | 70 | Automated decisions w/ risky inputs |
| No Human Review | +15 | Added to LLM base score |
| No Appeals Process | Concern flag | Flagged in human review |

---

## Score Combination Formula

### Step 1: Calculate Rule Engine Score

```
IF critical_violations exist (score ≥ 80):
    rule_score = MAX(critical_violation_scores)

ELIF violations exist:
    rule_score = Σ(score × confidence) / Σ(confidence)

ELIF potential_violations exist:
    rule_score = [Σ(score × confidence) / Σ(confidence)] × 0.8

ELSE:
    rule_score = 15  # Base minimal risk
```

### Step 2: Calculate LLM Score

```
llm_score = 30  # Base score

# Add risk factors (max +40)
llm_score += MIN(risk_factor_count × 10, 40)

# Subtract mitigating factors (max -25)
llm_score -= MIN(mitigating_factor_count × 8, 25)

# Add human review flag
IF requires_human_review:
    llm_score += 15
```

### Step 3: Adjust Weights

```
DEFAULT:
    rule_weight = 0.60
    llm_weight = 0.40

IF critical_violation:
    rule_weight = MIN(0.90, rule_weight + 0.20)
    llm_weight = MAX(0.10, llm_weight - 0.20)

IF llm_confidence < 0.50:
    llm_weight = llm_weight × 0.50
```

### Step 4: Combine

```
total_weight = rule_weight + llm_weight

final_score = (rule_score × rule_confidence × rule_weight +
               llm_score × llm_confidence × llm_weight) / total_weight

final_score = CLAMP(final_score, 0, 100)
```

---

## Human Review Triggers

Human review is recommended when:

| Condition | Threshold |
|-----------|-----------|
| Low confidence | < 50% average |
| High risk score | ≥ 60 |
| Escalated rules | Any (if score ≥ 40) |
| LLM recommends review | If score ≥ 40 |
| Conflicting signals | Rule vs LLM diff > 40 |

---

## Example Calculations

### Example 1: HIGH RISK Hiring System

**Input:**
- Race: Used in decisions
- Age: Used in decisions
- Zip Code: Used in decisions
- Name: Used in decisions
- Fully automated, no human review
- No appeals

**Violations Detected:**
| Field | Type | Score |
|-------|------|-------|
| Race | Protected Class | 90 |
| Age | Protected Class | 90 |
| Zip Code | Proxy | 75 |
| Name | Proxy | 75 |

**Calculation:**
```
Critical violations exist (90, 90)
rule_score = MAX(90, 90) = 90
rule_confidence = 0.95

llm_score = 30 + 40 (4 risk factors) + 15 (no human review) = 85
llm_confidence = 0.80

Weights (critical violation adjustment):
  rule_weight = 0.80 (was 0.60 + 0.20)
  llm_weight = 0.20 (was 0.40 - 0.20)

final = (90 × 0.95 × 0.80 + 85 × 0.80 × 0.20) / (0.80 + 0.20)
      = (68.4 + 13.6) / 1.0
      = 82

RESULT: 82 = CRITICAL
```

### Example 2: LOW RISK System

**Input:**
- No protected class data
- No proxy variables
- Human reviews all decisions
- Appeals available

**Violations Detected:**
None

**Calculation:**
```
No violations
rule_score = 15 (base minimal)
rule_confidence = 1.0

llm_score = 30 - 16 (2 mitigating factors) = 14
llm_confidence = 0.90

Weights (no adjustment):
  rule_weight = 0.60
  llm_weight = 0.40

final = (15 × 1.0 × 0.60 + 14 × 0.90 × 0.40) / (0.60 + 0.40)
      = (9.0 + 5.04) / 1.0
      = 14.04

RESULT: 14 = MINIMAL
```

---

## Applicable Laws by Category

### Hiring (Employment)
- Title VII of the Civil Rights Act
- Americans with Disabilities Act (ADA)
- Age Discrimination in Employment Act (ADEA)
- NYC Local Law 144 (AI in hiring)

### Lending
- Equal Credit Opportunity Act (ECOA)
- Fair Credit Reporting Act (FCRA)
- Regulation B

### Housing
- Fair Housing Act (FHA)
- ECOA (for housing-related credit)

### Insurance
- State Insurance Regulations
- FCRA

---

## Confidence Scoring

| Confidence Level | Description | Impact |
|------------------|-------------|--------|
| ≥ 85% | High | Full weight applied |
| 50-84% | Medium | Normal weight |
| < 50% | Low | Weight reduced by 50%, human review triggered |

---

## Key Design Principles

1. **Critical violations override** - A single 90-point violation will produce a CRITICAL score regardless of other factors

2. **Rule engine prioritized** - Deterministic rules (60%) weighted higher than LLM analysis (40%)

3. **Protected classes are highest risk** - Direct collection of race, age, gender, etc. scores 90 when used in decisions

4. **Proxy variables are serious** - Zip code, name, etc. score 75 when used in decisions

5. **Human review safety net** - Triggered for uncertain, high-risk, or conflicting assessments

6. **Confidence-weighted** - Low confidence scores are discounted in the final calculation
