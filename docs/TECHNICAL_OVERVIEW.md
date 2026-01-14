# Equitas: Technical Overview (Plain English)

## How Equitas Works - The Simple Version

Think of Equitas as a **spell-checker for discrimination**. Just like spell-check highlights misspelled words, Equitas highlights features in your software that could lead to discrimination lawsuits.

---

## The Three-Layer System

### Layer 1: The Rule Engine (The Checklist)

This is like a very thorough checklist that a civil rights lawyer would use.

**What it does:**
- Looks for "red flag" words in your feature description
- Checks if you're collecting data about race, age, gender, etc.
- Identifies "proxy variables" - things that seem innocent but correlate with protected groups

**Example:**
```
You type: "Our hiring tool uses zip code to filter candidates"

Rule Engine thinks:
"Zip code" → This correlates with race (redlining history)
"Filter candidates" → This is making employment decisions
"Hiring" → Employment law applies (Title VII)

Result: HIGH RISK - Proxy variable used in employment decisions
```

### Layer 2: The AI Analysis (The Second Opinion)

This is like having an AI assistant that reads your description and thinks about it more deeply.

**What it does:**
- Understands context and nuance
- Catches things the checklist might miss
- Provides explanations in plain English

**Example:**
```
You type: "We use facial recognition to verify identity"

AI thinks:
"Facial recognition has known accuracy problems for darker skin tones"
"This could cause disparate impact against Black applicants"
"Similar systems have been sued (EEOC v. [Company])"

Result: Adds context about real-world discrimination risks
```

### Layer 3: The Scoring Engine (The Calculator)

This combines the checklist findings and AI analysis into a single score.

**How it works:**
```
Rule Engine Score (60% of final) + AI Score (40% of final) = Final Score

0-19  = MINIMAL (Green - You're probably fine)
20-39 = LOW (Yellow - Minor concerns)
40-59 = MEDIUM (Orange - Should review)
60-79 = HIGH (Red - Significant risk)
80-100 = CRITICAL (Dark Red - Stop and fix now)
```

---

## What We Look For

### Protected Classes (Direct Discrimination)
These are characteristics you **cannot** use to make decisions about people:

| Protected Class | Laws That Apply |
|-----------------|-----------------|
| Race/Ethnicity | Title VII, FHA, ECOA |
| Sex/Gender | Title VII, ECOA |
| Age (40+) | ADEA |
| Religion | Title VII |
| Disability | ADA |
| National Origin | Title VII |
| Pregnancy | PDA |
| Genetic Information | GINA |

### Proxy Variables (Indirect Discrimination)
These seem neutral but often correlate with protected classes:

| Proxy Variable | Often Correlates With | Why It's Risky |
|----------------|----------------------|----------------|
| **Zip Code** | Race, National Origin | Residential segregation history |
| **Name** | Race, National Origin, Sex | Ethnic/gender patterns in names |
| **Credit Score** | Race | Historical lending discrimination |
| **Criminal History** | Race | Disparate enforcement patterns |
| **Employment Gaps** | Sex, Disability | Caregiving, medical leave |
| **School Name** | Race, Socioeconomic | Educational segregation |
| **Commute Distance** | Race | Residential patterns |

---

## The Technology Stack (What It's Built With)

### Frontend (What You See)
- **Single HTML file** - The entire user interface
- **Vanilla JavaScript** - No frameworks, fast loading
- **CSS** - Dark theme, responsive design
- **Hosted on Vercel** - Free, fast, global

### Backend (The Brains)
- **Python + FastAPI** - The server that does the analysis
- **Rule Engine** - YAML files with discrimination rules
- **Claude AI** - Anthropic's AI for nuanced analysis
- **Hosted on Render** - Cloud server

### How They Talk
```
┌─────────────┐         ┌─────────────┐
│   Browser   │  HTTP   │   Server    │
│  (Vercel)   │ ◄─────► │  (Render)   │
│             │  JSON   │             │
└─────────────┘         └─────────────┘
     │                        │
     │                        ├── Rule Engine
     │                        ├── AI Analysis
     │                        └── Scoring
     │
     └── Displays results
```

---

## How We Score Violations

### The Math (Simplified)

**Step 1: Find violations**
```
Each violation has a score:
- Collecting race for decisions = 90 points (CRITICAL)
- Using zip code for decisions = 75 points (HIGH)
- Missing bias testing = 65 points (HIGH)
```

**Step 2: Combine scores**
```
If multiple violations, we use the highest one
(One critical violation = critical overall)
```

**Step 3: Add AI input**
```
AI adds context:
- More risk factors found = score goes up
- Mitigating factors found = score goes down
```

**Step 4: Final score**
```
60% × Rule Score + 40% × AI Score = Final Score
```

---

## Why Two Systems? (Rules + AI)

| Rules Alone | AI Alone | Rules + AI Together |
|-------------|----------|---------------------|
| Fast but rigid | Flexible but slow | Fast AND flexible |
| Catches known patterns | Catches new patterns | Catches everything |
| 100% consistent | Can hallucinate | AI validates rules |
| No explanations | Great explanations | Best of both |

---

## Data Flow Example

**You enter:**
> "Hiring tool that screens resumes using zip code, name, and employment history. Decisions are automatic with no human review."

**System processes:**

```
1. TEXT EXTRACTION
   Found: zip_code, name, employment_history
   Decision type: automated
   Human review: none

2. RULE ENGINE
   zip_code → PROXY for race (75 pts)
   name → PROXY for national_origin (75 pts)
   employment_history → PROXY for sex/disability (55 pts)
   automated + no review → HIGH RISK factor

3. AI ANALYSIS
   "This system has multiple discrimination vectors..."
   Risk factors: 4
   Mitigating factors: 0
   Recommends human review: YES

4. SCORING
   Rule score: 75 (highest violation)
   AI score: 85 (many risk factors)
   Combined: 75×0.6 + 85×0.4 = 79

5. RESULT
   Score: 79
   Level: HIGH
   Human review: REQUIRED
```

---

## Security & Privacy

- **No data stored** - We don't save your feature descriptions
- **No training on your data** - Your input doesn't train our AI
- **HTTPS everywhere** - All communication encrypted
- **No accounts required** - Use anonymously

---

## Limitations (What We Don't Do)

1. **Not legal advice** - We're a screening tool, not a law firm
2. **Not 100% accurate** - Some edge cases may be missed
3. **US law focused** - Currently covers US federal civil rights laws
4. **Not real-time monitoring** - Point-in-time assessment only

---

## Future Roadmap

- [ ] EU AI Act compliance checking
- [ ] State-specific laws (California, Illinois, NYC)
- [ ] CI/CD integration (GitHub Actions, etc.)
- [ ] API for programmatic access
- [ ] Batch assessment for multiple features
- [ ] Historical tracking and trend analysis
