# Equitas: Monetization Plan

## Business Model Overview

**Freemium SaaS** with usage-based pricing for enterprise features.

---

## Tier Structure

### Free Tier (Public)
**Price:** $0/month

| Feature | Limit |
|---------|-------|
| Risk assessments | 10/month |
| Categories | All (Hiring, Lending, Housing) |
| Risk score | Yes |
| Basic recommendations | Top 3 |
| Data field detection | Yes |
| Export reports | No |
| API access | No |
| Support | Community only |

**Purpose:** Lead generation, product validation, brand awareness

---

### Professional Tier
**Price:** $49/month (or $470/year - 20% discount)

| Feature | Limit |
|---------|-------|
| Risk assessments | 100/month |
| Categories | All |
| Risk score | Yes |
| Full recommendations | Unlimited |
| Data field detection | Yes |
| Executive summary | Yes |
| Export reports | PDF, JSON |
| API access | 500 calls/month |
| Historical tracking | 90 days |
| Support | Email (48hr response) |

**Target:** Freelance consultants, small startups, individual compliance officers

---

### Team Tier
**Price:** $199/month (or $1,910/year)

| Feature | Limit |
|---------|-------|
| Risk assessments | 500/month |
| Team members | Up to 10 |
| Categories | All |
| Full recommendations | Unlimited |
| Export reports | PDF, JSON, CSV |
| API access | 2,500 calls/month |
| Historical tracking | 1 year |
| Compliance dashboard | Yes |
| Custom rule sets | 5 |
| Slack integration | Yes |
| Support | Email (24hr response) |

**Target:** Growing startups, mid-size compliance teams, consultancies

---

### Enterprise Tier
**Price:** Custom ($500-$5,000+/month based on usage)

| Feature | Limit |
|---------|-------|
| Risk assessments | Unlimited |
| Team members | Unlimited |
| Categories | All + Custom |
| API access | Unlimited |
| Historical tracking | Unlimited |
| Custom rule sets | Unlimited |
| White-label option | Available |
| SSO/SAML | Yes |
| On-premise deployment | Available |
| Custom integrations | Yes |
| Dedicated support | Yes (SLA) |
| Compliance certifications | SOC 2 (roadmap) |

**Target:** Fortune 500, regulated industries, government contractors

---

## Feature Breakdown by Tier

### Assessment Features

| Feature | Free | Pro | Team | Enterprise |
|---------|------|-----|------|------------|
| Basic risk score | ✓ | ✓ | ✓ | ✓ |
| Confidence intervals | - | ✓ | ✓ | ✓ |
| Executive summary | - | ✓ | ✓ | ✓ |
| Detailed violations | - | ✓ | ✓ | ✓ |
| Law references | Basic | Full | Full | Full + Custom |
| Recommendations | Top 3 | All | All + Priority | All + Custom |
| Extracted fields | ✓ | ✓ | ✓ | ✓ |
| Proxy detection | ✓ | ✓ | ✓ | ✓ |

### Reporting & Export

| Feature | Free | Pro | Team | Enterprise |
|---------|------|-----|------|------------|
| On-screen results | ✓ | ✓ | ✓ | ✓ |
| PDF export | - | ✓ | ✓ | ✓ |
| JSON export | - | ✓ | ✓ | ✓ |
| CSV export | - | - | ✓ | ✓ |
| Branded reports | - | - | - | ✓ |
| Audit trail | - | - | ✓ | ✓ |

### Collaboration

| Feature | Free | Pro | Team | Enterprise |
|---------|------|-----|------|------------|
| Single user | ✓ | ✓ | - | - |
| Team seats | - | - | 10 | Unlimited |
| Role-based access | - | - | ✓ | ✓ |
| Comments/notes | - | - | ✓ | ✓ |
| Approval workflows | - | - | - | ✓ |

### Integration

| Feature | Free | Pro | Team | Enterprise |
|---------|------|-----|------|------------|
| Web app | ✓ | ✓ | ✓ | ✓ |
| REST API | - | ✓ | ✓ | ✓ |
| Slack | - | - | ✓ | ✓ |
| Jira | - | - | - | ✓ |
| GitHub/GitLab | - | - | - | ✓ |
| Custom webhooks | - | - | - | ✓ |

---

## Revenue Projections

### Year 1 Assumptions
- Free users: 1,000
- Pro conversion: 5% (50 users)
- Team conversion: 1% (10 teams)
- Enterprise: 2 contracts

### Year 1 Revenue
| Tier | Users | Monthly | Annual |
|------|-------|---------|--------|
| Pro | 50 | $2,450 | $29,400 |
| Team | 10 | $1,990 | $23,880 |
| Enterprise | 2 | $2,000 | $24,000 |
| **Total** | | **$6,440** | **$77,280** |

### Year 3 Projections (with growth)
| Tier | Users | Monthly | Annual |
|------|-------|---------|--------|
| Pro | 500 | $24,500 | $294,000 |
| Team | 100 | $19,900 | $238,800 |
| Enterprise | 20 | $40,000 | $480,000 |
| **Total** | | **$84,400** | **$1,012,800** |

---

## Pricing Psychology

### Why These Price Points?

**$49 Pro:**
- Below the "needs manager approval" threshold at most companies
- Comparable to other dev tools (GitHub Copilot: $19, Linear: $8/user)
- Low enough for individuals to expense

**$199 Team:**
- 4x Pro, but 10 users = value proposition
- Per-seat equivalent: ~$20/user/month
- Competitive with compliance tools

**$500+ Enterprise:**
- Custom pricing allows value-based negotiation
- Anchors against $500/hr legal fees
- ROI story: One avoided lawsuit > years of subscription

---

## Upsell Triggers

### Free → Pro
- Hit 10 assessment limit
- Try to export a report
- Request detailed recommendations
- Try API access

### Pro → Team
- Add second team member
- Hit 100 assessment limit
- Need historical tracking > 90 days
- Request Slack integration

### Team → Enterprise
- Hit 500 assessment limit
- Need SSO/SAML
- Request custom rules
- Require compliance certifications
- Need SLA guarantees

---

## Add-On Revenue Streams

### 1. Compliance Audit Reports ($500-$2,000)
- Deep-dive assessment of entire product
- Certified compliance report for investors/auditors
- Includes remediation roadmap

### 2. Custom Rule Development ($2,000-$10,000)
- Industry-specific rule sets
- State/country law additions
- Company policy integration

### 3. Training & Certification ($200-$500/person)
- "Certified Equitas Practitioner" program
- Online course + exam
- Annual recertification

### 4. Consulting Referrals (15% commission)
- Partner with employment lawyers
- Refer complex cases
- Revenue share on referred business

### 5. API Overage ($0.10/assessment)
- Pay-per-use beyond tier limits
- Enables burst usage without tier upgrade

---

## Go-To-Market Strategy

### Phase 1: Product-Led Growth (Months 1-6)
1. Free tier drives organic signups
2. Content marketing (blog, LinkedIn, Twitter)
3. SEO for "AI discrimination," "hiring bias," etc.
4. Developer community engagement

### Phase 2: Sales-Assisted (Months 6-12)
1. Hire first sales rep
2. Target Series A+ startups with AI products
3. Partner with HR tech platforms
4. Conference presence (HR Tech, LegalTech)

### Phase 3: Enterprise Focus (Year 2+)
1. Build enterprise sales team
2. SOC 2 certification
3. Strategic partnerships (Big 4 consulting)
4. Government/regulated industry focus

---

## Competitive Positioning

### vs. Law Firms
| Factor | Equitas | Law Firm |
|--------|---------|----------|
| Cost | $49-500/mo | $500+/hr |
| Speed | Instant | Days/weeks |
| Scalability | Unlimited | Limited |
| Integration | Dev workflow | Separate process |
| Availability | 24/7 | Business hours |

### vs. Bias Audit Firms
| Factor | Equitas | Audit Firm |
|--------|---------|------------|
| Cost | $49-500/mo | $10K-100K/audit |
| Frequency | Continuous | Annual |
| Scope | Pre-deployment | Post-deployment |
| Actionability | Immediate fixes | Report only |

### vs. DIY/Open Source
| Factor | Equitas | DIY |
|--------|---------|-----|
| Setup time | 0 | Weeks |
| Maintenance | None | Ongoing |
| Updates | Automatic | Manual |
| Support | Included | None |
| Liability | Shared | All yours |

---

## Key Metrics to Track

### North Star Metric
**Assessments Run** - Indicates product value delivery

### Funnel Metrics
1. **Visitors → Signup** (target: 10%)
2. **Signup → First Assessment** (target: 50%)
3. **Free → Paid Conversion** (target: 5%)
4. **Monthly Churn** (target: <5%)

### Revenue Metrics
1. **MRR** (Monthly Recurring Revenue)
2. **ARPU** (Average Revenue Per User)
3. **LTV** (Lifetime Value)
4. **CAC** (Customer Acquisition Cost)
5. **LTV:CAC Ratio** (target: >3:1)

---

## Risk Mitigation

### Legal Disclaimer
- Clear "not legal advice" disclaimer
- Recommend attorney review for high-risk findings
- Liability limitation in ToS

### Accuracy Concerns
- Confidence intervals communicate uncertainty
- Human review recommendations for edge cases
- Continuous rule updates as law evolves

### Competition
- First-mover advantage in niche
- Network effects from rule improvements
- Switching costs from historical data

---

## Summary

| Metric | Target |
|--------|--------|
| Year 1 ARR | $77K |
| Year 3 ARR | $1M+ |
| Free users | 1,000+ |
| Paid conversion | 5-6% |
| Enterprise contracts | 2-20 |
| Gross margin | 80%+ |
