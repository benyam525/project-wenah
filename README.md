# Wenah - Civil Rights Compliance Framework

A Python-based ML framework that helps companies build responsible products by evaluating them against federal and state civil rights laws.

## Features

- **Risk Assessment Dashboard** - Evaluate products for civil rights compliance
- **Hybrid Analysis** - Combines deterministic rule engine with LLM-powered analysis
- **Multiple Categories** - Employment (Title VII, ADA), Housing (FHA), Consumer (ECOA, FCRA)
- **Design Guidance** - Get compliance recommendations during product development
- **Pre-launch Checks** - Verify compliance before going live

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run the API server
uvicorn wenah.api.main:app --reload

# Open http://localhost:8000/docs for API documentation
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/assess/quick` | Quick compliance assessment |
| `POST /api/v1/assess/risk` | Full risk assessment with LLM analysis |
| `POST /api/v1/guidance/design` | Get design-phase guidance |
| `POST /api/v1/check/prelaunch` | Pre-launch compliance check |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              Compliance Engine (Orchestrator)                │
│  ┌──────────────────┐       ┌──────────────────────────┐    │
│  │   Rule Engine    │       │   LLM Pipeline (RAG)     │    │
│  │  Deterministic   │──────▶│  Claude Analysis         │    │
│  │  Confidence: 1.0 │       │  + Guardrails            │    │
│  └──────────────────┘       └──────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

## Live Demo

- **Frontend Dashboard**: https://wenah-dashboard.vercel.app
- **API Documentation**: Available at `/docs` when running locally

## License

MIT
