"""
Wenah API - Civil Rights Compliance Assessment API.

Main FastAPI application providing endpoints for:
- Risk Assessment
- Design Guidance
- Pre-launch Compliance Checks
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from wenah.api.routes import assess_router, guidance_router, check_router
from wenah.api.schemas import (
    HealthResponse,
    DetailedHealthResponse,
    ComponentHealth,
    ReadinessResponse,
    LivenessResponse,
    MetricsResponse,
    APIInfoResponse,
    ErrorResponse,
    ValidationErrorResponse,
)
from wenah.config import Settings


# =============================================================================
# Application State for Metrics
# =============================================================================

class AppMetrics:
    """Simple in-memory metrics collector."""

    def __init__(self):
        self.start_time: datetime = datetime.now(timezone.utc)
        self.requests_total: int = 0
        self.requests_by_endpoint: dict[str, int] = {}
        self.response_times: list[float] = []
        self.errors: int = 0

    @property
    def uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()

    @property
    def average_response_time_ms(self) -> float:
        """Get average response time in milliseconds."""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)

    @property
    def error_rate(self) -> float:
        """Get error rate as a fraction."""
        if self.requests_total == 0:
            return 0.0
        return self.errors / self.requests_total


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    # Startup
    print("Starting Wenah Compliance API...")

    # Initialize metrics
    app.state.metrics = AppMetrics()

    # Initialize services (lazy loading is handled by singletons)
    # This is where you could pre-load vector stores, warm up models, etc.

    yield

    # Shutdown
    print("Shutting down Wenah Compliance API...")


# =============================================================================
# Application Factory
# =============================================================================

def create_app(settings: Settings | None = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        settings: Optional settings override for testing

    Returns:
        Configured FastAPI application
    """
    if settings is None:
        settings = Settings()

    app = FastAPI(
        title="Wenah Civil Rights Compliance API",
        description="""
# Wenah Compliance API

A comprehensive API for assessing civil rights compliance risks in software products,
particularly those involving AI/ML, automated decision-making, and data collection.

## Features

### Risk Assessment
Evaluate products against federal civil rights laws including:
- Title VII (Employment Discrimination)
- Americans with Disabilities Act (ADA)
- Fair Housing Act (FHA)
- Equal Credit Opportunity Act (ECOA)

### Design Guidance
Get proactive compliance guidance during product design:
- Protected class detection
- Proxy variable identification
- Algorithm design best practices
- Data collection recommendations

### Pre-launch Compliance
Verify compliance before product launch:
- Comprehensive compliance checklists
- Documentation requirements
- Sign-off tracking
- Monitoring recommendations

## Quick Start

1. Use `/assess/quick` for fast risk screening
2. Use `/guidance/field` to check individual data fields
3. Use `/check/prelaunch` for full compliance verification before launch

## Authentication

Currently no authentication is required. API rate limiting may apply.
        """,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Store settings in app state
    app.state.settings = settings

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(assess_router, prefix="/api/v1")
    app.include_router(guidance_router, prefix="/api/v1")
    app.include_router(check_router, prefix="/api/v1")

    # Register exception handlers
    register_exception_handlers(app)

    # Register health and info endpoints
    register_health_endpoints(app)

    return app


# =============================================================================
# Exception Handlers
# =============================================================================

def register_exception_handlers(app: FastAPI) -> None:
    """Register custom exception handlers."""

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle request validation errors."""
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation Error",
                "detail": exc.errors(),
                "code": "VALIDATION_ERROR",
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle unexpected exceptions."""
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "detail": str(exc),
                "code": "INTERNAL_ERROR",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


# =============================================================================
# Health and Info Endpoints
# =============================================================================

def register_health_endpoints(app: FastAPI) -> None:
    """Register health check and API info endpoints."""

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Health Check",
        description="Check if the API is running and healthy.",
    )
    async def health_check() -> HealthResponse:
        """Return API health status."""
        return HealthResponse(
            status="healthy",
            version="0.1.0",
            timestamp=datetime.now(timezone.utc),
        )

    @app.get(
        "/health/detailed",
        response_model=DetailedHealthResponse,
        tags=["Health"],
        summary="Detailed Health Check",
        description="Get detailed health status including component checks.",
    )
    async def detailed_health_check(request: Request) -> DetailedHealthResponse:
        """Return detailed API health status with component checks."""
        import time

        components: list[ComponentHealth] = []
        checks_passed = 0
        checks_failed = 0

        # Check rule engine
        try:
            start = time.time()
            from wenah.rules.rule_engine import get_rule_engine
            engine = get_rule_engine()
            latency = (time.time() - start) * 1000
            components.append(ComponentHealth(
                name="rule_engine",
                status="healthy",
                latency_ms=latency,
            ))
            checks_passed += 1
        except Exception as e:
            components.append(ComponentHealth(
                name="rule_engine",
                status="unhealthy",
                message=str(e),
            ))
            checks_failed += 1

        # Check scoring engine
        try:
            start = time.time()
            from wenah.core.scoring import get_scoring_engine
            engine = get_scoring_engine()
            latency = (time.time() - start) * 1000
            components.append(ComponentHealth(
                name="scoring_engine",
                status="healthy",
                latency_ms=latency,
            ))
            checks_passed += 1
        except Exception as e:
            components.append(ComponentHealth(
                name="scoring_engine",
                status="unhealthy",
                message=str(e),
            ))
            checks_failed += 1

        # Check design guidance
        try:
            start = time.time()
            from wenah.use_cases.design_guidance import get_design_guidance
            guidance = get_design_guidance()
            latency = (time.time() - start) * 1000
            components.append(ComponentHealth(
                name="design_guidance",
                status="healthy",
                latency_ms=latency,
            ))
            checks_passed += 1
        except Exception as e:
            components.append(ComponentHealth(
                name="design_guidance",
                status="unhealthy",
                message=str(e),
            ))
            checks_failed += 1

        # Determine overall status
        if checks_failed == 0:
            overall_status = "healthy"
        elif checks_passed > 0:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        metrics: AppMetrics = request.app.state.metrics

        return DetailedHealthResponse(
            status=overall_status,
            version="0.1.0",
            timestamp=datetime.now(timezone.utc),
            uptime_seconds=metrics.uptime_seconds,
            components=components,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )

    @app.get(
        "/health/live",
        response_model=LivenessResponse,
        tags=["Health"],
        summary="Liveness Probe",
        description="Kubernetes liveness probe - checks if the process is alive.",
    )
    async def liveness_probe() -> LivenessResponse:
        """Return liveness status for Kubernetes."""
        return LivenessResponse(alive=True)

    @app.get(
        "/health/ready",
        response_model=ReadinessResponse,
        tags=["Health"],
        summary="Readiness Probe",
        description="Kubernetes readiness probe - checks if the service is ready to accept traffic.",
    )
    async def readiness_probe(request: Request) -> ReadinessResponse:
        """Return readiness status for Kubernetes."""
        try:
            # Check that core services are available
            from wenah.rules.rule_engine import get_rule_engine
            from wenah.core.scoring import get_scoring_engine

            get_rule_engine()
            get_scoring_engine()

            return ReadinessResponse(ready=True)
        except Exception as e:
            return ReadinessResponse(ready=False, reason=str(e))

    @app.get(
        "/metrics",
        response_model=MetricsResponse,
        tags=["Monitoring"],
        summary="Basic Metrics",
        description="Get basic application metrics.",
    )
    async def get_metrics(request: Request) -> MetricsResponse:
        """Return application metrics."""
        metrics: AppMetrics = request.app.state.metrics

        return MetricsResponse(
            uptime_seconds=metrics.uptime_seconds,
            requests_total=metrics.requests_total,
            requests_by_endpoint=metrics.requests_by_endpoint,
            average_response_time_ms=metrics.average_response_time_ms,
            error_rate=metrics.error_rate,
        )

    @app.get(
        "/",
        response_model=APIInfoResponse,
        tags=["Info"],
        summary="API Information",
        description="Get basic information about the API.",
    )
    async def api_info() -> APIInfoResponse:
        """Return API information."""
        return APIInfoResponse(
            name="Wenah Civil Rights Compliance API",
            version="0.1.0",
            description="API for assessing civil rights compliance in software products",
            documentation_url="/docs",
            supported_categories=[
                "hiring",
                "lending",
                "housing",
                "insurance",
                "general",
            ],
            supported_laws=[
                "Title VII of the Civil Rights Act",
                "Americans with Disabilities Act (ADA)",
                "Age Discrimination in Employment Act (ADEA)",
                "Fair Housing Act (FHA)",
                "Equal Credit Opportunity Act (ECOA)",
                "Fair Credit Reporting Act (FCRA)",
            ],
        )

    @app.get(
        "/api/v1",
        tags=["Info"],
        summary="API v1 Root",
        description="Get API v1 endpoint listing.",
    )
    async def api_v1_root() -> dict[str, Any]:
        """Return API v1 endpoint listing."""
        return {
            "version": "v1",
            "endpoints": {
                "assessment": {
                    "full": "POST /api/v1/assess/risk",
                    "quick": "POST /api/v1/assess/quick",
                    "feature": "POST /api/v1/assess/feature",
                    "feature_quick": "POST /api/v1/assess/feature/quick",
                },
                "guidance": {
                    "design": "POST /api/v1/guidance/design",
                    "field": "POST /api/v1/guidance/field",
                    "algorithm": "POST /api/v1/guidance/algorithm",
                    "checklist": "GET /api/v1/guidance/checklist/{category}",
                    "protected_classes": "GET /api/v1/guidance/protected-classes",
                    "proxy_variables": "GET /api/v1/guidance/proxy-variables",
                },
                "prelaunch": {
                    "full": "POST /api/v1/check/prelaunch",
                    "quick": "POST /api/v1/check/quick",
                    "requirements": "GET /api/v1/check/requirements",
                    "checklist": "GET /api/v1/check/checklist",
                    "sign_offs": "GET /api/v1/check/sign-offs",
                    "monitoring": "GET /api/v1/check/monitoring",
                },
            },
            "documentation": "/docs",
        }


# =============================================================================
# Application Instance
# =============================================================================

# Create default application instance
app = create_app()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> None:
    """Run the API server using uvicorn."""
    import uvicorn

    settings = Settings()

    uvicorn.run(
        "wenah.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info",
    )


if __name__ == "__main__":
    main()
