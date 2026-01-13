"""
FastAPI application and routes.

Provides the main API application and factory function for testing.
"""

from wenah.api.main import app, create_app

__all__ = [
    "app",
    "create_app",
]
