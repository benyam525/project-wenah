"""
API routes package.

Exports all route routers for inclusion in the main application.
"""

from wenah.api.routes.assess import router as assess_router
from wenah.api.routes.check import router as check_router
from wenah.api.routes.guidance import router as guidance_router

__all__ = [
    "assess_router",
    "guidance_router",
    "check_router",
]
