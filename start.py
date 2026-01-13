#!/usr/bin/env python3
"""Startup script for Railway deployment."""
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Wenah API on port {port}")
    uvicorn.run(
        "wenah.api.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
