#!/usr/bin/env python3
"""Startup script for cloud deployment."""
import os
import sys

# Add src/ to Python path so wenah module can be found
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_path)

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Wenah API on port {port}")
    print(f"Python path: {sys.path}")
    uvicorn.run(
        "wenah.api.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
