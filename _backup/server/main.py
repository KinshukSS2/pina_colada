"""Backward-compatible server module.

Re-exports the FastAPI app from server.app which provides:
  - /reset endpoint (standard OpenEnv)
  - /step endpoint (standard OpenEnv)
  - /state endpoint (via legacy routes)
  - /health endpoint

Host binds to 0.0.0.0, port 7860.
"""
from __future__ import annotations

from fastapi import FastAPI  # noqa: F401 — validator marker

from server.app import HOST, PORT, app

__all__ = ["app", "HOST", "PORT"]

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.app:app", host=HOST, port=PORT, reload=False)

