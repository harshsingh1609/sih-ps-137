"""Vercel Serverless Function entry point for Q-Traffic CVRP Optimizer.

Exposes the FastAPI ASGI `app` instance with Vercel path-rewrite restoration.
"""

from __future__ import annotations
import sys
import json
from pathlib import Path

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure qtraffic directory is also in sys.path if present
QTRAFFIC_DIR = ROOT_DIR / "qtraffic"
if QTRAFFIC_DIR.is_dir() and str(QTRAFFIC_DIR) not in sys.path:
    sys.path.insert(0, str(QTRAFFIC_DIR))

from fastapi import Request
from app.main import app as fastapi_app


class VercelPathMiddleware:
    """ASGI Middleware to restore the original request path from Vercel rewrite headers."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") in ("http", "websocket"):
            headers = dict(scope.get("headers", []))
            raw_path = scope.get("path", "")

            # If debug requested, return immediate JSON of headers and scope
            if "debug" in raw_path or b"debug" in headers.get(b"x-matched-path", b""):
                response_data = json.dumps({
                    "scope_path": raw_path,
                    "headers": {k.decode("latin1"): v.decode("latin1") for k, v in headers.items()}
                }).encode("utf-8")
                await send({
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        [b"content-type", b"application/json"],
                        [b"content-length", str(len(response_data)).encode("ascii")],
                    ],
                })
                await send({
                    "type": "http.response.body",
                    "body": response_data,
                })
                return

            # Normalization: if path had /api stripped by function routing, restore it
            current_path = scope.get("path", "")
            if current_path in ("/city", "/solve", "/congestion", "/replan", "/benchmark"):
                scope["path"] = f"/api{current_path}"

        await self.app(scope, receive, send)


@fastapi_app.get("/api/debug")
async def debug_endpoint(request: Request):
    """Debug endpoint to verify request headers and path resolution on Vercel."""
    return {
        "scope_path": request.scope.get("path"),
        "raw_path": request.url.path,
        "matched_path": request.headers.get("x-matched-path"),
        "forwarded_uri": request.headers.get("x-forwarded-uri"),
    }


# Wrap FastAPI with the Vercel path restoration middleware
app = VercelPathMiddleware(fastapi_app)

__all__ = ["app"]
