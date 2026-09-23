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
    """ASGI Middleware to restore the original request path from Vercel rewrite headers.

    Vercel rewrites all /api/* traffic to /api/index.py (the handler file), but
    sets the original path in the `x-forwarded-uri` header. Without this middleware,
    FastAPI would see '/api/index.py' for every request and match nothing.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") in ("http", "websocket"):
            headers = dict(scope.get("headers", []))

            # x-forwarded-uri is set by Vercel to the original request URI
            # e.g. for a request to /api/city?foo=bar, this will be b"/api/city"
            forwarded_uri = headers.get(b"x-forwarded-uri", b"").decode("latin1")
            # Strip query string if present (path only)
            original_path = forwarded_uri.split("?")[0] if forwarded_uri else ""

            # Also check x-matched-path as secondary fallback
            matched_path = headers.get(b"x-matched-path", b"").decode("latin1").split("?")[0]

            # Determine which path to use
            if original_path and original_path not in ("/", ""):
                scope["path"] = original_path
            elif matched_path and matched_path not in ("/api/index.py", "/", ""):
                scope["path"] = matched_path
            # else: keep whatever scope["path"] already is

        await self.app(scope, receive, send)


@fastapi_app.get("/api/debug")
async def debug_endpoint(request: Request):
    """Debug endpoint to verify request headers and path resolution on Vercel."""
    return {
        "scope_path": request.scope.get("path"),
        "raw_path": request.url.path,
        "matched_path": request.headers.get("x-matched-path"),
        "forwarded_uri": request.headers.get("x-forwarded-uri"),
        "all_headers": dict(request.headers),
    }


# Wrap FastAPI with the Vercel path restoration middleware
app = VercelPathMiddleware(fastapi_app)

__all__ = ["app"]
