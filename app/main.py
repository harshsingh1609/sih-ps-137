"""FastAPI application entry point for SIH26137 Q-Traffic optimizer."""

from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from app.api.routes import router

app = FastAPI(
    title="Q-Traffic: Quantum-Inspired CVRP Optimizer",
    version="1.0.0",
    description="Quantum-behaved Particle Swarm Optimization for dynamic traffic route planning (SIH26137)",
)

# Enable CORS for local testing and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health():
    """System health check endpoint."""
    return {
        "status": "ok",
        "version": "1.0.0",
    }


# Include API routes
app.include_router(router)

# Mount static frontend directory — only in local dev, not on Vercel.
# On Vercel, the public/ directory is served by the platform as static files;
# mounting it here would conflict and intercept API routes.
import os as _os
_is_vercel = bool(_os.environ.get("VERCEL") or _os.environ.get("VERCEL_ENV"))

if not _is_vercel:
    STATIC_DIR = Path(__file__).resolve().parent / "static"
    if not (STATIC_DIR / "index.html").exists():
        PUBLIC_DIR = Path(__file__).resolve().parent.parent / "public"
        if (PUBLIC_DIR / "index.html").exists():
            STATIC_DIR = PUBLIC_DIR

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    if (STATIC_DIR / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled errors and return structured JSON response."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {str(exc)}"},
    )
