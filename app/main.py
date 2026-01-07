from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse

from app.api.v1.api import api_router
from app.config import Settings, get_settings


def get_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(title="Video Vision Analyzer", version="0.1.0")

    @app.get("/health", include_in_schema=False)
    async def health(settings: Settings = Depends(get_settings)) -> JSONResponse:
        if not settings.openai_api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
        return JSONResponse({"status": "ok"})

    app.include_router(api_router, prefix="/api/v1")
    return app


app = get_app()
