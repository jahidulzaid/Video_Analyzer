from fastapi import APIRouter

from app.api.v1.endpoints import video

api_router = APIRouter()
api_router.include_router(video.router, prefix="/video", tags=["video"])
