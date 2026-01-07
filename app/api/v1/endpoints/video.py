from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.config import Settings, get_settings
from app.schemas import VideoAnalysisResponse
from app.services.video_analyzer import VideoAnalyzer
from app.utils.video import save_upload_to_temp

router = APIRouter()


@router.post(
    "/analyze",
    response_model=VideoAnalysisResponse,
    summary="Analyze a video with GPT vision",
    response_description="Natural language summary of the uploaded video.",
)
async def analyze_video(
    file: UploadFile = File(..., description="Video file to analyze."),
    instruction: Optional[str] = Form(
        default=None,
        description="Optional instruction or question for the model (e.g. 'List key scenes').",
    ),
    frame_samples: Optional[int] = Form(
        default=None,
        description="Override minimum number of frames to sample (higher = more detail).",
    ),
    seconds_per_frame: Optional[float] = Form(
        default=None,
        description="Override target interval between sampled frames in seconds.",
    ),
    settings: Settings = Depends(get_settings),
) -> VideoAnalysisResponse:
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OPENAI_API_KEY is not configured.",
        )

    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Please upload a valid video file.")

    if frame_samples is not None and frame_samples <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="frame_samples must be positive.")
    if seconds_per_frame is not None and seconds_per_frame <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="seconds_per_frame must be positive.")

    analyzer = VideoAnalyzer(settings)
    temp_path: Path = await save_upload_to_temp(file)

    try:
        return await analyzer.analyze(
            temp_path,
            instruction,
            frame_samples=frame_samples,
            seconds_per_frame=seconds_per_frame,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    finally:
        temp_path.unlink(missing_ok=True)
