from typing import List, Optional

from pydantic import BaseModel


class VideoAnalysisResponse(BaseModel):
    summary: str
    frames_used: int
    model: str
    frame_timestamps: List[float] | None = None
    prompt: Optional[str] = None
    total_frames: Optional[int] = None
    video_duration_sec: Optional[float] = None
    sampling_interval_sec: Optional[float] = None
    requested_frame_samples: Optional[int] = None
