import asyncio
import base64
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2  # type: ignore
from fastapi import UploadFile


@dataclass
class FrameSample:
    """Represents a sampled frame that can be sent to the vision model."""

    index: int
    timestamp_sec: Optional[float]
    data_url: str


@dataclass
class VideoMetadata:
    """Basic video metadata used to derive sampling density."""

    frame_count: int
    fps: float
    duration_sec: Optional[float]


async def save_upload_to_temp(upload_file: UploadFile) -> Path:
    """Persist an uploaded file to a temporary location and return the path."""
    suffix = Path(upload_file.filename or "video").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
    await upload_file.seek(0)
    return Path(tmp.name)


def _frame_indices(total_frames: int, samples: int) -> List[int]:
    if total_frames <= 0 or samples <= 0:
        return []
    if samples >= total_frames:
        return list(range(total_frames))

    step = total_frames / samples
    return sorted({int(i * step) for i in range(samples)})


def _encode_frame(frame) -> str:
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise ValueError("Failed to encode frame.")
    b64_image = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_image}"


def get_video_metadata(video_path: Path) -> VideoMetadata:
    """Extract lightweight metadata to inform sampling density."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Unable to read the provided video file.")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        duration = (total_frames / fps) if fps > 0 else None
        return VideoMetadata(frame_count=total_frames, fps=fps, duration_sec=duration)
    finally:
        cap.release()


def sample_video_frames(
    video_path: Path, sample_count: int, *, indices: Optional[List[int]] = None
) -> List[FrameSample]:
    """Sample frames uniformly across a video and return data URLs."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Unable to read the provided video file.")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        indices = indices or _frame_indices(total_frames, sample_count)
        samples: List[FrameSample] = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if not success:
                continue
            timestamp = (idx / fps) if fps else None
            try:
                data_url = _encode_frame(frame)
            except ValueError:
                continue
            samples.append(FrameSample(index=idx, timestamp_sec=timestamp, data_url=data_url))

        return samples
    finally:
        cap.release()


async def sample_video_frames_async(
    video_path: Path, sample_count: int, *, indices: Optional[List[int]] = None
) -> List[FrameSample]:
    """Async wrapper to sample frames without blocking the event loop."""
    return await asyncio.to_thread(sample_video_frames, video_path, sample_count, indices=indices)
