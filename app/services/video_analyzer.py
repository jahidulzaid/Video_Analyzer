import asyncio
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from app.config import Settings
from app.schemas.video import VideoAnalysisResponse
from app.utils.video import FrameSample, VideoMetadata, get_video_metadata, sample_video_frames_async


class VideoAnalyzer:
    """Handles frame extraction and GPT vision calls for video understanding."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)

    async def analyze(
        self,
        video_path: Path,
        instruction: Optional[str] = None,
        frame_samples: Optional[int] = None,
        seconds_per_frame: Optional[float] = None,
    ) -> VideoAnalysisResponse:
        metadata: VideoMetadata = await asyncio.to_thread(get_video_metadata, video_path)
        sample_target, effective_interval = self._choose_sample_count(
            metadata, frame_samples_override=frame_samples, interval_override=seconds_per_frame
        )

        frames: List[FrameSample] = await sample_video_frames_async(video_path, sample_target)
        if not frames:
            raise ValueError("Could not sample frames from the provided video.")

        user_instruction = instruction.strip() if instruction else (
            "Provide a rich, chronological explanation of the video. Summarize intent and outcome, list scene changes, "
            "key actions, subjects/objects, and notable visual cues. Reference timestamps when visible."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert video analyst. Use the provided frames to reconstruct the story, noting scene "
                    "transitions, actions, and visual details. Provide detail without inventing elements not visible."
                ),
            },
            {
                "role": "user",
                "content": self._build_user_content(user_instruction, frames),
            },
        ]

        completion = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.settings.openai_model,
            messages=messages,
            max_tokens=self.settings.max_tokens,
        )

        content = completion.choices[0].message.content if completion.choices else ""
        return VideoAnalysisResponse(
            summary=content or "",
            frames_used=len(frames),
            model=self.settings.openai_model,
            frame_timestamps=[f.timestamp_sec for f in frames if f.timestamp_sec is not None],
            prompt=user_instruction,
            total_frames=metadata.frame_count or None,
            video_duration_sec=metadata.duration_sec,
            sampling_interval_sec=effective_interval,
            requested_frame_samples=sample_target,
        )

    def _choose_sample_count(
        self,
        metadata: VideoMetadata,
        *,
        frame_samples_override: Optional[int] = None,
        interval_override: Optional[float] = None,
    ) -> tuple[int, float]:
        """
        Decide how many frames to sample based on video length.

        - At least `frame_samples`
        - Roughly one frame per `seconds_per_frame`
        - Clamped by `max_frame_samples` and total frames
        """
        baseline = max(frame_samples_override or self.settings.frame_samples, 1)
        interval = interval_override or self.settings.seconds_per_frame
        time_based = baseline
        if metadata.duration_sec:
            time_based = int(metadata.duration_sec / interval) + 1
        target = max(baseline, time_based)

        if metadata.frame_count > 0:
            target = min(target, metadata.frame_count)

        target = min(target, self.settings.max_frame_samples)
        return max(target, 1), interval

    @staticmethod
    def _build_user_content(instruction: str, frames: List[FrameSample]):
        content = [{"type": "text", "text": instruction}]
        for frame in frames:
            content.append({"type": "image_url", "image_url": {"url": frame.data_url}})
        return content
