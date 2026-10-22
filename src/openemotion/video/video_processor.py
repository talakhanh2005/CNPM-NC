import math
import time
from pathlib import Path
from threading import RLock
from typing import Any

import cv2
import torch

from openemotion.config import (
    LOW_EMOTION_CONFIDENCE,
    SUPPORTED_VIDEO_EXTENSIONS,
    VIDEO_SAMPLE_FPS,
)
from openemotion.detection.face_detector import FaceDetector
from openemotion.model.inference import predict_emotion


class VideoProcessor:
    """Sample decoded video at one frame per second, classifying each visible face."""

    def __init__(
        self,
        model: torch.nn.Module,
        detector: FaceDetector,
        device: torch.device,
        inference_lock=None,
        low_confidence_threshold: float = LOW_EMOTION_CONFIDENCE,
    ):
        self.model = model
        self.detector = detector
        self.device = device
        self.inference_lock = inference_lock or RLock()
        self.low_confidence_threshold = low_confidence_threshold

    def process_video(
        self,
        video_path: str,
        fps_sample: float = VIDEO_SAMPLE_FPS,
        deadline: float | None = None,
        max_duration: float | None = None,
    ) -> dict[str, Any]:
        path = Path(video_path)
        if not path.is_file():
            return {
                "status": "failed",
                "error_code": "INVALID_VIDEO",
                "message": "Video file not found.",
            }
        if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            return {
                "status": "failed",
                "error_code": "UNSUPPORTED_FORMAT",
                "message": "Unsupported video format.",
            }
        if not math.isfinite(fps_sample) or fps_sample <= 0:
            raise ValueError("fps_sample must be finite and positive")
        if max_duration is not None and (
            not math.isfinite(max_duration) or max_duration <= 0
        ):
            raise ValueError("max_duration must be finite and positive")
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                return {
                    "status": "failed",
                    "error_code": "INVALID_VIDEO",
                    "message": "Cannot open video.",
                }
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if not math.isfinite(fps) or fps <= 0:
                return {
                    "status": "failed",
                    "error_code": "EMPTY_VIDEO",
                    "message": "Invalid video metadata.",
                }
            # MediaRecorder WebM files often omit an indexed frame count. OpenCV
            # then reports a negative sentinel even though frames decode normally,
            # so only use the metadata duration when the count is trustworthy.
            metadata_duration = (
                total_frames / fps
                if math.isfinite(total_frames) and total_frames > 0
                else None
            )
            if max_duration is not None and (
                metadata_duration is not None and metadata_duration > max_duration
            ):
                return {
                    "status": "failed",
                    "error_code": "VIDEO_TOO_LONG",
                    "message": "Video duration exceeds the configured limit.",
                }
            frame_interval = max(1, round(fps / fps_sample))
            raw_results = []
            frame_idx = 0
            while cap.isOpened():
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError("Video inference deadline exceeded")
                ret, frame = cap.read()
                if not ret:
                    break
                if max_duration is not None and frame_idx / fps > max_duration:
                    return {
                        "status": "failed",
                        "error_code": "VIDEO_TOO_LONG",
                        "message": "Video duration exceeds the configured limit.",
                    }
                if frame_idx % frame_interval == 0:
                    timestamp = frame_idx / fps
                    with self.inference_lock:
                        raw_results.extend(
                            self._analyze_frame(frame, timestamp, deadline)
                        )
                frame_idx += 1
            if frame_idx == 0:
                return {
                    "status": "failed",
                    "error_code": "EMPTY_VIDEO",
                    "message": "No decodable video frames.",
                }
            duration = metadata_duration if metadata_duration is not None else frame_idx / fps
            return {
                "status": "completed",
                "duration": duration,
                "total_frames_processed": len(raw_results),
                "raw_results": raw_results,
            }
        finally:
            # Release file handles even on inference errors/timeouts (Windows).
            cap.release()

    def _analyze_frame(self, frame, timestamp, deadline):
        faces = self.detector.detect_faces(frame)
        if not faces:
            return [{"timestamp": timestamp, "status": "no_face_detected"}]
        results = []
        for face in faces:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("Video inference deadline exceeded")
            x, y, w, h = face["bbox"]
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            emotion, confidence, probabilities = predict_emotion(
                self.model, crop, self.device
            )
            results.append(
                {
                    "timestamp": timestamp,
                    "face_id": face["face_id"],
                    "bbox": [x, y, w, h],
                    "emotion": emotion,
                    "confidence": round(confidence, 4),
                    "probabilities": {
                        key: round(value, 4) for key, value in probabilities.items()
                    },
                    "uncertain": confidence < self.low_confidence_threshold,
                }
            )
        return results
