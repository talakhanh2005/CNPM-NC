from pathlib import Path
from typing import List, Dict, Any
import cv2
import torch

from openemotion.detection.face_detector import FaceDetector
from openemotion.model.inference import predict_emotion


class VideoProcessor:
    """Xử lý video recording buổi học (5-10 phút, trích 1 frame/giây)."""

    def __init__(self, model: torch.nn.Module, detector: FaceDetector, device: torch.device):
        self.model = model
        self.detector = detector
        self.device = device

    def process_video(self, video_path: str, fps_sample: float = 1.0) -> Dict[str, Any]:
        """
        Đọc file video và phân tích cảm xúc theo từng giây.

        Returns:
            Dict chứa danh sách kết quả timestamped và thông tin video.
        """
        path = Path(video_path)
        if not path.exists():
            return {
                "status": "failed",
                "error_code": "INVALID_VIDEO",
                "message": f"Không tìm thấy file video tại: {video_path}",
            }

        valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        if path.suffix.lower() not in valid_extensions:
            return {
                "status": "failed",
                "error_code": "UNSUPPORTED_FORMAT",
                "message": f"Định dạng video {path.suffix} không được hỗ trợ.",
            }

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return {
                "status": "failed",
                "error_code": "INVALID_VIDEO",
                "message": "Không thể mở file video.",
            }

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            cap.release()
            return {
                "status": "failed",
                "error_code": "EMPTY_VIDEO",
                "message": "Video rỗng hoặc không có dữ liệu frame.",
            }

        duration = total_frames / fps
        frame_interval = max(1, int(round(fps / fps_sample)))

        raw_results: List[Dict[str, Any]] = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp = round(frame_idx / fps, 2)
                faces = self.detector.detect_faces(frame)

                if not faces:
                    raw_results.append({
                        "timestamp": timestamp,
                        "status": "no_face_detected",
                    })
                else:
                    for face_info in faces:
                        face_id = face_info["face_id"]
                        x, y, w, h = face_info["bbox"]
                        face_crop = frame[y:y + h, x:x + w]
                        if face_crop.size == 0:
                            continue

                        emotion, confidence, probs = predict_emotion(
                            self.model, face_crop, self.device
                        )

                        raw_results.append({
                            "timestamp": timestamp,
                            "face_id": face_id,
                            "bbox": [x, y, w, h],
                            "emotion": emotion,
                            "confidence": round(confidence, 4),
                            "probabilities": {k: round(v, 4) for k, v in probs.items()},
                            "uncertain": confidence < 0.4,
                        })

            frame_idx += 1

        cap.release()

        return {
            "status": "completed",
            "duration": round(duration, 2),
            "total_frames_processed": len(raw_results),
            "raw_results": raw_results,
        }
