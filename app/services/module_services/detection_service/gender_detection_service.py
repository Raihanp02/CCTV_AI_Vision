import logging
import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import Any, Dict, Optional
from pathlib import Path
from services.module_services.detection_service.base_detection import BaseDetection


logger = logging.getLogger(__name__)

GENDER_LABELS = ["Male", "Female"]


class GenderDetectionService(BaseDetection):
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        img_size: int = 260,
        mean: list[float] = [131.0912 / 255, 103.8827 / 255, 91.4953 / 255],
        std: list[float] | None = None,
    ):
        """
        Initialize Gender Detection Service with ONNX.

        Args:
            model_path: Path to ONNX model (default: models/gender.onnx)
            device: Device to use ('cuda', 'cpu', or 'auto')
            viewing_session_uuid: Optional viewing session UUID
        """

        self.img_size = img_size
        self.mean = mean
        self.std = std

        if model_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
            model_path = base_dir / "assets/models" / "gender.onnx"

        self.model_path = model_path
        self.ort_session = ort.InferenceSession(
                self.model_path
            )

    def detect(self, frame: list[np.ndarray], threshold: float = 0.6) -> Dict[str, Any]:
        input_tensor = self._preprocess(frame)

        scores = self.ort_session.run(None, {"input": input_tensor})[0]

        result = []
        for score in scores:
            dominant_idx = np.argmax(score)
            dominant_emotion = GENDER_LABELS[dominant_idx]

            exp_score = np.exp(score - np.max(score))
            probabilities = exp_score / exp_score.sum()

            confidence = float(probabilities[dominant_idx])

            if confidence <= threshold:
                continue
            
            result.append({
                "label": dominant_emotion,
                "confidence": round(confidence, 4),
            })

        return result

    def _preprocess(self, frames: list[np.ndarray]) -> np.ndarray:
        """
        frames: list[np.ndarray]  # BGR images
        mean, std: list or tuple of length 3 (RGB)
        """

        # Create blob (N, C, H, W)
        blob = cv2.dnn.blobFromImages(
            frames,
            scalefactor=1.0 / 255.0,
            size=(self.img_size, self.img_size),
            mean=self.mean,          # subtract mean
            swapRB=True,        # BGR -> RGB
            crop=False
        )

        # Divide by std (broadcasted)
        if self.std:
            for c in range(3):
                blob[:, c, :, :] /= self.std[c]

        return blob