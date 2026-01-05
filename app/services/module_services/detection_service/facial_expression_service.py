import logging
import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import Any, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

HSEMOTION_EMOTIONS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}

STANDARD_EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


class FacialExpressionService:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Initialize Facial Expression Recognizer with ONNX.

        Args:
            model_path: Path to ONNX model (default: models/emotion.onnx)
            device: Device to use ('cuda', 'cpu', or 'auto')
            viewing_session_uuid: Optional viewing session UUID
        """

        self.img_size = 260
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        if model_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            model_path = base_dir / "assets/models" / "emotion.onnx"

        self.model_path = model_path
        self.ort_session = ort.InferenceSession(
                self.model_path
            )

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect expression using ONNX Runtime.

        Args:
            face_image: Face image (BGR format)

        Returns:
            Expression detection result
        """
        try:

            input_tensor = self._preprocess(frame)

            scores = self.ort_session.run(None, {"input": input_tensor})[0][0]

            dominant_idx = np.argmax(scores)
            dominant_emotion = HSEMOTION_EMOTIONS[dominant_idx]

            exp_scores = np.exp(scores - np.max(scores))
            probabilities = exp_scores / exp_scores.sum()

            confidence = float(probabilities[dominant_idx])

            if confidence <= 0.4:
                return None
            
            result_dict = {
                "label": dominant_emotion,
                "confidence": round(confidence, 4),
            }

            return result_dict

        except Exception as e:
            print(f"ONNX detection error: {e}")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for ONNX model.

        Args:
            face_image: Face image (BGR format from OpenCV)

        Returns:
            Preprocessed image ready for ONNX inference
        """

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(rgb_image, (self.img_size, self.img_size))

        normalized = resized.astype(np.float32) / 255.0

        for i in range(3):
            normalized[..., i] = (normalized[..., i] - self.mean[i]) / self.std[i]

        preprocessed = normalized.transpose(2, 0, 1).astype(np.float32)[np.newaxis, ...]

        return preprocessed
