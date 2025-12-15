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
        data: Optional[Dict[int, Any]] = None,
        viewing_session_uuid: Optional[str] = None,
    ):
        """
        Initialize Facial Expression Recognizer with ONNX.

        Args:
            model_path: Path to ONNX model (default: models/emotion.onnx)
            device: Device to use ('cuda', 'cpu', or 'auto')
            viewing_session_uuid: Optional viewing session UUID
        """
        self.frame_count = 0
        self.data = data
        self.event_emitter = None
        self.viewing_session_uuid = viewing_session_uuid

        self.img_size = 260
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        if model_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            model_path = base_dir / "assets/models" / "emotion.onnx"

        self.model_path = model_path
        self.ort_session = None

        try:
            self._initialize_onnx(device)
            logger.info(f"ONNX emotion model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ONNX model: {e}")
            self.ort_session = None

    def set_emitter(self, emitter):
        """Set the event emitter for publishing results."""
        self.event_emitter = emitter

    def process_frame(
        self, camera_uuid: str, frame: np.ndarray, frame_number: int
    ) -> None:
        """
        Process frame for facial expression recognition and emit events.

        Args:
            camera_uuid: Unique identifier for camera
            frame: Video frame to process
            frame_number: Frame sequence number
        """
        try:
            detections = self.recognize_emotion(frame)

            if detections and self.event_emitter:
                self.event_emitter.emit(
                    camera_uuid=camera_uuid,
                    detections=detections,
                    viewing_session_uuid=self.viewing_session_uuid
                )

        except Exception as exc:
            logger.error(
                f"Error processing frame {frame_number} for camera {camera_uuid}: {exc}",
                exc_info=True,
            )

    def _initialize_onnx(self, device: str):
        """
        Initialize ONNX Runtime session with CUDA → CPU fallback.
        
        Priority order:
        1. Try CUDA if available
        2. Fallback to CPU automatically
        """
        try:
            if device == "auto":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"ONNX model not found at {self.model_path}. "
                    f"Download from https://github.com/Shohruh72/Emotion_onnx/releases/download/v.1.0.0/emotion.onnx"
                )

            self.ort_session = ort.InferenceSession(
                self.model_path, providers=providers
            )
            
            # Log which provider was actually selected
            actual_provider = self.ort_session.get_providers()[0]
            if "CUDA" in actual_provider:
                logger.info(f"Facial Expression: Using CUDA acceleration")
            else:
                logger.info(f"Facial Expression: Using CPU (CUDA not available)")
                
        except ImportError:
            logger.error(
                "onnxruntime not installed. Install with: pip install onnxruntime-gpu"
            )
            raise
        except Exception as e:
            logger.error(f"Error initializing ONNX: {e}")
            raise

    def _format_detections_for_tracker(self, detections, width, height):
        """
        Format InsightFace detections for SORT tracker.

        Args:
            detections: List of (x1, y1, x2, y2, confidence, face_obj)

        Returns:
            Tuple of (numpy array for tracker, list of face objects)
        """
        if not detections:
            return np.empty((0, 6), dtype=np.float32), []

        formatted = []
        face_objects = []

        for x1, y1, x2, y2, confidence, face_obj in detections:
            x1 = max(0, min(float(x1), width))
            y1 = max(0, min(float(y1), height))
            x2 = max(0, min(float(x2), width))
            y2 = max(0, min(float(y2), height))

            if y2 <= y1 or x2 <= x1:
                continue

            formatted.append([x1, y1, x2, y2, float(confidence), 0.0])
            face_objects.append(face_obj)

        if not formatted:
            return np.empty((0, 6), dtype=np.float32), []

        return np.array(formatted, dtype=np.float32), face_objects

    def _append_current_id(self, results, obj_id):
        cached = self.data.get(obj_id)
        if not cached:
            return

        x1 = cached["x1"]
        y1 = cached["y1"]
        x2 = cached["x2"]
        y2 = cached["y2"]
        confidence = cached["confidence"]
        expression = cached["expression"]
        results.append(
            {
                "person_id": obj_id,
                "bbox": {
                    "top": int(y1),
                    "left": int(x1),
                    "bottom": int(y2),
                    "right": int(x2),
                },
                "confidence": float(confidence),
                "expression": expression,
            }
        )

    def _safe_insert_limited(self, store, key, value, max_size):
        if key not in store and len(store) >= max_size:
            store.pop(next(iter(store)))
        store[key] = value

    def detect(self, frame):
        """
        Detect faces using InsightFace (buffalo_sc).
        Returns list of tuples: (x1, y1, x2, y2, confidence, face_object)
        """
        faces = self.face_app.get(frame)

        detections = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            confidence = float(face.det_score)

            detections.append((x1, y1, x2, y2, confidence, face))

        return detections

    def recognize_emotion(self, boxes, frame: np.ndarray, tolerance: float = 0.4):
        """Analyze facial expressions in the given frame."""

        self.frame_count += 1
        h, w = frame.shape[:2]
        detected_faces = []

        for idx, (
            x1,
            y1,
            x2,
            y2,
            obj_id,
            class_id,
            confidence_score,
        ) in enumerate(boxes):
            x1 = max(0, min(int(x1), w))
            y1 = max(0, min(int(y1), h))
            x2 = max(0, min(int(x2), w))
            y2 = max(0, min(int(y2), h))
            obj_id = int(obj_id)

            if y2 <= y1 or x2 <= x1:
                continue

            pad = 10
            crop_x1 = max(0, x1 - pad)
            crop_y1 = max(0, y1 - pad)
            crop_x2 = min(w, x2 + pad)
            crop_y2 = min(h, y2 + pad)

            face_region = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            match = None
            if face_region.size == 0:
                continue
            
            expression_entry = self.data.get(obj_id)
            if expression_entry is None or expression_entry.get("expression_confidence") is None:
                match = self._detect_with_onnx(face_region)
                confidence = max(0.0, match["confidence"])
                
                if expression_entry is None:
                    expression_entry = {
                        "tracked_id": obj_id,
                        "expression_confidence": confidence,
                        "expression": match["expression"]
                    }
                    self._safe_insert_limited(
                        self.data,
                        obj_id,
                        expression_entry,
                        max_size=50,
                    )
                else:
                    expression_entry["expression_confidence"] = confidence
                    expression_entry["expression"] = match["expression"]

            if expression_entry:
                detected_faces.append(
                    {
                        "person_id": expression_entry["tracked_id"],
                        "bbox": {
                            "top": y1,
                            "left": x1,
                            "bottom": y2,
                            "right": x2,
                        },
                        "expression": expression_entry["expression"],
                        "expression_confidence": float(expression_entry["expression_confidence"]),
                    }
                )

        return detected_faces

    def _detect_with_onnx(self, frame: np.ndarray) -> Dict[str, Any]:
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
