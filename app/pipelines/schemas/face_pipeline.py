from typing import TypedDict
from typing_extensions import NotRequired
import numpy as np

from services.monitoring_service.schema import AIServices
from services.module_services.tracker_service.schema import Predictions

class FaceDetection(TypedDict):
    boxes: np.ndarray
    landmarks: np.ndarray
    scores: np.ndarray

class FacialInfo(TypedDict):
    person_id: int
    bbox: list
    landmarks: np.ndarray
    face_crop: np.ndarray
    confidence: float
    tracked_status: bool

class DetectionResults(TypedDict):
    face_detections: NotRequired[list[FaceDetection]]
    facial_info: NotRequired[list[FacialInfo]]

class SplitDetectionFace(TypedDict):
    frame_id: list[int]
    frame: list[np.ndarray]
    services: AIServices
    detections: DetectionResults

class FaceResults(TypedDict):
    bbox: list
    id: int
    detections: Predictions

class FacePipelineResults(TypedDict):
    face_pipeline: list[list[FaceResults]]
