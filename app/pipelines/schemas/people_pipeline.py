from typing import TypedDict, Literal
from typing_extensions import NotRequired
import numpy as np

from services.monitoring_service.schema import AIServices

class PeopleDetection(TypedDict):
    boxes: np.ndarray

class PeopleCountingResult(TypedDict):
    bbox: list
    id: int
    type: Literal["in", "out", None]
    current_total: int

class DetectionResults(TypedDict):
    people_detections: NotRequired[list[PeopleDetection]]

class SplitDetectionPeople(TypedDict):
    frame_id: list[int]
    frame: list[np.ndarray]
    services: AIServices
    detections: DetectionResults

class PeoplePipelineResults(TypedDict):
    PEOPLE_COUNTING: list[list[PeopleCountingResult]]