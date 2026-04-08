from typing import TypedDict
import numpy as np
from services.monitoring_service.schema import AIServices
from pipelines.schemas.face_pipeline import FacePipelineResults
from pipelines.schemas.people_pipeline import PeoplePipelineResults

class StructuredInfo(TypedDict):
    frame: list[np.ndarray]
    frame_id: list[int]
    result: dict[str, FacePipelineResults | PeoplePipelineResults]
    services: AIServices

PipelinesResults = list[FacePipelineResults | PeoplePipelineResults]





