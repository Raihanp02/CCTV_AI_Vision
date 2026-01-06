from ..base_pipeline import BasePipeline
from app.services.module_services.tracker_service.trackerpool import TrackingPool
import numpy as np

class FacePipeline(BasePipeline):
    def __init__(self, face_detection, face_tracker: list[TrackingPool], feature: list[BasePipeline], num_camera: int):
        self.face_detection = face_detection
        self.face_tracker = face_tracker
        self.feature = feature
        self.num_camera = num_camera

    def process(self, frame_info: list[dict]):
        boxes, landmarks, scores = self.face_detection.detect(frame_info)


        boxes, landmarks = self.face_tracker.process_tracked_data(boxes, landmarks, scores)

        result = []
        for feature in self.feature:
            name = feature.name
            prediction = feature.process(boxes, landmarks, frame_info)
            result.extend(prediction)

        return result
