from sort.tracker import SortTracker
import numpy as np
from typing import Dict
from .base_tracker_service import BaseTrackerService

class FaceTrackerService(BaseTrackerService):
    def __init__(self, module = SortTracker(max_age=30, min_hits=5, iou_threshold=0.1)):
        super().__init__(module)

    def process_tracked_data(self, boxes, landmarks, scores):
        if boxes.size: 
            detections = np.hstack([boxes, scores.reshape(-1,1), np.zeros((boxes.shape[0], 1), dtype=np.float32)])
            boxes = self.module.update(detections, None)

        else:
            boxes = self.module.update(np.array([]), None)
            landmarks = np.array([])  # No landmarks when there are no boxes
            scores = np.array([])     # No scores when there are no boxes

        return {"boxes": boxes, "landmarks": landmarks, "scores": scores}

        