from sort.tracker import SortTracker
import numpy as np
from typing import Dict

class FaceTrackerService:
    def __init__(self, module = SortTracker(max_age=30, min_hits=5, iou_threshold=0.1)):
        self.tracked_data: Dict[int, any] = {}
        self.tracker = module

    def process_tracked_data(self, boxes, landmarks, scores):
        if boxes.size: 
            detections = np.hstack([boxes, scores.reshape(-1,1), np.zeros((boxes.shape[0], 1), dtype=np.float32)])
            boxes = self.tracker.update(detections, None)

            return boxes, landmarks
        
        else:
            return []

        