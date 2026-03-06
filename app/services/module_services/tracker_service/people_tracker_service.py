from libs.tracking.sort_tracker import SortTracker
import numpy as np
from typing import Dict
from .base_tracker_service import BaseTrackerService

class PeopleTrackerService(BaseTrackerService):
    def __init__(self, module = SortTracker(max_age=30, min_hits=5, iou_threshold=0.1)):
        super().__init__(module)

    def process_tracked_data(self, boxes):
        if boxes.size: 
            boxes = self.module.update(boxes, None)

        else:
            boxes = self.module.update(np.array([]), None)

        return {"boxes": boxes}