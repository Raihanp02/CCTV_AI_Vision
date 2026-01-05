from sort.tracker import SortTracker
import numpy as np
from typing import Dict

class FaceTrackerService:
    def __init__(self, module = SortTracker(max_age=30, min_hits=5, iou_threshold=0.1)):
        self.tracker = module

    def process_tracked_data(self, all_boxes, all_landmarks, all_scores):
        boxes_list = []
        landmarks_list = []

        for boxes, landmarks, scores in zip(all_boxes, all_landmarks, all_scores):
            if boxes.size: 
                detections = np.hstack([boxes, scores.reshape(-1,1), np.zeros((boxes.shape[0], 1), dtype=np.float32)])
                tracked_boxes = self.tracker.update(detections, None)

                boxes_list.append(tracked_boxes)
                landmarks_list.append(landmarks)

            else:
                boxes_list.append(np.array([]))
                landmarks_list.append(np.array([]))

            return boxes_list, landmarks_list

        