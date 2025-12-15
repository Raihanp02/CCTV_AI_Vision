from sort.tracker import SortTracker
import numpy as np
from typing import Dict

class TrackedDataService:
    def __init__(self, module = SortTracker(max_age=30, min_hits=5, iou_threshold=0.1)):
        self.tracked_data: Dict[int, any] = {}
        self.tracker = module

    def process_tracked_data(self, boxes, landmarks, scores, frame):
        if boxes.size: 
            detections = np.hstack([boxes, scores.reshape(-1,1), np.zeros((boxes.shape[0], 1), dtype=np.float32)])
            boxes = self.tracker.update(detections, None)

        tracked_results = []
        for box, lmks in zip(boxes, landmarks):
            x1, y1, x2, y2, obj_id, class_id, confidence_score = box
            x1, y1, x2, y2, obj_id = int(x1), int(y1), int(x2), int(y2), int(obj_id)

            existence = self.tracked_data.get(obj_id, {})
            gender_status = existence.get("predictions", {}).get("gender", None)
            expression_status = existence.get("predictions", {}).get("expression", None)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
                
            tracked_results.append({
                "person_id": int(obj_id),
                "bbox": [x1, y1, x2, y2],
                "landmarks" : lmks,
                "face_crop": face_crop,
                "confidence": float(confidence_score),
                "tracked_status": True if existence else False,
                "gender_status": True if gender_status else False,
                "expression_status": True if expression_status else False
            })

        return tracked_results
    
    def store_tracked_info(self, tracked_id: int, predictions):
        self._safe_insert_limited(self.tracked_data, tracked_id, predictions, max_size=10)
    
    def get_tracked_info(self, person_id):
        return self.tracked_data.get(person_id, None)
    
    def _safe_insert_limited(self, d, key, value, max_size):
        if key not in d and len(d) >= max_size:
            d.pop(next(iter(d)))  # remove oldest key
        d[key] = value
    


    

    

    