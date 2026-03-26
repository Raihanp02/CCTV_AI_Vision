from typing import Dict
import time
import threading

class TrackedInfoService:
    def __init__(self):
        self.tracked_data: Dict[int, any] = {}
        self.lock = threading.Lock()
        self.max_tracked_history = 20

    def init_track_info(self, tracked_id):
        data = {"tracked_id": tracked_id, "time_seen": time.monotonic(), "predictions": {"facial_expression":{}, "gender":{}}}
        self._safe_insert_limited(self.tracked_data, tracked_id, data)
    
    def get_tracked_info(self, person_id):
        with self.lock:
            return self.tracked_data.get(person_id, None)
    
    def update_prediction_info(self, id, prediction: dict, recognition_type: str):
        with self.lock:
            if id in self.tracked_data:
                self.tracked_data[id]["predictions"][recognition_type] = prediction
                self.tracked_data[id]["time_seen"] = time.monotonic()

    def _safe_insert_limited(self, d, key, value):
        with self.lock:
            if key not in d and len(d) >= self.max_tracked_history:
                d.pop(next(iter(d)))  # remove oldest key
            d[key] = value