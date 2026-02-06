from typing import Dict

class TrackedInfoService:
    def __init__(self):
        self.tracked_data: Dict[int, any] = {}

    def init_track_info(self, tracked_id):
        data = {"tracked_id": tracked_id, "predictions": {"expression":{}, "gender":{}}}
        self._safe_insert_limited(self.tracked_data, tracked_id, data, max_size=10)
    
    def get_tracked_info(self, person_id):
        return self.tracked_data.get(person_id, None)
    
    def update_prediction_info(self, id, prediction: dict, recognition_type: str):
        self.tracked_data[id]["predictions"][recognition_type] = prediction
    
    def _safe_insert_limited(self, d, key, value, max_size):
        if key not in d and len(d) >= max_size:
            d.pop(next(iter(d)))  # remove oldest key
        d[key] = value