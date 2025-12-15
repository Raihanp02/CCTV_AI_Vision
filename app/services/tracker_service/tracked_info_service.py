from typing import Dict

class TrackedInfoService:
    def __init__(self):
        self.tracked_data: Dict[int, any] = {}

    def store_tracked_info(self, tracked_id: int, predictions):
        self._safe_insert_limited(self.tracked_data, tracked_id, predictions, max_size=10)
    
    def get_tracked_info(self, person_id):
        return self.tracked_data.get(person_id, None)
    
    def _safe_insert_limited(self, d, key, value, max_size):
        if key not in d and len(d) >= max_size:
            d.pop(next(iter(d)))  # remove oldest key
        d[key] = value