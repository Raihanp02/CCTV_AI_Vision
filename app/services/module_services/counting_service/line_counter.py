from .line_object import LineObject
import threading

class LineCounter:
    def __init__(self, lines: LineObject):
        self.prev_centroids = {}
        self.total_count = []
        self.going_in = 0
        self.going_out = 0
        self.lines = lines
        self.lock = threading.Lock()

    def batch_crossing_line(self, tracked_objects, w, h):
        results = []            
        for i, obj in enumerate(tracked_objects):
            bx1, by1, bx2, by2, obj_id, class_id, confidence_score = (
                int(obj[0]),
                int(obj[1]),
                int(obj[2]),
                int(obj[3]),
                int(obj[4]),
                int(obj[5]),
                obj[6],
            )

            cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
            curr_centroid = [cx, cy]

            status = None
            if obj_id in self.prev_centroids:
                for line in self.lines:
                    limits = line.to_absolute(w,h)
                    prev = self.prev_centroids[obj_id]
                    side_prev = self._side(prev, limits)
                    side_curr = self._side(curr_centroid, limits)

                    status = self._crossing_direction(side_prev, side_curr, line)
                    if status:
                        break

            self._safe_insert_limited(
                self.prev_centroids, obj_id, curr_centroid, max_size=100
            )

            results.append(
                {
                    "bbox": [bx1, by1, bx2, by2],
                    "person_id": obj_id,
                    "type": status,
                    "confidence": confidence_score,
                    "total_count": self.going_in - self.going_out,
                    "current_total": len(self.total_count),
                }
            )

        return results
    
    def single_crossing_line(self, tracked_object, w, h):
        bx1, by1, bx2, by2, obj_id, class_id, confidence_score = (
                int(tracked_object[0]),
                int(tracked_object[1]),
                int(tracked_object[2]),
                int(tracked_object[3]),
                int(tracked_object[4]),
                int(tracked_object[5]),
                tracked_object[6],
            )
        
        cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
        curr_centroid = [cx, cy]

        status = None
        if obj_id in self.prev_centroids:
            for line in self.lines:
                limits = line.to_absolute(w,h)
                prev = self.prev_centroids[obj_id]
                side_prev = self._side(prev, limits)
                side_curr = self._side(curr_centroid, limits)

                status = self._crossing_direction(side_prev, side_curr, line)
                if status:
                    break

        self._safe_insert_limited(
            self.prev_centroids, obj_id, curr_centroid, max_size=100
        )

        return status
        
    def _crossing_direction(self, side_prev, side_curr, line) -> str | None:

        forward = side_prev <= 0 and side_curr > 0
        backward = side_prev > 0 and side_curr <= 0

        if line.direction_left_to_right:
            if forward:
                self.going_in += 1
                return "in"
            elif backward:
                self.going_out += 1
                return "out"
            else:
                return None
        else:
            if forward:
                self.going_out += 1
                return "out"
            elif backward:
                self.going_in += 1
                return "in"
            else:
                return None

    def _side(self, p, limits):
        """
        Positive means right side of the line direction
        Negative means left side of the line direction
        """
        x1, y1, x2, y2 = limits
        return (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1)

    def _safe_insert_limited(self, d, key, value, max_size):
        with self.lock:
            if key not in d and len(d) >= max_size:
                d.pop(next(iter(d)))
            d[key] = value