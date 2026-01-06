from app.services.module_services.detection_service.people_detection_service import PeopleDetectionService
from app.services.module_services.counting_service.line_counter import LineCounter
from app.services.module_services.counting_service.line_object import LineObject
from sort.tracker import SortTracker
from app.pipelines.base_pipeline import BasePipeline

class PeopleCountingPipeline(BasePipeline):
    name="people_counting"
    def __init__(self):
        self.people_detection = PeopleDetectionService()
        self.counter = LineCounter(lines = [LineObject(coordinate_relative=[0.5, 0, 0.5, 1], direction_left_to_right=True)])
        self.tracker = SortTracker()

    def process(self, frame):
        h, w = frame.shape[:2]

        detections = self.people_detection.detect(frame)
        boxes = self.tracker.update(detections, None)

        result = []
        for data in boxes:
            x1, y1, x2, y2, obj_id, class_id, confidence_score = (
                int(data[0]),
                int(data[1]),
                int(data[2]),
                int(data[3]),
                int(data[4]),
                int(data[5]),
                data[6],
            )

            status = self.counter.single_crossing_line(boxes, w, h)

            result.append({
                "detection_type": self.name,
                "bbox": [x1,y1,x2,y2],
                "id": class_id,
                "type": status if status else None,
                "current_total": self.get_current_total()
            })

        return result
    
    def get_current_total(self):
        return self.counter.going_in - self.counter.going_out


