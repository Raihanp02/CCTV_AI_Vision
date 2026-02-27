from services.module_services.detection_service.people_detection_service import PeopleDetectionService
from services.module_services.counting_service.line_counter import LineCounter
from services.module_services.counting_service.line_object import LineObject
from libs.tracking.sort_tracker import SortTracker
from pipelines.base_pipeline import BasePipeline
from pipelines.tracker_pipeline.base_tracker import BaseTrackerPipeline 
from ..utils import merge_for_detection, split_detection_results_columnar

from concurrent.futures import ThreadPoolExecutor

class PeopleCountingPipeline(BasePipeline):
    name="people_counting"
    def __init__(self,
                 tracker_pipeline: BaseTrackerPipeline,  
                 people_detection = PeopleDetectionService(),
                 counter = LineCounter(lines = [LineObject(coordinate_relative=[0.5, 0, 0.5, 1], direction_left_to_right=True)]),      
            ):
        self.people_detection = people_detection
        self.counter = counter
        self.tracker_pipeline = tracker_pipeline

    def process(self, frame_info: dict):
        h, w = frame_info["frame"][0].shape[:2]

        frame, meta = merge_for_detection(frame_info)
        detections = self.people_detection.detect(frame)

        split_detection = split_detection_results_columnar(detections, meta, "people_detections")
        self.tracker_pipeline.process_tracker(split_detection, None)

        with ThreadPoolExecutor() as executor:
            futures = {}
            for cam_id, value in split_detection.items():
                future = executor.submit(self._count_result, value["detections"]["people_detections"]["boxes"], w, h)
                futures[future] = cam_id   # keep mapping
            
            results = {}

            for future, cam_id in futures.items():
                result = future.result()
                results.setdefault(cam_id, []).append(result)
                
    def get_current_total(self):
        return self.counter.going_in - self.counter.going_out
    
    def _count_result(self, boxes, w, h):
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


