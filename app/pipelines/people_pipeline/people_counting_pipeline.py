from services.module_services.detection_service.people_detection_service import PeopleDetectionService
from services.module_services.counting_service.line_counter import LineCounter
from services.module_services.counting_service.line_object import LineObject
from libs.tracking.sort_tracker import SortTracker
from pipelines.base_pipeline import BasePipeline
from pipelines.tracker_pipeline.base_tracker import BaseTrackerPipeline 
from ..utils import merge_for_detection, split_detection_results_columnar

from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

class PeopleCountingPipeline(BasePipeline):
    name="PEOPLE_COUNTING"
    def __init__(self,
                 tracker_pipeline: BaseTrackerPipeline,  
                 people_detection = PeopleDetectionService(),
                 counter = LineCounter(lines = [LineObject(coordinate_relative=[0.5, 0, 0.5, 1], direction_left_to_right=True)]),      
            ):
        self.people_detection = people_detection
        self.counter = counter
        self.tracker_pipeline = tracker_pipeline

    def process(self, frame_info: dict):
        results = defaultdict(lambda: {
                PeopleCountingPipeline.name: []
            })
        cam_ids = list(frame_info.keys())
        
        h, w = frame_info[cam_ids[0]]["frame"][0].shape[:2]

        frame, meta = merge_for_detection(frame_info)
        detections = self.people_detection.detect(frame)

        split_detection = split_detection_results_columnar(detections, meta, "people_detections")
        self.tracker_pipeline.process_tracker(split_detection)

        with ThreadPoolExecutor() as executor:
            futures = {}
            for cam_id, value in split_detection.items():
                people_detections = value["detections"]["people_detections"]
                for people in people_detections:
                    bbox = people.get("boxes", [])
                    future = executor.submit(self._count_result, bbox, w, h)
                    futures[future] = cam_id   # keep mapping

            for future, cam_id in futures.items():
                result = future.result()
                results[cam_id][PeopleCountingPipeline.name].append(result)

        return results
                
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

            status = self.counter.single_crossing_line(data, w, h)

            result.append({
                "bbox": [x1,y1,x2,y2],
                "id": class_id,
                "type": status if status else None,
                "current_total": self.get_current_total()
            })

        return result


