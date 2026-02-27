from .base_tracker import BaseTrackerPipeline

class PeopleTrackerPipeline(BaseTrackerPipeline):
    def __init__(self, cam_id: list[str], tracker_module, tracked_data):
        super().__init__(cam_id, tracker_module, tracked_data)

    def process_tracker(self, info):
        for cam_id, value in info.items():
            temp_detection = []

            detections = value["detections"]["people_detections"]
            boxes = detections.get("boxes", [])
            result = self.tracker_modules[cam_id].process_tracked_data(
                boxes
            )

            temp_detection.append({
                "boxes": result["boxes"],
            })
            value["detections"]["people_detections"] = temp_detection