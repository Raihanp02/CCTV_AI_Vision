from .base_tracker import BaseTrackerPipeline

class FaceTrackerPipeline(BaseTrackerPipeline):
    def __init__(self, cam_id: list[str], tracker_module, tracked_data):
        super().__init__(cam_id, tracker_module, tracked_data)

    def process_tracker(self, info):
        for key, value in info.items():
            temp_detection = []

            detections = value["detections"]
            for detection in detections:
                result = self.tracker_modules[key].process_tracked_data(
                    detection["boxes"],
                    detection["landmarks"],
                    detection["scores"]
                )

                temp_detection.append({
                    "boxes": result["boxes"],
                    "landmarks": result["landmarks"],
                    "scores": result["scores"]
                })

            detections = temp_detection