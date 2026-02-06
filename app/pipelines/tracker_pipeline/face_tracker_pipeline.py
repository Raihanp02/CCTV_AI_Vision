from .base_tracker import BaseTrackerPipeline

class FaceTrackerPipeline(BaseTrackerPipeline):
    def __init__(self, cam_id: list[str], tracker_module, tracked_data):
        super().__init__(cam_id, tracker_module, tracked_data)

    def process_tracker(self, info):
        for cam_id, value in info.items():
            temp_detection = []

            detections = value["detections"]["face_detections"]
            for boxes, landmarks, scores in zip(detections.get("boxes", []), detections.get("landmarks", []), detections.get("scores", [])):
                result = self.tracker_modules[cam_id].process_tracked_data(
                    boxes,
                    landmarks,
                    scores
                )

                temp_detection.append({
                    "boxes": result["boxes"],
                    "landmarks": result["landmarks"],
                    "scores": result["scores"]
                })
            value["detections"]["face_detections"] = temp_detection