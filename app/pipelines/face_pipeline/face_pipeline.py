from ..base_pipeline import BasePipeline
import numpy as np
from ..utils import merge_for_detection, split_detection_results_columnar
from app.services.module_services.detection_service.base_detection import BaseDetection
from app.pipelines.tracker_pipeline.base_tracker import BaseTrackerPipeline 
from collections import defaultdict

class FacePipeline(BasePipeline):
    name = "face_pipeline"
    def __init__(self, face_detection, tracker_pipeline: BaseTrackerPipeline, features: list[BasePipeline], face_module: list[BaseDetection]):
        self.face_detection = face_detection
        self.tracker_pipeline = tracker_pipeline
        self.tracked_data = tracker_pipeline.tracked_data
        self.features = [
            feature(module=module, tracked_data=self.tracked_data)
            for (feature, module) in zip(features, face_module)
        ]

        self.module_name = [feature.name for feature in features]
        
    def process(self, frame_info: dict):
        frame, meta = merge_for_detection(frame_info)
        detections = self.face_detection.detect(frame)

        split_detection = split_detection_results_columnar(detections, meta, "face_detections")
        self.tracker_pipeline.process_tracker(split_detection)

        for feature in self.features:
            name = feature.name
            self._preprocess(split_detection, name)
            feature.process(split_detection)

        face_result = self._generate_face_result(split_detection)
        return face_result
    
    def _preprocess(self, info):

        for key, value in info.items():
            frames = value.get("frame")
            detections = value["detections"]

            face_detections = detections.get("face_detections")
            boxes = face_detections.get("boxes")
            landmarks = face_detections.get("landmarks")

            temp_results = []
            for box, lmks, frame in zip(boxes, landmarks, frames):
                x1, y1, x2, y2, obj_id, class_id, confidence_score = box
                x1, y1, x2, y2, obj_id = int(x1), int(y1), int(x2), int(y2), int(obj_id)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                existence = self.tracked_data[key].get(obj_id, {})

                temp = {
                    "person_id": int(obj_id),
                    "bbox": [x1, y1, x2, y2],
                    "landmarks" : lmks,
                    "face_crop": face_crop,
                    "confidence": float(confidence_score),
                    "tracked_status": True if existence else False,
                }
                for name in self.module_name:
                    information = existence.get("predictions", {}).get(name, None)
                    temp[name] = information if information else False

                detections["facial_info"].append(temp)

    def _generate_face_result(self, face_info):
        result = defaultdict(lambda: {
                FacePipeline.name: []
            })
        
        for cam_id, value in face_info.items():

            detections = value.get("detections")
            facial_info = detections.get("facial_info")

            for info in facial_info:
                id = info.get("person_id")
                bbox = info.get("bbox")
                confidence = info.get("confidence")

                temp = {
                    "bbox": bbox,
                    "id": id,
                }

                for name in self.module_name:
                    tracked_data = self.tracked_data[cam_id].get_tracked_info(id)
                    prediction = tracked_data[cam_id].get("predictions").get(name)

                    temp[name] = prediction if prediction else None

                result[cam_id][FacePipeline.name].append(temp)

            return result
