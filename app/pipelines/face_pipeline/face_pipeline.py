from concurrent.futures import ThreadPoolExecutor

from ..base_pipeline import BasePipeline
import numpy as np
from ..utils import merge_for_detection, split_detection_results_columnar
from services.module_services.detection_service.base_detection import BaseDetection
from pipelines.tracker_pipeline.base_tracker import BaseTrackerPipeline 
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

        self._preprocess(split_detection)
        with ThreadPoolExecutor() as executor:
            executor.map(lambda f: f.process(split_detection), self.features)

        face_result = self._generate_face_result(split_detection)
        return face_result
    
    def _preprocess(self, info):
        for key, value in info.items():
            frames = value.get("frame")
            detections = value["detections"]

            face_detections = detections.get("face_detections")

            final_results = []
            for face, frame in zip(face_detections, frames):
                bbox = face.get("boxes")
                result_per_frame = []
                for i, box in enumerate(bbox):
                    x1, y1, x2, y2, obj_id, class_id, confidence_score = box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size == 0:
                        continue

                    existence = self.tracked_data[key].tracked_data.get(obj_id, {})

                    temp = {
                        "person_id": int(obj_id),
                        "bbox": [x1, y1, x2, y2],
                        "landmarks" : face["landmarks"][i],
                        "face_crop": face_crop,
                        "confidence": float(confidence_score),
                        "tracked_status": True if existence else False,
                    }
                    for name in self.module_name:
                        information = existence.get("predictions", {}).get(name, None)
                        temp[name] = information if information else False

                    result_per_frame.append(temp)

                final_results.append(result_per_frame)

                value["detections"]["facial_info"] = final_results

    def _generate_face_result(self, face_info):
        result = defaultdict(lambda: {
                FacePipeline.name: []
            })
        
        for cam_id, value in face_info.items():

            detections = value.get("detections")
            facial_info = detections.get("facial_info")

            for info in facial_info:
                result_per_object = []
                for detail in info:
                    id = detail.get("person_id")
                    bbox = detail.get("bbox")
                    confidence = detail.get("confidence")

                    temp = {
                        "bbox": bbox,
                        "id": id,
                        "detections": {}
                    }

                    for name in self.module_name:
                        tracked_data = self.tracked_data[cam_id].get_tracked_info(id)
                        if tracked_data:
                            prediction = tracked_data.get("predictions", {}).get(name, "")
                            if prediction:
                                temp["detections"][name] = prediction if prediction else None
                    result_per_object.append(temp)
            
                result[cam_id][FacePipeline.name].append(result_per_object)
            return result
