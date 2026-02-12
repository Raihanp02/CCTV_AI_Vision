from ..base_pipeline import BasePipeline

import numpy as np
from collections import defaultdict
import time

class FacialExpressionPipeline(BasePipeline):
    name = "facial_expression"
    def __init__(self, tracked_data, module, detection_interval = 2.5):
        self.module = module
        self.tracked_data = tracked_data
        self.detection_interval = detection_interval
    
    def process(self, face_info):
        face_list = []
        id_list = []
        index_list = []
        cam_id_list = []

        for cam_id, value in face_info.items():
            detections = value.get("detections")
            facial_info = detections.get("facial_info")

            for i, info in enumerate(facial_info):
                face = info.get("face_crop")
                id = info.get("person_id")
                expression_status = info.get(FacialExpressionPipeline.name)

                last_seen = self.tracked_data[cam_id].tracked_data.get(id, {}).get("time_seen") if self.tracked_data[cam_id].tracked_data.get(id, {}).get("time_seen") else 0
                seen_interval = time.monotonic() - last_seen

                if not expression_status or (seen_interval > self.detection_interval):
                    face_list.append(face)
                    id_list.append(id)
                    index_list.append(i)
                    cam_id_list.append(cam_id)
            
        if face_list:
            prediction = self.module.detect(face_list)

            for index, predict in enumerate(prediction):
                cam_id = cam_id_list[index]
                detections = face_info[cam_id].get("detections")
                facial_info = detections.get("facial_info")

                id = id_list[index]
                face_info = facial_info[index_list[index]]
                track_status = face_info.get("tracked_status")
    
                if predict:
                    if not track_status:
                        self.tracked_data[cam_id].init_track_info(id)
                    self.tracked_data[cam_id].update_prediction_info(id, predict, FacialExpressionPipeline.name)

    
            
                    