from ..base_pipeline import BasePipeline
import numpy as np
from collections import defaultdict

class FacialExpressionPipeline(BasePipeline):
    name = "facial_expression"
    def __init__(self, tracked_data, module):
        self.module = module
        self.tracked_data = tracked_data
    
    def process(self, face_info):
        for cam_id, value in face_info.items():
            detections = value.get("detections")
            facial_info = detections.get("facial_info")

            face_list = []
            id_list = []

            for info in facial_info:
                face = info.get("face_crop")
                id = info.get("person_id")
                expression_status = info.get(FacialExpressionPipeline.name)

                if not expression_status:
                    face_list.append(face)
                    id_list.append(id)

            prediction = self.module.detect(face_list)

            for index, predict in enumerate(prediction):
                facial_info = facial_info[index]
                track_status = facial_info.get("tracked_status")
                if predict:
                    if not track_status:
                        self.tracked_data[cam_id].init_track_info(id_list[index])
                    self.tracked_data[cam_id].update_prediction_info(predict, FacialExpressionPipeline.name)

    
            
                    