from ..base_pipeline import BasePipeline

import numpy as np
from collections import defaultdict
import time

class FacialExpressionPipeline(BasePipeline):
    name = "EXPRESSION"
    def __init__(self, tracked_data, module, detection_interval = 2.5):
        self.module = module
        self.tracked_data = tracked_data
        self.detection_interval = detection_interval
    
    def process(self, face_infos):
        faces_to_process = []

        for cam_id, value in face_infos.items():

            if not getattr(value["services"], FacialExpressionPipeline.name):
                continue

            detections = value.get("detections")
            facial_info = detections.get("facial_info")

            for frame_index, info in enumerate(facial_info):
                for order_index, detail in enumerate(info):

                    face = detail.get("face_crop")
                    person_id = detail.get("person_id")
                    expression_status = detail.get(FacialExpressionPipeline.name)

                    last_seen = (
                        self.tracked_data[cam_id].tracked_data
                        .get(person_id, {})
                        .get("time_seen", 0)
                    )

                    seen_interval = time.monotonic() - last_seen

                    if not expression_status or seen_interval > self.detection_interval:
                        faces_to_process.append({
                            "face": face,
                            "person_id": person_id,
                            "frame_index": frame_index,
                            "order_index": order_index,
                            "cam_id": cam_id
                        })

        if faces_to_process:
            faces = [x["face"] for x in faces_to_process]
            predictions = self.module.detect(faces)

            for data, predict in zip(faces_to_process, predictions):
                cam_id = data["cam_id"]
                person_id = data["person_id"]
                frame_index = data["frame_index"]
                order_index = data["order_index"]

                detections = face_infos[cam_id].get("detections")
                facial_info = detections.get("facial_info")

                face_info = facial_info[frame_index][order_index]
                track_status = face_info.get("tracked_status")

                if predict:
                    if not track_status:
                        self.tracked_data[cam_id].init_track_info(person_id)

                    self.tracked_data[cam_id].update_prediction_info(
                        person_id,
                        predict,
                        FacialExpressionPipeline.name
                    )
    
                
                        