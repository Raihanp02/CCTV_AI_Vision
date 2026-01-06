from ..base_pipeline import BasePipeline
import numpy as np

class FacialExpressionPipeline(BasePipeline):
    name = "facial_expression"
    def __init__(self, tracked_data, module):
        self.tracked_data = tracked_data
        self.module = module

    def process(self, boxes, landmarks, frame: list[np.ndarray]):
        face_info = self._preprocess(boxes, landmarks, frame)
        
        result = []
        for data in face_info:
            id = data.get("person_id")
            bbox = data.get("bbox")
            landmark = data.get("landmark")
            confidence = data.get("confidence")
            face = data.get("face-crop")
            track_status = data.get("tracked_status")

            if not data.get("expression_status"):
                prediction = self.module.detect(face)
                if prediction:
                    if not track_status:
                        self.tracked_data.init_track_info(id)
                    self.tracked_data.update_prediction_info(prediction, "expression")
                    result.append({
                        "detection_type": self.name,
                        "bbox": bbox,
                        "id": id,
                        "label": prediction.get("label",""),
                        "confidence": prediction.get("confidence","")
                    })

            else:
                info = self.tracked_data.get_tracked_info(id)
                prediction = info.get("predictions").get("expression")
                result.append({
                        "type":self.name,
                        "bbox": bbox,
                        "id": id,
                        "label": prediction.get("label",""),
                        "confidence": prediction.get("confidence","")
                    })
            
        return result

    def _preprocess(self, boxes, landmarks, frame: list[np.ndarray]):
        tracked_results = []
        for box, lmks in zip(boxes, landmarks):
            x1, y1, x2, y2, obj_id, class_id, confidence_score = box
            x1, y1, x2, y2, obj_id = int(x1), int(y1), int(x2), int(y2), int(obj_id)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            existence = self.tracked_data.get(obj_id, {})
            gender_status = existence.get("predictions", {}).get("gender", None)
            expression_status = existence.get("predictions", {}).get("expression", None)
                
            tracked_results.append({
                "person_id": int(obj_id),
                "bbox": [x1, y1, x2, y2],
                "landmarks" : lmks,
                "face_crop": face_crop,
                "confidence": float(confidence_score),
                "tracked_status": True if existence else False,
                "gender_status": True if gender_status else False,
                "expression_status": True if expression_status else False
            })

        return tracked_results