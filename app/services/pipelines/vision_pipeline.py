from app.services.module_services.face_extract_service import FaceExtractService
from app.services.module_services.facial_expression_service import FacialExpressionService
from app.services.tracker_service.face_tracker_service import FaceTrackerService
from app.services.monitoring_service.cctv_service import CCTVService
from app.services.tracker_service.tracked_info_service import TrackedInfoService

class VisionPipeline:
    def __init__(self, source: CCTVService):
        self.source = source
        self.running = False
        self.face_extract = FaceExtractService()
        self.facial_expression = FacialExpressionService()
        self.tracker = FaceTrackerService()
        self.tracked_data = TrackedInfoService()

    def start(self):
        self.running = True
        self.source.start()

    def stop(self):
        self.running = False
        self.source.stop()

    def run(self):
        while self.running:
            frame = self.source.read()
            if frame is None:
                continue

            boxes, landmarks, scores = self.face_extract.detect(frame)
            boxes, landmarks = self.tracker.process_tracked_data(boxes, landmarks, scores)
            face_info = self._preprocess(boxes, landmarks, frame)
            
            for face in face_info:
                if not face.expression_status:
                    pass

                else:
                    pass


    def _preprocess(self, boxes, landmarks, frame):
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


