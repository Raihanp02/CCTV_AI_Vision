from app.services.module_services.face_extract_service import FaceExtractService
from app.services.module_services.facial_expression_service import FacialExpressionService
from app.services.tracker_service.face_tracker_service import FaceTrackerService
from app.services.monitoring_service.cctv_service import CCTVService
from app.services.tracker_service.tracked_info_service import TrackedInfoService
from .face_pipeline.facial_expression_pipeline import FacialExpressionPipeline
from app.services.module_services.draw_services import DrawServices

class VisionPipeline:
    def __init__(self, source: CCTVService):
        self.source = source
        self.running = False
        self.facial_pipeline = FacialExpressionPipeline(tracker=FaceTrackerService(), tracked_data=TrackedInfoService(), facial_expression=FacialExpressionService())
        self.tracked_data = TrackedInfoService()
        self.draw_service = DrawServices()

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

            info = self.facial_pipeline.process(frame)
            self.draw_service.draw_bbox(frame, info)

            result = {
                "information": info,
                "frame": frame
            }

            return result

            


