from app.services.module_services.detection_service.face_detection_service import FaceDetectionService
from app.services.module_services.detection_service.facial_expression_service import FacialExpressionService
from app.services.module_services.tracker_service.face_tracker_service import FaceTrackerService
from app.services.monitoring_service.cctv_service import CCTVService
from app.services.module_services.tracker_service.tracked_info_service import TrackedInfoService
from .face_pipeline.facial_expression_pipeline import FacialExpressionPipeline
from .face_pipeline.face_pipeline import FacePipeline
from .people_pipeline.people_counting_pipeline import PeopleCountingPipeline
from app.services.module_services.draw_services import DrawServices

class VisionPipeline:
    def __init__(self, source: CCTVService):
        # cctv & run control
        self.source = source
        self.running = False

        # services & pipelines
        self.tracked_data = TrackedInfoService()
        self.facial_expression_pipeline = FacialExpressionPipeline(tracked_data=self.tracked_data, facial_expression=FacialExpressionService())
        self.face_pipeline = FacePipeline(face_detection=FaceDetectionService(), face_tracker=FaceTrackerService(), feature=[self.facial_pipeline])
        self.people_counting_pipeline = PeopleCountingPipeline()
        self.draw_service = DrawServices()

        # frame information & buffer
        self.frame_count = 0
        self.latest_information = None

    def start(self):
        self.running = True
        self.source.start()

    def stop(self):
        self.running = False
        self.source.stop()

    def run(self):
        while self.running:
            frame = self.source.read()
            self.frame_count += 1
            if frame is None:
                continue

            info = self.face_pipeline.process(frame)
            self.draw_service.draw_bbox(frame, info)

            result = {
                "information": info,
                "frame": frame,
                "frame_id": self.frame_count
            }

            self.latest_information = result

            


