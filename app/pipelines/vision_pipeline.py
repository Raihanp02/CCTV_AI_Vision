from app.services.module_services.detection_service.face_detection_service import FaceDetectionService
from app.services.module_services.detection_service.facial_expression_service import FacialExpressionService
from app.services.module_services.tracker_service.face_tracker_service import FaceTrackerService
from app.services.monitoring_service.cctv_service import CCTVService
from app.services.module_services.tracker_service.tracked_info_service import TrackedInfoService
from .face_pipeline.facial_expression_pipeline import FacialExpressionPipeline
from .face_pipeline.face_pipeline import FacePipeline
from .people_pipeline.people_counting_pipeline import PeopleCountingPipeline
from app.services.module_services.draw_services import DrawServices
from queue import Queue

class VisionPipeline:
    def __init__(self, 
                 source: list[CCTVService],
                 face_pipeline: FacePipeline,
                 people_counting_pipeline: PeopleCountingPipeline,
                 draw_service: DrawServices,):
        # cctv & run control
        self.source = source
        self.running = False

        # services & pipelines
        self.face_pipeline = face_pipeline
        self.people_counting_pipeline = people_counting_pipeline
        self.draw_service = draw_service

        # frame information buffer
        self.vision_buffer = Queue(maxsize=self.source[0].max_buffer_size)

    def start(self):
        self.running = True
        for cctv in self.source:
            cctv.start()

    def stop(self):
        self.running = False
        for cctv in self.source:
            cctv.stop()

    def run(self):
        while self.running:
            frame_info = self.source.read()
            if frame_info is None:
                continue
            
            frame = frame_info.get("frame")
            info = self.face_pipeline.process(frame)
            self.draw_service.draw_bbox(frame, info)

            result = {
                "information": info,
                "frame": frame,
                "frame_id": frame_info.get("frame_id"),
                "camera_id": frame_info.get("camera_id"),
                "camera_url": frame_info.get("camera_url")
            }

            self.vision_buffer.put(result)

            


