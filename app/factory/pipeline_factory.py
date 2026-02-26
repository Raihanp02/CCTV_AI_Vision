from app.services.monitoring_service.cctv_service import CCTVService
from app.pipelines.vision_orchestration import VisionPipeline
from app.pipelines.face_pipeline.face_pipeline import FacePipeline
from app.pipelines.tracker_pipeline.face_tracker_pipeline import FaceTrackerPipeline
from app.pipelines.face_pipeline.facial_expression_pipeline import FacialExpressionPipeline
from app.services.module_services.detection_service.face_detection_service import FaceDetectionService
from app.services.module_services.detection_service.facial_expression_service import FacialExpressionService
from app.services.module_services.tracker_service.face_tracker_service import FaceTrackerService
from app.services.module_services.tracker_service.tracked_info_service import TrackedInfoService
from app.services.module_services.draw_services import DrawServices

class PipelineFactory:
    @staticmethod
    def create_face_pipeline(cam_id: str):
        face_detection = FaceDetectionService()

        tracker_pipeline = FaceTrackerPipeline(
            cam_id=[cam_id],
            tracker_module=FaceTrackerService(),
            tracked_data=TrackedInfoService()
        )

        facial_expression_service = FacialExpressionService()

        face_pipeline = FacePipeline(
            face_detection=face_detection,
            tracker_pipeline=tracker_pipeline,
            features=[FacialExpressionPipeline],
            face_module=[facial_expression_service]
        )

        return face_pipeline

    @staticmethod
    def create_vision_pipeline(source):
        return VisionPipeline(
            source=source,
            pipelines=[PipelineFactory.create_face_pipeline("cam_1")],
            draw_service=DrawServices()
        )