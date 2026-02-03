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

from app.inference_gateway.cv2_testing.stream import StreamVideo

from queue import Queue

buffer = Queue(maxsize=1)

cam_1 = CCTVService(camera_url=0, camera_id="cam_1", buffer=buffer)

vision_pipeline = VisionPipeline(
    source=[cam_1],
    pipelines=[FacePipeline(
        face_detection=FaceDetectionService(),
        tracker_pipeline=FaceTrackerPipeline(cam_id=["cam_1"], tracker_module=FaceTrackerService, tracked_data=TrackedInfoService),
        features=[FacialExpressionPipeline],
        face_module=[FacialExpressionService()]
    )],
    draw_service=DrawServices(),
)

vision_pipeline.start()

streamer = StreamVideo(vision_pipeline, fps=30)
streamer.start()