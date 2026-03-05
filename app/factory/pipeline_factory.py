from services.monitoring_service.cctv_service import CCTVService
from pipelines.vision_orchestration import VisionPipeline
from pipelines.face_pipeline.face_pipeline import FacePipeline
from pipelines.tracker_pipeline.face_tracker_pipeline import FaceTrackerPipeline
from pipelines.face_pipeline.facial_expression_pipeline import FacialExpressionPipeline
from pipelines.people_pipeline.people_counting_pipeline import PeopleCountingPipeline
from pipelines.tracker_pipeline.people_tracker_pipeline import PeopleTrackerPipeline

from services.module_services.detection_service.face_detection_service import FaceDetectionService
from services.module_services.detection_service.facial_expression_service import FacialExpressionService
from services.module_services.tracker_service.face_tracker_service import FaceTrackerService
from services.module_services.tracker_service.people_tracker_service import PeopleTrackerService
from services.module_services.tracker_service.tracked_info_service import TrackedInfoService
from services.module_services.detection_service.people_detection_service import PeopleDetectionService
from services.module_services.counting_service.line_counter import LineCounter
from services.module_services.counting_service.line_object import LineObject
from services.module_services.draw_services import DrawServices

class PipelineFactory:
    @staticmethod
    def create_pipeline(cctvs: list[CCTVService]):
        pipelines = []
        
        dict_services = {"face_pipeline": {}, "people_counting": None}
        
        for cctv in cctvs:
            if cctv.services.EXPRESSION:
                dict_services["face_pipeline"] = {"expression": True}
            if cctv.services.PEOPLE_COUNTING:
                dict_services["people_counting"] = True

        if dict_services["face_pipeline"]:
            pipelines.append(PipelineFactory.create_face_pipeline(dict_services["face_pipeline"], cctvs))
    
        if dict_services["people_counting"] == True:
            pipelines.append(PipelineFactory.create_people_counting_pipeline(cctvs))
            
        return pipelines
            
    @staticmethod
    def create_face_pipeline(face_features: dict, cctvs: list[CCTVService]):
        face_detection = FaceDetectionService()

        tracker_pipeline = FaceTrackerPipeline(
            cam_id=[cctv.camera_id for cctv in cctvs],
            tracker_module=FaceTrackerService,
            tracked_data=TrackedInfoService
        )

        features = []
        face_module = []

        if face_features.get("expression"):
            features.append(FacialExpressionPipeline)
            face_module.append(FacialExpressionService())

        face_pipeline = FacePipeline(
            face_detection=face_detection,
            tracker_pipeline=tracker_pipeline,
            features=features,
            face_module=face_module
        )

        return face_pipeline

    @staticmethod
    def create_vision_pipeline(source):
        pipelines = PipelineFactory.create_pipeline(source)
        return VisionPipeline(
            source=source,
            pipelines=pipelines,
            draw_service=DrawServices()
        )
    
    @staticmethod
    def create_people_counting_pipeline(cctvs: list[CCTVService], counter = LineCounter(lines=[LineObject(coordinate_relative=[0.5, 0, 0.5, 1], direction_left_to_right=True)])):
        tracker_pipeline = PeopleTrackerPipeline(
            cam_id=[cctv.camera_id for cctv in cctvs if cctv.services.PEOPLE_COUNTING],
            tracker_module=PeopleTrackerService,
        )

        people_counting_pipeline = PeopleCountingPipeline(
            tracker_pipeline=tracker_pipeline,
            people_detection=PeopleDetectionService(),
            counter=counter
        )

        return people_counting_pipeline