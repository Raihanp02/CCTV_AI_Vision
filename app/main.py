from services.monitoring_service.cctv_service import CCTVService
from services.monitoring_service.schema import AIServices
from factory.pipeline_factory import PipelineFactory

from inference_gateway.cv2_testing.stream import StreamVideo

from queue import Queue

buffer = Queue(maxsize=1)

cam_1 = CCTVService(camera_url=0, 
                    camera_id="cam_1", 
                    buffer=buffer, 
                    services=AIServices(GENDER=True, EXPRESSION=True))

vision_pipeline = PipelineFactory.create_vision_pipeline(source=[cam_1])

vision_pipeline.start()

streamer = StreamVideo(vision_pipeline, fps=30)
streamer.start()