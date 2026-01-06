from app.services.monitoring_service.cctv_service import CCTVService
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
        self.frame_buffer = self.source[0].buffer
        self.batch_size = self.source[0].max_buffer_size
        self.vision_buffer = Queue(maxsize=self.batch_size)

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
            if self.frame_buffer.qsize() >= self.batch_size:
                frame_info = self._drain_queue(self.frame_buffer)

                if frame_info:
                    info = self.face_pipeline.process(frame_info)
                    self.draw_service.draw_bbox(frame_info.get("frame"), info)

                    result = {
                        "information": info,
                        "frame": frame_info.get("frame"),
                        "frame_id": frame_info.get("frame_id"),
                        "camera_id": frame_info.get("camera_id"),
                        "camera_url": frame_info.get("camera_url")
                    }

                    self.vision_buffer.put(result)

    def _drain_queue(self, q):
        with q.mutex:
            items = list(q.queue)
            if items:
                q.queue.clear()

                # maintain Queue invariants
                q.unfinished_tasks = 0
                q.all_tasks_done.notify_all()

        return items