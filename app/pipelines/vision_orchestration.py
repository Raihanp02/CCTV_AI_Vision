from app.services.monitoring_service.cctv_service import CCTVService
from .base_pipeline import BasePipeline
from app.services.module_services.draw_services import DrawServices
from queue import Queue
from collections import defaultdict 
from .executor_strategy import ThreadExecutorStrategy, Base

class VisionPipeline:
    def __init__(self, 
                 source: list[CCTVService],
                 pipelines: list[BasePipeline],
                 draw_service: DrawServices,
                 executor_strategy: Base =ThreadExecutorStrategy()):
        # cctv & run control
        self.source = source
        self.running = False

        # services & pipelines
        self.pipelines = pipelines
        self.draw_service = draw_service
        self.executor_strategy = executor_strategy

        # frame information buffer
        self.frame_buffer = self.source[0].buffer
        self.batch_size = self.source[0].max_buffer_size
        self.vision_buffer = Queue(maxsize=self.batch_size)

    def start(self):
        self.running = True
        for cctv in self.source:
            cctv.start()
        self.run()

    def stop(self):
        self.running = False
        for cctv in self.source:
            cctv.stop()

    def run(self):
        while self.running:
            if self.frame_buffer.qsize() >= self.batch_size:
                frame_info = self._drain_queue(self.frame_buffer)

                if frame_info:
                    frame_info = self._restructure_frame(frame_info)
                    
                    results = self.executor_strategy.execute(self.pipelines, frame_info)

                    #merge results back to frame_info
                    self._merge_results(frame_info, results)

                    # draw results
                    self.draw_service.process(frame_info)

                    # push to vision buffer
                    self.vision_buffer.put(frame_info)
    
    def _drain_queue(self, q):
        with q.mutex:
            items = list(q.queue)
            if items:
                q.queue.clear()

                # maintain Queue invariants
                q.unfinished_tasks = 0
                q.all_tasks_done.notify_all()

        return items
    
    def _restructure_frame(self, frame_info_list: list[dict]):
        grouped = defaultdict(lambda: {
            "frame": [],
            "frame_id": [],
            "result": {}
        })

        for item in frame_info_list:
            cam_id = item["camera_id"]
            grouped[cam_id]["frame"].append(item["frame"])
            grouped[cam_id]["frame_id"].append(item["frame_id"])

        return grouped

    def _merge_results(self, frame_info, results):
        for items in results:
            for cam_id, value in items.items():
                frame_info[cam_id]["result"].update(value)