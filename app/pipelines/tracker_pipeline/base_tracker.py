from abc import ABC, abstractmethod

class BaseTrackerPipeline(ABC):
    def __init__(self, cam_id: list[str], tracker_module, tracked_data):
        self.num_camera = len(cam_id)
        self.tracker_module = [tracker_module() for _ in range(self.num_camera)]

        if tracked_data:
            temp = [tracked_data() for _ in range(self.num_camera)]
            self.tracked_data = {v: k for v, k in zip(cam_id, temp)}

    @abstractmethod
    def process_tracker(self, *args, **kwargs):
        pass