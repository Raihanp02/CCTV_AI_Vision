from abc import ABC, abstractmethod
from app.services.module_services.tracker_service.base_tracker_service import BaseTrackerService
from app.services.module_services.tracker_service.tracked_info_service import TrackedInfoService

class BaseTrackerPipeline(ABC):
    def __init__(self, cam_id: list[str], tracker_module: type[BaseTrackerService], tracked_data: type[TrackedInfoService]):
        self.num_camera = len(cam_id)
        self.tracker_modules = [tracker_module() for _ in range(self.num_camera)]

        if tracked_data:
            temp = [tracked_data() for _ in range(self.num_camera)]
            self.tracked_data = {v: k for v, k in zip(cam_id, temp)}

    @abstractmethod
    def process_tracker(self, *args, **kwargs):
        pass