from abc import ABC, abstractmethod

class BaseTrackerService(ABC):
    def __init__(self, module):
        self.module = module

    @abstractmethod
    def process_tracked_data(self, *args, **kwargs):
        pass