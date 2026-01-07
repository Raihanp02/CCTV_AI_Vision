from abc import ABC, abstractmethod

class BaseDetection(ABC):
    @abstractmethod
    def detect(self, frame):
        pass