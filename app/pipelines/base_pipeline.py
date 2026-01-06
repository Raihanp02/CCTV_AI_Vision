from abc import ABC, abstractmethod

class BasePipeline(ABC):
    @abstractmethod
    def process(self, frame):
        pass