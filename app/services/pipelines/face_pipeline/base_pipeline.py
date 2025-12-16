from abc import ABC, abstractmethod

class BaseFacePipeline:
    def __init__(self, module,tracked_data):
        self.tracked_data = tracked_data
        self.module = module

    @abstractmethod
    def process(self, frame):
        pass

    @abstractmethod
    def _preprocess(self, boxes, landmarks, frame):
        pass