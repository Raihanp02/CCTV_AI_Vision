from abc import ABC, abstractmethod

class FacePipeline:
    def __init__(self, tracker, tracked_data, module):
        self.tracker = tracker
        self.tracked_data = tracked_data
        self.module = module

    @abstractmethod
    def process(self, frame):
        pass

    @abstractmethod
    def _preprocess(self, boxes, landmarks, frame):
        pass