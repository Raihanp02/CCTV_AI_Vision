from .base_pipeline import BaseFacePipeline

class FacePipeline:
    def __init__(self, face_detection, face_tracker, feature: list[BaseFacePipeline]):
        self.face_detection = face_detection
        self.face_tracker = face_tracker
        self.feature = feature

    def process(self, frame):
        boxes, landmarks, scores = self.face_detection.detect(frame)
        boxes, landmarks = self.face_tracker.process_tracked_data(boxes, landmarks, scores)

        result = []
        for feature in self.feature:
            name = feature.name
            prediction = feature.process(boxes, landmarks, frame)
            result.extend(prediction)

        return result
