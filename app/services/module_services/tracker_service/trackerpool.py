from typing import Callable

TrackerFactory = Callable[[], object]

class TrackingPool:
    def __init__(self, tracker_factory: TrackerFactory, num_trackers: int):
        self.trackers = [tracker_factory() for _ in range(num_trackers)]

    def run_all(self, detections):
        for t in self.trackers:
            t.update(detections)