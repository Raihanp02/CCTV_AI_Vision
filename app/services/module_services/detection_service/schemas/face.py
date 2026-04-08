from typing import TypedDict
import numpy as np

class FaceDetectResult(TypedDict):
    boxes: list[np.ndarray]
    landmarks: list[np.ndarray]
    scores: list[np.ndarray]