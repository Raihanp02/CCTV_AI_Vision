from typing import TypedDict
import numpy as np

class PeopleDetectResult(TypedDict):
    boxes: list[np.ndarray]
    