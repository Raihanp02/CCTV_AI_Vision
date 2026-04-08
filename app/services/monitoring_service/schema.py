from pydantic import BaseModel
from typing import TypedDict
import numpy as np

class AIServices(BaseModel):
    EXPRESSION: bool = False
    GENDER: bool = False
    PEOPLE_COUNTING: bool = False

class CCTVDict(TypedDict):
    services: AIServices
    camera_id: str
    camera_url: str | int
    frame_id: int
    frame: np.ndarray