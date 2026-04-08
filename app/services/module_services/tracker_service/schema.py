from datetime import datetime

from pydantic import BaseModel, Field, Optional
from typing import TypedDict, NotRequired

"""
example:
{
    1 : {
        "tracked_id": 1,
        "predictions": {
            "gender": {"label": "male", "confidence": 0.93},
            "facial_expression": {"label": "happy", "confidence": 0.87}
        }
    }
}
"""

class PredictionItem(TypedDict):
    label: str
    confidence: float

class Predictions(TypedDict):
    gender: NotRequired[PredictionItem]
    facial_expression: NotRequired[PredictionItem]

class TrackedDataSchema(TypedDict):
    tracked_id: int
    predictions: Predictions
