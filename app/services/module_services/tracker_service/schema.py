from datetime import datetime

from pydantic import BaseModel, Field, Optional
from typing import List, Dict

"""
example:
{
    1 : {
        "tracked_id": 1,
        "predictions": {
            "gender": {"label": "male", "confidence": 0.93},
            "expression": {"label": "happy", "confidence": 0.87}
        }
    }
}
"""

class PredictionItem(BaseModel):
    label: str
    confidence: float

class Predictions(BaseModel):
    gender: Optional[PredictionItem] = None
    expression: Optional[PredictionItem] = None

class TrackedDataSchema(BaseModel):
    tracked_id: int = Field(..., description="Unique identifier for the tracked data")
    predictions: Predictions
