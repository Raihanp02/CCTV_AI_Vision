from datetime import datetime

from pydantic import BaseModel, Field, Optional
from typing import List, Dict

class PredictionItem(BaseModel):
    label: str
    confidence: float

class Predictions(BaseModel):
    gender: Optional[PredictionItem] = None
    expression: Optional[PredictionItem] = None

class TrackedDataSchema(BaseModel):
    tracked_id: int = Field(..., description="Unique identifier for the tracked data")
    predictions: Predictions
