from pydantic import BaseModel

class AIServices(BaseModel):
    EXPRESSION: bool = False
    GENDER: bool = False
    PEOPLE_COUNTING: bool = False