from core.database.base import BaseModel
from sqlalchemy import Column, Integer, String, Date
from sqlalchemy.orm import relationship, Mapped

class Person(BaseModel):
    __tablename__ = 'persons'
    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(Integer, unique=True, nullable=False)
    name = Column(String, nullable=True)