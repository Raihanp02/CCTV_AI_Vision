from core.database.base import BaseModel
from sqlalchemy import Column, Integer, String, Date, ForeignKey
from sqlalchemy.orm import relationship, Mapped
from pgvector.sqlalchemy import Vector

class PersonFace(BaseModel):
    __tablename__ = 'person_faces'

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.person_id"), index=True, nullable=False)
    image_path = Column(String, nullable=False)
    embedding = Column(Vector(128), nullable=False)