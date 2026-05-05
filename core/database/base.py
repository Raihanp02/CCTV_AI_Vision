from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Date, DateTime
from datetime import datetime
from sqlalchemy_mixins import AllFeaturesMixin

Base = declarative_base()

class BaseModel(Base, AllFeaturesMixin):
    __abstract__ = True
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now)
    deleted_at = Column(DateTime, nullable=True)