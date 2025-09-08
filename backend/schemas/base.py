# schemas/base.py
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class BaseSchema(BaseModel):
    class Config:
        from_attributes = True

class TimestampMixin(BaseModel):
    timestamp: Optional[datetime] = None# Base schemas
