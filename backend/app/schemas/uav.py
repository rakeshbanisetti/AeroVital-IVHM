
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class UAVCreate(BaseModel):
    uav_code: str = Field(min_length=1, max_length=50)
    model_name: str = Field(min_length=1, max_length=100)
    manufacturer: str | None = Field(default=None, max_length=100)


class UAVResponse(BaseModel):
    id: int
    uav_code: str
    model_name: str
    manufacturer: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
