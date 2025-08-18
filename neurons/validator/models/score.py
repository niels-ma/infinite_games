from datetime import datetime
from enum import IntEnum
from typing import Any, Optional

from pydantic import BaseModel, field_validator


class ScoresExportedStatus(IntEnum):
    NOT_EXPORTED = 0
    EXPORTED = 1


class ScoresModel(BaseModel):
    event_id: str
    miner_uid: int
    miner_hotkey: str
    prediction: float
    event_score: float
    metagraph_score: Optional[float] = None
    other_data: Optional[str] = None
    created_at: Optional[datetime] = None
    spec_version: int
    processed: Optional[bool] = False
    exported: Optional[bool] = False
    alternative_metagraph_score: Optional[float] = None
    alternative_other_data: Optional[str] = None
    alternative_processed: Optional[bool] = False

    @property
    def primary_key(self):
        return [
            "event_id",
            "miner_uid",
            "miner_hotkey",
        ]

    @field_validator("processed", mode="before")
    def parse_processed_as_bool(cls, v: Any) -> bool:
        # If the DB returns an integer, convert it to boolean
        if isinstance(v, int):
            return bool(v)
        return v

    @field_validator("exported", mode="before")
    def parse_exported_as_bool(cls, v: Any) -> bool:
        # If the DB returns an integer, convert it to boolean
        if isinstance(v, int):
            return bool(v)
        return v


SCORE_FIELDS = ScoresModel.model_fields.keys()
