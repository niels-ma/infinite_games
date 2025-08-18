import typing
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class IfGamesEvent(BaseModel):
    event_id: str
    market_type: str
    title: str
    description: str
    event_metadata: typing.Optional[dict] = None
    created_at: datetime
    cutoff: datetime


class GetEventsResponse(BaseModel):
    count: typing.Optional[int] = None
    items: typing.List[IfGamesEvent]

    model_config = ConfigDict(from_attributes=True, extra="ignore")


class IfGamesEventDeleted(BaseModel):
    event_id: str
    market_type: str
    created_at: datetime
    deleted_at: datetime


class GetEventsDeletedResponse(BaseModel):
    count: typing.Optional[int] = None
    items: typing.List[IfGamesEventDeleted]

    model_config = ConfigDict(from_attributes=True, extra="ignore")


class IfGamesEventResolved(BaseModel):
    event_id: str
    market_type: str
    created_at: datetime
    answer: int = Field(..., ge=0, le=1)
    resolved_at: datetime
    # No need to type as datetime since is converted to string to persist in the DB
    forecasts: dict[str, float]


class GetEventsResolvedResponse(BaseModel):
    count: typing.Optional[int] = None
    items: typing.List[IfGamesEventResolved]

    model_config = ConfigDict(from_attributes=True, extra="ignore")


class MinerPrediction(BaseModel):
    unique_event_id: str
    provider_type: str
    prediction: float
    interval_start_minutes: int
    interval_datetime: datetime
    interval_agg_prediction: float
    interval_agg_count: int
    miner_hotkey: str
    miner_uid: int
    validator_hotkey: str
    validator_uid: int
    submitted_at: datetime

    # To be dropped
    title: typing.Optional[str]
    outcome: typing.Optional[float] = Field(None, ge=0, le=1)

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class PostPredictionsRequestBody(BaseModel):
    submissions: typing.Optional[typing.List[MinerPrediction]]

    # To be dropped
    events: typing.Optional[None] = Field(None)


class MinerScore(BaseModel):
    event_id: str
    prediction: float
    answer: float = Field(..., json_schema_extra={"ge": 0, "le": 1})
    miner_hotkey: str
    miner_uid: int
    miner_score: float
    miner_effective_score: float
    validator_hotkey: str
    validator_uid: int
    spec_version: typing.Optional[str] = "0.0.0"
    registered_date: typing.Optional[datetime]
    scored_at: typing.Optional[datetime]

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class PostScoresRequestBody(BaseModel):
    results: typing.List[MinerScore] = Field(..., min_length=1)
