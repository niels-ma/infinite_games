from typing import Literal

from pydantic import BaseModel, Field

from neurons.validator.models.event import EventsModel
from neurons.validator.models.prediction import PredictionsModel


class HealthCheckResponse(BaseModel):
    status: Literal["OK"]


class GetEventResponse(EventsModel):
    # Needed to not expose the forecasts in the API response
    forecasts: str = Field(exclude=True)


class GetEventCommunityPrediction(BaseModel):
    event_id: str
    community_prediction: None | float
    community_prediction_lr: None | float


class GetEventsCommunityPredictions(BaseModel):
    count: int
    community_predictions: list[GetEventCommunityPrediction]


class GetEventPredictions(BaseModel):
    count: int
    predictions: list[PredictionsModel]
