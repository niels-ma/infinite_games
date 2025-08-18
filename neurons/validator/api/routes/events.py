from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from neurons.validator.api.routes.utils import get_lr_predictions_events
from neurons.validator.api.types import ApiRequest
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.api import (
    GetEventCommunityPrediction,
    GetEventPredictions,
    GetEventResponse,
    GetEventsCommunityPredictions,
)
from neurons.validator.tasks.train_cp_model import TOP_N_RANKS
from neurons.validator.utils.common.interval import get_interval_start_minutes
from neurons.validator.utils.config import IfgamesEnvType
from neurons.validator.utils.logger.logger import api_logger

router = APIRouter()


@router.get(
    "/{event_id}",
)
async def get_event(event_id: str, request: ApiRequest) -> GetEventResponse:
    db_operations: DatabaseOperations = request.state.db_operations

    unique_event_id = f"ifgames-{event_id}"

    event = await db_operations.get_event(unique_event_id)

    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    return GetEventResponse.model_validate(event.model_dump())


@router.get(
    "/{event_id}/community_prediction",
)
async def get_event_community_prediction(
    event_id: str, request: ApiRequest
) -> GetEventCommunityPrediction:
    db_operations: DatabaseOperations = request.state.db_operations
    env: IfgamesEnvType = request.state.env
    top_n_ranks: int = TOP_N_RANKS[env]

    unique_event_id = f"ifgames-{event_id}"

    event = await db_operations.get_event(unique_event_id=unique_event_id)

    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    current_interval = get_interval_start_minutes()

    community_predictions_by_id = await db_operations.get_wa_predictions_events(
        unique_event_ids=[unique_event_id], interval_start_minutes=current_interval
    )

    try:
        lr_community_predictions_by_id = await get_lr_predictions_events(
            unique_event_ids=[unique_event_id],
            interval_start_minutes=current_interval,
            top_n_ranks=top_n_ranks,
            db_operations=db_operations,
        )
    except Exception:
        api_logger.exception("Error getting LR predictions")
        lr_community_predictions_by_id = {}

    return {
        "event_id": unique_event_id.removeprefix("ifgames-"),
        "community_prediction": community_predictions_by_id.get(unique_event_id),
        "community_prediction_lr": lr_community_predictions_by_id.get(unique_event_id),
    }


@router.get(
    "/community_predictions/",
)
async def get_events_community_predictions(
    event_ids: Annotated[list[str], Query()], request: ApiRequest
) -> GetEventsCommunityPredictions:
    db_operations: DatabaseOperations = request.state.db_operations
    env: IfgamesEnvType = request.state.env
    top_n_ranks: int = TOP_N_RANKS[env]

    if len(event_ids) > 20:
        raise HTTPException(
            status_code=422,
            detail="Maximum 20 events ids allowed",
        )

    current_interval = get_interval_start_minutes()

    unique_event_ids = [f"ifgames-{event_id}" for event_id in event_ids]

    community_predictions_by_id = await db_operations.get_wa_predictions_events(
        unique_event_ids=unique_event_ids, interval_start_minutes=current_interval
    )

    try:
        lr_community_predictions_by_id = await get_lr_predictions_events(
            unique_event_ids=unique_event_ids,
            interval_start_minutes=current_interval,
            top_n_ranks=top_n_ranks,
            db_operations=db_operations,
        )
    except Exception:
        api_logger.exception("Error getting LR predictions")
        lr_community_predictions_by_id = {}

    community_predictions = [
        {
            "event_id": unique_event_id.removeprefix("ifgames-"),
            "community_prediction": community_predictions_by_id.get(unique_event_id),
            "community_prediction_lr": lr_community_predictions_by_id.get(unique_event_id),
        }
        for unique_event_id in unique_event_ids
    ]

    return {"count": len(community_predictions), "community_predictions": community_predictions}


@router.get(
    "/{event_id}/predictions",
)
async def get_predictions(event_id: str, request: ApiRequest) -> GetEventPredictions:
    db_operations: DatabaseOperations = request.state.db_operations

    unique_event_id = f"ifgames-{event_id}"

    event = await db_operations.get_event(unique_event_id=unique_event_id)

    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    current_interval = get_interval_start_minutes()

    predictions = await db_operations.get_predictions_for_event(
        unique_event_id=unique_event_id, interval_start_minutes=current_interval
    )

    return {"count": len(predictions), "predictions": predictions}
