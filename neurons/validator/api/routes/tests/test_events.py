from unittest.mock import ANY

import pytest
from httpx import AsyncClient

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.utils.common.interval import get_interval_start_minutes


class TestEventsRoutes:
    @pytest.fixture
    async def mock_event(self, test_db_operations: DatabaseOperations):
        event_id = "event_id"
        unique_event_id = f"ifgames-{event_id}"

        events = [
            EventsModel(
                unique_event_id=unique_event_id,
                event_id=event_id,
                market_type="market_type",
                event_type="event_type",
                description="desc",
                status=EventStatus.SETTLED,
                metadata="{}",
                created_at="2012-12-02T14:30:00Z",
            ),
        ]

        await test_db_operations.upsert_events(events=events)

        return {"events": events}

    async def test_get_event(
        self, test_api_client: AsyncClient, mock_event: dict[str, list[EventsModel]]
    ):
        event = mock_event["events"][0]

        response = await test_api_client.get(f"/api/events/{event.event_id}")

        assert response.status_code == 200
        assert response.json() == {
            "event_id": "event_id",
            "unique_event_id": "ifgames-event_id",
            "created_at": "2012-12-02T14:30:00Z",
            "cutoff": None,
            "deleted_at": None,
            "description": "desc",
            "event_type": "event_type",
            "exported": False,
            "local_updated_at": ANY,
            "market_type": "market_type",
            "metadata": "{}",
            "outcome": None,
            "processed": False,
            "registered_date": ANY,
            "resolved_at": None,
            "status": 3,
        }

        # Test not found event
        response = await test_api_client.get("/api/events/not_found_event_id")

        assert response.status_code == 404
        assert response.json() == {"detail": "Event not found"}

    async def test_get_event_auth(
        self,
        test_api_client_no_auth: AsyncClient,
    ):
        response = await test_api_client_no_auth.get("/api/events/event_id")

        assert response.status_code == 401
        assert response.json() == {"detail": "Unauthorized"}

    async def test_get_event_community_prediction(
        self,
        test_api_client: AsyncClient,
        mock_event: dict[str, list[EventsModel]],
    ):
        # Test not found event
        response = await test_api_client.get("/api/events/not_found_event_id/community_prediction")

        assert response.status_code == 404
        assert response.json() == {"detail": "Event not found"}

        event = mock_event["events"][0]

        # Test no community prediction for the event yet
        response = await test_api_client.get(f"/api/events/{event.event_id}/community_prediction")

        assert response.status_code == 200
        assert response.json() == {
            "community_prediction": None,
            "community_prediction_lr": None,
            "event_id": event.event_id,
        }

    async def test_get_event_community_prediction_auth(
        self,
        test_api_client_no_auth: AsyncClient,
    ):
        response = await test_api_client_no_auth.get("/api/events/event_id/community_prediction")

        assert response.status_code == 401
        assert response.json() == {"detail": "Unauthorized"}

    async def test_get_community_predictions(
        self,
        test_api_client: AsyncClient,
    ):
        # Test / route belongs to get event
        response = await test_api_client.get("/api/events/community_predictions")

        assert response.status_code == 404
        assert response.json() == {"detail": "Event not found"}

        # Test get community predictions needs event ids query params
        response = await test_api_client.get("/api/events/community_predictions/")

        assert response.status_code == 422
        assert response.json() == {
            "detail": [
                {
                    "input": None,
                    "loc": [
                        "query",
                        "event_ids",
                    ],
                    "msg": "Input should be a valid list",
                    "type": "list_type",
                },
            ],
        }

        # Test get community predictions needs event ids query params
        response = await test_api_client.get(
            "/api/events/community_predictions/?event_ids=event1&event_ids=event2"
        )

        assert response.status_code == 200
        assert response.json() == {
            "community_predictions": [
                {
                    "community_prediction": None,
                    "community_prediction_lr": None,
                    "event_id": "event1",
                },
                {
                    "community_prediction": None,
                    "community_prediction_lr": None,
                    "event_id": "event2",
                },
            ],
            "count": 2,
        }

    async def test_get_community_predictions_max_events(
        self,
        test_api_client: AsyncClient,
    ):
        query_params = "&".join([f"event_ids=event{i}" for i in range(1, 21)])

        response = await test_api_client.get(f"/api/events/community_predictions/?{query_params}")

        assert response.status_code == 200

        query_params = query_params + "&event_ids=event21"

        response = await test_api_client.get(f"/api/events/community_predictions/?{query_params}")

        assert response.status_code == 422

    async def test_get_community_predictions_auth(
        self,
        test_api_client_no_auth: AsyncClient,
    ):
        response = await test_api_client_no_auth.get("/api/events/community_prediction/")

        assert response.status_code == 401
        assert response.json() == {"detail": "Unauthorized"}

    async def test_get_event_predictions(
        self,
        test_api_client: AsyncClient,
        mock_event: dict[str, list[EventsModel]],
        test_db_operations: DatabaseOperations,
    ):
        # Test not found event
        response = await test_api_client.get("/api/events/not_found_event_id/predictions")

        assert response.status_code == 404
        assert response.json() == {"detail": "Event not found"}

        event = mock_event["events"][0]

        # Test no predictions for the event yet
        response = await test_api_client.get(f"/api/events/{event.event_id}/predictions")

        assert response.status_code == 200
        assert response.json() == {"count": 0, "predictions": []}

        # Test past and current interval predictions
        old_interval = 1
        current_interval = get_interval_start_minutes()

        predictions = [
            (event.unique_event_id, "neuronHotkey_99", 99, 1, old_interval, 1),
            (event.unique_event_id, "neuronHotkey_2", 2, 1, current_interval, 1),
        ]

        await test_db_operations.upsert_predictions(predictions)

        response = await test_api_client.get(f"/api/events/{event.event_id}/predictions")

        assert response.status_code == 200
        assert response.json() == {
            "count": 1,
            "predictions": [
                {
                    "exported": False,
                    "interval_agg_prediction": 1.0,
                    "interval_count": 1,
                    "interval_start_minutes": current_interval,
                    "latest_prediction": 1.0,
                    "miner_hotkey": "neuronHotkey_2",
                    "miner_uid": 2,
                    "submitted": ANY,
                    "unique_event_id": event.unique_event_id,
                    "updated_at": ANY,
                },
            ],
        }

    async def test_get_event_predictions_auth(
        self,
        test_api_client_no_auth: AsyncClient,
    ):
        response = await test_api_client_no_auth.get("/api/events/event_id/predictions")

        assert response.status_code == 401
        assert response.json() == {"detail": "Unauthorized"}
