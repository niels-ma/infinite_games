import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from aioresponses import aioresponses
from bittensor_wallet import Wallet

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.if_games.client import IfGamesClient
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.tasks.resolve_events import ResolveEvents
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class TestResolveEventsTask:
    @pytest.fixture
    def db_operations_mock(self):
        db_operations = AsyncMock(spec=DatabaseOperations)

        return db_operations

    @pytest.fixture
    def api_client_mock(self):
        api_client = AsyncMock(spec=IfGamesClient)

        return api_client

    @pytest.fixture
    def bt_wallet(self):
        hotkey_mock = MagicMock()
        hotkey_mock.sign = MagicMock(side_effect=lambda x: x.encode("utf-8"))
        hotkey_mock.ss58_address = "ss58_address"

        bt_wallet = MagicMock(spec=Wallet)
        bt_wallet.get_hotkey = MagicMock(return_value=hotkey_mock)

        return bt_wallet

    @pytest.fixture
    def db_operations(self, db_client: DatabaseClient):
        logger = MagicMock(spec=InfiniteGamesLogger)

        return DatabaseOperations(db_client=db_client, logger=logger)

    @pytest.fixture
    def logger_mock(self):
        return MagicMock(spec=InfiniteGamesLogger)

    @pytest.fixture
    def resolve_events_task(
        self, db_operations: DatabaseOperations, bt_wallet: Wallet, logger_mock: InfiniteGamesLogger
    ):
        api_client = IfGamesClient(env="test", logger=logger_mock, bt_wallet=bt_wallet)

        return ResolveEvents(
            interval_seconds=60.0,
            db_operations=db_operations,
            api_client=api_client,
            logger=logger_mock,
            page_size=10,
        )

    async def test_run_no_pending_events(
        self,
        db_operations: DatabaseOperations,
        api_client_mock: MagicMock,
        logger_mock: InfiniteGamesLogger,
    ):
        """Test the run method when there are no pending events."""

        # Arrange
        resolve_events_task = ResolveEvents(
            interval_seconds=60.0,
            db_operations=db_operations,
            api_client=api_client_mock,
            logger=logger_mock,
            page_size=10,
        )

        # Act
        await resolve_events_task.run()

        # Assert
        api_client_mock.get_resolved_events.assert_not_called()

    async def test_resolved_from_last_resolved(
        self, db_operations: DatabaseOperations, resolve_events_task: ResolveEvents
    ):
        # Arrange
        events = [
            EventsModel(
                unique_event_id="unique_1",
                event_id="event_id_1",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome=None,
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unique_2",
                event_id="event_id_2",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome=None,
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events=events)

        current_time = datetime.now(timezone.utc)
        event_1_resolved_at = current_time.isoformat()
        event_2_resolved_at = (current_time + timedelta(seconds=100)).isoformat()

        await db_operations.resolve_event(
            event_id=events[0].event_id, outcome=1, resolved_at=event_1_resolved_at, forecasts="{}"
        )
        await db_operations.resolve_event(
            event_id=events[1].event_id,
            outcome=1,
            resolved_at=event_2_resolved_at,
            forecasts=str({"2000-12-02T14:30:00Z": 0.98}),
        )

        api_response = {"count": 0, "items": []}

        # Mock API response
        with aioresponses() as mocked:
            backdated_resolved_since = (
                (datetime.fromisoformat(event_2_resolved_at) - timedelta(hours=1))
                .isoformat()
                .replace("+00:00", "Z")
            )

            mocked.get(
                f"/api/v2/events/resolved?resolved_since={backdated_resolved_since}&offset=0&limit=10",
                status=200,
                body=json.dumps(api_response).encode("utf-8"),
            )

            # Act
            await resolve_events_task.run()

        # Assert
        assert True is True

    async def test_resolved_from_first_created(
        self, db_operations: DatabaseOperations, resolve_events_task: ResolveEvents
    ):
        # Arrange
        events = [
            EventsModel(
                unique_event_id="unique_1",
                event_id="event_id_1",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome=None,
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unique_2",
                event_id="event_id_2",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome=None,
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="1512-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events=events)

        api_response = {"count": 0, "items": []}

        # Mock API response
        with aioresponses() as mocked:
            backdated_resolved_since = (
                (events[1].created_at - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
            )

            mocked.get(
                f"/api/v2/events/resolved?resolved_since={backdated_resolved_since}&offset=0&limit=10",
                status=200,
                body=json.dumps(api_response).encode("utf-8"),
            )

            # Act
            await resolve_events_task.run()

        # Assert
        assert True is True

    async def test_resolve_events(
        self,
        db_client: DatabaseClient,
        db_operations: DatabaseOperations,
        resolve_events_task: ResolveEvents,
    ):
        """Test that task resolves correctly pending events."""
        # Arrange
        resolve_events_task.page_size = 1

        events = [
            EventsModel(
                unique_event_id="unique_1",
                event_id="event_id_1",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome=None,
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unique_2",
                event_id="event_id_2",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome=None,
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="1900-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unique_3",
                event_id="event_id_3",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome=None,
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unique_4",
                event_id="event_id_4",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome=None,
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events=events)

        api_response_1 = {
            "count": 1,
            "items": [
                {
                    "event_id": events[3].event_id,
                    "market_type": "MARKET_TYPE",
                    "answer": 1,
                    "created_at": "2012-09-01T20:43:02Z",
                    "resolved_at": "2012-09-10T20:43:02Z",
                    "forecasts": {"2012-09-01T20:43:02Z": 0.07, "2012-09-01T22:43:02Z": 0.51},
                }
            ],
        }

        api_response_2 = {
            "count": 1,
            "items": [
                {
                    "event_id": events[2].event_id,
                    "market_type": "MARKET_TYPE",
                    "answer": 0,
                    "created_at": "2012-09-01T20:43:02Z",
                    "resolved_at": "2024-09-11T20:43:02Z",
                    "forecasts": {},
                }
            ],
        }

        api_response_3 = {
            "count": 0,
            "items": [],
        }

        api_response_4 = {
            "count": 1,
            "items": [
                {
                    "event_id": events[1].event_id,
                    "market_type": "MARKET_TYPE",
                    "answer": 1,
                    "created_at": "2012-09-01T20:43:02Z",
                    "resolved_at": "2024-09-10T20:43:02Z",
                    "forecasts": {"2012-09-01T20:43:02Z": 0.07, "2012-09-01T22:43:02Z": 0.51},
                }
            ],
        }

        api_response_5 = {
            "count": 0,
            "items": [],
        }

        # Mock API response
        with aioresponses() as mocked:

            def backdate(iso_date: str):
                return (
                    (datetime.fromisoformat(iso_date) - timedelta(hours=1))
                    .isoformat()
                    .replace("+00:00", "Z")
                )

            first_request_resolved_since = backdate(events[1].created_at.isoformat())

            second_request_resolved_since = backdate("2024-09-11T20:43:02+00:00")

            mocked.get(
                f"/api/v2/events/resolved?resolved_since={first_request_resolved_since}&offset=0&limit=1",
                status=200,
                body=json.dumps(api_response_1).encode("utf-8"),
            )

            mocked.get(
                f"/api/v2/events/resolved?resolved_since={first_request_resolved_since}&offset=1&limit=1",
                status=200,
                body=json.dumps(api_response_2).encode("utf-8"),
            )

            mocked.get(
                f"/api/v2/events/resolved?resolved_since={first_request_resolved_since}&offset=2&limit=1",
                status=200,
                body=json.dumps(api_response_3).encode("utf-8"),
            )

            mocked.get(
                f"/api/v2/events/resolved?resolved_since={second_request_resolved_since}&offset=0&limit=1",
                status=200,
                body=json.dumps(api_response_4).encode("utf-8"),
            )

            mocked.get(
                f"/api/v2/events/resolved?resolved_since={second_request_resolved_since}&offset=1&limit=1",
                status=200,
                body=json.dumps(api_response_5).encode("utf-8"),
            )

            # Act
            await resolve_events_task.run()

            # Assert
            response = await db_client.many(
                """
                    SELECT status, outcome, resolved_at, forecasts from events
                """
            )

            assert len(response) == 4
            assert response[0][0] == str(EventStatus.PENDING.value)
            assert response[1][0] == str(EventStatus.PENDING.value)
            assert response[2][0] == str(EventStatus.SETTLED.value)
            assert response[3][0] == str(EventStatus.SETTLED.value)

            # Events 3 & 4 are settled
            assert response[2][1] == "0"
            assert response[2][2] == "2024-09-11 20:43:02+00:00"
            assert response[3][1] == "1"
            assert response[3][2] == "2012-09-10 20:43:02+00:00"

            # Check forecasts
            assert response[0][3] == "{}"
            assert response[1][3] == "{}"
            assert response[2][3] == "{}"
            assert response[3][3] == '{"2012-09-01T20:43:02Z": 0.07, "2012-09-01T22:43:02Z": 0.51}'

            # Act run again
            await resolve_events_task.run()

            # Assert
            response = await db_client.many(
                """
                    SELECT status, outcome, resolved_at, forecasts from events
                """
            )

            assert len(response) == 4
            assert response[0][0] == str(EventStatus.PENDING.value)
            assert response[1][0] == str(EventStatus.SETTLED.value)
            assert response[2][0] == str(EventStatus.SETTLED.value)
            assert response[3][0] == str(EventStatus.SETTLED.value)

            # Event 2 is settled now
            assert response[1][1] == "1"
            assert response[1][2] == "2024-09-10 20:43:02+00:00"

            # Check forecasts
            assert response[0][3] == "{}"
            assert response[1][3] == '{"2012-09-01T20:43:02Z": 0.07, "2012-09-01T22:43:02Z": 0.51}'
            assert response[2][3] == "{}"
            assert response[3][3] == '{"2012-09-01T20:43:02Z": 0.07, "2012-09-01T22:43:02Z": 0.51}'

            # Ensure valid JSON
            assert json.loads(response[3][3]) == {
                "2012-09-01T20:43:02Z": 0.07,
                "2012-09-01T22:43:02Z": 0.51,
            }
