import json
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from aioresponses import aioresponses
from bittensor_wallet import Wallet

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.if_games.client import IfGamesClient
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.tasks.delete_events import DeleteEvents
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class TestDeleteEventsTask:
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

    @pytest.fixture(scope="function")
    async def db_client(self):
        temp_db = tempfile.NamedTemporaryFile(delete=False)
        db_path = temp_db.name
        temp_db.close()

        logger = MagicMock(spec=InfiniteGamesLogger)

        db_client = DatabaseClient(db_path, logger)

        await db_client.migrate()

        return db_client

    @pytest.fixture
    def db_operations(self, db_client: DatabaseClient):
        logger = MagicMock(spec=InfiniteGamesLogger)

        return DatabaseOperations(db_client=db_client, logger=logger)

    @pytest.fixture
    def logger_mock(self):
        return MagicMock(spec=InfiniteGamesLogger)

    @pytest.fixture
    def delete_events_task(
        self, db_operations: DatabaseOperations, bt_wallet: Wallet, logger_mock: InfiniteGamesLogger
    ):
        api_client = IfGamesClient(env="test", logger=logger_mock, bt_wallet=bt_wallet)

        return DeleteEvents(
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
        resolve_events_task = DeleteEvents(
            interval_seconds=60.0,
            db_operations=db_operations,
            api_client=api_client_mock,
            logger=logger_mock,
            page_size=10,
        )

        # Act
        await resolve_events_task.run()

        # Assert
        api_client_mock.get_events_deleted.assert_not_called()

    async def test_delete_events(
        self,
        db_client: DatabaseClient,
        db_operations: DatabaseOperations,
        delete_events_task: DeleteEvents,
    ):
        """Test that task deletes correctly deleted events."""
        # Arrange
        delete_events_task.page_size = 1

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
                created_at="1950-12-02T14:30:00+00:00",
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
                created_at="1900-12-02T14:30:00+00:00",
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
                    "deleted_at": "2012-09-10T20:43:02Z",
                }
            ],
        }

        api_response_2 = {
            "count": 1,
            "items": [
                {
                    "event_id": events[2].event_id,
                    "deleted_at": "2024-09-11T20:43:02Z",
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
                    "deleted_at": "2024-09-05T20:43:02Z",
                }
            ],
        }

        api_response_5 = {
            "count": 0,
            "items": [],
        }

        # Mock API response
        with aioresponses() as mocked:
            first_request_deleted_since = "1900-12-02T14:30:00Z"

            second_request_deleted_since = "2024-09-11T20:43:02Z"

            mocked.get(
                f"/api/v2/events/deleted?deleted_since={first_request_deleted_since}&offset=0&limit=1",
                status=200,
                body=json.dumps(api_response_1).encode("utf-8"),
            )

            mocked.get(
                f"/api/v2/events/deleted?deleted_since={first_request_deleted_since}&offset=1&limit=1",
                status=200,
                body=json.dumps(api_response_2).encode("utf-8"),
            )

            mocked.get(
                f"/api/v2/events/deleted?deleted_since={first_request_deleted_since}&offset=2&limit=1",
                status=200,
                body=json.dumps(api_response_3).encode("utf-8"),
            )

            mocked.get(
                f"/api/v2/events/deleted?deleted_since={second_request_deleted_since}&offset=0&limit=1",
                status=200,
                body=json.dumps(api_response_4).encode("utf-8"),
            )

            mocked.get(
                f"/api/v2/events/deleted?deleted_since={second_request_deleted_since}&offset=1&limit=1",
                status=200,
                body=json.dumps(api_response_5).encode("utf-8"),
            )

            # Act
            await delete_events_task.run()

            # Assert
            response = await db_client.many(
                """
                    SELECT event_id, status, deleted_at from events
                """
            )

            # Event 3 and 4 are soft deleted, events 1 and 2 are left
            assert len(response) == 4
            assert response == [
                (events[0].event_id, str(EventStatus.PENDING.value), None),
                (events[1].event_id, str(EventStatus.PENDING.value), None),
                (events[2].event_id, str(EventStatus.DELETED.value), "2024-09-11 20:43:02+00:00"),
                (events[3].event_id, str(EventStatus.DELETED.value), "2012-09-10 20:43:02+00:00"),
            ]

            # Act run again
            await delete_events_task.run()

            # Assert
            response = await db_client.many(
                """
                    SELECT event_id, status, deleted_at from events
                """
            )

            # Event 2 is soft deleted too, event 1 is left
            assert len(response) == 4
            assert response == [
                (events[0].event_id, str(EventStatus.PENDING.value), None),
                (events[1].event_id, str(EventStatus.DELETED.value), "2024-09-05 20:43:02+00:00"),
                (events[2].event_id, str(EventStatus.DELETED.value), "2024-09-11 20:43:02+00:00"),
                (events[3].event_id, str(EventStatus.DELETED.value), "2012-09-10 20:43:02+00:00"),
            ]
