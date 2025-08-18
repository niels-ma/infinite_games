import json
import math
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from aioresponses import aioresponses
from bittensor_wallet import Wallet

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.if_games.client import IfGamesClient
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.models.if_games_client import GetEventsResponse, IfGamesEvent
from neurons.validator.tasks.pull_events import PullEvents
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class TestPullEventsTask:
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
    def pull_events_task(self, db_client: DatabaseClient, bt_wallet: Wallet):
        logger = MagicMock(spec=InfiniteGamesLogger)
        db_operations = DatabaseOperations(db_client=db_client, logger=logger)
        api_client = IfGamesClient(env="test", logger=logger, bt_wallet=bt_wallet)

        return PullEvents(
            interval_seconds=60.0,
            db_operations=db_operations,
            api_client=api_client,
            page_size=1,
        )

    def test_parse_event(self, db_operations_mock, api_client_mock):
        """Test the parse_event method for correctness."""
        # Arrange
        event = IfGamesEvent.model_validate(
            {
                "event_id": "123",
                "title": "Test Event?",
                "description": "This is a test.",
                "end_date": 1700003600,
                "created_at": 1699996400,
                "answer": None,
                "market_type": "TYpe1",
                "cutoff": 1699996800,
                "event_metadata": {"topics": [], "trigger_name": None},
            }
        )

        pull_events_task = PullEvents(
            interval_seconds=60.0,
            db_operations=db_operations_mock,
            api_client=api_client_mock,
            page_size=100,
        )

        # Act
        parsed_event = pull_events_task.parse_event(event)

        # Assert
        assert parsed_event == EventsModel(
            unique_event_id=f"ifgames-{event.event_id}",
            event_id=event.event_id,
            market_type="ifgames",
            event_type=event.market_type.lower(),
            registered_date=None,
            description="Test Event? ==Further Information==: This is a test.",
            outcome=None,
            local_updated_at=None,
            status=EventStatus.PENDING,
            metadata='{"topics": [], "trigger_name": null}',
            processed=False,
            exported=False,
            created_at=datetime.fromtimestamp(1699996400, tz=timezone.utc),
            cutoff=datetime.fromtimestamp(1699996800, tz=timezone.utc),
            resolved_at=None,
            deleted_at=None,
        )

    def test_parse_event_no_metadata(self, db_operations_mock, api_client_mock):
        # Arrange
        event = IfGamesEvent.model_validate(
            {
                "event_id": "123",
                "title": "Test Event?",
                "description": "This is a test.",
                "end_date": 1700003600,
                "created_at": 1699996400,
                "answer": None,
                "market_type": "TYpe1",
                "cutoff": 1699996800,
                "event_metadata": None,
            }
        )

        pull_events_task = PullEvents(
            interval_seconds=60.0,
            db_operations=db_operations_mock,
            api_client=api_client_mock,
            page_size=100,
        )

        # Act
        parsed_event = pull_events_task.parse_event(event)

        # Assert
        assert parsed_event == EventsModel(
            unique_event_id=f"ifgames-{event.event_id}",
            event_id=event.event_id,
            market_type="ifgames",
            event_type=event.market_type.lower(),
            registered_date=None,
            description="Test Event? ==Further Information==: This is a test.",
            outcome=None,
            local_updated_at=None,
            status=EventStatus.PENDING,
            metadata="{}",
            processed=False,
            exported=False,
            created_at=datetime.fromtimestamp(1699996400, tz=timezone.utc),
            cutoff=datetime.fromtimestamp(1699996800, tz=timezone.utc),
            resolved_at=None,
            deleted_at=None,
        )

    async def test_run_no_events(self, db_operations_mock, api_client_mock):
        """Test the run method when there are no events."""
        # Arrange
        page_size = 100

        db_operations_mock.get_last_event_from.return_value = None

        api_client_mock.get_events.return_value = GetEventsResponse.model_validate({"items": []})

        pull_events_task = PullEvents(
            interval_seconds=60.0,
            db_operations=db_operations_mock,
            api_client=api_client_mock,
            page_size=page_size,
        )

        # Act
        await pull_events_task.run()

        # Assert
        api_client_mock.get_events.assert_called_with(0, 0, page_size)
        db_operations_mock.upsert_events.assert_not_called()

    async def test_start_from_empty(self, db_client: DatabaseClient, pull_events_task: PullEvents):
        """Test that pulls events from 0 and iterates until the end."""
        # Arrange
        mock_response_data_1 = {
            "count": -5,
            "items": [
                {
                    "event_id": "0x123456789abcdef123456789abcdef123456789abcdef123456789abcdef1234",
                    "cutoff": (datetime.now(timezone.utc) + timedelta(minutes=1)).timestamp(),
                    "title": "Will Tesla stock price reach $2000 by 2025?",
                    "description": (
                        "This market will resolve to 'Yes' if the closing stock price of Tesla reaches or exceeds $2000 "
                        "on any trading day in 2025. Otherwise, this market will resolve to 'No'.\n\n"
                        "Resolution source: NASDAQ official data."
                    ),
                    "market_type": "BINARY",
                    "created_at": 1733200000,
                    "end_date": 1733620000,
                    "event_metadata": {},
                },
            ],
        }

        mock_response_data_2 = {
            "count": -5,
            "items": [
                {
                    "event_id": "0xabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef",
                    "cutoff": (datetime.now(timezone.utc) + timedelta(minutes=1)).timestamp(),
                    "title": "Will AI surpass human intelligence by 2030?",
                    "description": (
                        "This market will resolve to 'Yes' if credible sources, including major AI researchers, "
                        "announce that AI has surpassed general human intelligence before December 31, 2030."
                    ),
                    "market_type": "POLYMARKET",
                    "created_at": 1733210000,
                    "end_date": 1733621000,
                    "event_metadata": {},
                },
            ],
        }

        mock_response_data_3 = {
            "count": -5,
            "items": [],
        }

        # Mock API response
        with aioresponses() as mocked:
            mocked.get(
                "/api/v2/events?from_date=0&offset=0&limit=1",
                status=200,
                body=json.dumps(mock_response_data_1).encode("utf-8"),
            )

            mocked.get(
                "/api/v2/events?from_date=0&offset=1&limit=1",
                status=200,
                body=json.dumps(mock_response_data_2).encode("utf-8"),
            )

            mocked.get(
                "/api/v2/events?from_date=0&offset=2&limit=1",
                status=200,
                body=json.dumps(mock_response_data_3).encode("utf-8"),
            )

            # Act
            await pull_events_task.run()

        # Assert
        response = await db_client.many(
            """
                SELECT * from events
            """
        )

        assert len(response) == 2
        assert response[0][1] == mock_response_data_1["items"][0]["event_id"]
        assert response[1][1] == mock_response_data_2["items"][0]["event_id"]

    async def test_start_from_last(self, db_client: DatabaseClient, pull_events_task: PullEvents):
        """Test that pulls events from where it left and iterates until the end."""
        # Arrange
        mock_response_data_1 = {
            "count": -5,
            "items": [
                {
                    "event_id": "0x123456789abcdef123456789abcdef123456789abcdef123456789abcdef1234",
                    "cutoff": (datetime.now(timezone.utc) + timedelta(minutes=1)).timestamp(),
                    "title": "Will Tesla stock price reach $2000 by 2025?",
                    "description": (
                        "This market will resolve to 'Yes' if the closing stock price of Tesla reaches or exceeds $2000 "
                        "on any trading day in 2025. Otherwise, this market will resolve to 'No'.\n\n"
                        "Resolution source: NASDAQ official data."
                    ),
                    "market_type": "BINARY",
                    "created_at": 1733200000,
                    "end_date": 1733620000,
                    "event_metadata": {},
                },
            ],
        }

        mock_response_data_2 = {
            "count": -5,
            "items": [],
        }

        mock_response_data_3 = {
            "count": -5,
            "items": [
                {
                    "event_id": "0xabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef",
                    "cutoff": (datetime.now(timezone.utc) + timedelta(minutes=1)).timestamp(),
                    "title": "Will AI surpass human intelligence by 2030?",
                    "description": (
                        "This market will resolve to 'Yes' if credible sources, including major AI researchers, "
                        "announce that AI has surpassed general human intelligence before December 31, 2030."
                    ),
                    "market_type": "POLYMARKET",
                    "created_at": 1733210000,
                    "end_date": 1733621000,
                    "event_metadata": {},
                },
            ],
        }

        mock_response_data_4 = {
            "count": -5,
            "items": [],
        }

        from_last = math.floor(
            (
                datetime.fromtimestamp(mock_response_data_1["items"][0]["created_at"])
                - timedelta(seconds=1)
            ).timestamp()
        )

        # Mock API response
        with aioresponses() as mocked:
            mocked.get(
                "/api/v2/events?from_date=0&offset=0&limit=1",
                status=200,
                body=json.dumps(mock_response_data_1).encode("utf-8"),
            )

            mocked.get(
                "/api/v2/events?from_date=0&offset=1&limit=1",
                status=200,
                body=json.dumps(mock_response_data_2).encode("utf-8"),
            )

            mocked.get(
                f"/api/v2/events?from_date={from_last}&offset=0&limit=1",
                status=200,
                body=json.dumps(mock_response_data_3).encode("utf-8"),
            )

            mocked.get(
                f"/api/v2/events?from_date={from_last}&offset=1&limit=1",
                status=200,
                body=json.dumps(mock_response_data_4).encode("utf-8"),
            )

            # Act
            await pull_events_task.run()
            # Run a second pass
            await pull_events_task.run()

        # Assert
        response = await db_client.many(
            """
                SELECT * from events
            """
        )

        assert len(response) == 2
        assert response[0][1] == mock_response_data_1["items"][0]["event_id"]
        assert response[1][1] == mock_response_data_3["items"][0]["event_id"]
