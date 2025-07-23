import asyncio
import json
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import ANY, MagicMock

import pytest
from aiosqlite import IntegrityError

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.db.tests.test_utils import TestDbOperationsBase
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.models.miner import MinersModel
from neurons.validator.models.prediction import PredictionExportedStatus
from neurons.validator.models.score import ScoresModel
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class TestDbOperationsPart1(TestDbOperationsBase):
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
    async def db_operations(self, db_client: DatabaseClient):
        logger = MagicMock(spec=InfiniteGamesLogger)

        db_operations = DatabaseOperations(db_client=db_client, logger=logger)

        return db_operations

    def test_parse_rows_valid_data(self, db_operations: DatabaseOperations):
        valid_data = [
            {
                "miner_uid": "123",
                "miner_hotkey": "abc",
                "registered_date": datetime(2024, 1, 1, 11, 0, 1),
                "is_validating": True,
                "validator_permit": True,
            },
            {
                "miner_uid": "456",
                "miner_hotkey": "def",
                "registered_date": datetime(2024, 1, 1, 11, 0, 1),
                "is_validating": False,
                "validator_permit": False,
            },
        ]

        result = db_operations._parse_rows(MinersModel, valid_data)

        assert len(result) == 2
        assert isinstance(result[0], MinersModel)

        assert result[0].miner_uid == "123"
        assert result[1].miner_uid == "456"

    def test_parse_rows_invalid_data_not_throw(self, db_operations: DatabaseOperations):
        invalid_data = [
            # Missing required fields
            {"miner_uid": "123", "miner_hotkey": "abc"},
            {
                "miner_uid": "456",
                "miner_hotkey": "def",
                "registered_date": datetime(2024, 1, 1, 11, 0, 1),
                "is_validating": False,
                "validator_permit": False,
            },
        ]

        result = db_operations._parse_rows(MinersModel, invalid_data)

        # Only one valid row should be parsed
        assert len(result) == 1

        assert isinstance(result[0], MinersModel)
        assert result[0].miner_uid == "456"

        db_operations.logger.exception.assert_called_once_with(
            "Error parsing model",
            extra={"row": {"miner_uid": "123", "miner_hotkey": "abc"}, "model": "MinersModel"},
        )

    def test_parse_rows_invalid_data_throw(self, db_operations: DatabaseOperations):
        invalid_data = [{"miner_uid": "123", "miner_hotkey": "abc"}]

        with pytest.raises(Exception):
            db_operations._parse_rows(MinersModel, invalid_data, throw_on_error=True)

            db_operations.logger.exception.assert_not_called()

    def test_parse_rows_empty_data(self, db_operations: DatabaseOperations):
        empty_data = []

        result = db_operations._parse_rows(MinersModel, empty_data)

        assert len(result) == 0
        assert isinstance(result, list)

    async def test_delete_event(self, db_operations: DatabaseOperations, db_client: DatabaseClient):
        event_id_to_keep = "event1"
        event_id_to_delete = "event2"

        events = [
            EventsModel(
                unique_event_id="unique1",
                event_id=event_id_to_keep,
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-02T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unique2",
                event_id=event_id_to_delete,
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-02T14:30:00+00:00",
            ),
        ]

        deleted_at = datetime.now(timezone.utc).isoformat()

        await db_operations.upsert_events(events=events)

        # Set local updated at null to check it is updated by the delete operation
        await db_client.update(
            """
                UPDATE events SET local_updated_at = NULL
            """
        )

        result = await db_operations.delete_event(
            event_id=event_id_to_delete, deleted_at=deleted_at
        )

        assert len(result) == 1
        assert result[0][0] == event_id_to_delete

        # Verify the event still exists but is marked as DELETED and has deleted_at set
        result = await db_client.many(
            """
                SELECT event_id, status, deleted_at, local_updated_at FROM events
            """
        )

        assert len(result) == 2
        assert result[0][0] == event_id_to_keep
        assert result[0][1] == str(EventStatus.SETTLED.value)
        assert result[0][2] is None  # deleted_at should be None for non-deleted event
        assert result[0][3] is None

        assert result[1][0] == event_id_to_delete
        assert result[1][1] == str(EventStatus.DELETED.value)
        assert result[1][2] == deleted_at.replace(
            "Z", "+00:00"
        )  # deleted_at should be set for deleted event
        assert isinstance(result[1][3], str) is True

    async def test_delete_predictions_processed_unprocessed_events(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        events = [
            EventsModel(
                unique_event_id="processed_event_prediction_id",
                event_id="event1",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-01-01T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unprocessed_event_prediction_id",
                event_id="event2",
                market_type="truncated_market2",
                event_type="market_2",
                description="desc2",
                outcome="outcome2",
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="2012-12-02T14:30:00+00:00",
                cutoff="2001-01-01T14:30:00+00:00",
            ),
        ]

        predictions = [
            (
                "processed_event_prediction_id",
                "neuronHotkey_2",
                2,
                1,
                10,
                1,
            ),
            ("unprocessed_event_prediction_id", "neuronHotkey_2", 2, 1, 10, 1),
        ]

        await db_operations.upsert_events(events=events)
        await db_operations.upsert_predictions(predictions=predictions)

        # Mark processed event as processed
        await db_client.update(
            """
                UPDATE
                    events
                SET
                    processed = ?,
                    resolved_at = ?
                WHERE
                    unique_event_id = ?
            """,
            [
                True,
                (datetime.now(timezone.utc) - timedelta(days=4, hours=1)).isoformat(),
                events[0].unique_event_id,
            ],
        )

        # Mark all predictions as exported but not exported one
        await db_client.update(
            """
                UPDATE
                    predictions
                SET
                    exported = ?
                """,
            [PredictionExportedStatus.EXPORTED],
        )

        # delete
        result = await db_operations.delete_predictions(batch_size=100)

        assert len(result) == 1
        assert result[0][0] == 1

        result = await db_client.many(
            """
                SELECT unique_event_id FROM predictions ORDER BY ROWID ASC
            """
        )

        # Should have deleted the processed event prediction
        assert len(result) == 1
        assert result[0][0] == "unprocessed_event_prediction_id"

    async def test_delete_predictions_discarded_and_deleted_events(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        events = [
            EventsModel(
                unique_event_id="discarded_event_prediction_id",
                event_id="event3",
                market_type="truncated_market3",
                event_type="market_3",
                description="desc2",
                outcome="outcome3",
                status=EventStatus.DISCARDED,
                metadata='{"key": "value"}',
                created_at="2012-12-02T14:30:00+00:00",
                cutoff="2001-01-01T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="deleted_event_prediction_id",
                event_id="event3",
                market_type="truncated_market3",
                event_type="market_3",
                description="desc2",
                outcome="outcome3",
                status=EventStatus.DELETED,
                metadata='{"key": "value"}',
                created_at="2012-12-02T14:30:00+00:00",
                cutoff="2001-01-01T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="pending_event_prediction_id",
                event_id="event3",
                market_type="truncated_market3",
                event_type="market_3",
                description="desc2",
                outcome="outcome3",
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="2012-12-02T14:30:00+00:00",
                cutoff="2001-01-01T14:30:00+00:00",
            ),
        ]

        predictions = [
            (
                "discarded_event_prediction_id",
                "neuronHotkey_2",
                2,
                1,
                10,
                1,
            ),
            (
                "deleted_event_prediction_id",
                "neuronHotkey_2",
                2,
                1,
                10,
                1,
            ),
            (
                "pending_event_prediction_id",
                "neuronHotkey_2",
                2,
                1,
                10,
                1,
            ),
        ]

        await db_operations.upsert_events(events=events)
        await db_operations.upsert_predictions(predictions=predictions)

        # Mark all predictions as exported
        await db_client.update(
            """
                UPDATE
                    predictions
                SET
                    exported = ?
                """,
            [PredictionExportedStatus.EXPORTED],
        )

        # delete
        result = await db_operations.delete_predictions(batch_size=100)

        assert len(result) == 2
        assert result == [(1,), (2,)]

        result = await db_client.many(
            """
                SELECT unique_event_id FROM predictions ORDER BY ROWID ASC
            """
        )

        assert len(result) == 1
        assert result[0][0] == "pending_event_prediction_id"

    async def test_delete_predictions_exported_unexported(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        events = [
            EventsModel(
                unique_event_id="exported_prediction_event_id",
                event_id="event1",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-01-01T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="not_exported_prediction_event_id",
                event_id="event4",
                market_type="truncated_market4",
                event_type="market_4",
                description="desc4",
                outcome="outcome4",
                status=EventStatus.DELETED,
                metadata='{"key": "value"}',
                created_at="2012-12-02T14:30:00+00:00",
                cutoff="2001-01-01T14:30:00+00:00",
            ),
        ]

        predictions = [
            (
                "exported_prediction_event_id",
                "neuronHotkey_1",
                1,
                1,
                10,
                1,
            ),
            (
                "not_exported_prediction_event_id",
                "neuronHotkey_2",
                2,
                1,
                10,
                1,
            ),
        ]

        await db_operations.upsert_events(events=events)
        await db_operations.upsert_predictions(predictions=predictions)

        # Mark all events as processed but unprocessed and discarded ones
        await db_client.update(
            """
                UPDATE
                    events
                SET
                    processed = ?,
                    resolved_at = ?
            """,
            [
                True,
                (datetime.now(timezone.utc) - timedelta(days=4, hours=1)).isoformat(),
            ],
        )

        # Mark all predictions as exported but not exported one
        await db_client.update(
            """
                UPDATE
                    predictions
                SET
                    exported = ?
                WHERE
                    unique_event_id NOT IN ('not_exported_prediction_event_id')
                """,
            [PredictionExportedStatus.EXPORTED],
        )

        # delete
        result = await db_operations.delete_predictions(batch_size=100)

        assert len(result) == 1
        assert result[0][0] == 1

        result = await db_client.many(
            """
                SELECT unique_event_id FROM predictions ORDER BY ROWID ASC
            """
        )

        # Should have unexported prediction left
        assert len(result) == 1
        assert result[0][0] == "not_exported_prediction_event_id"

    async def test_delete_predictions_batch_size(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        events = [
            EventsModel(
                unique_event_id="event_id_1",
                event_id="event1",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.DISCARDED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-01-01T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="event_id_2",
                event_id="event2",
                market_type="truncated_market2",
                event_type="market_2",
                description="desc2",
                outcome="outcome2",
                status=EventStatus.DISCARDED,
                metadata='{"key": "value"}',
                created_at="2012-12-02T14:30:00+00:00",
                cutoff="2001-01-01T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="event_id_3",
                event_id="event3",
                market_type="truncated_market3",
                event_type="market_3",
                description="desc2",
                outcome="outcome3",
                status=EventStatus.DISCARDED,
                metadata='{"key": "value"}',
                created_at="2012-12-02T14:30:00+00:00",
                cutoff="2001-01-01T14:30:00+00:00",
            ),
        ]

        predictions = [
            ("event_id_1", "neuronHotkey_1", 1, 1, 10, 1),
            ("event_id_2", "neuronHotkey_2", 2, 1, 10, 1),
            (
                "event_id_3",
                "neuronHotkey_2",
                2,
                1,
                10,
                1,
            ),
        ]

        await db_operations.upsert_events(events=events)
        await db_operations.upsert_predictions(predictions=predictions)

        # Mark all events as processed
        await db_client.update(
            """
                UPDATE
                    events
                SET
                    processed = ?,
                    resolved_at = ?
            """,
            [
                True,
                (datetime.now(timezone.utc) - timedelta(days=4, hours=1)).isoformat(),
            ],
        )
        # Mark all predictions as exported
        await db_client.update(
            """
                UPDATE
                    predictions
                SET
                    exported = ?
                """,
            [PredictionExportedStatus.EXPORTED],
        )

        # delete with batch size 2
        result = await db_operations.delete_predictions(batch_size=2)

        assert len(result) == 2

        # Check the rows are deleted in ASC order
        assert result[0][0] == 1
        assert result[1][0] == 2

        result = await db_client.many(
            """
                SELECT unique_event_id FROM predictions ORDER BY ROWID ASC
            """
        )

        # Should have 1 prediction left
        assert len(result) == 1
        assert result[0][0] == "event_id_3"

        # Delete the rest
        result = await db_operations.delete_predictions(batch_size=1000)

        assert len(result) == 1
        assert result[0][0] == 3

        result = await db_client.many(
            """
                SELECT unique_event_id FROM predictions ORDER BY ROWID ASC
            """
        )

        # Should have no predictions left
        assert len(result) == 0

    async def test_get_event(self, db_operations: DatabaseOperations):
        unique_event_id = "unique1"

        events = [
            EventsModel(
                unique_event_id="unique1",
                event_id="event1",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-02T14:30:00+00:00",
            )
        ]

        await db_operations.upsert_events(events=events)

        result = await db_operations.get_event(unique_event_id)

        assert isinstance(result, EventsModel)
        assert result.unique_event_id == "unique1"
        assert result.event_id == "event1"

    async def test_get_event_no_event(self, db_operations: DatabaseOperations):
        unique_event_id = "unique_1"

        result = await db_operations.get_event(unique_event_id=unique_event_id)

        assert result is None

    async def test_get_events_last_resolved_at(self, db_operations: DatabaseOperations):
        events = [
            EventsModel(
                unique_event_id="unique1",
                event_id="event1",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="1900-12-02T14:30:00+00:00",
                cutoff="2024-12-03",
            ),
            EventsModel(
                unique_event_id="unique2",
                event_id="event2",
                market_type="truncated_market2",
                event_type="market_2",
                description="desc2",
                outcome="outcome2",
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="3000-12-02T14:30:00+00:00",
                cutoff="2024-12-03",
            ),
            EventsModel(
                unique_event_id="unique3",
                event_id="event3",
                market_type="truncated_market3",
                event_type="market_3",
                description="desc3",
                outcome="outcome3",
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="1950-12-02T14:30:00+00:00",
                cutoff="2024-12-03",
            ),
        ]

        current_time = datetime.now(timezone.utc)
        current_time_iso = current_time.isoformat()
        future_time_iso = (current_time + timedelta(seconds=1)).isoformat()

        await db_operations.upsert_events(events=events)

        # Resolve events 1 and 3
        await db_operations.resolve_event(
            event_id="event1", outcome=1, resolved_at=current_time_iso
        )
        await db_operations.resolve_event(event_id="event3", outcome=1, resolved_at=future_time_iso)

        result = await db_operations.get_events_last_resolved_at()

        assert result == future_time_iso

    async def test_get_events_last_resolved_at_no_events(self, db_operations: DatabaseOperations):
        result = await db_operations.get_events_last_resolved_at()

        assert result is None

    async def test_get_events_pending_first_created_at(self, db_operations: DatabaseOperations):
        events = [
            EventsModel(
                unique_event_id="unique1",
                event_id="resolved_event",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="1900-12-02T14:30:00+00:00",
                cutoff="2024-12-03",
            ),
            EventsModel(
                unique_event_id="unique2",
                event_id="pending_event_new",
                market_type="truncated_market2",
                event_type="market_2",
                description="desc2",
                outcome="outcome2",
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="3000-12-02T14:30:00+00:00",
                cutoff="2024-12-03",
            ),
            EventsModel(
                unique_event_id="unique3",
                event_id="pending_event_old",
                market_type="truncated_market3",
                event_type="market_3",
                description="desc3",
                outcome="outcome3",
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="1950-12-02T14:30:00+00:00",
                cutoff="2024-12-03",
            ),
        ]

        await db_operations.upsert_events(events=events)

        result = await db_operations.get_events_pending_first_created_at()

        assert result == "1950-12-02 14:30:00+00:00"

    async def test_get_events_pending_first_created_at_no_events(
        self, db_operations: DatabaseOperations
    ):
        result = await db_operations.get_events_pending_first_created_at()

        assert result is None

    async def test_get_last_event_from(self, db_operations: DatabaseOperations):
        created_at = "2000-12-02T14:30:00+00:00"

        events = [
            EventsModel(
                unique_event_id="unique1",
                event_id="event1",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.DELETED,
                metadata='{"key": "value"}',
                created_at=created_at,
                cutoff="2024-12-03",
            )
        ]

        await db_operations.upsert_events(events=events)

        result = await db_operations.get_last_event_from()

        assert result == created_at.replace("T", " ")

    async def test_get_last_event_from_no_events(self, db_operations: DatabaseOperations):
        result = await db_operations.get_last_event_from()

        assert result is None

    async def test_get_events_to_predict(self, db_operations: DatabaseOperations):
        event_to_predict_id = "event3"

        cutoff_now = datetime.now(timezone.utc).isoformat()
        cutoff_future = (datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat()

        events = [
            EventsModel(
                unique_event_id="unique1",
                event_id="event1",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome=None,
                status=EventStatus.DISCARDED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unique2",
                event_id="event2",
                market_type="truncated_market2",
                event_type="market_2",
                description="desc2",
                outcome="outcome2",
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="2012-12-02T14:30:00+00:00",
                cutoff=cutoff_now,
            ),
            EventsModel(
                unique_event_id="unique3",
                event_id=event_to_predict_id,
                market_type="truncated_market3",
                event_type="market_3",
                description="desc3",
                outcome="outcome2",
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="2012-12-02T14:30:00+00:00",
                cutoff=cutoff_future,
            ),
            EventsModel(
                unique_event_id="unique4",
                event_id="event4",
                market_type="truncated_market4",
                event_type="market_4",
                description="desc2",
                outcome="outcome2",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="2012-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events=events)

        result = await db_operations.get_events_to_predict()

        assert len(result) == 1
        assert result[0][0] == event_to_predict_id

    async def test_get_predictions_for_event(self, db_operations: DatabaseOperations):
        unique_event_id_1 = "unique_event_id_1"
        unique_event_id_2 = "unique_event_id_2"

        current_interval = 10

        events = [
            EventsModel(
                unique_event_id=unique_event_id_1,
                event_id=unique_event_id_1,
                market_type="truncated_market1",
                event_type="market1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id=unique_event_id_2,
                event_id=unique_event_id_2,
                market_type="truncated_market1",
                event_type="market1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events=events)

        predictions = [
            (unique_event_id_1, "neuronHotkey_99", 99, 1, current_interval, 1),
            (unique_event_id_2, "neuronHotkey_2", 2, 1, current_interval, 1),
            (
                unique_event_id_1,
                "neuronHotkey_248",
                248,
                0.5,
                current_interval,
                1,
            ),
            (
                unique_event_id_1,
                "neuronHotkey_0",
                0,
                0.5,
                current_interval,
                1,
            ),
            (unique_event_id_1, "neuronHotkey_0", 0, 0.5, current_interval - 1, 1),
        ]

        await db_operations.upsert_predictions(predictions)

        result = await db_operations.get_predictions_for_event(
            unique_event_id=unique_event_id_1, interval_start_minutes=current_interval
        )

        assert len(result) == 3
        assert result[0].unique_event_id == "unique_event_id_1"
        assert result[0].miner_uid == 0
        assert result[0].latest_prediction == 0.5

        assert result[1].unique_event_id == "unique_event_id_1"
        assert result[1].miner_uid == 99
        assert result[1].latest_prediction == 1

        assert result[2].unique_event_id == "unique_event_id_1"
        assert result[2].miner_uid == 248
        assert result[2].latest_prediction == 0.5

    async def test_get_predictions_for_event_predictions(self, db_operations: DatabaseOperations):
        unique_event_id = "unique_event_id_1"
        interval_start_minutes = 999

        result = await db_operations.get_predictions_for_event(
            unique_event_id=unique_event_id, interval_start_minutes=interval_start_minutes
        )

        assert len(result) == 0

    async def test_get_predictions_to_export(
        self, db_client: DatabaseClient, db_operations: DatabaseOperations
    ):
        events = [
            EventsModel(
                unique_event_id="unique_event_id_1",
                event_id="event_1",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.DELETED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-02T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unique_event_id_2",
                event_id="event_2",
                market_type="truncated_market2",
                event_type="market_2",
                description="desc2",
                outcome="outcome2",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-02T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unique_event_id_3",
                event_id="event_3",
                market_type="truncated_market3",
                event_type="market_3",
                description="desc3",
                outcome="outcome3",
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-02T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unique_event_id_4",
                event_id="event_4",
                market_type="truncated_market4",
                event_type="market_4",
                description="desc4",
                outcome="outcome4",
                status=EventStatus.DISCARDED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-02T14:30:00+00:00",
            ),
        ]

        predictions = [
            (
                "unique_event_id_1",
                "neuronHotkey_1",
                1,
                1.0,
                # interval_start_minutes
                10,
                1.0,
            ),
            (
                "unique_event_id_2",
                "neuronHotkey_2",
                2,
                1.0,
                # interval_start_minutes
                10,
                1.0,
            ),
            (
                "unique_event_id_3",
                "neuronHotkey_3",
                3,
                1.0,
                # interval_start_minutes
                10,
                1.0,
            ),
            (
                "unique_event_id_4",
                "neuronHotkey_4",
                4,
                1.0,
                # interval_start_minutes
                # This prediction wont be ready to be exported
                11,
                1.0,
            ),
        ]

        await db_operations.upsert_events(events=events)

        await db_operations.upsert_predictions(predictions)

        # Mark one prediction as exported
        await db_client.update(
            """
                UPDATE predictions SET exported = ? WHERE unique_event_id = ?
            """,
            [PredictionExportedStatus.EXPORTED, "unique_event_id_2"],
        )

        current_interval_minutes = 11

        result = await db_operations.get_predictions_to_export(
            current_interval_minutes=current_interval_minutes, batch_size=1
        )

        assert len(result) == 1
        assert result[0][1] == "unique_event_id_1"

        result = await db_operations.get_predictions_to_export(
            current_interval_minutes=current_interval_minutes, batch_size=20
        )

        assert len(result) == 2

    async def test_mark_predictions_as_exported(
        self, db_client: DatabaseClient, db_operations: DatabaseOperations
    ):
        events = [
            EventsModel(
                unique_event_id="unique_event_id_1",
                event_id="unique_event_id_1",
                market_type="truncated_market1",
                event_type="market1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unique_event_id_2",
                event_id="unique_event_id_2",
                market_type="truncated_market1",
                event_type="market1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unique_event_id_3",
                event_id="unique_event_id_3",
                market_type="truncated_market1",
                event_type="market1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events=events)

        predictions = [
            ("unique_event_id_1", "neuronHotkey_1", 1, 1.0, 10, 1.0),
            ("unique_event_id_2", "neuronHotkey_2", 2, 1.0, 10, 1.0),
            ("unique_event_id_3", "neuronHotkey_3", 3, 1.0, 10, 1.0),
        ]

        await db_operations.upsert_predictions(predictions=predictions)

        result = await db_operations.mark_predictions_as_exported(ids=["2"])

        # Updated row id is correct
        assert result[0][0] == 2

        result = await db_client.many(
            """
                SELECT
                    ROWID,
                    unique_event_id,
                    exported
                FROM
                    predictions
            """
        )

        assert result[1][1] == "unique_event_id_2"
        assert result[0][2] == PredictionExportedStatus.NOT_EXPORTED
        assert result[1][2] == PredictionExportedStatus.EXPORTED
        assert result[2][2] == PredictionExportedStatus.NOT_EXPORTED

    async def test_resolve_event(
        self, db_client: DatabaseClient, db_operations: DatabaseOperations
    ):
        event_id = "event1"
        outcome = 1
        resolved_at = "2000-12-31T14:30:00+00:00"
        prev_resolved_at = "2000-12-30T14:30:00+00:00"

        events = [
            EventsModel(
                unique_event_id="unique1",
                event_id=event_id,
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
                unique_event_id="unique2",
                event_id="event2",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome=None,
                status=EventStatus.DISCARDED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unique3",
                event_id="event3",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome=None,
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events=events)
        await db_client.update(
            "UPDATE events SET resolved_at = ? WHERE event_id = ?",
            [prev_resolved_at, "event3"],
        )

        await db_operations.resolve_event(
            event_id=event_id,
            outcome=outcome,
            resolved_at=resolved_at,
        )

        result = await db_client.many(
            """
                SELECT event_id, status, outcome, resolved_at, local_updated_at
                FROM events
                ORDER BY event_id
            """
        )

        assert len(result) == 3
        assert result[0][0] == event_id
        assert result[0][1] == str(EventStatus.SETTLED.value)
        assert result[0][2] == str(outcome)
        assert result[0][3] == resolved_at
        # the other two events were not resolved now
        assert result[1][1] == str(EventStatus.DISCARDED.value)
        assert result[1][3] is None
        assert result[2][1] == str(EventStatus.SETTLED.value)
        assert result[2][3] == prev_resolved_at
        assert isinstance(result[0][4], str)

    async def test_resolve_event_not_resolving_again(
        self, db_client: DatabaseClient, db_operations: DatabaseOperations
    ):
        event_id = "event1"
        outcome = 1
        resolved_at = "2000-12-31T14:30:00+00:00"

        events = [
            EventsModel(
                unique_event_id="unique1",
                event_id=event_id,
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

        await db_operations.resolve_event(
            event_id=event_id,
            outcome=outcome,
            resolved_at=resolved_at,
        )

        outcome_2 = 0
        resolved_at_2 = "1900-12-31T14:30:00+00:00"

        await db_operations.resolve_event(
            event_id=event_id,
            outcome=outcome_2,
            resolved_at=resolved_at_2,
        )

        result = await db_client.many(
            """
                SELECT event_id, status, outcome, resolved_at, local_updated_at FROM events
            """
        )

        # Assert event not resolved with new values
        assert len(result) == 1
        assert result[0][0] == event_id
        assert result[0][2] == str(outcome)
        assert result[0][3] == resolved_at

    async def test_upsert_events(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        events = [
            EventsModel(
                unique_event_id="unique1",
                event_id="event1",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.DISCARDED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-01-01T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unique2",
                event_id="event2",
                market_type="truncated_market2",
                event_type="market_2",
                description="desc2",
                outcome="outcome2",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="2012-12-02T14:30:00+00:00",
                cutoff="2001-01-01T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events=events)

        result = await db_client.many(
            """
                SELECT event_id, cutoff FROM events
            """
        )

        assert len(result) == 2

        # Assert event id
        assert result[0][0] == "event1"
        assert result[1][0] == "event2"

        # Assert cutoff
        assert datetime.fromisoformat(result[0][1]) == events[0].cutoff
        assert datetime.fromisoformat(result[1][1]) == events[1].cutoff

    async def test_upsert_no_events(self, db_operations: DatabaseOperations):
        """Test upsert not failing with empty list."""
        events = []

        await db_operations.upsert_events(events=events)

    async def test_upsert_miners(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        miners = [
            (
                "uid1",
                "hotkey1",
                "ip1",
                "2000-12-02T14:30:00+00:00",
                1,
                False,
                True,
                "ip1",
                1,
            ),
            (
                "uid2",
                "hotkey2",
                "ip2",
                "2000-12-02T14:30:00+00:00",
                2,
                True,
                False,
                "ip2",
                2,
            ),
        ]

        await db_operations.upsert_miners(miners)

        result = await db_client.many(
            """
                SELECT
                    miner_hotkey,
                    miner_uid,
                    registered_date,
                    last_updated,
                    is_validating,
                    validator_permit
                FROM
                    miners
            """
        )

        assert len(result) == 2

        # Assert miner hotkey
        assert result[0][0] == "hotkey1"
        assert result[1][0] == "hotkey2"

        # Assert miner uid
        assert result[0][1] == "uid1"
        assert result[1][1] == "uid2"

        # Assert registered_date
        assert result[0][2] == "2000-12-02T14:30:00+00:00"
        assert result[1][2] == "2000-12-02T14:30:00+00:00"

        # Assert last_registration
        assert result[0][3] == ANY
        assert result[1][3] == ANY

        # Assert is_validating
        assert result[0][4] == 0
        assert result[1][4] == 1

        # Assert validator_permit
        assert result[0][5] == 1
        assert result[1][5] == 0

        # TODO: Test update fields on conflict

    async def test_upsert_predictions(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        interval_start_minutes = 5

        neuron_1_answer = "1"
        neuron_2_answer = "0.5"

        events = [
            EventsModel(
                unique_event_id="unique_event_id_1",
                event_id="unique_event_id_1",
                market_type="truncated_market1",
                event_type="market1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unique_event_id_2",
                event_id="unique_event_id_2",
                market_type="truncated_market1",
                event_type="market1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events=events)

        predictions = [
            [
                "unique_event_id_1",
                "neuronHotkey_1",
                1,
                neuron_1_answer,
                interval_start_minutes,
                neuron_1_answer,
            ],
            [
                "unique_event_id_2",
                "neuronHotkey_2",
                2,
                neuron_2_answer,
                interval_start_minutes,
                neuron_2_answer,
            ],
        ]

        # Insert
        await db_operations.upsert_predictions(predictions)

        result_1 = await db_client.many(
            """
                SELECT
                    unique_event_id,
                    miner_uid,
                    miner_hotkey,
                    latest_prediction,
                    interval_agg_prediction,
                    interval_count,
                    submitted,
                    updated_at
                FROM
                    predictions
            """
        )

        assert len(result_1) == 2

        # Assert event id
        assert result_1[0][0] == predictions[0][0]
        assert result_1[1][0] == predictions[1][0]

        # Assert neuron uid
        assert result_1[0][1] == predictions[0][2]
        assert result_1[1][1] == predictions[1][2]

        # Assert neuron hotkey
        assert result_1[0][2] == predictions[0][1]
        assert result_1[1][2] == predictions[1][1]

        # Assert prediction
        assert result_1[0][3] == float(neuron_1_answer)
        assert result_1[1][3] == float(neuron_2_answer)

        # Assert interval_agg_prediction
        assert result_1[0][4] == float(neuron_1_answer)
        assert result_1[1][4] == float(neuron_2_answer)

        # Assert interval_count
        assert result_1[0][5] == 1
        assert result_1[1][5] == 1

        # Assert submitted
        assert result_1[0][6] == ANY
        assert result_1[1][6] == ANY

        # Assert updated_at
        assert result_1[0][6] == result_1[0][7]
        assert result_1[1][6] == result_1[1][7]

        # Change predictions
        predictions[0][3] = float(neuron_1_answer) * 3
        predictions[1][3] = float(neuron_2_answer) * 3

        # Change interval_agg_prediction
        predictions[0][5] = float(neuron_1_answer) * 3
        predictions[1][5] = float(neuron_2_answer) * 3

        # wait 1s for updated at to change
        await asyncio.sleep(1)

        # Upsert
        await db_operations.upsert_predictions(predictions)

        result_2 = await db_client.many(
            """
                SELECT
                    unique_event_id,
                    miner_uid,
                    miner_hotkey,
                    latest_prediction,
                    interval_agg_prediction,
                    interval_count,
                    submitted,
                    updated_at
                FROM
                    predictions
            """
        )

        assert len(result_2) == 2

        # Assert event id
        assert result_2[0][0] == result_1[0][0]
        assert result_2[1][0] == result_1[1][0]

        # Assert neuron uid
        assert result_2[0][1] == result_1[0][1]
        assert result_2[1][1] == result_1[1][1]

        # Assert neuron hotkey
        assert result_2[0][2] == result_1[0][2]
        assert result_2[1][2] == result_1[1][2]

        # Assert prediction
        assert result_2[0][3] == float(neuron_1_answer) * 3
        assert result_2[1][3] == float(neuron_2_answer) * 3

        # Assert interval_agg_prediction
        assert result_2[0][4] == float(neuron_1_answer) * 2
        assert result_2[1][4] == float(neuron_2_answer) * 2

        # Assert interval_count
        assert result_2[0][5] == 2
        assert result_2[1][5] == 2

        # Assert submitted
        assert result_2[0][6] == result_1[0][6]
        assert result_2[1][6] == result_1[1][6]

        # Assert updated_at gets updated
        assert result_2[0][7] != result_1[0][7]
        assert result_2[1][7] != result_1[1][7]

    async def test_upsert_predictions_fk_constraint(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        predictions = [
            [
                "unique_event_id_1",
                "neuronHotkey_1",
                10,
                0.5,
                5,
                0.5,
            ],
        ]

        # Insert
        with pytest.raises(IntegrityError):
            await db_operations.upsert_predictions(predictions=predictions)

        result = await db_client.many(
            """
                SELECT
                    *
                FROM
                    predictions
            """
        )

        assert len(result) == 0

    async def test_get_events_for_scoring(self, db_operations: DatabaseOperations):
        expected_event_id = "event1"

        events = [
            EventsModel(
                unique_event_id="unique1",
                event_id=expected_event_id,
                market_type="truncated_market1",
                event_type="market1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unique2",
                event_id="event2",
                market_type="truncated_market2",
                event_type="market2",
                description="desc2",
                outcome=None,
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="2012-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events)

        result = await db_operations.get_events_for_scoring()

        assert len(result) == 1
        assert result[0].event_id == expected_event_id
        assert result[0].status == EventStatus.SETTLED

    async def test_get_predictions_for_scoring(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        expected_event_id = "_event1"

        events = [
            EventsModel(
                unique_event_id="unique_event_id_1",
                event_id="unique_event_id_1",
                market_type="truncated_market1",
                event_type="market1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id=expected_event_id,
                event_id=expected_event_id,
                market_type="truncated_market1",
                event_type="market1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events=events)

        predictions = [
            (
                "unique_event_id_1",
                "neuronHotkey_1",
                1,
                1.0,
                10,
                1.0,
            ),
            (expected_event_id, "neuronHotkey_2", 2, 1.0, 10, 1.0),
        ]

        await db_operations.upsert_predictions(predictions)

        all_predictions = await db_client.many(
            """
                SELECT unique_event_id FROM predictions ORDER BY unique_event_id
            """
        )
        assert len(all_predictions) == 2
        assert all_predictions[0][0] == expected_event_id
        assert all_predictions[1][0] == "unique_event_id_1"

        result = await db_operations.get_predictions_for_scoring(unique_event_id=expected_event_id)

        assert len(result) == 1
        assert result[0].unique_event_id == expected_event_id

    async def test_get_miners_last_registration(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        miner_1 = MinersModel(
            miner_hotkey="hotkey1",
            miner_uid="uid1",
            registered_date=datetime(2024, 1, 1, 10, 0, 0),
            is_validating=False,
            validator_permit=False,
        )

        miner_2 = MinersModel(
            miner_hotkey="hotkey2",
            miner_uid="uid2",
            registered_date=datetime(2024, 1, 1, 10, 0, 0),
            is_validating=False,
            validator_permit=True,
        )

        miner_1_replaced = MinersModel(
            miner_hotkey="hotkey2",
            miner_uid="uid1",
            registered_date=datetime(2024, 1, 1, 11, 0, 1),
            is_validating=True,
            validator_permit=True,
        )

        miners = [miner_1, miner_2, miner_1_replaced]
        await db_client.insert_many(
            """
            INSERT INTO miners (miner_hotkey, miner_uid, registered_date, is_validating, validator_permit)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    m.miner_hotkey,
                    m.miner_uid,
                    m.registered_date,
                    m.is_validating,
                    m.validator_permit,
                )
                for m in miners
            ],
        )

        all_miners = await db_client.many(
            """
                SELECT miner_uid, miner_hotkey, registered_date FROM miners
                ORDER BY miner_uid, registered_date
            """
        )

        assert len(all_miners) == 3
        assert all_miners[0][0] == miner_1.miner_uid
        assert all_miners[0][1] == miner_1.miner_hotkey
        assert all_miners[1][1] == miner_1_replaced.miner_hotkey
        assert datetime.fromisoformat(all_miners[1][2]) == miner_1_replaced.registered_date
        assert all_miners[2][0] == miner_2.miner_uid

        result = await db_operations.get_miners_last_registration()
        assert len(result) == 2
        assert result[0].miner_uid == miner_1_replaced.miner_uid
        assert result[0].miner_hotkey == miner_1_replaced.miner_hotkey
        assert result[0].registered_date == miner_1_replaced.registered_date
        assert result[1].miner_uid == miner_2.miner_uid
        assert result[1].miner_hotkey == miner_2.miner_hotkey
        assert result[1].registered_date == miner_2.registered_date

    async def test_mark_event_as_processed(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        unique_event_id = "unique_event1"

        events = [
            EventsModel(
                unique_event_id=unique_event_id,
                event_id="event1",
                market_type="truncated_market1",
                event_type="market1",
                description="desc1",
                outcome=None,
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unique_event2",
                event_id="event2",
                market_type="truncated_market2",
                event_type="market2",
                description="desc1",
                outcome=None,
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events=events)

        await db_operations.mark_event_as_processed(unique_event_id=unique_event_id)

        result = await db_client.many(
            """
                SELECT unique_event_id, processed, exported FROM events ORDER BY unique_event_id
            """
        )

        assert len(result) == 2
        assert result[0][0] == unique_event_id
        assert result[0][1] == 1
        assert result[0][2] == 0
        assert result[1][0] == "unique_event2"
        assert result[1][1] == 0
        assert result[1][2] == 0

    async def test_mark_event_as_exported(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        unique_event_id = "unique_event1"

        events = [
            EventsModel(
                unique_event_id=unique_event_id,
                event_id="event1",
                market_type="truncated_market1",
                event_type="market1",
                description="desc1",
                outcome=None,
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unique_event2",
                event_id="event2",
                market_type="truncated_market2",
                event_type="market2",
                description="desc1",
                outcome=None,
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events=events)

        await db_operations.mark_event_as_exported(unique_event_id=unique_event_id)

        result = await db_client.many(
            """
                SELECT unique_event_id, exported FROM events ORDER BY unique_event_id
            """
        )

        assert len(result) == 2
        assert result[0][0] == unique_event_id
        assert result[0][1] == 1
        assert result[1][0] == "unique_event2"
        assert result[1][1] == 0

    @pytest.mark.parametrize(
        "scores, expected_tuples",
        [
            (
                [
                    ScoresModel(
                        event_id="evt1",
                        miner_uid=1,
                        miner_hotkey="hk1",
                        prediction=0.75,
                        event_score=0.85,
                        spec_version=1,
                    ),
                    ScoresModel(
                        event_id="evt2",
                        miner_uid=2,
                        miner_hotkey="hk2",
                        prediction=0.65,
                        event_score=0.80,
                        spec_version=1,
                    ),
                ],
                [
                    ("evt1", 1, "hk1", 0.75, 0.85, 1),
                    ("evt2", 2, "hk2", 0.65, 0.80, 1),
                ],
            ),
        ],
    )
    async def test_insert_peer_scores(
        self,
        db_operations: DatabaseOperations,
        db_client: DatabaseClient,
        scores: list[ScoresModel],
        expected_tuples: list[tuple],
    ):
        expected_timestamp = datetime.now()
        await db_operations.insert_peer_scores(scores)

        result = await db_client.many(
            """
                SELECT
                    event_id,
                    miner_uid,
                    miner_hotkey,
                    prediction,
                    event_score,
                    spec_version
                FROM scores
                ORDER BY event_id
            """
        )

        for i, row in enumerate(result):
            assert row == expected_tuples[i]

        result = await db_client.many("""SELECT created_at, processed, exported FROM scores""")
        for row in result:
            actual_timestamp = datetime.fromisoformat(row[0])
            assert actual_timestamp > expected_timestamp - timedelta(seconds=5)
            assert actual_timestamp < expected_timestamp + timedelta(seconds=5)
            assert row[1] == 0
            assert row[2] == 0

    async def test_get_events_for_metagraph_scoring(
        self,
        db_operations: DatabaseOperations,
        db_client: DatabaseClient,
    ):
        valid_event_id = "metagraph_event"
        processed_event_id = "processed_event"

        now = datetime.now(timezone.utc)
        ten_seconds_ago = now - timedelta(seconds=10)

        # The query should pick the minimum row id, not min created
        scores = [
            ScoresModel(
                event_id=processed_event_id,
                miner_uid=3,
                miner_hotkey="hk3",
                prediction=0.75,
                event_score=0.80,
                spec_version=1,
            ),
            ScoresModel(
                event_id=valid_event_id,
                miner_uid=1,
                miner_hotkey="hk1",
                prediction=0.85,
                event_score=0.90,
                spec_version=1,
            ),
            ScoresModel(
                event_id=valid_event_id,
                miner_uid=2,
                miner_hotkey="hk2",
                prediction=0.80,
                event_score=0.88,
                spec_version=1,
            ),
        ]

        await db_operations.insert_peer_scores(scores)

        # set the timestamps and processed status
        await db_client.update(
            "UPDATE scores SET created_at = ?, processed = ? WHERE event_id = ?",
            [now.isoformat(), False, valid_event_id],
        )
        await db_client.update(
            "UPDATE scores SET created_at = ?, processed = ? WHERE event_id = ?",
            [now.isoformat(), True, processed_event_id],
        )
        # for rowid 2 put timestamp to 10 seconds ago
        await db_client.update(
            "UPDATE scores SET created_at = ? WHERE rowid = 3",
            [ten_seconds_ago.isoformat()],
        )

        # confirm right setup
        raw_scores = await db_client.many(
            "SELECT event_id, rowid, created_at, processed FROM scores"
        )
        assert len(raw_scores) == 3
        assert raw_scores[1][0] == valid_event_id
        assert raw_scores[1][3] == 0
        assert raw_scores[2][0] == valid_event_id
        assert raw_scores[2][3] == 0
        assert raw_scores[2][2] == ten_seconds_ago.isoformat()
        assert raw_scores[0][0] == processed_event_id
        assert raw_scores[0][3] == 1

        result = await db_operations.get_events_for_metagraph_scoring(max_events=1000)

        assert len(result) == 1
        assert result[0]["event_id"] == valid_event_id
        assert result[0]["min_row_id"] == 2

    async def test_set_metagraph_peer_scores(
        self,
        db_operations: DatabaseOperations,
        db_client: DatabaseClient,
    ):
        # Only basic test here, more detailed tests are in the task tests.
        previous_score = ScoresModel(
            event_id="other_event",
            miner_uid=10,
            miner_hotkey="hk10",
            prediction=0.7,
            event_score=0.75,
            spec_version=1,
        )
        current_score = ScoresModel(
            event_id="test_event",
            miner_uid=20,
            miner_hotkey="hk20",
            prediction=0.9,
            event_score=0.95,
            spec_version=1,
        )
        await db_operations.insert_peer_scores([previous_score, current_score])

        events = [
            EventsModel(
                unique_event_id="unique1",
                event_id="test_event",
                market_type="truncated_market1",
                event_type="market1",
                description="desc1",
                outcome="1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unique2",
                event_id="other_event",
                market_type="truncated_market2",
                event_type="market2",
                description="desc2",
                outcome="0",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="2012-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events)

        # confirm right setup
        raw_scores = await db_client.many("SELECT event_id FROM scores")
        assert len(raw_scores) == 2
        assert raw_scores[0][0] == "other_event"
        assert raw_scores[1][0] == "test_event"
        raw_events = await db_client.many("SELECT event_id FROM events")
        assert len(raw_events) == 2
        assert raw_events[0][0] == "test_event"
        assert raw_events[1][0] == "other_event"

        updated = await db_operations.set_metagraph_peer_scores(event_id="test_event", n_events=5)
        assert updated == []

        # Verify that only the current event row was updated - test_event is the second row.
        actual_rows = await db_client.many(
            "SELECT event_id, processed, metagraph_score, other_data FROM scores ORDER BY event_id",
            use_row_factory=True,
        )
        assert len(actual_rows) == 2
        assert actual_rows[0]["event_id"] == "other_event"
        assert actual_rows[0]["processed"] == 0
        assert actual_rows[0]["metagraph_score"] is None
        assert actual_rows[0]["other_data"] is None

        assert actual_rows[1]["event_id"] == "test_event"
        assert actual_rows[1]["processed"] == 1
        assert actual_rows[1]["metagraph_score"] == 1.0
        assert actual_rows[1]["other_data"] is not None

        other_data = json.loads(actual_rows[1]["other_data"])
        assert other_data["sum_weighted_peer_score"] == 2.85
        assert other_data["sum_weight"] == 4.5
        assert other_data["count_peer_score"] == 6
        assert other_data["true_count_peer_score"] == 1
        assert other_data["avg_peer_score"] == pytest.approx(0.633333, abs=1e-6)
        assert other_data["sqmax_avg_peer_score"] == pytest.approx(0.401111, abs=1e-6)
