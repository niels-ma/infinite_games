from datetime import datetime, timedelta, timezone
from unittest.mock import ANY

import pytest

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.db.tests.test_utils import TestDbOperationsBase
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.models.score import SCORE_FIELDS, ScoresExportedStatus, ScoresModel


class TestDbOperationsPart2(TestDbOperationsBase):
    async def test_get_peer_scored_events_for_export_basic(self, db_operations, db_client):
        export_event_id = "export_event"
        non_export_event_id = "non_export_event"

        now = datetime.now(timezone.utc)

        # For export_event, we want processed=1 and exported=0.
        score_export = ScoresModel(
            event_id=export_event_id,
            miner_uid=1,
            miner_hotkey="hk1",
            prediction=0.85,
            event_score=0.90,
            spec_version=1,
        )
        # For non_export_event, set exported=1 so it should not be picked up.
        score_non_export = ScoresModel(
            event_id=non_export_event_id,
            miner_uid=2,
            miner_hotkey="hk2",
            prediction=0.75,
            event_score=0.80,
            spec_version=1,
        )

        await db_operations.insert_peer_scores([score_export, score_non_export])
        await db_client.update(
            "UPDATE scores SET processed = ?, exported = ?, created_at = ? WHERE event_id = ?",
            [1, ScoresExportedStatus.NOT_EXPORTED, now.isoformat(), export_event_id],
        )
        await db_client.update(
            "UPDATE scores SET processed = ?, exported = ?, created_at = ? WHERE event_id = ?",
            [1, ScoresExportedStatus.EXPORTED, now.isoformat(), non_export_event_id],
        )

        # Prepare corresponding events in the events table.
        export_event = EventsModel(
            unique_event_id="unique_export_event",
            event_id=export_event_id,
            market_type="market_type1",
            event_type="type1",
            description="Exportable event",
            outcome="1",
            status=EventStatus.SETTLED,
            metadata='{"key": "value"}',
            resolved_at=now.isoformat(),
        )
        non_export_event = EventsModel(
            unique_event_id="unique_non_export_event",
            event_id=non_export_event_id,
            market_type="market_type2",
            event_type="type2",
            description="Non-exportable event",
            outcome="0",
            status=EventStatus.SETTLED,
            metadata='{"key": "value"}',
            resolved_at=now.isoformat(),
        )

        await db_operations.upsert_events([export_event, non_export_event])

        raw_scores = await db_client.many(
            "SELECT event_id, ROWID, processed, exported FROM scores ORDER BY ROWID ASC",
            use_row_factory=True,
        )
        assert len(raw_scores) >= 2
        assert raw_scores[0]["event_id"] == export_event_id
        assert raw_scores[0]["processed"] == 1
        assert raw_scores[0]["exported"] == 0
        assert raw_scores[1]["event_id"] == non_export_event_id
        assert raw_scores[1]["processed"] == 1
        assert raw_scores[1]["exported"] == 1

        result = await db_operations.get_peer_scored_events_for_export(max_events=1000)
        assert len(result) == 1
        event = result[0]
        assert event.event_id == export_event_id
        assert event.unique_event_id == "unique_export_event"
        assert event.market_type == "market_type1"
        assert event.event_type == "type1"
        assert event.description == "Exportable event"
        assert event.outcome == "1"
        assert event.status == EventStatus.SETTLED

    async def test_get_peer_scored_events_for_export_invalid_data(self, db_operations, db_client):
        # Create a fake row with missing required field(s) so that EventsModel parsing fails.
        invalid_row = {"unique_event_id": "unique_invalid", "market_type": "market_type_invalid"}

        async def fake_many(*args, **kwargs):
            return [invalid_row]

        db_client.many = fake_many

        logged_exceptions = []

        def fake_logger_exception(msg, extra):
            logged_exceptions.append((msg, extra))

        db_operations.logger.exception = fake_logger_exception

        result = await db_operations.get_peer_scored_events_for_export(max_events=1000)

        assert result == []
        assert len(logged_exceptions) >= 1
        for msg, extra in logged_exceptions:
            assert "Error parsing model" in msg
            assert "row" in extra
            assert "model" in extra

    async def test_get_peer_scores_for_export_basic(self, db_operations, db_client):
        event_id = "test_event"

        score1 = ScoresModel(
            event_id=event_id,
            miner_uid=1,
            miner_hotkey="hk1",
            prediction=0.8,
            event_score=0.85,
            spec_version=1,
        )
        score2 = ScoresModel(
            event_id=event_id,
            miner_uid=2,
            miner_hotkey="hk2",
            prediction=0.9,
            event_score=0.95,
            spec_version=1,
        )
        other_score = ScoresModel(
            event_id="other_event",
            miner_uid=3,
            miner_hotkey="hk3",
            prediction=0.7,
            event_score=0.75,
            spec_version=1,
        )
        await db_operations.insert_peer_scores([score1, score2, other_score])
        await db_client.update(
            "UPDATE scores SET processed = ? WHERE event_id = ?",
            [1, event_id],
        )
        await db_client.update(
            "UPDATE scores SET processed = ? WHERE event_id = ?",
            [0, "other_event"],
        )

        scores = await db_operations.get_peer_scores_for_export(event_id)

        assert isinstance(scores, list)
        assert len(scores) == 2
        for score in scores:
            assert score.event_id == event_id
            assert score.processed == 1

    async def test_get_peer_scores_for_export_invalid_data(self, db_operations, db_client):
        # Test that the method can handle invalid data.
        event_id = "test_event_invalid"
        invalid_row = {field: "invalid" for field in SCORE_FIELDS if field != "event_id"}

        async def fake_many(*args, **kwargs):
            return [invalid_row]

        db_client.many = fake_many

        logged_exceptions = []

        def fake_logger_exception(msg, extra):
            logged_exceptions.append((msg, extra))

        db_operations.logger.exception = fake_logger_exception

        scores = await db_operations.get_peer_scores_for_export(event_id)
        assert scores == []
        assert len(logged_exceptions) >= 1
        for msg, extra in logged_exceptions:
            assert "Error parsing model" in msg
            assert "row" in extra
            assert "model" in extra

    async def test_mark_peer_scores_as_exported(self, db_operations, db_client):
        events = ["event1", "event2", "event3"]
        scores = [
            ScoresModel(
                event_id=event_id,
                miner_uid=1,
                miner_hotkey="hk1",
                prediction=0.80,
                event_score=0.85,
                spec_version=1,
            )
            for event_id in events
        ]

        await db_operations.insert_peer_scores(scores)

        initial = await db_client.many(
            "SELECT event_id, exported FROM scores ORDER BY ROWID ASC",
            use_row_factory=True,
        )
        assert len(initial) == len(events)
        for row, ev in zip(initial, events):
            assert row["exported"] == 0
            assert row["event_id"] == ev

        # Call the method under test.
        update_result = await db_operations.mark_peer_scores_as_exported(events[1])
        assert update_result == []

        updated = await db_client.many(
            "SELECT event_id, exported FROM scores ORDER BY ROWID ASC",
            use_row_factory=True,
        )
        assert len(updated) == len(events)
        for row, ev in zip(updated, events):
            if ev == events[1]:
                assert row["exported"] == 1
            else:
                assert row["exported"] == 0

    async def test_get_last_metagraph_scores(self, db_operations, db_client):
        created_at = datetime.now(timezone.utc) - timedelta(days=1)
        scores_list = [
            ScoresModel(
                event_id="expected_event_id_1",
                miner_uid=3,
                miner_hotkey="hk3",
                prediction=0.75,
                event_score=0.80,
                metagraph_score=1.0,
                created_at=created_at,
                spec_version=1,
                processed=True,
            ),
            ScoresModel(
                event_id="expected_event_id_2",
                miner_uid=3,
                miner_hotkey="hk3",
                prediction=0.75,
                event_score=0.40,
                metagraph_score=0.9,
                created_at=created_at,
                spec_version=1,
                processed=True,
            ),
            ScoresModel(
                event_id="expected_event_id_3",
                miner_uid=3,
                miner_hotkey="hk3",
                prediction=0.75,
                event_score=0.60,
                metagraph_score=0.835,
                created_at=created_at,
                spec_version=1,
                processed=True,
            ),
            ScoresModel(
                event_id="expected_event_id_2",
                miner_uid=4,
                miner_hotkey="hk4",
                prediction=0.75,
                event_score=0.40,
                metagraph_score=0.1,
                created_at=created_at,
                spec_version=1,
                processed=True,
            ),
            ScoresModel(
                event_id="expected_event_id_1",
                miner_uid=4,
                miner_hotkey="hk4",
                prediction=0.75,
                event_score=0.40,
                metagraph_score=0.165,
                created_at=created_at,
                spec_version=1,
                processed=True,
            ),
            ScoresModel(
                event_id="expected_event_id_2",
                miner_uid=5,
                miner_hotkey="hk5",
                prediction=0.75,
                event_score=-0.40,
                metagraph_score=0.0,
                created_at=created_at,
                spec_version=1,
                processed=True,
            ),
        ]
        # insert scores
        sql = f"""
            INSERT INTO scores ({', '.join(SCORE_FIELDS)})
            VALUES ({', '.join(['?'] * len(SCORE_FIELDS))})
        """
        score_tuples = [
            tuple(getattr(score, field) for field in SCORE_FIELDS) for score in scores_list
        ]
        await db_client.insert_many(sql, score_tuples)

        inserted_scores = await db_client.many("SELECT * FROM scores")
        assert len(inserted_scores) == len(scores_list)

        # get last metagraph scores
        last_metagraph_scores = await db_operations.get_last_metagraph_scores()
        assert len(last_metagraph_scores) == 3

        assert last_metagraph_scores[0].event_id == "expected_event_id_3"
        assert last_metagraph_scores[0].miner_uid == 3
        assert last_metagraph_scores[0].miner_hotkey == "hk3"
        assert last_metagraph_scores[0].metagraph_score == 0.835

        assert last_metagraph_scores[1].event_id == "expected_event_id_1"
        assert last_metagraph_scores[1].miner_uid == 4
        assert last_metagraph_scores[1].miner_hotkey == "hk4"
        assert last_metagraph_scores[1].metagraph_score == 0.165

        assert last_metagraph_scores[2].event_id == "expected_event_id_2"
        assert last_metagraph_scores[2].miner_uid == 5
        assert last_metagraph_scores[2].miner_hotkey == "hk5"
        assert last_metagraph_scores[2].metagraph_score == 0.0

    async def test_get_last_alternative_metagraph_scores(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        created_at = datetime.now(timezone.utc) - timedelta(days=1)
        scores_list = [
            ScoresModel(
                event_id="expected_event_id_1",
                miner_uid=3,
                miner_hotkey="hk3",
                prediction=0.75,
                event_score=0.80,
                alternative_metagraph_score=1.0,
                created_at=created_at,
                spec_version=1,
                alternative_processed=True,
            ),
            ScoresModel(
                event_id="expected_event_id_2",
                miner_uid=3,
                miner_hotkey="hk3",
                prediction=0.75,
                event_score=0.40,
                alternative_metagraph_score=0.9,
                created_at=created_at,
                spec_version=1,
                alternative_processed=True,
            ),
            ScoresModel(
                event_id="expected_event_id_3",
                miner_uid=3,
                miner_hotkey="hk3",
                prediction=0.75,
                event_score=0.60,
                alternative_metagraph_score=0.835,
                created_at=created_at,
                spec_version=1,
                alternative_processed=True,
            ),
            ScoresModel(
                event_id="expected_event_id_2",
                miner_uid=4,
                miner_hotkey="hk4",
                prediction=0.75,
                event_score=0.40,
                alternative_metagraph_score=0.1,
                created_at=created_at,
                spec_version=1,
                alternative_processed=True,
            ),
            ScoresModel(
                event_id="expected_event_id_1",
                miner_uid=4,
                miner_hotkey="hk4",
                prediction=0.75,
                event_score=0.40,
                alternative_metagraph_score=0.165,
                created_at=created_at,
                spec_version=1,
                alternative_processed=True,
            ),
            ScoresModel(
                event_id="expected_event_id_2",
                miner_uid=5,
                miner_hotkey="hk5",
                prediction=0.75,
                event_score=-0.40,
                alternative_metagraph_score=0.0,
                created_at=created_at,
                spec_version=1,
                alternative_processed=True,
            ),
        ]
        # insert scores
        sql = f"""
            INSERT INTO scores ({', '.join(SCORE_FIELDS)})
            VALUES ({', '.join(['?'] * len(SCORE_FIELDS))})
        """
        score_tuples = [
            tuple(getattr(score, field) for field in SCORE_FIELDS) for score in scores_list
        ]
        await db_client.insert_many(sql, score_tuples)

        inserted_scores = await db_client.many("SELECT * FROM scores")
        assert len(inserted_scores) == len(scores_list)

        # get last alternative metagraph scores
        last_alternative_metagraph_scores = (
            await db_operations.get_last_alternative_metagraph_scores()
        )

        assert [score.model_dump() for score in last_alternative_metagraph_scores] == [
            {
                "alternative_metagraph_score": 0.835,
                "alternative_other_data": None,
                "alternative_processed": True,
                "created_at": ANY,
                "event_id": "expected_event_id_3",
                "event_score": 0.6,
                "exported": False,
                "metagraph_score": None,
                "miner_hotkey": "hk3",
                "miner_uid": 3,
                "other_data": None,
                "prediction": 0.75,
                "processed": False,
                "spec_version": 1,
            },
            {
                "alternative_metagraph_score": 0.165,
                "alternative_other_data": None,
                "alternative_processed": True,
                "created_at": ANY,
                "event_id": "expected_event_id_1",
                "event_score": 0.4,
                "exported": False,
                "metagraph_score": None,
                "miner_hotkey": "hk4",
                "miner_uid": 4,
                "other_data": None,
                "prediction": 0.75,
                "processed": False,
                "spec_version": 1,
            },
            {
                "alternative_metagraph_score": 0.0,
                "alternative_other_data": None,
                "alternative_processed": True,
                "created_at": ANY,
                "event_id": "expected_event_id_2",
                "event_score": -0.4,
                "exported": False,
                "metagraph_score": None,
                "miner_hotkey": "hk5",
                "miner_uid": 5,
                "other_data": None,
                "prediction": 0.75,
                "processed": False,
                "spec_version": 1,
            },
        ]

    async def test_mark_event_as_discarded(self, db_operations, db_client):
        now = datetime.now(timezone.utc)
        discarded_unique_event_id = "discarded_event"
        non_discarded_unique_event_id = "non_discarded_event"
        export_event = EventsModel(
            unique_event_id=discarded_unique_event_id,
            event_id="ev1",
            market_type="market_type1",
            event_type="type1",
            description="Bad event",
            outcome="1",
            status=EventStatus.SETTLED,
            metadata='{"key": "value"}',
            resolved_at=now.isoformat(),
        )
        non_export_event = EventsModel(
            unique_event_id=non_discarded_unique_event_id,
            event_id="ev2",
            market_type="market_type2",
            event_type="type2",
            description="Good event",
            outcome="0",
            status=EventStatus.SETTLED,
            metadata='{"key": "value"}',
            resolved_at=now.isoformat(),
        )

        await db_operations.upsert_events([export_event, non_export_event])

        initial = await db_client.many(
            "SELECT unique_event_id, status FROM events ORDER BY ROWID ASC",
            use_row_factory=True,
        )
        assert len(initial) == 2
        assert initial[0]["unique_event_id"] == discarded_unique_event_id
        # status column is text type :|
        assert initial[0]["status"] == str(EventStatus.SETTLED.value)
        assert initial[1]["unique_event_id"] == non_discarded_unique_event_id
        assert initial[1]["status"] == str(EventStatus.SETTLED.value)

        update_result = await db_operations.mark_event_as_discarded(discarded_unique_event_id)
        assert update_result == []

        updated = await db_client.many(
            "SELECT unique_event_id, status FROM events ORDER BY ROWID ASC",
            use_row_factory=True,
        )
        assert len(updated) == 2
        assert updated[0]["unique_event_id"] == discarded_unique_event_id
        assert updated[0]["status"] == str(EventStatus.DISCARDED.value)
        assert updated[1]["unique_event_id"] == non_discarded_unique_event_id
        assert updated[1]["status"] == str(EventStatus.SETTLED.value)

    async def test_vacuum_database(self, db_operations):
        result = await db_operations.vacuum_database(500)

        assert result is None

    async def test_delete_scores_orphan_scores(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        events = [
            # Orphan scores
        ]

        scores = [
            ScoresModel(
                event_id="exported_score_event",
                miner_uid=1,
                miner_hotkey="hk1",
                prediction=0.75,
                event_score=0.85,
                spec_version=1,
            ),
            ScoresModel(
                event_id="non_exported_score_event",
                miner_uid=2,
                miner_hotkey="hk2",
                prediction=0.65,
                event_score=0.80,
                spec_version=1,
            ),
        ]

        await db_operations.upsert_events(events=events)
        await db_operations.insert_peer_scores(scores)

        # Mark score as exported
        await db_client.update(
            """
                UPDATE
                    scores
                SET
                    exported = ?
                WHERE
                    event_id = ?
                """,
            [ScoresExportedStatus.EXPORTED, scores[0].event_id],
        )

        # delete
        result = await db_operations.delete_scores(batch_size=100)

        assert len(result) == 2

        result = await db_client.many(
            """
                SELECT event_id FROM scores ORDER BY ROWID ASC
            """
        )

        assert len(result) == 0

    async def test_delete_scores_discarded_events(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        events = [
            EventsModel(
                unique_event_id="discarded_event_id",
                event_id="discarded_event_id",
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
                unique_event_id="pending_event_id",
                event_id="pending_event_id",
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

        scores = [
            ScoresModel(
                event_id="discarded_event_id",
                miner_uid=1,
                miner_hotkey="hk1",
                prediction=0.75,
                event_score=0.85,
                spec_version=1,
            ),
            ScoresModel(
                event_id="pending_event_id",
                miner_uid=2,
                miner_hotkey="hk2",
                prediction=0.65,
                event_score=0.80,
                spec_version=1,
            ),
        ]

        await db_operations.upsert_events(events=events)
        await db_operations.insert_peer_scores(scores)

        # Mark all scores as exported
        await db_client.update(
            """
                UPDATE
                    scores
                SET
                    exported = ?
                """,
            [ScoresExportedStatus.EXPORTED],
        )

        # delete
        result = await db_operations.delete_scores(batch_size=100)

        assert len(result) == 1
        assert result == [(1,)]

        result = await db_client.many(
            """
                SELECT event_id FROM scores ORDER BY ROWID ASC
            """
        )

        assert len(result) == 1
        assert result[0][0] == "pending_event_id"

    async def test_delete_scores_deleted_events(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        events = [
            EventsModel(
                unique_event_id="deleted_event_id_1",
                event_id="deleted_event_id_1",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.DELETED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-01-01T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="deleted_event_id_2",
                event_id="deleted_event_id_2",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.DELETED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-01-01T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="pending_event_id",
                event_id="pending_event_id",
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

        scores = [
            ScoresModel(
                event_id="deleted_event_id_1",
                miner_uid=1,
                miner_hotkey="hk1",
                prediction=0.75,
                event_score=0.85,
                spec_version=1,
            ),
            ScoresModel(
                event_id="pending_event_id",
                miner_uid=2,
                miner_hotkey="hk2",
                prediction=0.65,
                event_score=0.80,
                spec_version=1,
            ),
        ]

        await db_operations.upsert_events(events=events)
        await db_operations.insert_peer_scores(scores=scores)

        # Mark all scores as exported
        await db_client.update(
            """
                UPDATE
                    scores
                SET
                    exported = ?
                """,
            [ScoresExportedStatus.EXPORTED],
        )

        scores_not_exported = [
            ScoresModel(
                event_id="deleted_event_id_2",
                miner_uid=1,
                miner_hotkey="hk1",
                prediction=0.75,
                event_score=0.85,
                spec_version=1,
            )
        ]

        await db_operations.insert_peer_scores(scores=scores_not_exported)

        # delete
        result = await db_operations.delete_scores(batch_size=100)

        assert len(result) == 2
        assert result == [(1,), (3,)]

        result = await db_client.many(
            """
                SELECT event_id FROM scores ORDER BY ROWID ASC
            """
        )

        assert len(result) == 1
        assert result[0][0] == "pending_event_id"

    async def test_delete_scores_processed_events(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        events = [
            EventsModel(
                unique_event_id="processed_event_old_id",
                event_id="processed_event_old_id",
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
                unique_event_id="processed_event_old_non_exported_score_id",
                event_id="processed_event_old_non_exported_score_id",
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
                unique_event_id="processed_event_new_id",
                event_id="processed_event_new_id",
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
                unique_event_id="non_processed_event_id",
                event_id="non_processed_event_id",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-01-01T14:30:00+00:00",
            ),
        ]

        scores = [
            ScoresModel(
                event_id="processed_event_old",
                miner_uid=1,
                miner_hotkey="hk1",
                prediction=0.75,
                event_score=0.85,
                spec_version=1,
            ),
            ScoresModel(
                event_id="processed_event_old_non_exported_score_id",
                miner_uid=1,
                miner_hotkey="hk1",
                prediction=0.75,
                event_score=0.85,
                spec_version=1,
            ),
            ScoresModel(
                event_id="processed_event_new_id",
                miner_uid=1,
                miner_hotkey="hk1",
                prediction=0.75,
                event_score=0.85,
                spec_version=1,
            ),
            ScoresModel(
                event_id="non_processed_event_id",
                miner_uid=1,
                miner_hotkey="hk1",
                prediction=0.75,
                event_score=0.85,
                spec_version=1,
            ),
        ]

        await db_operations.upsert_events(events=events)
        await db_operations.insert_peer_scores(scores)

        # Mark events as processed
        await db_client.update(
            """
                UPDATE
                    events
                SET
                    processed = ?,
                    resolved_at = ?
                WHERE
                    event_id LIKE 'processed_event_%'
            """,
            [
                True,
                (datetime.now(timezone.utc) - timedelta(days=14, hours=1)).isoformat(),
            ],
        )

        # Update resolved at for old processed events
        await db_client.update(
            """
                UPDATE
                    events
                SET
                    resolved_at = ?
                WHERE
                    event_id LIKE 'processed_event_old_%'
            """,
            [
                (datetime.now(timezone.utc) - timedelta(days=15, hours=1)).isoformat(),
            ],
        )

        # Mark scores exported
        await db_client.update(
            """
                UPDATE
                    scores
                SET
                    exported = ?
                WHERE
                    event_id != ?
                """,
            [ScoresExportedStatus.EXPORTED, "processed_event_old_non_exported_score_id"],
        )

        # delete
        result = await db_operations.delete_scores(batch_size=100)

        assert len(result) == 1
        # assert row id is returned
        assert result[0][0] == 1

        result = await db_client.many(
            """
                SELECT event_id FROM scores ORDER BY ROWID ASC
            """
        )

        # Should have unexported score left
        assert len(result) == 3
        assert result[0][0] == "processed_event_old_non_exported_score_id"
        assert result[1][0] == "processed_event_new_id"
        assert result[2][0] == "non_processed_event_id"

    async def test_delete_scores_batch_size(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        events = [
            EventsModel(
                unique_event_id="event_id_1",
                event_id="event_id_1",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.DELETED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-01-01T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="event_id_2",
                event_id="event_id_2",
                market_type="truncated_market2",
                event_type="market_2",
                description="desc2",
                outcome="outcome2",
                status=EventStatus.DELETED,
                metadata='{"key": "value"}',
                created_at="2012-12-02T14:30:00+00:00",
                cutoff="2001-01-01T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="event_id_3",
                event_id="event_id_3",
                market_type="truncated_market3",
                event_type="market_3",
                description="desc3",
                outcome="outcome3",
                status=EventStatus.DELETED,
                metadata='{"key": "value"}',
                created_at="2012-12-02T14:30:00+00:00",
                cutoff="2001-01-01T14:30:00+00:00",
            ),
        ]

        scores = [
            ScoresModel(
                event_id="event_id_1",
                miner_uid=1,
                miner_hotkey="hk1",
                prediction=0.75,
                event_score=0.85,
                spec_version=1,
            ),
            ScoresModel(
                event_id="event_id_2",
                miner_uid=2,
                miner_hotkey="hk2",
                prediction=0.65,
                event_score=0.80,
                spec_version=1,
            ),
            ScoresModel(
                event_id="event_id_3",
                miner_uid=3,
                miner_hotkey="hk3",
                prediction=0.70,
                event_score=0.75,
                spec_version=1,
            ),
        ]

        await db_operations.upsert_events(events=events)
        await db_operations.insert_peer_scores(scores)

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
                (datetime.now(timezone.utc) - timedelta(days=15, hours=1)).isoformat(),
            ],
        )
        # Mark all scores as exported
        await db_client.update(
            """
                UPDATE
                    scores
                SET
                    exported = ?
                """,
            [ScoresExportedStatus.EXPORTED],
        )

        # delete with batch size 2
        result = await db_operations.delete_scores(batch_size=2)

        assert len(result) == 2

        # Check the rows are deleted in ASC order
        assert result[0][0] == 1
        assert result[1][0] == 2

        result = await db_client.many(
            """
                SELECT event_id FROM scores ORDER BY ROWID ASC
            """
        )

        # Should have 1 score left
        assert len(result) == 1
        assert result[0][0] == "event_id_3"

        # delete again
        result = await db_operations.delete_scores(batch_size=2)

        assert len(result) == 1

        result = await db_client.many(
            """
                SELECT event_id FROM scores ORDER BY ROWID ASC
            """
        )

        # Should have no scores left
        assert len(result) == 0

    async def test_delete_scores_resolved_at_condition(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        # Create events with different resolved_at timestamps

        events = [
            EventsModel(
                unique_event_id="old_event_id",
                event_id="old_event_id",
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
                unique_event_id="recent_event_id",
                event_id="recent_event_id",
                market_type="truncated_market2",
                event_type="market_2",
                description="desc2",
                outcome="outcome2",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="2012-12-02T14:30:00+00:00",
                cutoff="2012-12-02T14:30:00+00:00",
            ),
        ]

        scores = [
            ScoresModel(
                event_id="old_event_id",
                miner_uid=1,
                miner_hotkey="hk1",
                prediction=0.75,
                event_score=0.85,
                spec_version=1,
            ),
            ScoresModel(
                event_id="recent_event_id",
                miner_uid=2,
                miner_hotkey="hk2",
                prediction=0.65,
                event_score=0.80,
                spec_version=1,
            ),
        ]

        await db_operations.upsert_events(events=events)
        await db_operations.insert_peer_scores(scores)

        # Mark events as processed

        # Set events processed and resolved at
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
                (
                    datetime.now(timezone.utc) - timedelta(days=15, hours=1)
                ).isoformat(),  # Resolved 15 days ago
                events[0].unique_event_id,
            ],
        )

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
                (datetime.now(timezone.utc) - timedelta(days=3)).isoformat(),  # Resolved 3 days ago
                events[1].unique_event_id,
            ],
        )

        # Mark all scores as exported
        await db_client.update(
            """
                UPDATE
                    scores
                SET
                    exported = ?
            """,
            [ScoresExportedStatus.EXPORTED],
        )

        # Try to delete scores
        result = await db_operations.delete_scores(batch_size=100)

        # Only the score for the old resolved event should be deleted
        assert len(result) == 1
        assert result[0][0] == 1  # ROWID of the old event's score

        # Verify remaining scores
        remaining_scores = await db_client.many(
            """
                SELECT event_id FROM scores ORDER BY ROWID ASC
            """
        )

        assert len(remaining_scores) == 1
        assert remaining_scores[0][0] == "recent_event_id"  # Score for recent event should remain

    async def test_get_wa_predictions_events(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        unique_event_id_1 = "unique_event_id_1"
        unique_event_id_2 = "unique_event_id_2"
        unique_event_ids = [unique_event_id_1, unique_event_id_2, "fake_unique_event_id"]

        wa_predictions = await db_operations.get_wa_predictions_events(
            unique_event_ids=unique_event_ids, interval_start_minutes=1
        )

        # No predictions yet
        assert len(wa_predictions) == 0

        events = [
            EventsModel(
                unique_event_id=unique_event_id_1,
                event_id="event_id_1",
                market_type="truncated_market",
                event_type="market",
                description="desc",
                outcome="0",
                status=3,
                metadata='{"key": "value"}',
                created_at="2024-12-02T14:30:00+00:00",
                cutoff="2024-12-26T11:30:00+00:00",
                resolved_at="2024-12-30T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id=unique_event_id_2,
                event_id="event_id_2",
                market_type="truncated_market",
                event_type="market",
                description="desc",
                outcome="0",
                status=3,
                metadata='{"key": "value"}',
                created_at="2024-12-02T14:30:00+00:00",
                cutoff="2024-12-26T11:30:00+00:00",
                resolved_at="2024-12-30T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events=events)

        scores = [
            ScoresModel(
                event_id="event_id_1",
                miner_uid=i,
                miner_hotkey=f"hk{i}",
                prediction=0.5,
                event_score=0.5,
                spec_version=1,
            )
            for i in range(50)
        ] + [
            ScoresModel(
                event_id="event_id_1",
                miner_uid=i,
                miner_hotkey=f"hk{i}",
                prediction=0.5,
                event_score=0.5,
                spec_version=1,
            )
            for i in range(50)
        ]

        await db_operations.insert_peer_scores(scores)

        await db_client.update(
            "UPDATE scores SET processed = 1, metagraph_score = miner_uid * 0.01",
        )

        # insert predictions like 0.0, 0.01, 0.02, 0.03, ..., 0.99 for calculations
        predictions = [
            (
                unique_event_id_1,
                f"hk{i}",
                i,
                1.0,
                10,
                i * 0.01,
            )
            for i in range(100)
        ] + [
            (
                unique_event_id_2,
                f"hk{i}",
                i,
                1.0,
                10,
                i * 0.01,
            )
            for i in range(100)
        ]

        await db_operations.upsert_predictions(predictions=predictions)
        # validate inserted predictions
        inserted_predictions = await db_client.many(
            "SELECT * FROM predictions", use_row_factory=True
        )
        assert [int(p["miner_uid"]) for p in inserted_predictions] == list(range(100)) * 2
        assert [p["interval_agg_prediction"] for p in inserted_predictions] == [
            i * 0.01 for i in range(100)
        ] * 2

        # No prediction for interval
        wa_predictions = await db_operations.get_wa_predictions_events(
            unique_event_ids=unique_event_ids, interval_start_minutes=11
        )

        assert len(wa_predictions) == 0

        wa_predictions = await db_operations.get_wa_predictions_events(
            unique_event_ids=unique_event_ids,
            interval_start_minutes=10,
        )
        # top 10 miners should be 40-49 with metagraph scores 0.4-0.49
        # their prediction should be 0.40-0.49
        assert len(wa_predictions) == 2

        expected_wa_prediction = sum([(i * 0.01) ** 2 for i in range(40, 50)]) / sum(
            [i * 0.01 for i in range(40, 50)]
        )

        assert wa_predictions[unique_event_id_1] == pytest.approx(expected_wa_prediction)
        assert wa_predictions[unique_event_id_2] == pytest.approx(expected_wa_prediction)

        # check for metagraph scores all 0s
        await db_client.update(
            "UPDATE scores SET metagraph_score = 0.0",
        )

        predictions = [
            (
                unique_event_id_1,
                f"hk{i}",
                i,
                1,
                12,  # interval_start_minutes
                i * 0.01,
            )
            for i in range(10)
        ]
        await db_operations.upsert_predictions(predictions=predictions)

        wa_predictions = await db_operations.get_wa_predictions_events(
            unique_event_ids=[unique_event_id_1],
            interval_start_minutes=12,
        )
        assert len(wa_predictions) == 1

        expected_simple_avg = sum([i * 0.01 for i in range(10)]) / 10
        assert wa_predictions[unique_event_id_1] == pytest.approx(expected_simple_avg)

    async def test_get_wa_predictions_events_no_predictions(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        unique_event_id = "unique_event_id_1"
        unique_event_ids = [unique_event_id]

        scores = [
            ScoresModel(
                event_id="event_id_1",
                miner_uid=i,
                miner_hotkey=f"hk{i}",
                prediction=0.5,
                event_score=0.5,
                spec_version=1,
            )
            for i in range(20)
        ]

        await db_operations.insert_peer_scores(scores)

        await db_client.update(
            "UPDATE scores SET processed = 1, metagraph_score = miner_uid * 0.01",
        )

        wa_predictions = await db_operations.get_wa_predictions_events(
            unique_event_ids=unique_event_ids,
            interval_start_minutes=10,
        )

        assert len(wa_predictions) == 0
