import tempfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from bittensor_wallet import Wallet
from freezegun import freeze_time

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.if_games.client import IfGamesClient
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.models.if_games_client import MinerScore, PostScoresRequestBody
from neurons.validator.models.score import ScoresModel
from neurons.validator.tasks.export_scores import ExportScores
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class TestExportScores:
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
    def bt_wallet(self):
        hotkey_mock = MagicMock()
        hotkey_mock.sign = MagicMock(side_effect=lambda x: x.encode("utf-8"))
        hotkey_mock.ss58_address = "hotkey2"

        bt_wallet = MagicMock(spec=Wallet)
        bt_wallet.get_hotkey = MagicMock(return_value=hotkey_mock)
        bt_wallet.hotkey.ss58_address = "hotkey2"

        return bt_wallet

    @pytest.fixture
    def sample_event(self) -> EventsModel:
        return EventsModel(
            unique_event_id="unique_event_id",
            event_id="event_id",
            market_type="test_market",
            event_type="test_type",
            registered_date=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            description="""This is a test event description""",
            outcome="1",
            metadata='{"market_type": "test_real_market"}',
            status=EventStatus.SETTLED,
            processed=False,
            exported=False,
            created_at=datetime(2025, 1, 2, 3, 0, 0, tzinfo=timezone.utc),
            cutoff=datetime(2025, 1, 2, 4, 0, 0, tzinfo=timezone.utc),
            resolved_at=None,
        )

    @pytest.fixture
    def export_scores_task(
        self,
        db_operations: DatabaseOperations,
        bt_wallet: Wallet,  # type: ignore
    ):
        api_client = IfGamesClient(
            env="test", logger=MagicMock(spec=InfiniteGamesLogger), bt_wallet=bt_wallet
        )
        logger = MagicMock(spec=InfiniteGamesLogger)

        with freeze_time("2025-01-02 03:00:00"):
            return ExportScores(
                interval_seconds=60.0,
                page_size=100,
                db_operations=db_operations,
                api_client=api_client,
                logger=logger,
                validator_uid=2,
                validator_hotkey=bt_wallet.hotkey.ss58_address,
            )

    def test_init(self, export_scores_task):
        unit = export_scores_task

        assert isinstance(unit, ExportScores)
        assert unit.interval == 60.0
        assert unit.interval_seconds == 60.0
        assert unit.page_size == 100
        assert unit.errors_count == 0
        assert unit.validator_uid == 2
        assert unit.validator_hotkey == "hotkey2"

    def test_prepare_scores_payload_success(
        self, export_scores_task: ExportScores, sample_event: EventsModel
    ):
        event = sample_event
        now = datetime(2025, 1, 2, 3, 0, 0, tzinfo=timezone.utc)
        score = ScoresModel(
            event_id=event.event_id,
            miner_uid=1,
            miner_hotkey="hk1",
            prediction=0.95,
            event_score=0.90,
            metagraph_score=1.0,
            other_data='{"extra": "data"}',
            spec_version=1,
            created_at=now,
        )

        payload = export_scores_task.prepare_scores_payload(event, [score])

        assert isinstance(payload, PostScoresRequestBody)

        results = payload.results
        assert len(results) == 1

        result = results[0]
        assert isinstance(result, MinerScore)

        assert result == MinerScore(
            event_id=event.event_id,
            prediction=score.prediction,
            answer=float(event.outcome),
            miner_hotkey=score.miner_hotkey,
            miner_uid=score.miner_uid,
            miner_score=score.event_score,
            miner_effective_score=score.metagraph_score,
            validator_hotkey=export_scores_task.validator_hotkey,
            validator_uid=export_scores_task.validator_uid,
            spec_version="1039",
            registered_date=event.registered_date,
            scored_at=score.created_at,
        )

        # score.other_data is not included in the payload; also dates nullified
        score.other_data = None
        score.spec_version = 1040

        payload = export_scores_task.prepare_scores_payload(event, [score])

        result = payload.results[0]
        assert result.spec_version == "1040"

    async def test_export_scores_to_backend(self, export_scores_task: ExportScores, monkeypatch):
        unit = export_scores_task
        unit.api_client.post_scores = AsyncMock(return_value=True)

        dummy_payload = PostScoresRequestBody.model_validate(
            {
                "results": [
                    {
                        "event_id": "event_export",
                        "prediction": 1,
                        "answer": 1,
                        "miner_hotkey": "miner_hotkey",
                        "miner_uid": 1,
                        "miner_score": 1,
                        "miner_effective_score": 1,
                        "validator_hotkey": "validator_hotkey",
                        "validator_uid": 2,
                        "spec_version": "1.3.3",
                        "registered_date": datetime.now(),
                        "scored_at": datetime.now(),
                    }
                ]
            }
        )

        await export_scores_task.export_scores_to_backend(dummy_payload)
        export_scores_task.logger.debug.assert_called_with(
            "Exported scores.",
            extra={
                "event_id": dummy_payload.results[0].event_id,
                "n_scores": len(dummy_payload.results),
            },
        )

        assert export_scores_task.errors_count == 0
        assert unit.api_client.post_scores.call_count == 1
        assert unit.api_client.post_scores.call_args.kwargs["body"] == dummy_payload

        # mock with side effect
        unit.api_client.post_scores = AsyncMock(side_effect=Exception("Simulated failure"))
        with pytest.raises(Exception, match="Simulated failure"):
            await export_scores_task.export_scores_to_backend(dummy_payload)

        assert unit.api_client.post_scores.call_count == 1

    async def test_run_no_scored_events(
        self, export_scores_task: ExportScores, db_operations: DatabaseOperations
    ):
        db_operations.get_peer_scored_events_for_export = AsyncMock(return_value=[])
        export_scores_task.logger.debug = MagicMock()

        await export_scores_task.run()
        export_scores_task.logger.debug.assert_any_call("No peer scored events to export scores.")
        assert export_scores_task.errors_count == 0

    async def test_run_no_peer_scores_for_event(
        self,
        export_scores_task: ExportScores,
        db_operations: DatabaseOperations,
        db_client: DatabaseClient,
        sample_event: EventsModel,
    ):
        unit = export_scores_task
        event = sample_event
        await db_operations.upsert_events([event])

        score = ScoresModel(
            event_id=event.event_id,
            miner_uid=2,
            miner_hotkey="hk2",
            prediction=0.75,
            event_score=0.80,
            spec_version=1,
        )

        await db_operations.insert_peer_scores([score])
        await db_client.update(
            "UPDATE scores SET processed = ?",
            [
                1,
            ],
        )
        unit.db_operations.get_peer_scores_for_export = AsyncMock(return_value=[])

        await unit.run()
        unit.logger.warning.assert_called_with(
            "No peer scores found for event.",
            extra={"event_id": event.event_id},
        )
        assert unit.logger.debug.call_args_list[1][0][0] == "Export scores task completed."
        assert unit.logger.debug.call_args_list[1][1]["extra"] == {"errors_count": 1}

    async def test_run_no_payload(
        self,
        export_scores_task: ExportScores,
        db_operations: DatabaseOperations,
        db_client: DatabaseClient,
        sample_event: EventsModel,
    ):
        unit = export_scores_task
        unit.api_client.post_scores = AsyncMock(return_value=True)
        unit.db_operations.get_peer_scores_for_export = AsyncMock(return_value=[])

        event = sample_event
        await db_operations.upsert_events([event])

        score = ScoresModel(
            event_id=event.event_id,
            miner_uid=2,
            miner_hotkey="hk2",
            prediction=0.75,
            event_score=0.80,
            spec_version=1,
        )

        await db_operations.insert_peer_scores([score])

        await db_client.update(
            "UPDATE scores SET processed = ?",
            [
                1,
            ],
        )
        unit.prepare_scores_payload = MagicMock(return_value=None)

        await unit.run()

        unit.prepare_scores_payload.assert_not_called()

        unit.logger.warning.assert_called_with(
            "No peer scores found for event.",
            extra={"event_id": event.event_id},
        )
        assert unit.logger.debug.call_args_list[1][0][0] == "Export scores task completed."
        assert unit.logger.debug.call_args_list[1][1]["extra"] == {"errors_count": 1}

    async def test_run_export_exception(
        self,
        export_scores_task: ExportScores,
        db_operations: DatabaseOperations,
        db_client: DatabaseClient,
        sample_event: EventsModel,
    ):
        unit = export_scores_task
        unit
        unit.api_client.post_scores = AsyncMock(side_effect=Exception("Simulated failure"))
        event = sample_event
        await db_operations.upsert_events([event])

        score = ScoresModel(
            event_id=event.event_id,
            miner_uid=2,
            miner_hotkey="hk2",
            prediction=0.75,
            event_score=0.80,
            spec_version=1,
        )

        await db_operations.insert_peer_scores([score])
        await db_client.update(
            "UPDATE scores SET processed = ?, metagraph_score = ?, other_data = ?",
            [
                1,
                1.0,
                '{"extra": "data"}',
            ],
        )

        await unit.run()
        unit.logger.exception.assert_called_with(
            "Failed to export scores.",
            extra={"event_id": event.event_id},
        )
        assert unit.logger.debug.call_args_list[1][0][0] == "Export scores task completed."
        assert unit.logger.debug.call_args_list[1][1]["extra"] == {"errors_count": 1}

    async def test_run_e2e(
        self,
        export_scores_task: ExportScores,
        db_operations: DatabaseOperations,
        db_client: DatabaseClient,
        sample_event: EventsModel,
    ):
        unit = export_scores_task
        unit.api_client.post_scores = AsyncMock(return_value=True)

        event = sample_event
        event_2 = event.model_copy(deep=True)
        event_2.event_id = "event_id_2"
        event_2.unique_event_id = "unique_event_id_2"
        await db_operations.upsert_events([event, event_2])
        events_inserted = await db_client.many("SELECT * FROM events", use_row_factory=True)
        assert len(events_inserted) == 2

        score_1 = ScoresModel(
            event_id=event.event_id,
            miner_uid=2,
            miner_hotkey="hk2",
            prediction=0.75,
            event_score=0.80,
            spec_version=1,
        )
        score_2 = ScoresModel(
            event_id=event.event_id,
            miner_uid=3,
            miner_hotkey="hk3",
            prediction=0.85,
            event_score=0.90,
            spec_version=1,
        )
        score_3 = score_1.model_copy(deep=True)
        score_3.event_id = event_2.event_id

        await db_operations.insert_peer_scores([score_1, score_2, score_3])
        await db_client.update(
            "UPDATE scores SET processed = ?, metagraph_score = ?, other_data = ?",
            [
                1,
                1.0,
                '{"extra": "data"}',
            ],
        )

        await unit.run()
        updated_scores = await db_client.many(
            "SELECT * FROM scores",
            use_row_factory=True,
        )
        for row in updated_scores:
            assert row["exported"] == 1

        updated_events = await db_client.many(
            "SELECT * FROM events",
            use_row_factory=True,
        )
        for row in updated_events:
            assert row["exported"] == 1

        unit.logger.debug.assert_any_call(
            "Found peer scored events to export scores.", extra={"n_events": 2}
        )

        unit.logger.debug.assert_any_call(
            "Export scores task completed.", extra={"errors_count": 0}
        )

        assert unit.api_client.post_scores.call_count == 2
