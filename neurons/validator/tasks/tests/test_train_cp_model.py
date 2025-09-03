import base64
import json
import pickle
from datetime import datetime, timedelta, timezone
from time import sleep
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.models.score import ScoresModel
from neurons.validator.tasks.train_cp_model import MODEL_NAME, TrainCommunityPredictionModel
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class TestTrainCPModel:
    @pytest.fixture
    def db_operations(self, db_client: DatabaseClient):
        logger = MagicMock(spec=InfiniteGamesLogger)

        return DatabaseOperations(db_client=db_client, logger=logger)

    @pytest.fixture
    def train_cp_model_task(
        self,
        db_operations: DatabaseOperations,
    ):
        logger = MagicMock(spec=InfiniteGamesLogger)

        return TrainCommunityPredictionModel(
            interval_seconds=60.0,
            db_operations=db_operations,
            logger=logger,
            env="prod",
        )

    @pytest.fixture(scope="function")
    async def seed_data(
        self,
        db_client: DatabaseClient,
        db_operations: DatabaseOperations,
    ):
        unique_event_id_1 = "unique_event_id_1"
        unique_event_id_2 = "unique_event_id_2"
        now = datetime.now(timezone.utc)
        now_minus_4h = now - timedelta(hours=4)

        scores = [
            ScoresModel(
                event_id=unique_event_id_1,
                miner_uid=i,
                miner_hotkey=f"hk{i}",
                prediction=1 - i * 0.01,
                event_score=0.5,
                spec_version=1,
            )
            for i in range(100)
        ] + [
            ScoresModel(
                event_id=unique_event_id_2,
                miner_uid=i,
                miner_hotkey=f"hk{i}",
                prediction=i * 0.01,
                event_score=0.5,
                spec_version=1,
            )
            for i in range(100)
        ]

        await db_operations.insert_peer_scores(scores)

        await db_client.update(
            "UPDATE scores SET processed = 1, metagraph_score = miner_uid * 0.01",
        )
        # move the first event to 4h ago to be in the previous batch
        await db_client.update(
            f"""UPDATE scores SET created_at = '{now_minus_4h.isoformat()}'"""
            + f""" WHERE event_id = '{unique_event_id_1}'""",
        )

        events = [
            EventsModel(
                unique_event_id=unique_event_id_1,
                event_id=unique_event_id_1,
                market_type="market_type1",
                event_type="type1",
                description="Some event",
                outcome="1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                resolved_at=now.isoformat(),
            ),
            EventsModel(
                unique_event_id=unique_event_id_2,
                event_id=unique_event_id_2,
                market_type="market_type2",
                event_type="type2",
                description="Some event",
                outcome="0",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                resolved_at=now.isoformat(),
            ),
        ]

        await db_operations.upsert_events(events)

    def test_init(self, train_cp_model_task: TrainCommunityPredictionModel):
        unit = train_cp_model_task

        assert isinstance(unit, TrainCommunityPredictionModel)
        assert unit.interval_seconds == 60.0
        assert unit.errors_count == 0

    async def test_get_data(self, train_cp_model_task: TrainCommunityPredictionModel, seed_data):
        unit = train_cp_model_task

        data = await unit.get_data()

        assert not data.empty
        assert len(data) == 200

        assert data.select_dtypes(include=["int32"]).columns.tolist() == [
            "miner_uid",
            "miner_rank",
            "prev_batch_miner_rank",
            "event_rank",
            "batch_rank",
            "event_batch_rank",
            "outcome",
        ]

        assert data.select_dtypes(include=["float32"]).columns.tolist() == [
            "prev_metagraph_score",
            "agg_prediction",
            "metagraph_score",
        ]
        assert data.select_dtypes(include=["object"]).columns.tolist() == [
            "miner_hotkey",
            "event_id",
        ]
        assert data.select_dtypes(include=["datetime64[ns]"]).columns.tolist() == [
            "scoring_batch",
        ]

    async def test_pivot_top_n(self, train_cp_model_task: TrainCommunityPredictionModel, seed_data):
        data = await train_cp_model_task.get_data()

        n = 100
        pivoted_data = TrainCommunityPredictionModel.pivot_top_n(data, n=n, train=True)
        assert not pivoted_data.empty

        assert len(pivoted_data) == 1
        assert len(pivoted_data.columns) == 3 + n

        assert pivoted_data.columns.tolist() == [
            "event_id",
            "batch_rank",
            "outcome",
        ] + [f"prediction_rank_{i}" for i in range(1, n + 1)]

        assert pivoted_data.select_dtypes(include=["float32"]).columns.tolist() == [
            f"prediction_rank_{i}" for i in range(1, n + 1)
        ]

        assert pivoted_data["event_id"].iloc[0] == "unique_event_id_2"
        assert pivoted_data["batch_rank"].iloc[0] == 1
        assert pivoted_data["outcome"].iloc[0] == 0

        for i in range(1, n + 1):
            assert pivoted_data[f"prediction_rank_{i}"].iloc[0] == 1 - i * 0.01

    def test_pivot_top_n_imputes_single_nan(self, train_cp_model_task):
        # Prepare a minimal raw DataFrame for a single event with 3 ranks,
        # but leave rank 2's prediction as NaN.
        df = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "batch_rank": 1,
                    "prev_batch_miner_rank": 1,
                    "agg_prediction": 1.0,
                    "outcome": 0,
                },
                {
                    "event_id": "e1",
                    "batch_rank": 1,
                    "prev_batch_miner_rank": 2,
                    "agg_prediction": 2.0,
                    "outcome": 0,
                },
                {
                    "event_id": "e1",
                    "batch_rank": 1,
                    "prev_batch_miner_rank": 3,
                    "agg_prediction": 3.0,
                    "outcome": 0,
                },
                # e2 missing rank 2
                {
                    "event_id": "e2",
                    "batch_rank": 1,
                    "prev_batch_miner_rank": 1,
                    "agg_prediction": 4.0,
                    "outcome": 0,
                },
                {
                    "event_id": "e2",
                    "batch_rank": 1,
                    "prev_batch_miner_rank": 3,
                    "agg_prediction": 6.0,
                    "outcome": 0,
                },
            ]
        )

        wide = TrainCommunityPredictionModel.pivot_top_n(df, n=3, train=True)
        pred_cols = [f"prediction_rank_{i}" for i in range(1, 4)]
        assert set(pred_cols).issubset(wide.columns), "pivot dropped all-NaN columns!"
        assert not wide[pred_cols].isna().any().any(), "pivot_top_n left NaNs after imputation"

        assert wide["prediction_rank_2"].iloc[1] == pytest.approx((4.0 + 6.0) / 2)

    def test_train_model(self, train_cp_model_task):
        wide_df = pd.DataFrame(
            {
                "event_id": ["e1", "e2", "e3", "e4"],
                "batch_rank": [1, 1, 1, 1],
                "outcome": [0, 1, 0, 1],
                "prediction_rank_1": [0.1, 0.6, 0.2, 0.8],
                "prediction_rank_2": [0.2, 0.8, 0.3, 0.7],
            }
        )

        model = TrainCommunityPredictionModel.train_model(wide_df)

        assert isinstance(model, LogisticRegression)
        assert hasattr(model, "coef_")
        assert model.coef_.shape == (1, 2)  # one row, two features
        assert hasattr(model, "intercept_")

        # Coefficients should be positive
        coefs = model.coef_.ravel()
        assert np.all(coefs >= 0)

        # And predict_proba should respect that ordering
        X_test = np.array([[0.15, 0.25], [0.85, 0.75]])
        X_test = pd.DataFrame(X_test, columns=["prediction_rank_1", "prediction_rank_2"])
        probs = model.predict_proba(X_test)[:, 1]
        assert probs.shape == (2,)
        assert probs[1] >= probs[0]

    async def test_save_model(self, train_cp_model_task, db_client):
        model = LogisticRegression(solver="liblinear").fit([[0], [1]], [0, 1])
        await train_cp_model_task.save_model(model, other_data=None)

        rows = await db_client.many(
            "SELECT * FROM models",
            use_row_factory=True,
        )
        assert len(rows) == 1
        assert rows[0]["name"] == MODEL_NAME
        assert rows[0]["other_data"] is None

        b64 = rows[0]["model_blob"]
        loaded: LogisticRegression = pickle.loads(base64.b64decode(b64))
        assert isinstance(loaded, LogisticRegression)
        assert np.allclose(loaded.coef_, model.coef_)

        # one more time with other_data - avoid created_at collision
        sleep(1)
        other = {"accuracy": 0.75, "notes": "unit test"}
        await train_cp_model_task.save_model(model, other_data=other)

        rows = await db_client.many(
            "SELECT * FROM models ORDER BY created_at DESC",
            use_row_factory=True,
        )
        assert len(rows) == 2
        assert rows[0]["name"] == MODEL_NAME
        assert rows[0]["other_data"] == json.dumps(other)

        b64 = rows[0]["model_blob"]
        loaded: LogisticRegression = pickle.loads(base64.b64decode(b64))
        assert isinstance(loaded, LogisticRegression)
        assert np.allclose(loaded.coef_, model.coef_)

    async def test_run(
        self,
        train_cp_model_task: TrainCommunityPredictionModel,
        seed_data,
        db_operations,
        db_client,
    ):
        # insert more data to have more classes
        unique_event_id_3 = "unique_event_id_3"
        now = datetime.now(timezone.utc)
        await db_operations.upsert_events(
            [
                EventsModel(
                    unique_event_id=unique_event_id_3,
                    event_id=unique_event_id_3,
                    market_type="market_type1",
                    event_type="type1",
                    description="Some event",
                    outcome="1",
                    status=EventStatus.SETTLED,
                    metadata='{"key": "value"}',
                    resolved_at=now.isoformat(),
                ),
            ]
        )
        await db_operations.insert_peer_scores(
            [
                ScoresModel(
                    event_id=unique_event_id_3,
                    miner_uid=i,
                    miner_hotkey=f"hk{i}",
                    prediction=1 - i * 0.01,
                    event_score=0.5,
                    spec_version=1,
                )
                for i in range(100)
            ]
        )

        await db_client.update(
            "UPDATE scores SET processed = 1, metagraph_score = miner_uid * 0.01",
        )

        await train_cp_model_task.run()

        debug_calls = train_cp_model_task.logger.debug.call_args_list
        assert len(debug_calls) == 4
        assert debug_calls[0][0][0] == "Data fetched successfully from the database."
        assert debug_calls[1][0][0] == "Model trained successfully."
        assert debug_calls[2][0][0] == "Community Prediction model saved successfully to DB."
        assert debug_calls[3][0][0] == "Community Prediction model training completed successfully."

        rows = await db_client.many(
            "SELECT * FROM models ORDER BY created_at DESC",
            use_row_factory=True,
        )
        assert len(rows) == 1
        assert rows[0]["name"] == MODEL_NAME
        assert rows[0]["other_data"] is None
        assert rows[0]["created_at"] is not None

        b64 = rows[0]["model_blob"]
        loaded: LogisticRegression = pickle.loads(base64.b64decode(b64.encode("ascii")))
        assert isinstance(loaded, LogisticRegression)
        assert len(loaded.coef_.ravel()) == 100

    async def test_run_no_data(self, train_cp_model_task: TrainCommunityPredictionModel):
        # should not have been called
        train_cp_model_task.pivot_top_n = MagicMock()
        train_cp_model_task.train_model = MagicMock()
        train_cp_model_task.save_model = MagicMock()

        await train_cp_model_task.run()

        train_cp_model_task.logger.error.assert_called_once_with(
            "No data to train the community prediction model."
        )
        train_cp_model_task.pivot_top_n.assert_not_called()
        train_cp_model_task.train_model.assert_not_called()
        train_cp_model_task.save_model.assert_not_called()
