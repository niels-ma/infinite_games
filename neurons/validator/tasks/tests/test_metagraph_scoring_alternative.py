import math
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
from bittensor.core.metagraph import AsyncMetagraph
from pandas.testing import assert_frame_equal

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.models.score import AlternativeMetagraphScore
from neurons.validator.tasks.metagraph_scoring_alternative import MetagraphScoringAlternative
from neurons.validator.tasks.peer_scoring import PeerScoring
from neurons.validator.utils.cluster_selector.cluster_selector import ClusterSelector
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class TestMetagraphScoringAlternative:
    @pytest.fixture
    def db_operations(self, db_client: DatabaseClient):
        logger = MagicMock(spec=InfiniteGamesLogger)

        return DatabaseOperations(db_client=db_client, logger=logger)

    async def test_get_metagraph_neurons_and_owner(self):
        mock_metagraph = MagicMock(spec=AsyncMetagraph, sync=AsyncMock(), owner_hotkey="hotkey10")
        mock_db_operations = MagicMock(spec=DatabaseOperations)
        mock_logger = MagicMock(spec=InfiniteGamesLogger)

        mock_metagraph.uids = [torch.tensor(1), np.array(5), np.int64(10), torch.tensor(15)]
        mock_metagraph.hotkeys = {1: "hotkey1", 5: "hotkey5", 10: "hotkey10", 15: "hotkey15"}

        alternative_scoring_task = MetagraphScoringAlternative(
            interval_seconds=60.0,
            cluster_selector_cls=ClusterSelector,
            db_operations=mock_db_operations,
            metagraph=mock_metagraph,
            logger=mock_logger,
        )

        (
            neurons,
            owner,
        ) = await alternative_scoring_task.get_metagraph_neurons_and_owner()

        mock_metagraph.sync.assert_awaited_once_with(lite=True)

        # Verify returned DataFrame
        assert isinstance(neurons, pd.DataFrame)
        assert len(neurons) == 4

        # Verify the actual data
        expected_data = [
            {"miner_uid": 1, "miner_hotkey": "hotkey1"},
            {"miner_uid": 5, "miner_hotkey": "hotkey5"},
            {"miner_uid": 10, "miner_hotkey": "hotkey10"},
            {"miner_uid": 15, "miner_hotkey": "hotkey15"},
        ]

        for i, row in neurons.iterrows():
            assert row["miner_uid"] == expected_data[i]["miner_uid"]
            assert row["miner_hotkey"] == expected_data[i]["miner_hotkey"]

        # Verify owner_uid
        assert isinstance(owner, dict)
        assert owner["uid"] == 10
        assert owner["hotkey"] == "hotkey10"

    async def test_get_metagraph_neurons_and_owner_empty_metagraph(self):
        mock_metagraph = MagicMock(spec=AsyncMetagraph, sync=AsyncMock(), owner_hotkey="hotkey_x")
        mock_db_operations = MagicMock(spec=DatabaseOperations)
        mock_logger = MagicMock(spec=InfiniteGamesLogger)

        mock_metagraph.uids = []
        mock_metagraph.hotkeys = {}

        alternative_scoring_task = MetagraphScoringAlternative(
            interval_seconds=60.0,
            cluster_selector_cls=ClusterSelector,
            db_operations=mock_db_operations,
            metagraph=mock_metagraph,
            logger=mock_logger,
        )

        with pytest.raises(AssertionError, match="Owner uid not found in metagraph uids"):
            await alternative_scoring_task.get_metagraph_neurons_and_owner()

        mock_metagraph.sync.assert_awaited_once_with(lite=True)

    async def test_get_ranked_predictions(self):
        mock_metagraph = MagicMock(spec=AsyncMetagraph)
        mock_logger = MagicMock(spec=InfiniteGamesLogger)
        mock_db_operations = MagicMock(spec=DatabaseOperations)

        mock_db_operations.get_predictions_ranked.return_value = (
            # event_id,
            # event_rank,
            # outcome,
            # miner_uid,
            # miner_hotkey,
            # prediction
            ("event1", 1, "1", 11, "hotkey_11", 0.33),
            ("event1", 1, "1", 77, "hotkey_77", 0.89),
            ("event2", 2, "0", 11, "hotkey_11", 0.21),
        )

        alternative_scoring_task = MetagraphScoringAlternative(
            interval_seconds=60.0,
            cluster_selector_cls=ClusterSelector,
            db_operations=mock_db_operations,
            metagraph=mock_metagraph,
            logger=mock_logger,
        )

        result = await alternative_scoring_task.get_ranked_predictions()

        moving_window = 150
        mock_db_operations.get_predictions_ranked.assert_awaited_once_with(
            moving_window=moving_window
        )

        assert result.to_dict(orient="records") == [
            {
                "event_id": "event1",
                "event_rank": 1,
                "miner_hotkey": "hotkey_11",
                "miner_uid": 11,
                "outcome": "1",
                "prediction": 0.33,
            },
            {
                "event_id": "event1",
                "event_rank": 1,
                "miner_hotkey": "hotkey_77",
                "miner_uid": 77,
                "outcome": "1",
                "prediction": 0.89,
            },
            {
                "event_id": "event2",
                "event_rank": 2,
                "miner_hotkey": "hotkey_11",
                "miner_uid": 11,
                "outcome": "0",
                "prediction": 0.21,
            },
        ]

    async def test_get_internal_forecasts_mixed_forecasts(self, db_operations: DatabaseOperations):
        mock_metagraph = MagicMock(spec=AsyncMetagraph)
        mock_logger = MagicMock(spec=InfiniteGamesLogger)

        events = [
            EventsModel(
                unique_event_id="ifgames-event1",
                event_id="event1",
                market_type="truncated_market1",
                event_type="market1",
                description="desc1",
                outcome="1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value1"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
                forecasts='{"2025-01-20T16:10:15Z": 0.7, "2025-01-20T16:15:20Z": 0.8, "2025-01-20T16:20:25Z": 0.9}',
            ),
            EventsModel(
                unique_event_id="ifgames-event2",
                event_id="event2",
                market_type="truncated_market",
                event_type="market",
                description="desc",
                outcome="1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="2015-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
                forecasts="{}",
            ),
            EventsModel(
                unique_event_id="ifgames-event3",
                event_id="event3",
                market_type="truncated_market",
                event_type="market",
                description="desc",
                outcome="1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="2018-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
                forecasts='{"2025-01-20T16:10:15Z": 0.6}',
            ),
        ]

        await db_operations.upsert_events(events=events)

        alternative_scoring_task = MetagraphScoringAlternative(
            interval_seconds=60.0,
            cluster_selector_cls=ClusterSelector,
            db_operations=db_operations,
            metagraph=mock_metagraph,
            logger=mock_logger,
        )

        event_ids = ["event1", "event2", "event3", "fake_event"]
        result = await alternative_scoring_task.get_internal_forecasts(event_ids=event_ids)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["event_id", "prediction"]
        assert len(result) == 2  # Only events with valid forecasts should be included

        # Verify the results
        result_dict = result.set_index("event_id")["prediction"].to_dict()

        def weighted_average(values: list[float]):
            n = len(values)

            weights = [PeerScoring.power_decay_weight(i, n) for i in range(n)]

            sum_forecasts_weighted = sum(v * w for v, w in zip(values, weights))
            sum_weights = sum(weights)

            return sum_forecasts_weighted / sum_weights

        # event1: 0.7 / 0.8 / 0.9
        assert result_dict["event1"] == weighted_average([0.7, 0.8, 0.9])

        # event4: 0.6
        assert result_dict["event3"] == 0.6

        # event2 should not be in results due to None/empty forecasts
        assert "event2" not in result_dict

    async def test_get_internal_forecasts_empty_list(self, db_operations: DatabaseOperations):
        mock_metagraph = MagicMock(spec=AsyncMetagraph)
        mock_logger = MagicMock(spec=InfiniteGamesLogger)

        events = [
            EventsModel(
                unique_event_id="ifgames-event1",
                event_id="event1",
                market_type="truncated_market1",
                event_type="market1",
                description="desc1",
                outcome="1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value1"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
                forecasts='{"2025-01-20T16:10:15Z": 0.6}',
            ),
        ]

        await db_operations.upsert_events(events=events)

        alternative_scoring_task = MetagraphScoringAlternative(
            interval_seconds=60.0,
            cluster_selector_cls=ClusterSelector,
            db_operations=db_operations,
            metagraph=mock_metagraph,
            logger=mock_logger,
        )

        # Empty list
        event_ids = []
        result = await alternative_scoring_task.get_internal_forecasts(event_ids=event_ids)

        # Verify returned DataFrame is empty but has correct structure
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["event_id", "prediction"]
        assert len(result) == 0

    async def test_get_alternative_metagraph_scores(self):
        mock_metagraph = MagicMock(spec=AsyncMetagraph)
        mock_logger = MagicMock(spec=InfiniteGamesLogger)
        mock_db_operations = MagicMock(spec=DatabaseOperations)

        alternative_scoring_task = MetagraphScoringAlternative(
            interval_seconds=60.0,
            cluster_selector_cls=ClusterSelector,
            db_operations=mock_db_operations,
            metagraph=mock_metagraph,
            logger=mock_logger,
        )

        selected_clusters_credits = pd.DataFrame(
            {
                "miner_uid": [11, 22, 33, 44],
                "miner_hotkey": ["hotkey_11", "hotkey_22", "hotkey_33", "hotkey_44"],
                "round_credit_per_miner": [0.3, 0.4, 0.2, math.nan],
                "cluster_id": ["cluster_id_1", "cluster_id_2", "cluster_id_3", "cluster_id_3"],
                "coefficients": [
                    "coefficients_1",
                    "coefficients_2",
                    "coefficients_3",
                    "coefficients_3",
                ],
                "cluster_credit": [
                    "cluster_credit_1",
                    "cluster_credit_2",
                    "cluster_credit_3",
                    "cluster_credit_3",
                ],
                "cluster_credit_adjusted": [
                    "cluster_credit_adjusted_1",
                    "cluster_credit_adjusted_2",
                    "cluster_credit_adjusted_3",
                    "cluster_credit_adjusted_3",
                ],
                "miner_count": ["miner_count_1", "miner_count_2", "miner_count_3", "miner_count_3"],
                "scaled_std": ["scaled_std_1", "scaled_std_2", "scaled_std_3", "scaled_std_3"],
                "scaled_log_loss": [
                    "scaled_log_loss_1",
                    "scaled_log_loss_2",
                    "scaled_log_loss_3",
                    "scaled_log_loss_3",
                ],
            }
        )

        owner = {"uid": 99, "hotkey": "owner_hotkey"}

        result = await alternative_scoring_task.get_alternative_metagraph_scores(
            selected_clusters_credits=selected_clusters_credits, owner=owner
        )

        assert result == [
            AlternativeMetagraphScore(
                miner_uid=11,
                miner_hotkey="hotkey_11",
                alternative_metagraph_score=0.3,
                alternative_other_data='{"cluster_id": "cluster_id_1", "coefficients": "coefficients_1", "cluster_credit": "cluster_credit_1", "cluster_credit_adjusted": "cluster_credit_adjusted_1", "miner_count": "miner_count_1", "scaled_std": "scaled_std_1", "scaled_log_loss": "scaled_log_loss_1"}',
            ),
            AlternativeMetagraphScore(
                miner_uid=22,
                miner_hotkey="hotkey_22",
                alternative_metagraph_score=0.4,
                alternative_other_data='{"cluster_id": "cluster_id_2", "coefficients": "coefficients_2", "cluster_credit": "cluster_credit_2", "cluster_credit_adjusted": "cluster_credit_adjusted_2", "miner_count": "miner_count_2", "scaled_std": "scaled_std_2", "scaled_log_loss": "scaled_log_loss_2"}',
            ),
            AlternativeMetagraphScore(
                miner_uid=33,
                miner_hotkey="hotkey_33",
                alternative_metagraph_score=0.2,
                alternative_other_data='{"cluster_id": "cluster_id_3", "coefficients": "coefficients_3", "cluster_credit": "cluster_credit_3", "cluster_credit_adjusted": "cluster_credit_adjusted_3", "miner_count": "miner_count_3", "scaled_std": "scaled_std_3", "scaled_log_loss": "scaled_log_loss_3"}',
            ),
            AlternativeMetagraphScore(
                miner_uid=44,
                miner_hotkey="hotkey_44",
                alternative_metagraph_score=None,
                alternative_other_data='{"cluster_id": "cluster_id_3", "coefficients": "coefficients_3", "cluster_credit": "cluster_credit_3", "cluster_credit_adjusted": "cluster_credit_adjusted_3", "miner_count": "miner_count_3", "scaled_std": "scaled_std_3", "scaled_log_loss": "scaled_log_loss_3"}',
            ),
            AlternativeMetagraphScore(
                miner_uid=99,
                miner_hotkey="owner_hotkey",
                # Owner burns the rest 1 - (0.3 + 0.4 + 0.2)
                alternative_metagraph_score=0.10000000000000009,
                alternative_other_data='{"type": "subnet_owner"}',
            ),
            # NaN credits miners are filtered out
        ]

    async def test_get_alternative_metagraph_scores_with_none_and_nan_credits(self):
        mock_metagraph = MagicMock(spec=AsyncMetagraph)
        mock_logger = MagicMock(spec=InfiniteGamesLogger)
        mock_db_operations = MagicMock(spec=DatabaseOperations)

        alternative_scoring_task = MetagraphScoringAlternative(
            interval_seconds=60.0,
            cluster_selector_cls=ClusterSelector,
            db_operations=mock_db_operations,
            metagraph=mock_metagraph,
            logger=mock_logger,
        )

        selected_clusters_credits = pd.DataFrame(
            {
                "miner_uid": [22, 55],
                "miner_hotkey": ["hotkey_22", "hotkey_55"],
                "round_credit_per_miner": [None, math.nan],
                "cluster_id": [1, 3],
                "coefficients": [1, 3],
                "cluster_credit": [1, 3],
                "cluster_credit_adjusted": [1, 3],
                "miner_count": [1, 3],
                "scaled_std": [1, 3],
                "scaled_log_loss": [1, 3],
            }
        )

        owner = {"uid": 99, "hotkey": "owner_hotkey"}

        result = await alternative_scoring_task.get_alternative_metagraph_scores(
            selected_clusters_credits=selected_clusters_credits, owner=owner
        )

        assert result == [
            AlternativeMetagraphScore(
                miner_uid=22,
                miner_hotkey="hotkey_22",
                alternative_metagraph_score=None,
                alternative_other_data='{"cluster_id": 1, "coefficients": 1, "cluster_credit": 1, "cluster_credit_adjusted": 1, "miner_count": 1, "scaled_std": 1, "scaled_log_loss": 1}',
            ),
            AlternativeMetagraphScore(
                miner_uid=55,
                miner_hotkey="hotkey_55",
                alternative_metagraph_score=None,
                alternative_other_data='{"cluster_id": 3, "coefficients": 3, "cluster_credit": 3, "cluster_credit_adjusted": 3, "miner_count": 3, "scaled_std": 3, "scaled_log_loss": 3}',
            ),
            AlternativeMetagraphScore(
                # Owner burns all
                miner_uid=99,
                miner_hotkey="owner_hotkey",
                alternative_metagraph_score=1.0,
                alternative_other_data='{"type": "subnet_owner"}',
            ),
        ]

    async def test_get_alternative_metagraph_scores_owner_in_clusters(self):
        mock_metagraph = MagicMock(spec=AsyncMetagraph)
        mock_logger = MagicMock(spec=InfiniteGamesLogger)
        mock_db_operations = MagicMock(spec=DatabaseOperations)

        alternative_scoring_task = MetagraphScoringAlternative(
            interval_seconds=60.0,
            cluster_selector_cls=ClusterSelector,
            db_operations=mock_db_operations,
            metagraph=mock_metagraph,
            logger=mock_logger,
        )

        selected_clusters_credits = pd.DataFrame(
            {
                "miner_uid": [11, 99, 33],  # 99 is the owner
                "miner_hotkey": ["hotkey_11", "owner_hotkey", "hotkey_33"],
                "round_credit_per_miner": [0.5, 0.1, 0.5],
                "cluster_id": [1, 1, 1],
                "coefficients": [1, 1, 1],
                "cluster_credit": [1, 1, 1],
                "cluster_credit_adjusted": [1, 1, 1],
                "miner_count": [1, 1, 1],
                "scaled_std": [1, 1, 1],
                "scaled_log_loss": [1, 1, 1],
            }
        )

        owner = {"uid": 99, "hotkey": "owner_hotkey"}

        result = await alternative_scoring_task.get_alternative_metagraph_scores(
            selected_clusters_credits=selected_clusters_credits, owner=owner
        )

        assert result == [
            AlternativeMetagraphScore(
                miner_uid=11,
                miner_hotkey="hotkey_11",
                alternative_metagraph_score=0.5,
                alternative_other_data='{"cluster_id": 1, "coefficients": 1, "cluster_credit": 1, "cluster_credit_adjusted": 1, "miner_count": 1, "scaled_std": 1, "scaled_log_loss": 1}',
            ),
            AlternativeMetagraphScore(
                miner_uid=33,
                miner_hotkey="hotkey_33",
                alternative_metagraph_score=0.5,
                alternative_other_data='{"cluster_id": 1, "coefficients": 1, "cluster_credit": 1, "cluster_credit_adjusted": 1, "miner_count": 1, "scaled_std": 1, "scaled_log_loss": 1}',
            ),
            AlternativeMetagraphScore(
                miner_uid=99,
                miner_hotkey="owner_hotkey",
                # Credits to burn is recalculated as 1 - (0.5 + 0.5)
                alternative_metagraph_score=0.0,
                alternative_other_data='{"type": "subnet_owner"}',
            ),
        ]

        # Verify logger call
        mock_logger.error.assert_called_once()
        error_call_args = mock_logger.error.call_args

        assert "Owner should have no credits from clusters" in error_call_args[0][0]
        assert error_call_args[1]["extra"]["owner_id"] == 99
        assert error_call_args[1]["extra"]["owner_hotkey"] == "owner_hotkey"
        assert error_call_args[1]["extra"]["miner_hotkey"] == "owner_hotkey"
        assert error_call_args[1]["extra"]["round_credit_per_miner"] == 0.1

    async def test_get_alternative_metagraph_scores_owner_in_clusters_nan(self):
        mock_metagraph = MagicMock(spec=AsyncMetagraph)
        mock_logger = MagicMock(spec=InfiniteGamesLogger)
        mock_db_operations = MagicMock(spec=DatabaseOperations)

        alternative_scoring_task = MetagraphScoringAlternative(
            interval_seconds=60.0,
            cluster_selector_cls=ClusterSelector,
            db_operations=mock_db_operations,
            metagraph=mock_metagraph,
            logger=mock_logger,
        )

        selected_clusters_credits = pd.DataFrame(
            {
                "miner_uid": [11, 99, 33],  # 99 is the owner
                "miner_hotkey": ["hotkey_11", "owner_hotkey", "hotkey_33"],
                "round_credit_per_miner": [0.5, math.nan, 0.3],
                "cluster_id": [1, 1, 1],
                "coefficients": [1, 1, 1],
                "cluster_credit": [1, 1, 1],
                "cluster_credit_adjusted": [1, 1, 1],
                "miner_count": [1, 1, 1],
                "scaled_std": [1, 1, 1],
                "scaled_log_loss": [1, 1, 1],
            }
        )

        owner = {"uid": 99, "hotkey": "owner_hotkey"}

        result = await alternative_scoring_task.get_alternative_metagraph_scores(
            selected_clusters_credits=selected_clusters_credits, owner=owner
        )

        assert result == [
            AlternativeMetagraphScore(
                miner_uid=11,
                miner_hotkey="hotkey_11",
                alternative_metagraph_score=0.5,
                alternative_other_data='{"cluster_id": 1, "coefficients": 1, "cluster_credit": 1, "cluster_credit_adjusted": 1, "miner_count": 1, "scaled_std": 1, "scaled_log_loss": 1}',
            ),
            AlternativeMetagraphScore(
                miner_uid=33,
                miner_hotkey="hotkey_33",
                alternative_metagraph_score=0.3,
                alternative_other_data='{"cluster_id": 1, "coefficients": 1, "cluster_credit": 1, "cluster_credit_adjusted": 1, "miner_count": 1, "scaled_std": 1, "scaled_log_loss": 1}',
            ),
            AlternativeMetagraphScore(
                miner_uid=99,
                miner_hotkey="owner_hotkey",
                # Credits to burn is recalculated as 1 - (0.5 + 0.3)
                alternative_metagraph_score=0.19999999999999996,
                alternative_other_data='{"type": "subnet_owner"}',
            ),
        ]

        # Verify logger call
        mock_logger.debug.assert_called_once()
        debug_call_args = mock_logger.debug.call_args

        assert "Owner should have no credits from clusters" in debug_call_args[0][0]
        assert debug_call_args[1]["extra"]["owner_id"] == 99
        assert debug_call_args[1]["extra"]["owner_hotkey"] == "owner_hotkey"
        assert debug_call_args[1]["extra"]["miner_hotkey"] == "owner_hotkey"
        assert debug_call_args[1]["extra"]["round_credit_per_miner"] is None

    async def test_run_no_events_exists_to_process(self):
        mock_metagraph = MagicMock(spec=AsyncMetagraph, sync=AsyncMock())
        mock_logger = MagicMock(spec=InfiniteGamesLogger)
        mock_db_operations = MagicMock(spec=DatabaseOperations)

        mock_db_operations.count_events_for_alternative_metagraph_scoring.return_value = 0

        alternative_scoring_task = MetagraphScoringAlternative(
            interval_seconds=60.0,
            cluster_selector_cls=ClusterSelector,
            db_operations=mock_db_operations,
            metagraph=mock_metagraph,
            logger=mock_logger,
        )

        # Run the method
        await alternative_scoring_task.run()

        mock_logger.add_context.assert_called_once_with(
            {"source_task": "metagraph-scoring-alternative"}
        )

        mock_db_operations.count_events_for_alternative_metagraph_scoring.assert_awaited_once_with()

        mock_logger.debug.assert_called_once_with("No events to calculate metagraph scores.")

        mock_metagraph.sync.assert_not_awaited()
        mock_db_operations.get_predictions_ranked.assert_not_awaited()
        mock_db_operations.get_events.assert_not_awaited()
        mock_db_operations.update_alternative_metagraph_scores.assert_not_awaited()
        mock_db_operations.mark_scores_as_alternative_processed_where_not_processed.assert_not_awaited()

    async def test_run(self):
        mock_metagraph = MagicMock(spec=AsyncMetagraph)
        mock_logger = MagicMock(spec=InfiniteGamesLogger)

        mock_db_operations = MagicMock(spec=DatabaseOperations)
        mock_db_operations.count_events_for_alternative_metagraph_scoring = AsyncMock(
            return_value=9
        )
        mock_db_operations.get_predictions_ranked = AsyncMock()
        mock_db_operations.update_alternative_metagraph_scores = AsyncMock()
        mock_db_operations.mark_scores_as_alternative_processed_where_not_processed = AsyncMock()

        mock_metagraph.uids = [torch.tensor(1), torch.tensor(2), torch.tensor(99)]
        mock_metagraph.hotkeys = {1: "hotkey1", 2: "hotkey2", 99: "owner_hotkey"}
        mock_metagraph.owner_hotkey = "owner_hotkey"
        mock_metagraph.block = torch.tensor(1000)

        mock_metagraph.sync = AsyncMock()

        mock_ranked_predictions = pd.DataFrame(
            {
                "event_id": ["event1", "event2"],
                "event_rank": [1, 2],
                "outcome": ["1", "0"],
                "miner_uid": [1, 2],
                "miner_hotkey": ["hotkey1", "hotkey2"],
                "prediction": [0.7, 0.3],
            }
        )
        mock_db_operations.get_predictions_ranked.return_value = mock_ranked_predictions

        mock_internal_forecasts = pd.DataFrame(
            {"event_id": ["event1", "event2"], "prediction": [0.8, 0.4]}
        )

        class MockClusterSelector(ClusterSelector):
            call_args = None  # For test assertions
            initiated_count = 0

            calculate_selected_clusters_credits_called_count = 0

            def __init__(self, **kwargs):
                MockClusterSelector.initiated_count += 1
                MockClusterSelector.call_args = kwargs

            def calculate_selected_clusters_credits(self):
                MockClusterSelector.calculate_selected_clusters_credits_called_count += 1

                return pd.DataFrame(
                    {
                        "miner_uid": [1, 2],
                        "miner_hotkey": ["hotkey1", "hotkey2"],
                        "round_credit_per_miner": [0.6, 0.3],
                        "cluster_id": ["cluster_id_1", "cluster_id_2"],
                        "coefficients": [
                            "coefficients_1",
                            "coefficients_2",
                        ],
                        "cluster_credit": [
                            "cluster_credit_1",
                            "cluster_credit_2",
                        ],
                        "cluster_credit_adjusted": [
                            "cluster_credit_adjusted_1",
                            "cluster_credit_adjusted_2",
                        ],
                        "miner_count": ["miner_count_1", "miner_count_2"],
                        "scaled_std": ["scaled_std_1", "scaled_std_2"],
                        "scaled_log_loss": [
                            "scaled_log_loss_1",
                            "scaled_log_loss_2",
                        ],
                    }
                )

        expected_alternative_scores = [
            AlternativeMetagraphScore(
                miner_uid=1,
                miner_hotkey="hotkey1",
                alternative_metagraph_score=0.6,
                alternative_other_data='{"cluster_id": "cluster_id_1", "coefficients": "coefficients_1", "cluster_credit": "cluster_credit_1", "cluster_credit_adjusted": "cluster_credit_adjusted_1", "miner_count": "miner_count_1", "scaled_std": "scaled_std_1", "scaled_log_loss": "scaled_log_loss_1"}',
            ),
            AlternativeMetagraphScore(
                miner_uid=2,
                miner_hotkey="hotkey2",
                alternative_metagraph_score=0.3,
                alternative_other_data='{"cluster_id": "cluster_id_2", "coefficients": "coefficients_2", "cluster_credit": "cluster_credit_2", "cluster_credit_adjusted": "cluster_credit_adjusted_2", "miner_count": "miner_count_2", "scaled_std": "scaled_std_2", "scaled_log_loss": "scaled_log_loss_2"}',
            ),
            AlternativeMetagraphScore(
                miner_uid=99,
                miner_hotkey="owner_hotkey",
                alternative_metagraph_score=0.10000000000000009,
                alternative_other_data='{"type": "subnet_owner"}',
            ),
        ]

        # Build task
        task = MetagraphScoringAlternative(
            interval_seconds=60.0,
            cluster_selector_cls=MockClusterSelector,
            db_operations=mock_db_operations,
            metagraph=mock_metagraph,
            logger=mock_logger,
        )

        task.get_internal_forecasts = AsyncMock(return_value=mock_internal_forecasts)

        # Run
        await task.run()

        # Assert
        mock_logger.add_context.assert_called_once_with(
            {"source_task": "metagraph-scoring-alternative"}
        )

        mock_db_operations.count_events_for_alternative_metagraph_scoring.assert_awaited_once()

        mock_metagraph.sync.assert_awaited_once_with(lite=True)

        mock_db_operations.get_predictions_ranked.assert_awaited_once_with(moving_window=150)

        # Assert ClusterSelector was instantiated with correct args
        assert MockClusterSelector.initiated_count == 1
        kwargs = MockClusterSelector.call_args

        assert kwargs["random_seed"] == 9
        assert_frame_equal(kwargs["ranked_predictions"], mock_ranked_predictions, check_dtype=False)
        assert_frame_equal(kwargs["internal_forecasts"], mock_internal_forecasts, check_dtype=False)
        latest_df = kwargs["latest_metagraph_neurons"]
        assert_frame_equal(
            latest_df,
            pd.DataFrame(
                {"miner_uid": [1, 2, 99], "miner_hotkey": ["hotkey1", "hotkey2", "owner_hotkey"]}
            ),
            check_dtype=False,
            check_like=True,
        )

        assert MockClusterSelector.calculate_selected_clusters_credits_called_count == 1

        mock_db_operations.update_alternative_metagraph_scores.assert_awaited_once_with(
            alternative_metagraph_scores=expected_alternative_scores
        )

        mock_db_operations.mark_scores_as_alternative_processed_where_not_processed.assert_awaited_once()
