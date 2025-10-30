import time
from datetime import datetime, timedelta, timezone
from unittest.mock import ANY, AsyncMock, MagicMock

import bittensor as bt
import numpy as np
import pandas as pd
import pytest
import torch
from bittensor.core.metagraph import AsyncMetagraph
from bittensor_wallet import Wallet
from freezegun import freeze_time

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.score import SCORE_FIELDS, ScoresModel
from neurons.validator.tasks.set_weights import SetWeights, SWNames
from neurons.validator.utils.common.interval import BLOCK_DURATION
from neurons.validator.utils.logger.logger import InfiniteGamesLogger
from neurons.validator.version import __spec_version__ as spec_version


class TestSetWeights:
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
    def set_weights_task(
        self,
        db_operations: DatabaseOperations,
        bt_wallet: Wallet,  # type: ignore
    ):
        metagraph = MagicMock(spec=AsyncMetagraph)
        metagraph.sync = AsyncMock()
        subtensor = MagicMock(spec=bt.AsyncSubtensor)

        # Mock metagraph attributes
        metagraph.uids = torch.tensor([1, 2, 3], dtype=torch.int32).to("cpu")
        metagraph.hotkeys = ["hotkey1", "hotkey2", "hotkey3"]
        metagraph.n = torch.tensor(3, dtype=torch.int32).to("cpu")

        # Mock subtensor methods
        subtensor.min_allowed_weights.return_value = 1  # Set minimum allowed weights
        subtensor.max_weight_limit.return_value = 10  # Set maximum weight limit
        subtensor.weights_rate_limit = AsyncMock(return_value=100)  # Set weights rate limit
        subtensor.network = "mock"

        logger = MagicMock(spec=InfiniteGamesLogger)

        with freeze_time("2024-12-27 07:00:00"):
            return SetWeights(
                interval_seconds=60.0,
                db_operations=db_operations,
                logger=logger,
                metagraph=metagraph,
                netuid=155,
                subtensor=subtensor,
                wallet=bt_wallet,
            )

    def test_init(self, set_weights_task: SetWeights):
        unit = set_weights_task

        assert isinstance(unit, SetWeights)

        assert unit.interval_seconds == 60.0
        assert unit.db_operations is not None
        assert unit.logger is not None
        assert unit.metagraph is not None
        assert unit.netuid == 155
        assert unit.subtensor is not None
        assert unit.wallet is not None

        assert unit.spec_version == spec_version

    async def test_metagraph_lite_sync(self, set_weights_task: SetWeights):
        unit = set_weights_task

        unit.metagraph.uids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        unit.metagraph.hotkeys = ["hotkey1", "hotkey2", "hotkey3", "hotkey4"]

        await unit.metagraph_lite_sync()

        assert unit.current_miners_df.miner_uid.tolist() == [1, 2, 3, 4]
        assert unit.current_miners_df.miner_hotkey.tolist() == [
            "hotkey1",
            "hotkey2",
            "hotkey3",
            "hotkey4",
        ]
        assert unit.current_miners_df.index.size == 4
        assert unit.current_hotkeys == ["hotkey1", "hotkey2", "hotkey3", "hotkey4"]
        assert unit.current_uids.tolist() == [1, 2, 3, 4]
        assert unit.n_hotkeys == 4

    @pytest.mark.parametrize(
        "delta,expected",
        [
            (99 * BLOCK_DURATION, False),
            (101 * BLOCK_DURATION, True),
        ],
    )
    async def test_time_to_set_weights(self, set_weights_task: SetWeights, delta, expected):
        now = time.time()
        set_weights_task.last_set_weights_at = now - delta

        result = await set_weights_task.time_to_set_weights()

        assert result is expected

    async def test_filter_last_scores(self, set_weights_task: SetWeights):
        unit = set_weights_task

        # Need to call sync to load instance data
        await unit.metagraph_lite_sync()

        last_metagraph_scores = [
            ScoresModel(
                miner_uid=1,
                miner_hotkey="hotkey1",
                alternative_metagraph_score=0.8,
                event_score=0.9,
                prediction=0.9,
                event_id="e1",
                spec_version=1,
                created_at=datetime.now(timezone.utc) - timedelta(minutes=3),
            ),
            ScoresModel(
                miner_uid=2,
                miner_hotkey="hotkey2",
                alternative_metagraph_score=None,
                event_score=0.7,
                prediction=0.7,
                event_id="e2",
                spec_version=1,
                created_at=datetime.now(timezone.utc) - timedelta(minutes=2),
            ),
            ScoresModel(
                miner_uid=4,
                miner_hotkey="hotkey4",
                alternative_metagraph_score=0.5,
                event_score=0.4,
                prediction=0.4,
                event_id="e3",
                spec_version=1,
                created_at=datetime.now(timezone.utc),
            ),
        ]

        filtered_scores = unit.filter_last_scores(last_metagraph_scores)

        assert len(filtered_scores) == 3
        assert filtered_scores.loc[0].miner_uid == 1
        assert filtered_scores.loc[1].miner_uid == 2
        assert filtered_scores.loc[2].miner_uid == 3
        assert filtered_scores.loc[0].miner_hotkey == "hotkey1"
        assert filtered_scores.loc[1].miner_hotkey == "hotkey2"
        assert filtered_scores.loc[2].miner_hotkey == "hotkey3"
        assert filtered_scores.loc[0].alternative_metagraph_score == 0.8
        assert filtered_scores.loc[1].alternative_metagraph_score == 0.0
        assert filtered_scores.loc[2].alternative_metagraph_score == 0.0

        expected_stats = {
            "len_last_metagraph_scores": 3,
            "len_filtered_scores": 3,
            "len_current_miners": 3,
            "len_valid_meta_scores": 1,
            "len_valid_event_scores": 2,
            "distinct_events": 3,  # null counts as distinct
            "distinct_spec_version": 2,
            "distinct_created_at": 3,
        }

        assert unit.logger.debug.call_count == 1
        assert unit.logger.debug.call_args[0][0] == "Stats for filter last scores"
        assert unit.logger.debug.call_args[1]["extra"] == expected_stats

    @pytest.mark.parametrize(
        "data,raises",
        [
            # Valid case: unique miner_uid and non-zero metagraph scores.
            (
                {
                    SWNames.miner_uid: [1, 2, 3],
                    SWNames.miner_hotkey: ["hotkey1", "hotkey2", "hotkey3"],
                    SWNames.alternative_metagraph_score: [0.8, 0.9, 1.0],
                },
                False,
            ),
            # Duplicate miner_uid: should raise assertion.
            (
                {
                    SWNames.miner_uid: [1, 1, 2],
                    SWNames.miner_hotkey: ["hotkey1", "hotkey11111", "hotkey2"],
                    SWNames.alternative_metagraph_score: [0.8, 0.9, 1.0],
                },
                True,
            ),
            # All metagraph_score 0: should raise.
            (
                {
                    SWNames.miner_uid: [1, 2, 3],
                    SWNames.miner_hotkey: ["hotkey1", "hotkey2", "hotkey3"],
                    SWNames.alternative_metagraph_score: [0.0, 0.0, 0.0],
                },
                True,
            ),
            # Missing miner_hotkey: should raise.
            (
                {
                    SWNames.miner_uid: [1, 2, 3],
                    SWNames.miner_hotkey: ["hotkey1", None, "hotkey3"],
                    SWNames.alternative_metagraph_score: [0.8, 0.9, 1.0],
                },
                True,
            ),
        ],
    )
    async def test_check_scores_insanity(self, set_weights_task: SetWeights, data, raises):
        # Need to sync to load task instance data
        await set_weights_task.metagraph_lite_sync()

        df = pd.DataFrame(data)

        if raises:
            with pytest.raises(AssertionError):
                set_weights_task.check_scores_sanity(df)
        else:
            assert set_weights_task.check_scores_sanity(df) is True

    def test_renormalize_weights(self, set_weights_task: SetWeights):
        data = {
            SWNames.miner_uid: [1, 2, 3],
            SWNames.miner_hotkey: ["hotkey1", "hotkey2", "hotkey3"],
            SWNames.alternative_metagraph_score: [0.8, 0.9, 1.3],
        }
        df = pd.DataFrame(data)
        normalized = set_weights_task.renormalize_weights(df)

        assert SWNames.raw_weights in normalized.columns
        total = sum(data[SWNames.alternative_metagraph_score])
        expected = [score / total for score in data[SWNames.alternative_metagraph_score]]

        pd.testing.assert_series_equal(
            normalized[SWNames.raw_weights],
            pd.Series(expected, name=SWNames.raw_weights),
        )

        set_weights_task.logger.debug.assert_called()

        assert (
            set_weights_task.logger.debug.call_args[0][0]
            == "Top 5 and bottom 5 miners by raw_weights"
        )
        assert set_weights_task.logger.debug.call_args[1]["extra"]["sum_scores"] == total

    async def test_preprocess_weights_success(self, set_weights_task: SetWeights, monkeypatch):
        data = {
            SWNames.miner_uid: [1, 2, 3],
            SWNames.miner_hotkey: ["hotkey1", "hotkey2", "hotkey3"],
            SWNames.raw_weights: [0.2, 0.3, 0.5],
        }
        normalized = pd.DataFrame(data)

        uids, weights = await set_weights_task.preprocess_weights(normalized)

        # Expect the same non-zero entries.
        torch.testing.assert_close(
            weights, torch.tensor(data[SWNames.raw_weights], dtype=torch.float)
        )
        torch.testing.assert_close(uids, torch.tensor(data[SWNames.miner_uid], dtype=torch.int))

    async def test_preprocess_weights_edge_cases(self, set_weights_task, monkeypatch):
        data = {
            SWNames.miner_uid: [1, 2, 3],
            SWNames.miner_hotkey: ["hotkey1", "hotkey2", "hotkey3"],
            SWNames.raw_weights: [0.2, 0.3, 0.0],
        }
        normalized = pd.DataFrame(data)

        uids, weights = await set_weights_task.preprocess_weights(normalized)
        torch.testing.assert_close(weights, torch.tensor([0.4, 0.6], dtype=torch.float))
        torch.testing.assert_close(uids, torch.tensor([1, 2], dtype=torch.int))

        # Weights get normalized
        assert set_weights_task.logger.warning.call_count == 1
        assert (
            set_weights_task.logger.warning.call_args[0][0]
            == "Processed weights do not match the original weights."
        )

        set_weights_task.logger.warning.reset_mock()

        # Force return empty tensors
        monkeypatch.setattr(
            "neurons.validator.tasks.set_weights.process_weights",
            lambda uids, weights, **kwargs: (torch.tensor([]), torch.tensor([])),
        )

        with pytest.raises(
            expected_exception=ValueError,
            match="Failed to process the weights - received None or empty tensors.",
        ):
            await set_weights_task.preprocess_weights(normalized)

        assert set_weights_task.logger.error.call_count == 1
        assert (
            set_weights_task.logger.error.call_args[0][0]
            == "Failed to process the weights - received None or empty tensors."
        )

        set_weights_task.logger.error.reset_mock()

        # Force return different UIDs
        monkeypatch.setattr(
            "neurons.validator.tasks.set_weights.process_weights",
            lambda uids, weights, **kwargs: (torch.tensor([1, 2, 4]), weights),
        )
        with pytest.raises(
            expected_exception=ValueError, match="Processed UIDs do not match the original UIDs."
        ):
            await set_weights_task.preprocess_weights(normalized)

        assert set_weights_task.logger.error.call_count == 1
        assert (
            set_weights_task.logger.error.call_args[0][0]
            == "Processed UIDs do not match the original UIDs."
        )

        set_weights_task.logger.error.reset_mock()

        # Force return different weights
        monkeypatch.setattr(
            "neurons.validator.tasks.set_weights.process_weights",
            lambda uids, weights, **kwargs: (uids, torch.tensor([0.2, 0.3, 0.4])),
        )

        normalized.loc[2, SWNames.raw_weights] = 0.41
        uids, weights = await set_weights_task.preprocess_weights(normalized)

        assert torch.equal(uids, torch.tensor([1, 2, 3], dtype=torch.int))
        torch.testing.assert_close(weights, torch.tensor([0.2, 0.3, 0.4], dtype=torch.float))

        assert set_weights_task.logger.warning.call_count == 1
        assert set_weights_task.logger.error.call_count == 0
        assert (
            set_weights_task.logger.warning.call_args[0][0]
            == "Processed weights do not match the original weights."
        )

    @pytest.mark.parametrize(
        "successful,sw_msg,expected_log",
        [
            (True, "Success", "Weights set successfully."),
            (False, "No attempt made: network busy", "Failed to set the weights."),  # warning case
            (
                False,
                "Other failure",
                "Failed to set the weights.",
            ),  # error case
        ],
    )
    async def test_subtensor_set_weights(
        self, set_weights_task: SetWeights, successful, sw_msg, expected_log
    ):
        processed_uids = torch.tensor([1, 2, 3])
        processed_weights = torch.tensor([0.2, 0.3, 0.5])
        set_weights_task.subtensor.set_weights = AsyncMock(return_value=(successful, sw_msg))
        set_weights_task.logger.debug.reset_mock()
        set_weights_task.logger.warning.reset_mock()
        set_weights_task.logger.error.reset_mock()

        await set_weights_task.subtensor_set_weights(processed_uids, processed_weights)

        expected_arguments = {
            "uids": processed_uids,
            "weights": processed_weights,
            "netuid": set_weights_task.netuid,
            "wallet": set_weights_task.wallet,
            "version_key": set_weights_task.spec_version,
            "wait_for_inclusion": True,
            "wait_for_finalization": True,
            "max_retries": 5,
        }

        set_weights_task.subtensor.set_weights.assert_awaited_once_with(**expected_arguments)

        if successful:
            set_weights_task.logger.debug.assert_called_with(
                expected_log,
                extra={"last_set_weights_at": ANY},
            )
        else:
            extra = {
                "fail_msg": sw_msg,
                "processed_uids[:10]": [1, 2, 3],
                "processed_weights[:10]": ANY,
            }
            if "No attempt made" in sw_msg:
                set_weights_task.logger.warning.assert_called_with(expected_log, extra=extra)
            else:
                set_weights_task.logger.error.assert_called_with(expected_log, extra=extra)

    @pytest.mark.asyncio
    async def test_run_successful_x(self, set_weights_task: SetWeights, monkeypatch, db_client):
        unit = set_weights_task
        unit.metagraph.owner_hotkey = "hk3"
        unit.metagraph.uids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
        unit.metagraph.hotkeys = ["hk0", "hk1", "hk2", "hk3", "hk4"]
        unit.last_set_weights_at = time.time() - 101 * BLOCK_DURATION

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

        unit.subtensor.set_weights = AsyncMock(return_value=(True, "Success"))

        # run the task
        await unit.run()

        assert unit.logger.debug.call_count == 2
        assert unit.logger.warning.call_count == 0
        assert unit.logger.error.call_count == 0

        debug_calls = unit.logger.debug.call_args_list
        assert debug_calls[0][0][0] == "Attempting to set the weights - enough blocks passed."
        assert debug_calls[0][1]["extra"]["blocks_since_last_attempt"] >= 100

        assert debug_calls[1][0][0] == "Weights set successfully."

        assert unit.subtensor.set_weights.call_count == 1
        assert unit.subtensor.set_weights.call_args.kwargs["uids"].tolist() == [3]
        torch.testing.assert_close(
            unit.subtensor.set_weights.call_args.kwargs["weights"],
            torch.tensor([1.0], dtype=torch.float),
        )

    def test_get_owner_neuron(self, set_weights_task: SetWeights):
        set_weights_task.metagraph.owner_hotkey = "hotkey10"
        set_weights_task.metagraph.uids = [
            torch.tensor(1),
            np.array(5),
            np.int64(10),
            torch.tensor(15),
        ]
        set_weights_task.metagraph.hotkeys = {
            1: "hotkey1",
            5: "hotkey5",
            10: "hotkey10",
            15: "hotkey15",
        }

        owner = set_weights_task.get_owner_neuron()

        # Verify owner_uid
        assert isinstance(owner, dict)
        assert owner["uid"] == 10
        assert owner["hotkey"] == "hotkey10"

    def test_get_owner_neuron_empty_metagraph(self, set_weights_task: SetWeights):
        set_weights_task.metagraph.owner_hotkey = None
        set_weights_task.metagraph.uids = []
        set_weights_task.metagraph.hotkeys = {}

        with pytest.raises(AssertionError, match="Owner uid not found in metagraph uids"):
            set_weights_task.get_owner_neuron()
