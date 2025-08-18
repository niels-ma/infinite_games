import time
from datetime import datetime, timedelta, timezone
from unittest.mock import ANY, MagicMock

import bittensor as bt
import pandas as pd
import pytest
import torch
import torch.test
from bittensor.core.metagraph import MetagraphMixin
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
        metagraph = MagicMock(spec=MetagraphMixin)
        metagraph.sync = MagicMock()
        subtensor = MagicMock(spec=bt.Subtensor)

        # Mock metagraph attributes
        metagraph.uids = torch.tensor([1, 2, 3], dtype=torch.int32).to("cpu")
        metagraph.hotkeys = ["hotkey1", "hotkey2", "hotkey3"]
        metagraph.n = torch.tensor(3, dtype=torch.int32).to("cpu")

        # Mock subtensor methods
        subtensor.min_allowed_weights.return_value = 1  # Set minimum allowed weights
        subtensor.max_weight_limit.return_value = 10  # Set maximum weight limit
        subtensor.weights_rate_limit.return_value = 100  # Set weights rate limit
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

        assert unit.current_hotkeys == ["hotkey1", "hotkey2", "hotkey3"]
        assert unit.current_uids.tolist() == [1, 2, 3]
        assert unit.n_hotkeys == 3
        assert len(unit.current_miners_df) == 3
        assert unit.current_miners_df[SWNames.miner_uid].tolist() == [1, 2, 3]

        assert unit.weights_rate_limit == 100
        assert unit.spec_version == spec_version

    def test_metagraph_lite_sync(self, set_weights_task: SetWeights):
        unit = set_weights_task

        assert unit.current_miners_df.miner_uid.tolist() == [1, 2, 3]
        assert unit.current_hotkeys == ["hotkey1", "hotkey2", "hotkey3"]

        unit.metagraph.uids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        unit.metagraph.hotkeys = ["hotkey1", "hotkey2", "hotkey3", "hotkey4"]

        unit.metagraph_lite_sync()
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
    def test_time_to_set_weights(self, set_weights_task: SetWeights, delta, expected):
        now = time.time()
        set_weights_task.last_set_weights_at = now - delta
        result = set_weights_task.time_to_set_weights()
        assert result is expected

    def test_filter_last_scores(self, set_weights_task: SetWeights):
        unit = set_weights_task
        assert unit.current_miners_df.miner_uid.tolist() == [1, 2, 3]

        last_metagraph_scores = [
            ScoresModel(
                miner_uid=1,
                miner_hotkey="hotkey1",
                metagraph_score=0.8,
                event_score=0.9,
                prediction=0.9,
                event_id="e1",
                spec_version=1,
                created_at=datetime.now() - timedelta(minutes=3),
            ),
            ScoresModel(
                miner_uid=2,
                miner_hotkey="hotkey2",
                metagraph_score=None,
                event_score=0.7,
                prediction=0.7,
                event_id="e2",
                spec_version=1,
                created_at=datetime.now() - timedelta(minutes=2),
            ),
            ScoresModel(
                miner_uid=4,
                miner_hotkey="hotkey4",
                metagraph_score=0.5,
                event_score=0.4,
                prediction=0.4,
                event_id="e3",
                spec_version=1,
                created_at=datetime.now(),
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
        assert filtered_scores.loc[0].metagraph_score == 0.8
        assert filtered_scores.loc[1].metagraph_score == 0.0
        assert filtered_scores.loc[2].metagraph_score == 0.0

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
                    SWNames.metagraph_score: [0.8, 0.9, 1.0],
                },
                False,
            ),
            # Duplicate miner_uid: should raise assertion.
            (
                {
                    SWNames.miner_uid: [1, 1, 2],
                    SWNames.miner_hotkey: ["hotkey1", "hotkey11111", "hotkey2"],
                    SWNames.metagraph_score: [0.8, 0.9, 1.0],
                },
                True,
            ),
            # All metagraph_score 0: should raise.
            (
                {
                    SWNames.miner_uid: [1, 2, 3],
                    SWNames.miner_hotkey: ["hotkey1", "hotkey2", "hotkey3"],
                    SWNames.metagraph_score: [0.0, 0.0, 0.0],
                },
                True,
            ),
            # Missing miner_hotkey: should raise.
            (
                {
                    SWNames.miner_uid: [1, 2, 3],
                    SWNames.miner_hotkey: ["hotkey1", None, "hotkey3"],
                    SWNames.metagraph_score: [0.8, 0.9, 1.0],
                },
                True,
            ),
        ],
    )
    def test_check_scores_insanity(self, set_weights_task: SetWeights, data, raises):
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
            SWNames.metagraph_score: [0.8, 0.9, 1.3],
        }
        df = pd.DataFrame(data)
        normalized = set_weights_task.renormalize_weights(df)

        assert SWNames.raw_weights in normalized.columns
        total = sum(data[SWNames.metagraph_score])
        expected = [score / total for score in data[SWNames.metagraph_score]]

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

    def test_preprocess_weights_success(self, set_weights_task: SetWeights, monkeypatch):
        data = {
            SWNames.miner_uid: [1, 2, 3],
            SWNames.miner_hotkey: ["hotkey1", "hotkey2", "hotkey3"],
            SWNames.raw_weights: [0.2, 0.3, 0.5],
        }
        normalized = pd.DataFrame(data)

        # Patch bt.utils.weight_utils.process_weights_for_netuid to return the inputs.
        import bittensor.utils.weight_utils as wu

        monkeypatch.setattr(
            wu, "process_weights_for_netuid", lambda uids, weights, **kwargs: (uids, weights)
        )
        uids, weights = set_weights_task.preprocess_weights(normalized)
        # Expect the same non-zero entries.
        torch.testing.assert_close(
            weights, torch.tensor(data[SWNames.raw_weights], dtype=torch.float)
        )
        torch.testing.assert_close(uids, torch.tensor(data[SWNames.miner_uid], dtype=torch.int))

    def test_preprocess_weights_edge_cases(self, set_weights_task, monkeypatch):
        data = {
            SWNames.miner_uid: [1, 2, 3],
            SWNames.miner_hotkey: ["hotkey1", "hotkey2", "hotkey3"],
            SWNames.raw_weights: [0.2, 0.3, 0.0],
        }
        normalized = pd.DataFrame(data)
        import bittensor.utils.weight_utils as wu

        # returns only first two entries
        monkeypatch.setattr(
            wu,
            "process_weights_for_netuid",
            lambda uids, weights, **kwargs: (uids[weights != 0], weights[weights != 0]),
        )
        uids, weights = set_weights_task.preprocess_weights(normalized)
        torch.testing.assert_close(weights, torch.tensor([0.2, 0.3], dtype=torch.float))
        torch.testing.assert_close(uids, torch.tensor([1, 2], dtype=torch.int))

        # return empty tensors
        monkeypatch.setattr(
            wu,
            "process_weights_for_netuid",
            lambda uids, weights, **kwargs: (torch.tensor([]), torch.tensor([])),
        )
        with pytest.raises(ValueError):
            set_weights_task.preprocess_weights(normalized)

        assert set_weights_task.logger.error.call_count == 1
        assert (
            set_weights_task.logger.error.call_args[0][0]
            == "Failed to process the weights - received None or empty tensors."
        )
        set_weights_task.logger.error.reset_mock()

        # return different UIDs
        monkeypatch.setattr(
            wu,
            "process_weights_for_netuid",
            lambda uids, weights, **kwargs: (torch.tensor([1, 2, 4]), weights),
        )
        with pytest.raises(ValueError):
            set_weights_task.preprocess_weights(normalized)

        assert set_weights_task.logger.error.call_count == 1
        assert (
            set_weights_task.logger.error.call_args[0][0]
            == "Processed UIDs do not match the original UIDs."
        )
        set_weights_task.logger.error.reset_mock()

        # return different weights
        monkeypatch.setattr(
            wu,
            "process_weights_for_netuid",
            lambda uids, weights, **kwargs: (uids, torch.tensor([0.2, 0.3, 0.4])),
        )
        normalized.loc[2, SWNames.raw_weights] = 0.41
        uids, weights = set_weights_task.preprocess_weights(normalized)
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
    def test_subtensor_set_weights(
        self, set_weights_task: SetWeights, successful, sw_msg, expected_log
    ):
        processed_uids = torch.tensor([1, 2, 3])
        processed_weights = torch.tensor([0.2, 0.3, 0.5])
        set_weights_task.subtensor.set_weights.return_value = (successful, sw_msg)
        set_weights_task.logger.debug.reset_mock()
        set_weights_task.logger.warning.reset_mock()
        set_weights_task.logger.error.reset_mock()

        set_weights_task.subtensor_set_weights(processed_uids, processed_weights)
        if successful:
            set_weights_task.logger.debug.assert_called_with(
                "Weights set successfully.",
                extra={"last_set_weights_at": ANY},
            )
        else:
            extra = {
                "fail_msg": sw_msg,
                "processed_uids[:10]": [1, 2, 3],
                "processed_weights[:10]": ANY,
            }
            if "No attempt made" in sw_msg:
                set_weights_task.logger.warning.assert_called_with(
                    "Failed to set the weights.", extra=extra
                )
            else:
                set_weights_task.logger.error.assert_called_with(
                    "Failed to set the weights.", extra=extra
                )

    @pytest.mark.asyncio
    async def test_run_successful(self, set_weights_task: SetWeights, monkeypatch, db_client):
        unit = set_weights_task
        unit.metagraph.uids = torch.tensor([2, 3, 4, 5], dtype=torch.int32)
        unit.metagraph.hotkeys = ["hk2", "hk3", "hk4", "hk5"]
        unit.last_set_weights_at = time.time() - 101 * BLOCK_DURATION

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

        # Patch bt.utils.weight_utils.process_weights_for_netuid to return the inputs.
        import bittensor.utils.weight_utils as wu

        monkeypatch.setattr(
            wu,
            "process_weights_for_netuid",
            lambda uids, weights, **kwargs: (uids[weights != 0], weights[weights != 0]),
        )

        unit.subtensor.set_weights.return_value = (True, "Success")

        # run the task
        await unit.run()

        assert unit.logger.debug.call_count == 4
        assert unit.logger.warning.call_count == 0
        assert unit.logger.error.call_count == 0

        debug_calls = unit.logger.debug.call_args_list
        assert debug_calls[0][0][0] == "Attempting to set the weights - enough blocks passed."
        assert debug_calls[0][1]["extra"]["blocks_since_last_attempt"] >= 100

        assert debug_calls[1][0][0] == "Stats for filter last scores"
        assert debug_calls[1][1]["extra"]["len_last_metagraph_scores"] == 3
        assert debug_calls[1][1]["extra"]["len_filtered_scores"] == 4
        assert debug_calls[1][1]["extra"]["len_current_miners"] == 4
        assert debug_calls[1][1]["extra"]["len_valid_meta_scores"] == 3

        assert debug_calls[2][0][0] == "Top 5 and bottom 5 miners by raw_weights"
        assert debug_calls[2][1]["extra"]["sum_scores"] == 1.0
        assert list(debug_calls[2][1]["extra"]["top_5"][SWNames.miner_uid].values()) == [3, 4, 2, 5]

        assert debug_calls[3][0][0] == "Weights set successfully."

        assert unit.subtensor.set_weights.call_count == 1
        assert unit.subtensor.set_weights.call_args.kwargs["uids"].tolist() == [3, 4]
        torch.testing.assert_close(
            unit.subtensor.set_weights.call_args.kwargs["weights"],
            torch.tensor([0.8350, 0.1650], dtype=torch.float),
        )
