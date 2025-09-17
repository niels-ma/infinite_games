import copy
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
import torch
from bittensor import AsyncSubtensor
from bittensor.core.metagraph import AsyncMetagraph
from bittensor.utils.weight_utils import process_weights
from bittensor_wallet.wallet import Wallet

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.common.converters import pydantic_models_to_dataframe
from neurons.validator.utils.common.interval import BLOCK_DURATION
from neurons.validator.utils.logger.logger import InfiniteGamesLogger
from neurons.validator.version import __spec_version__ as spec_version


# this is just for avoiding typos in column names
@dataclass
class SWNames:
    miner_uid: str = "miner_uid"
    miner_hotkey: str = "miner_hotkey"
    event_score: str = "event_score"
    alternative_metagraph_score: str = "alternative_metagraph_score"
    event_id: str = "event_id"
    spec_version_name: str = "spec_version"
    created_at: str = "created_at"
    raw_weights: str = "raw_weights"


class SetWeights(AbstractTask):
    interval: float
    db_operations: DatabaseOperations
    logger: InfiniteGamesLogger
    netuid: int
    subtensor: AsyncSubtensor
    wallet: Wallet  # type: ignore

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        logger: InfiniteGamesLogger,
        metagraph: AsyncMetagraph,
        netuid: int,
        subtensor: AsyncSubtensor,
        wallet: Wallet,  # type: ignore
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        self.interval = interval_seconds
        self.db_operations = db_operations
        self.logger = logger

        self.metagraph = metagraph
        self.netuid = netuid
        self.subtensor = subtensor
        self.wallet = wallet

        self.current_hotkeys = None
        self.n_hotkeys = None
        self.current_uids = None
        self.current_miners_df = None

        self.last_set_weights_at = round(time.time())
        self.spec_version = spec_version

    @property
    def name(self):
        return "set-weights"

    @property
    def interval_seconds(self):
        return self.interval

    async def metagraph_lite_sync(self):
        # sync the metagraph in lite mode - # duplicate of PeerScoring.metagraph_lite_sync
        await self.metagraph.sync(lite=True)

        #  WARNING! hotkeys is a list[str] and uids is a torch.tensor
        self.current_hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        self.n_hotkeys = len(self.current_hotkeys)
        self.current_uids = copy.deepcopy(self.metagraph.uids)
        self.current_miners_df = pd.DataFrame(
            {
                "miner_hotkey": self.current_hotkeys,
                "miner_uid": self.current_uids.tolist(),
            }
        )

    async def time_to_set_weights(self):
        weights_rate_limit = await self.subtensor.weights_rate_limit(netuid=self.netuid)

        # do not attempt to set weights more often than the rate limit
        blocks_since_last_attempt = (
            round(time.time()) - self.last_set_weights_at
        ) // BLOCK_DURATION

        last_set_weights_at_dt = datetime.fromtimestamp(
            self.last_set_weights_at, tz=timezone.utc
        ).isoformat(timespec="seconds")

        if blocks_since_last_attempt < weights_rate_limit:
            self.logger.debug(
                "Not setting the weights - not enough blocks passed.",
                extra={
                    "last_set_weights_at": last_set_weights_at_dt,
                    "blocks_since_last_attempt": blocks_since_last_attempt,
                    "weights_rate_limit": weights_rate_limit,
                },
            )
            return False
        else:
            self.logger.debug(
                "Attempting to set the weights - enough blocks passed.",
                extra={
                    "last_set_weights_at": last_set_weights_at_dt,
                    "blocks_since_last_attempt": blocks_since_last_attempt,
                    "weights_rate_limit": weights_rate_limit,
                },
            )
            # reset the last set weights time here to avoid attempts rate limit
            self.last_set_weights_at = round(time.time())

            return True

    def filter_last_scores(self, last_metagraph_scores) -> pd.DataFrame:
        # this is re-normalizing the weights for the current miners

        filtered_scores = pydantic_models_to_dataframe(last_metagraph_scores)
        # merge the current metagraph with the last metagraph scores
        filtered_scores = pd.merge(
            self.current_miners_df,
            filtered_scores,
            on=[SWNames.miner_uid, SWNames.miner_hotkey],
            how="left",
        )

        # some stats for logging
        stats = {
            "len_last_metagraph_scores": len(last_metagraph_scores),
            "len_filtered_scores": len(filtered_scores),
            "len_current_miners": len(self.current_miners_df),
            "len_valid_meta_scores": len(
                filtered_scores.dropna(subset=[SWNames.alternative_metagraph_score])
            ),
            "len_valid_event_scores": len(filtered_scores.dropna(subset=[SWNames.event_score])),
            "distinct_events": len(filtered_scores[SWNames.event_id].unique()),
            "distinct_spec_version": len(filtered_scores[SWNames.spec_version_name].unique()),
            "distinct_created_at": len(filtered_scores[SWNames.created_at].unique()),
        }
        self.logger.debug("Stats for filter last scores", extra=stats)

        filtered_scores = filtered_scores[
            [SWNames.miner_uid, SWNames.miner_hotkey, SWNames.alternative_metagraph_score]
        ]
        data_types = {
            SWNames.miner_uid: "int",
            SWNames.miner_hotkey: "str",
            SWNames.alternative_metagraph_score: "float",
        }
        filtered_scores = filtered_scores.astype(data_types)
        filtered_scores[SWNames.alternative_metagraph_score] = filtered_scores[
            SWNames.alternative_metagraph_score
        ].fillna(0.0)

        return filtered_scores

    def check_scores_sanity(self, filtered_scores: pd.DataFrame) -> bool:
        # Do some sanity checks before and throw assert exceptions if there are issues

        # we should have unique miner_uids
        assert filtered_scores[
            SWNames.miner_uid
        ].is_unique, (
            f"miner_uids are not unique: {filtered_scores[SWNames.miner_uid].value_counts()[:5]}"
        )
        # we should have the same miner_uids as the current metagraph
        assert set(filtered_scores[SWNames.miner_uid]) == set(
            self.current_miners_df[SWNames.miner_uid]
        ), "The miner_uids are not the same as the current metagraph"

        # metagraph scores should not be all 0.0
        assert (
            not filtered_scores[SWNames.alternative_metagraph_score].eq(0.0).all()
        ), "All metagraph_scores are 0.0. This is not expected."

        # metagraph scores should not sum up to 0.0
        # redundant, they are positive and not all 0.0
        assert (
            not filtered_scores[SWNames.alternative_metagraph_score].sum() == 0.0
        ), "The sum of metagraph_scores is 0.0."

        # no NaNs/nulls in filtered_scores in any column
        assert not filtered_scores.isnull().values.any(), "There are NaNs in filtered_scores."

        return True

    def renormalize_weights(self, filtered_scores: pd.DataFrame) -> pd.DataFrame:
        # this is re-normalizing the weights for the current miners
        normalized_scores = filtered_scores.copy()

        # normalize the metagraph scores - guaranteed that sum is strictly positive
        normalized_scores[SWNames.raw_weights] = normalized_scores[
            SWNames.alternative_metagraph_score
        ].div(normalized_scores[SWNames.alternative_metagraph_score].sum())

        # for debug, log top 5 and bottom 5 miners by raw_weights
        top_5 = normalized_scores.nlargest(5, SWNames.raw_weights)
        bottom_5 = normalized_scores.nsmallest(5, SWNames.raw_weights)
        self.logger.debug(
            "Top 5 and bottom 5 miners by raw_weights",
            extra={
                "top_5": top_5.to_dict(),
                "bottom_5": bottom_5.to_dict(),
                "sum_scores": normalized_scores[SWNames.alternative_metagraph_score].sum(),
            },
        )

        return normalized_scores

    async def preprocess_weights(
        self, normalized_scores: pd.DataFrame
    ) -> tuple[torch.Tensor, torch.Tensor]:
        miner_uids_tf = torch.tensor(
            normalized_scores[SWNames.miner_uid].values, dtype=torch.int
        ).to("cpu")
        raw_weights_tf = torch.tensor(
            normalized_scores[SWNames.raw_weights].values, dtype=torch.float
        ).to("cpu")

        min_allowed_weights = await self.subtensor.min_allowed_weights(netuid=self.netuid)
        max_weight_limit = await self.subtensor.max_weight_limit(netuid=self.netuid)

        processed_uids, processed_weights = process_weights(
            uids=miner_uids_tf,
            weights=raw_weights_tf,
            num_neurons=self.metagraph.n,
            min_allowed_weights=min_allowed_weights,
            max_weight_limit=max_weight_limit,
        )

        if (
            processed_uids is None
            or processed_weights is None
            or processed_uids.numel() == 0
            or processed_weights.numel() == 0
        ):
            self.logger.error(
                "Failed to process the weights - received None or empty tensors.",
                extra={
                    "processed_uids[:10]": (
                        processed_uids.tolist()[:10] if processed_uids is not None else None
                    ),
                    "processed_weights[:10]": (
                        processed_weights.tolist()[:10] if processed_weights is not None else None
                    ),
                },
            )
            raise ValueError("Failed to process the weights - received None or empty tensors.")

        # process_weights excludes the zero weights
        mask = raw_weights_tf != 0
        if not torch.equal(processed_uids, miner_uids_tf[mask]):
            self.logger.error(
                "Processed UIDs do not match the original UIDs.",
                extra={
                    "processed_uids[:10]": processed_uids.tolist()[:10],
                    "original_uids[:10]": miner_uids_tf.tolist()[:10],
                },
            )
            raise ValueError("Processed UIDs do not match the original UIDs.")

        if (
            not torch.isclose(processed_weights, raw_weights_tf[mask], atol=1e-5, rtol=1e-5)
            .all()
            .item()
        ):
            self.logger.warning(
                "Processed weights do not match the original weights.",
                extra={
                    "processed_weights[:10]": [
                        round(w, 5) for w in processed_weights.tolist()[:10]
                    ],
                    "original_weights[:10]": [round(w, 5) for w in raw_weights_tf.tolist()[:10]],
                },
            )

        return processed_uids, processed_weights

    async def subtensor_set_weights(
        self, processed_uids: torch.Tensor, processed_weights: torch.Tensor
    ):
        successful, sw_msg = await self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.netuid,
            uids=processed_uids,
            weights=processed_weights,
            version_key=self.spec_version,
            wait_for_inclusion=False,
            wait_for_finalization=False,  # can take more than 5 mins when True
            max_retries=5,
        )
        if not successful:
            extra = {
                "fail_msg": sw_msg,
                "processed_uids[:10]": processed_uids.tolist()[:10],
                "processed_weights[:10]": processed_weights.tolist()[:10],
            }
            log_msg = "Failed to set the weights."
            if "No attempt made" in sw_msg:
                # do not consider this as an error - pollutes the logs
                self.logger.warning(log_msg, extra=extra)
            else:
                self.logger.error(log_msg, extra=extra)
        else:
            self.logger.debug(
                "Weights set successfully.",
                extra={
                    "last_set_weights_at": datetime.fromtimestamp(
                        self.last_set_weights_at, tz=timezone.utc
                    ).isoformat(timespec="seconds")
                },
            )

    async def run(self):
        await self.metagraph_lite_sync()

        can_set_weights = await self.time_to_set_weights()

        if not can_set_weights:
            return

        last_alternative_metagraph_scores = (
            await self.db_operations.get_last_alternative_metagraph_scores()
        )

        if last_alternative_metagraph_scores is None:
            raise ValueError("Failed to get the last metagraph scores.")

        filtered_scores = self.filter_last_scores(last_alternative_metagraph_scores)

        self.check_scores_sanity(filtered_scores)

        normalized_scores = self.renormalize_weights(filtered_scores)

        uids, weights = await self.preprocess_weights(normalized_scores)

        await self.subtensor_set_weights(processed_uids=uids, processed_weights=weights)
