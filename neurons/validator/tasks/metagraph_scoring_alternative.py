import asyncio
import inspect
import json

import numpy as np
import pandas as pd
from bittensor.core.metagraph import AsyncMetagraph

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.score import AlternativeMetagraphScore
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.tasks.peer_scoring import PeerScoring
from neurons.validator.utils.cluster_selector.cluster_selector import ClusterSelector
from neurons.validator.utils.common.converters import torch_or_numpy_to_int
from neurons.validator.utils.logger.logger import InfiniteGamesLogger

# how many previous events to consider for the moving average
EVENTS_MOVING_WINDOW = 150


class MetagraphScoringAlternative(AbstractTask):
    interval: float
    cluster_selector_cls: type["ClusterSelector"]
    db_operations: DatabaseOperations
    logger: InfiniteGamesLogger

    def __init__(
        self,
        interval_seconds: float,
        cluster_selector_cls: type["ClusterSelector"],
        db_operations: DatabaseOperations,
        metagraph: AsyncMetagraph,
        logger: InfiniteGamesLogger,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        # Validate cluster_selector_cls
        if not inspect.isclass(cluster_selector_cls) or not issubclass(
            cluster_selector_cls, ClusterSelector
        ):
            raise TypeError(
                "cluster_selector_cls must be a ClusterSelector class (subclass of ClusterSelector)."
            )

        # Validate metagraph
        if not isinstance(metagraph, AsyncMetagraph):
            raise TypeError("metagraph must be an instance of AsyncMetagraph.")

        # Validate logger
        if not isinstance(logger, InfiniteGamesLogger):
            raise TypeError("logger must be an instance of InfiniteGamesLogger.")

        self.interval = interval_seconds
        self.cluster_selector_cls = cluster_selector_cls
        self.db_operations = db_operations
        self.metagraph = metagraph
        self.logger = logger

    @property
    def name(self):
        return "metagraph-scoring-alternative"

    @property
    def interval_seconds(self):
        return self.interval

    async def run(self):
        self.logger.add_context({"source_task": self.name})

        count_events_to_process = (
            await self.db_operations.count_events_for_alternative_metagraph_scoring()
        )

        if count_events_to_process < 1:
            self.logger.debug("No events to calculate metagraph scores.")

            return

        self.logger.debug(
            "Events to calculate metagraph scores found", extra={"count": count_events_to_process}
        )

        (
            metagraph_neurons,
            owner,
        ) = await self.get_metagraph_neurons_and_owner()

        ranked_predictions = await self.get_ranked_predictions()

        event_ids = ranked_predictions["event_id"].unique().tolist()
        internal_forecasts = await self.get_internal_forecasts(event_ids=event_ids)

        cluster_selector = self.cluster_selector_cls(
            ranked_predictions=ranked_predictions,
            latest_metagraph_neurons=metagraph_neurons,
            internal_forecasts=internal_forecasts,
            random_seed=count_events_to_process,
        )

        selected_clusters_credits = await asyncio.to_thread(
            cluster_selector.calculate_selected_clusters_credits
        )

        alternative_metagraph_scores = await self.get_alternative_metagraph_scores(
            selected_clusters_credits=selected_clusters_credits, owner=owner
        )

        await self.db_operations.update_alternative_metagraph_scores(
            alternative_metagraph_scores=alternative_metagraph_scores
        )

        await self.db_operations.mark_scores_as_alternative_processed_where_not_processed()

    async def get_metagraph_neurons_and_owner(self):
        await self.metagraph.sync(lite=True)

        neurons = []

        owner_uid = None
        owner_hotkey = self.metagraph.owner_hotkey

        for uid in self.metagraph.uids:
            int_uid = torch_or_numpy_to_int(uid)
            hotkey = self.metagraph.hotkeys[int_uid]

            neurons.append({"miner_uid": int_uid, "miner_hotkey": hotkey})

            if hotkey == owner_hotkey:
                owner_uid = int_uid

        neurons = pd.DataFrame(neurons, columns=["miner_uid", "miner_hotkey"])

        assert owner_uid is not None, "Owner uid not found in metagraph uids"

        return neurons, {"uid": owner_uid, "hotkey": owner_hotkey}

    async def get_ranked_predictions(self):
        predictions_ranked = await self.db_operations.get_predictions_ranked(
            moving_window=EVENTS_MOVING_WINDOW
        )

        return pd.DataFrame(
            predictions_ranked,
            columns=[
                "event_id",
                "event_rank",
                "outcome",
                "miner_uid",
                "miner_hotkey",
                "prediction",
            ],
        )

    async def get_internal_forecasts(self, event_ids: list[str]):
        unique_event_ids = [f"ifgames-{event_id}" for event_id in event_ids]

        events = await self.db_operations.get_events(unique_event_ids=unique_event_ids)

        events_forecasts = []

        # Calculate predictions avg for each event
        for event in events:
            if not event.forecasts:
                continue

            event_forecasts = json.loads(event.forecasts)

            event_forecasts_values = list(event_forecasts.values())

            n_intervals = len(event_forecasts_values)

            if n_intervals:
                sum_weights = 0
                sum_predictions_weighted = 0

                for idx, prediction in enumerate(event_forecasts_values):
                    weight = PeerScoring.power_decay_weight(idx=idx, n_intervals=n_intervals)

                    sum_predictions_weighted += prediction * weight
                    sum_weights += weight

                events_forecasts.append(
                    {
                        "event_id": event.event_id,
                        "prediction": sum_predictions_weighted / sum_weights,
                    }
                )

        return pd.DataFrame(events_forecasts, columns=["event_id", "prediction"])

    async def get_alternative_metagraph_scores(
        self, selected_clusters_credits: pd.DataFrame, owner: dict
    ):
        # Calculate credits to burn
        sum_round_credits = selected_clusters_credits["round_credit_per_miner"].sum()
        owner_credits_to_burn = 1 - sum_round_credits

        credits = (
            selected_clusters_credits[
                [
                    "miner_uid",
                    "miner_hotkey",
                    "round_credit_per_miner",
                    "cluster_id",
                    "coefficients",
                    "cluster_credit",
                    "cluster_credit_adjusted",
                    "miner_count",
                    "scaled_std",
                    "scaled_log_loss",
                ]
            ]
            .replace({np.nan: None})
            .to_dict(orient="records")
        )

        alternative_metagraph_scores: list[AlternativeMetagraphScore] = []

        owner_uid = owner["uid"]
        owner_hotkey = owner["hotkey"]

        for row in credits:
            miner_uid = row["miner_uid"]
            miner_hotkey = row["miner_hotkey"]
            miner_credits = row["round_credit_per_miner"]

            if miner_uid == owner_uid:
                # Recalculate credits to burn
                sum_round_credits = sum_round_credits - (miner_credits or 0)
                owner_credits_to_burn = 1 - sum_round_credits

                logger_method = (
                    self.logger.error if miner_credits not in (0, None) else self.logger.debug
                )

                logger_method(
                    "Owner should have no credits from clusters",
                    extra={
                        "owner_id": owner_uid,
                        "owner_hotkey": owner_hotkey,
                        "miner_hotkey": miner_hotkey,
                        "round_credit_per_miner": miner_credits,
                    },
                )

                continue

            alternative_metagraph_scores.append(
                AlternativeMetagraphScore(
                    miner_uid=miner_uid,
                    miner_hotkey=miner_hotkey,
                    alternative_metagraph_score=miner_credits,
                    alternative_other_data=json.dumps(
                        {
                            "cluster_id": row["cluster_id"],
                            "coefficients": row["coefficients"],
                            "cluster_credit": row["cluster_credit"],
                            "cluster_credit_adjusted": row["cluster_credit_adjusted"],
                            "miner_count": row["miner_count"],
                            "scaled_std": row["scaled_std"],
                            "scaled_log_loss": row["scaled_log_loss"],
                        }
                    ),
                )
            )

        alternative_metagraph_scores.append(
            AlternativeMetagraphScore(
                miner_uid=owner_uid,
                miner_hotkey=owner_hotkey,
                alternative_metagraph_score=owner_credits_to_burn,
                alternative_other_data=json.dumps({"type": "subnet_owner"}),
            )
        )

        return alternative_metagraph_scores
