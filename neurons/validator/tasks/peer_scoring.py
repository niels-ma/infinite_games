import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial

import numpy as np
import pandas as pd
from bittensor.core.metagraph import AsyncMetagraph

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.event import EventsModel
from neurons.validator.models.prediction import PredictionsModel
from neurons.validator.models.score import ScoresModel
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.common.converters import pydantic_models_to_dataframe
from neurons.validator.utils.common.interval import (
    AGGREGATION_INTERVAL_LENGTH_MINUTES,
    align_to_interval,
    minutes_since_epoch,
    to_utc,
)
from neurons.validator.utils.logger.logger import InfiniteGamesLogger
from neurons.validator.version import __spec_version__ as spec_version

# controls the clipping of predictions [CLIP_EPS, 1 - CLIP_EPS]
CLIP_EPS = 1e-2
# controls the distance mean-min answer penalty for miners which are unresponsive
UPTIME_PENALTY_DISTANCE = 1 / 3

DEFAULT_POWER_DECAY_WEIGHT_EXPONENT = 2


# this is just for avoiding typos in column names
@dataclass
class PSNames:
    miner_uid: str = "miner_uid"
    miner_hotkey: str = "miner_hotkey"
    registered_date: str = "registered_date"
    miner_registered_minutes: str = "miner_registered_minutes"
    interval_idx: str = "interval_idx"
    interval_start: str = "interval_start"
    interval_end: str = "interval_end"
    weight: str = "weight"
    interval_agg_prediction: str = "interval_agg_prediction"
    log_score: str = "log_score"
    group_sum: str = "group_sum"
    group_count: str = "group_count"
    mean_log_score_others: str = "mean_log_score_others"
    peer_score: str = "peer_score"
    weighted_peer_score: str = "weighted_peer_score"
    weighted_prediction: str = "weighted_prediction"
    weighted_prediction_sum: str = "weighted_prediction_sum"
    weighted_peer_score_sum: str = "weighted_peer_score_sum"
    weight_sum: str = "weight_sum"
    rema_prediction: str = "rema_prediction"
    rema_peer_score: str = "rema_peer_score"


class PeerScoring(AbstractTask):
    interval: float
    page_size: int
    db_operations: DatabaseOperations
    logger: InfiniteGamesLogger

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        metagraph: AsyncMetagraph,
        logger: InfiniteGamesLogger,
        page_size: int = 100,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        # get current hotkeys and uids
        # regularly update these during and after each event scoring
        self.metagraph = metagraph
        self.current_hotkeys = None
        self.n_hotkeys = None
        self.current_uids = None
        self.current_miners_df = None

        self.spec_version = spec_version

        self.interval = interval_seconds
        self.db_operations = db_operations
        self.miners_last_reg = None

        self.errors_count = 0
        self.logger = logger
        self.page_size = page_size

    @property
    def name(self):
        return "peer-scoring"

    @property
    def interval_seconds(self):
        return self.interval

    async def metagraph_lite_sync(self):
        # sync the metagraph in lite mode
        await self.metagraph.sync(lite=True)
        #  WARNING! hotkeys is a list[str] and uids is a torch.tensor
        self.current_hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        self.n_hotkeys = len(self.current_hotkeys)
        self.current_uids = copy.deepcopy(self.metagraph.uids)
        self.current_miners_df = pd.DataFrame(
            {
                PSNames.miner_hotkey: self.current_hotkeys,
                PSNames.miner_uid: self.current_uids.tolist(),
            }
        )

    async def miners_last_reg_sync(self) -> bool:
        miners_last_reg_rows = await self.db_operations.get_miners_last_registration()
        if not miners_last_reg_rows:
            self.errors_count += 1
            self.logger.error("No miners found in the DB, skipping scoring!")
            return False

        miners_last_reg = pydantic_models_to_dataframe(miners_last_reg_rows)
        # for some reason, someone decided to store the miner_uid as a string in the DB
        miners_last_reg[PSNames.miner_uid] = miners_last_reg[PSNames.miner_uid].astype(
            pd.Int64Dtype()
        )

        # inner join to current miners
        self.miners_last_reg = pd.merge(
            miners_last_reg,
            self.current_miners_df,
            on=[PSNames.miner_uid, PSNames.miner_hotkey],
            how="inner",
        )

        if self.miners_last_reg.empty:
            self.logger.error(
                "No overlap in miners between DB and metagraph, skipping scoring!",
                extra={
                    "db_hotkeys[:10]": self.current_miners_df[PSNames.miner_hotkey][:10].tolist(),
                    "metagraph_hotkeys[:10]": miners_last_reg[PSNames.miner_hotkey][:10].tolist(),
                },
            )
            return False

        # calculate the reg_data as minutes since reference data
        self.miners_last_reg[PSNames.miner_registered_minutes] = (
            self.miners_last_reg[PSNames.registered_date].apply(to_utc).apply(minutes_since_epoch)
        )

        return True

    @staticmethod
    def set_right_cutoff(input_event: EventsModel):
        event = copy.deepcopy(input_event)
        effective_cutoff = min(
            to_utc(input_event.cutoff), to_utc(input_event.resolved_at), datetime.now(timezone.utc)
        )

        event.cutoff = effective_cutoff
        event.resolved_at = effective_cutoff
        event.registered_date = to_utc(event.registered_date)
        return event

    @staticmethod
    def power_decay_weight(
        idx: int, n_intervals: int, exponent: int = DEFAULT_POWER_DECAY_WEIGHT_EXPONENT
    ):
        """
        General power-law decay:
        w (i) = 1 - (i / (n_intervals - 1)) ** exponent
        """
        if n_intervals <= 1:
            return 1.0

        x = idx / (n_intervals - 1)

        return 1 - x**exponent

    def get_intervals_df(self, event_registered_start_minutes, event_cutoff_start_minutes):
        n_intervals = (
            event_cutoff_start_minutes - event_registered_start_minutes
        ) // AGGREGATION_INTERVAL_LENGTH_MINUTES
        if n_intervals <= 0:
            self.logger.error(
                "n_intervals computed to be <= 0",
                extra={
                    "n_intervals": n_intervals,
                    "effective_cutoff_start_minutes": event_cutoff_start_minutes,
                    "registered_date_start_minutes": event_registered_start_minutes,
                },
            )
            return pd.DataFrame(
                columns=[
                    PSNames.interval_idx,
                    PSNames.interval_start,
                    PSNames.interval_end,
                    PSNames.weight,
                ]
            )

        intervals = pd.DataFrame({PSNames.interval_idx: np.arange(n_intervals)})
        intervals[PSNames.interval_start] = (
            event_registered_start_minutes
            + intervals[PSNames.interval_idx] * AGGREGATION_INTERVAL_LENGTH_MINUTES
        )
        intervals[PSNames.interval_end] = (
            intervals[PSNames.interval_start] + AGGREGATION_INTERVAL_LENGTH_MINUTES
        )
        # Reverse exponential MA weights:
        intervals[PSNames.weight] = intervals[PSNames.interval_idx].apply(
            lambda idx: self.power_decay_weight(idx, n_intervals)
        )

        return intervals

    def prepare_predictions_df(
        self, predictions: list[PredictionsModel], miners: pd.DataFrame
    ) -> pd.DataFrame:
        # consider predictions only for valid miners
        predictions_df = pydantic_models_to_dataframe(predictions)
        predictions_df.rename(
            columns={
                "interval_start_minutes": PSNames.interval_start,
            },
            inplace=True,
        )
        predictions_df[PSNames.miner_uid] = predictions_df[PSNames.miner_uid].astype(
            pd.Int64Dtype()
        )
        predictions_df = pd.merge(
            miners[[PSNames.miner_uid, PSNames.miner_hotkey]],
            predictions_df,
            on=[PSNames.miner_uid, PSNames.miner_hotkey],
            how="left",
        )
        predictions_df[PSNames.interval_agg_prediction] = predictions_df[
            PSNames.interval_agg_prediction
        ].clip(CLIP_EPS, 1 - CLIP_EPS)
        return predictions_df

    def get_interval_scores_base(
        self, predictions_df: pd.DataFrame, miners: pd.DataFrame, intervals: pd.DataFrame
    ) -> pd.DataFrame:
        # all miners should have a row for each interval, then left join with predictions
        miners["key"] = 1
        intervals["key"] = 1
        miners_intervals = pd.merge(
            miners[
                [
                    "key",
                    PSNames.miner_uid,
                    PSNames.miner_hotkey,
                    PSNames.miner_registered_minutes,
                ]
            ],
            intervals,
            on="key",
        )

        interval_scores_df = pd.merge(
            miners_intervals,
            predictions_df,
            on=[PSNames.miner_uid, PSNames.miner_hotkey, PSNames.interval_start],
            how="left",
        )
        # keep only columns needed for scoring
        interval_scores_df = interval_scores_df[
            [
                PSNames.miner_uid,
                PSNames.miner_hotkey,
                PSNames.miner_registered_minutes,
                PSNames.interval_idx,
                PSNames.interval_start,
                PSNames.interval_end,
                PSNames.weight,
                PSNames.interval_agg_prediction,
            ]
        ]
        return interval_scores_df

    def return_empty_scores_df(self, reason: str, event_id: str) -> pd.DataFrame:
        self.errors_count += 1
        self.logger.error(
            reason,
            extra={
                "event_id": event_id,
            },
        )
        return pd.DataFrame(
            columns=[
                PSNames.miner_uid,
                PSNames.miner_hotkey,
                PSNames.rema_prediction,
                PSNames.rema_peer_score,
            ]
        )

    @staticmethod
    def log_score(
        prediction: float,
        outcome: int,
    ) -> float:
        if outcome == 1:
            return np.log(prediction)
        else:
            return np.log(1 - prediction)

    @staticmethod
    def inverse_log_score(
        log_score: float,
        outcome: int,
    ) -> float:
        if outcome == 1:
            return np.exp(log_score)
        else:
            return 1 - np.exp(log_score)

    def fill_unresponsive_miners(
        self, interval_scores: pd.DataFrame, outcome_round: int
    ) -> pd.DataFrame:
        interval_scores_df = interval_scores.copy()
        interval_scores_df[PSNames.interval_agg_prediction] = interval_scores_df[
            PSNames.interval_agg_prediction
        ].astype("Float64")

        wrong_outcome = 1 - abs(outcome_round - CLIP_EPS)
        # for miners with registered_date_minutes < interval_start but no answer:
        unresponsive_miners = (
            interval_scores_df[PSNames.miner_registered_minutes]
            < interval_scores_df[PSNames.interval_start]
        ) & (interval_scores_df[PSNames.interval_agg_prediction].isnull())

        grouped = interval_scores_df.groupby(PSNames.interval_idx)
        mean_prediction = grouped[PSNames.interval_agg_prediction].transform("mean")
        # Determine the worst prediction per interval:
        # - If outcome_round is 1, a lower prediction is worse so we use the minimum.
        # - If outcome_round is 0, a higher prediction is worse so we use the maximum.
        if outcome_round == 1:
            worst_prediction = grouped[PSNames.interval_agg_prediction].transform("min")
        else:
            worst_prediction = grouped[PSNames.interval_agg_prediction].transform("max")

        # imputed prediction is the mean plus 1/3 of the difference between worst and mean.
        imputed_prediction = (
            mean_prediction + (worst_prediction - mean_prediction) * UPTIME_PENALTY_DISTANCE
        )

        # In case there are no responsive miners in an interval, fallback to the totally wrong answer.
        imputed_prediction = imputed_prediction.fillna(wrong_outcome)

        interval_scores_df.loc[
            unresponsive_miners, PSNames.interval_agg_prediction
        ] = imputed_prediction[unresponsive_miners]

        return interval_scores_df

    def peer_score_intervals(
        self, interval_scores: pd.DataFrame, outcome_round: int
    ) -> pd.DataFrame:
        worst_log_score = np.log(CLIP_EPS)  # worst possible log score

        # fill unresponsive miners
        interval_scores_df = self.fill_unresponsive_miners(interval_scores, outcome_round)

        # calculate the log score for each interval
        partial_log_score = partial(PeerScoring.log_score, outcome=outcome_round)
        interval_scores_df[PSNames.log_score] = interval_scores_df[
            PSNames.interval_agg_prediction
        ].apply(partial_log_score)

        # group by interval and calculate mean_log_score of other miners except the current one
        interval_scores_df[PSNames.group_sum] = interval_scores_df.groupby(PSNames.interval_idx)[
            PSNames.log_score
        ].transform("sum")
        # use count not size to avoid NaNs!!!
        interval_scores_df[PSNames.group_count] = interval_scores_df.groupby(PSNames.interval_idx)[
            PSNames.log_score
        ].transform("count")

        interval_scores_df[PSNames.mean_log_score_others] = np.where(
            interval_scores_df[PSNames.log_score].isna(),
            # For null log_score, use the full group's mean (if count > 0)
            np.where(
                interval_scores_df[PSNames.group_count] > 0,
                interval_scores_df[PSNames.group_sum] / interval_scores_df[PSNames.group_count],
                worst_log_score,  # should not happen - in case all miners are new for an interval
            ),
            # For non-null log_score, subtract current value and reduce the count by one,
            # but only if there is at least one "other" value
            np.where(
                interval_scores_df[PSNames.group_count] > 1,
                (interval_scores_df[PSNames.group_sum] - interval_scores_df[PSNames.log_score])
                / (interval_scores_df[PSNames.group_count] - 1),
                worst_log_score,  # in case all miners but one are new for an interval
            ),
        )

        # fill null log_scores with mean_log_score_others
        interval_scores_df[PSNames.log_score] = interval_scores_df[PSNames.log_score].astype(
            "Float64"
        )
        interval_scores_df[PSNames.log_score] = interval_scores_df[PSNames.log_score].fillna(
            interval_scores_df[PSNames.mean_log_score_others]
        )
        # fill null interval_agg_predictions with reverse of log_scores
        partial_inverse_log_score = partial(PeerScoring.inverse_log_score, outcome=outcome_round)
        interval_scores_df[PSNames.interval_agg_prediction] = interval_scores_df[
            PSNames.interval_agg_prediction
        ].fillna(interval_scores_df[PSNames.log_score].apply(partial_inverse_log_score))

        # calculate the peer score
        interval_scores_df[PSNames.peer_score] = (
            interval_scores_df[PSNames.log_score]
            - interval_scores_df[PSNames.mean_log_score_others]
        )
        interval_scores_df[PSNames.weighted_peer_score] = (
            interval_scores_df[PSNames.peer_score] * interval_scores_df[PSNames.weight]
        )
        interval_scores_df[PSNames.weighted_prediction] = (
            interval_scores_df[PSNames.interval_agg_prediction] * interval_scores_df[PSNames.weight]
        )

        return interval_scores_df

    def reduce_scored_intervals_df(self, scored_intervals_df: pd.DataFrame) -> pd.DataFrame:
        # group by miner and calculate the reverse exponential MA of peer scores
        scores_df = (
            scored_intervals_df.groupby([PSNames.miner_uid, PSNames.miner_hotkey])
            .agg(
                weighted_prediction_sum=(PSNames.weighted_prediction, "sum"),
                weighted_peer_score_sum=(PSNames.weighted_peer_score, "sum"),
                weight_sum=(PSNames.weight, "sum"),
            )
            .reset_index()
        )
        scores_df[PSNames.rema_prediction] = (
            scores_df[PSNames.weighted_prediction_sum] / scores_df[PSNames.weight_sum]
        )
        scores_df[PSNames.rema_peer_score] = (
            scores_df[PSNames.weighted_peer_score_sum] / scores_df[PSNames.weight_sum]
        )

        return scores_df[
            [
                PSNames.miner_uid,
                PSNames.miner_hotkey,
                PSNames.rema_prediction,
                PSNames.rema_peer_score,
            ]
        ]

    async def peer_score_event(
        self, event: EventsModel, predictions: list[PredictionsModel]
    ) -> pd.DataFrame:
        # outcome is text in DB :|
        outcome = float(event.outcome)
        outcome_round = int(round(outcome))  # for safety, should be 0 or 1

        # Convert cutoff and now to minutes since epoch, then align them to the interval start
        event_cutoff_minutes = minutes_since_epoch(event.cutoff)
        event_cutoff_start_minutes = align_to_interval(event_cutoff_minutes)

        # Determine when we started, based on registered_date
        event_registered_minutes = minutes_since_epoch(event.registered_date)
        event_registered_start_minutes = align_to_interval(event_registered_minutes)

        intervals = self.get_intervals_df(
            event_registered_start_minutes=event_registered_start_minutes,
            event_cutoff_start_minutes=event_cutoff_start_minutes,
        )
        if intervals.empty:
            await self.db_operations.mark_event_as_discarded(unique_event_id=event.unique_event_id)
            return self.return_empty_scores_df(
                "No intervals to score - event discarded.", event.event_id
            )

        # do not score miners which registered after the effective cutoff - 1 interval
        miners = self.miners_last_reg[
            self.miners_last_reg[PSNames.miner_registered_minutes]
            <= event_cutoff_start_minutes - AGGREGATION_INTERVAL_LENGTH_MINUTES
        ].copy()
        if miners.empty:
            return self.return_empty_scores_df("No miners to score.", event.event_id)

        # prepare predictions
        predictions_df = self.prepare_predictions_df(predictions=predictions, miners=miners)
        if predictions_df.empty:
            return self.return_empty_scores_df("No predictions to score.", event.event_id)

        interval_scores_df = self.get_interval_scores_base(
            predictions_df=predictions_df, miners=miners, intervals=intervals
        )

        scored_intervals_df = self.peer_score_intervals(
            interval_scores=interval_scores_df, outcome_round=outcome_round
        )

        scores_df = self.reduce_scored_intervals_df(scored_intervals_df)
        return scores_df

    async def export_peer_scores_to_db(self, scores_df: pd.DataFrame, event_id: str):
        sanitized_scores = scores_df.copy()
        fill_values = {
            PSNames.miner_uid: -1,  # should not happen
            PSNames.miner_hotkey: "unknown",  # should not happen
            PSNames.rema_prediction: -998,  # marker for missing predictions
            PSNames.rema_peer_score: 0.0,
        }
        sanitized_scores.fillna(value=fill_values, inplace=True)

        records = sanitized_scores.to_dict(orient="records")
        scores = []
        for record in records:
            try:
                score = ScoresModel(
                    event_id=event_id,
                    miner_uid=record[PSNames.miner_uid],
                    miner_hotkey=record[PSNames.miner_hotkey],
                    prediction=record[PSNames.rema_prediction],
                    event_score=record[PSNames.rema_peer_score],
                    spec_version=self.spec_version,
                )
                scores.append(score)
            except Exception:
                self.errors_count += 1
                self.logger.error(
                    "Error while creating a score record.",
                    extra={"record": record},
                )

        if not scores:
            self.errors_count += 1
            self.logger.error("No scores to export.", extra={"event_id": event_id})
            return

        await self.db_operations.insert_peer_scores(scores)

    async def run(self):
        await self.metagraph_lite_sync()

        miners_synced = await self.miners_last_reg_sync()
        if not miners_synced:
            return

        # TODO: do not score more than page_size=100 events at a time.
        events_to_score = await self.db_operations.get_events_for_scoring(
            # max_events=self.page_size
        )
        if not events_to_score:
            self.logger.debug("No events to calculate peer scores.")
        else:
            self.logger.debug(
                "Found events to calculate peer scores.", extra={"n_events": len(events_to_score)}
            )

            for event in events_to_score:
                unique_event_id = event.unique_event_id
                event_id = event.event_id
                event = self.set_right_cutoff(event)
                self.logger.debug(
                    "Calculating peer scores for an event.",
                    extra={
                        "event_id": event_id,
                        "event_registered_date": event.registered_date.isoformat(),
                        "event_cutoff": event.cutoff.isoformat(),
                        "event_resolved_at": event.resolved_at.isoformat(),
                    },
                )

                predictions = await self.db_operations.get_predictions_for_scoring(
                    unique_event_id=unique_event_id
                )
                if not predictions:
                    self.errors_count += 1
                    self.logger.error(
                        "There are no predictions for a settled event - discarding.",
                        extra={"event_id": event_id},
                    )
                    await self.db_operations.mark_event_as_discarded(
                        unique_event_id=unique_event_id
                    )
                    continue

                scores_df = await self.peer_score_event(event, predictions)
                if scores_df.empty:
                    self.logger.error(
                        "Peer scores could not be calculated for an event.",
                        extra={"event_id": event_id},
                    )
                    continue
                else:
                    self.logger.debug(
                        "Peer scores calculated, sample below.",
                        extra={
                            "event_id": event_id,
                            "scores": scores_df.head(n=5).to_dict(orient="index"),
                            "len_scores": len(scores_df),
                        },
                    )

                await self.export_peer_scores_to_db(scores_df, event_id)
                await self.db_operations.mark_event_as_processed(unique_event_id=unique_event_id)

        self.logger.debug(
            "Peer Scoring run finished. Resetting errors count.",
            extra={"errors_count_in_logs": self.errors_count},
        )
        self.errors_count = 0
