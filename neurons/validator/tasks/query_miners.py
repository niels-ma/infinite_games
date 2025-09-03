import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable

from bittensor.core.chain_data import AxonInfo
from bittensor.core.dendrite import DendriteMixin
from bittensor.core.metagraph import AsyncMetagraph

from neurons.protocol import EventPrediction, EventPredictionSynapse
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.reasoning import ReasoningModel
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.common.async_iterator import async_iterator
from neurons.validator.utils.common.converters import torch_or_numpy_to_int
from neurons.validator.utils.common.interval import get_interval_start_minutes
from neurons.validator.utils.config import IfgamesEnvType
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


@dataclass
class ExtendedAxonInfo(AxonInfo):
    is_validating: bool = False
    validator_permit: bool = False


AxonInfoByUidType = dict[int, ExtendedAxonInfo]
SynapseResponseByUidType = dict[int, EventPredictionSynapse]

REASONING_LENGTH_LIMIT = 10000


class QueryMiners(AbstractTask):
    interval: float
    db_operations: DatabaseOperations
    dendrite: DendriteMixin
    env: IfgamesEnvType
    logger: InfiniteGamesLogger

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        dendrite: DendriteMixin,
        metagraph: AsyncMetagraph,
        env: IfgamesEnvType,
        logger: InfiniteGamesLogger,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        # Validate dendrite
        if not isinstance(dendrite, DendriteMixin):
            raise TypeError("dendrite must be an instance of DendriteMixin.")

        # Validate metagraph
        if not isinstance(metagraph, AsyncMetagraph):
            raise TypeError("metagraph must be an instance of AsyncMetagraph.")

        # Validate env
        if not isinstance(env, str):
            raise TypeError("env must be an instance of str.")

        # Validate logger
        if not isinstance(logger, InfiniteGamesLogger):
            raise TypeError("logger must be an instance of InfiniteGamesLogger.")

        self.interval = interval_seconds
        self.db_operations = db_operations
        self.dendrite = dendrite
        self.metagraph = metagraph
        self.env = env
        self.logger = logger

    @property
    def name(self):
        return "query-miners"

    @property
    def interval_seconds(self):
        return self.interval

    async def run(self):
        # Get events to predict
        events = await self.db_operations.get_events_to_predict()

        if not len(events):
            return

        # Build synapse
        synapse = self.make_predictions_synapse(events)

        # Sync metagraph & store the current block
        await self.metagraph.sync(lite=True)
        block = torch_or_numpy_to_int(self.metagraph.block)

        # Get axons to query
        all_axons, filtered_axons = self.get_axons()

        # Store miners
        await self.store_miners(block=block, axons=all_axons)

        if not len(filtered_axons):
            return

        # Query neurons
        predictions_synapses: SynapseResponseByUidType = await self.query_neurons(
            axons_by_uid=filtered_axons, synapse=synapse
        )

        interval_start_minutes = get_interval_start_minutes()

        # Store predictions
        await self.store_predictions_and_reasonings(
            interval_start_minutes=interval_start_minutes,
            neurons_predictions=predictions_synapses,
        )

    def get_axons(self) -> tuple[AxonInfoByUidType, AxonInfoByUidType]:
        all_axons: AxonInfoByUidType = {}
        filtered_axons: AxonInfoByUidType = {}

        seen_cold_keys: set[str] = set()
        seen_ips: set[str] = set()

        for uid in self.metagraph.uids:
            int_uid = torch_or_numpy_to_int(uid)
            axon = self.metagraph.axons[int_uid]

            if axon is None:
                continue

            is_validating = True if self.metagraph.validator_trust[uid].float() > 0.0 else False
            validator_permit = torch_or_numpy_to_int(self.metagraph.validator_permit[uid]) > 0

            extended_axon = ExtendedAxonInfo(
                **asdict(axon), is_validating=is_validating, validator_permit=validator_permit
            )

            all_axons[int_uid] = extended_axon

            # Filtering logic
            if axon.is_serving:
                if self.env == "prod":
                    # Only allow unique cold keys
                    if axon.coldkey in seen_cold_keys:
                        self.logger.debug(
                            "Duplicate axon cold key found",
                            extra={"uid": int_uid, "coldkey": axon.coldkey},
                        )

                        continue

                    # Only allow unique ips
                    if axon.ip in seen_ips:
                        self.logger.debug(
                            "Duplicate axon IP found",
                            extra={"uid": int_uid, "ip": axon.ip},
                        )

                        continue

                filtered_axons[int_uid] = extended_axon

                seen_cold_keys.add(axon.coldkey)
                seen_ips.add(axon.ip)

        return all_axons, filtered_axons

    def make_predictions_synapse(self, events: Iterable[tuple[any]]) -> EventPredictionSynapse:
        compiled_events: dict[str, EventPrediction] = {}

        for event in events:
            event_id = event[0]
            truncated_market_type = event[1]
            event_type = event[2]
            description = event[3]
            cutoff = int(datetime.fromisoformat(event[4]).timestamp()) if event[4] else None
            metadata = {**json.loads(event[5])}

            compiled_events[f"{truncated_market_type}-{event_id}"] = EventPrediction(
                event_id=event_id,
                market_type=event_type,
                probability=None,
                reasoning=None,
                miner_answered=False,
                description=description,
                cutoff=cutoff,
                metadata=metadata,
            )

        return EventPredictionSynapse(events=compiled_events)

    def parse_neuron_predictions(
        self,
        interval_start_minutes: int,
        uid: int,
        neuron_predictions: EventPredictionSynapse,
    ) -> tuple[list, list[ReasoningModel]]:
        axon_hotkey = self.metagraph.axons[uid].hotkey

        predictions_to_insert = []
        reasonings_to_insert: list[ReasoningModel] = []

        # Iterate over all event predictions
        for unique_event_id, event_prediction in neuron_predictions.events.items():
            answer = event_prediction.probability

            # Drop null predictions
            if answer is None:
                continue

            prediction = (
                unique_event_id,
                axon_hotkey,
                uid,
                answer,
                interval_start_minutes,
                answer,
            )

            predictions_to_insert.append(prediction)

            reasoning = event_prediction.reasoning

            if reasoning is not None:
                if len(reasoning) > REASONING_LENGTH_LIMIT:
                    reasoning = reasoning[:REASONING_LENGTH_LIMIT] + "--TRUNCATED--"

                reasoning_model = ReasoningModel(
                    event_id=unique_event_id,
                    miner_uid=uid,
                    miner_hotkey=axon_hotkey,
                    reasoning=reasoning,
                )

                reasonings_to_insert.append(reasoning_model)

        return predictions_to_insert, reasonings_to_insert

    async def query_neurons(self, axons_by_uid: AxonInfoByUidType, synapse: EventPredictionSynapse):
        timeout = 120

        start_time = time.time()

        tasks: list[asyncio.Task[EventPredictionSynapse]] = []

        async for axon in async_iterator(axons_by_uid.values()):
            tasks.append(
                asyncio.create_task(
                    self.dendrite.call(
                        target_axon=axon,
                        synapse=synapse.model_copy(),
                        deserialize=False,
                        timeout=timeout,
                    )
                )
            )

        responses = await asyncio.gather(*tasks)

        elapsed_time_ms = round((time.time() - start_time) * 1000)

        self.logger.debug(
            "Miners queried",
            extra={"miners_count": len(axons_by_uid), "elapsed_time_ms": elapsed_time_ms},
        )

        responses_by_uid: SynapseResponseByUidType = {}
        axons_uids = list(axons_by_uid.keys())

        for uid, response in zip(axons_uids, responses):
            if not response.is_success:
                self.logger.debug(
                    "Miner response error",
                    extra={
                        "miner_uid": uid,
                        "miner_hotkey": axons_by_uid[uid].hotkey,
                        "is_success": response.is_success,
                        "axon_status_code": response.axon.status_code,
                        "axon_status_message": response.axon.status_message,
                        "dendrite_status_code": response.dendrite.status_code,
                        "dendrite_status_message": response.dendrite.status_message,
                    },
                )

            responses_by_uid[uid] = response

        return responses_by_uid

    async def store_miners(self, block: int, axons: AxonInfoByUidType):
        miners_count_in_db = await self.db_operations.get_miners_count()

        registered_date = (
            datetime.now().isoformat()
            if miners_count_in_db > 0
            else datetime(year=2024, month=1, day=1).isoformat()
        )

        miners = [
            (
                uid,
                axon.hotkey,
                axon.ip,
                registered_date,
                block,
                axon.is_validating,
                axon.validator_permit,
                axon.ip,
                block,
            )
            for uid, axon in axons.items()
        ]

        await self.db_operations.upsert_miners(miners=miners)

        self.logger.debug("Miners stored", extra={"miners_count": len(miners)})

    async def store_predictions_and_reasonings(
        self, interval_start_minutes: int, neurons_predictions: SynapseResponseByUidType
    ):
        # For each neuron predictions
        for uid, neuron_predictions in neurons_predictions.items():
            # Parse neuron predictions for insert
            (
                parsed_neuron_predictions_for_insertion,
                parsed_neuron_reasonings_for_insertion,
            ) = self.parse_neuron_predictions(
                interval_start_minutes=interval_start_minutes,
                uid=uid,
                neuron_predictions=neuron_predictions,
            )

            if len(parsed_neuron_predictions_for_insertion) > 0:
                # Batch upsert neuron predictions
                await self.db_operations.upsert_predictions(
                    predictions=parsed_neuron_predictions_for_insertion
                )

                self.logger.debug(
                    "Predictions stored",
                    extra={
                        "neuron_uid": uid,
                        "predictions_count": len(parsed_neuron_predictions_for_insertion),
                    },
                )

            if len(parsed_neuron_reasonings_for_insertion) > 0:
                # Batch upsert neuron reasonings
                await self.db_operations.upsert_reasonings(
                    reasonings=parsed_neuron_reasonings_for_insertion
                )

                self.logger.debug(
                    "Reasonings stored",
                    extra={
                        "neuron_uid": uid,
                        "reasonings_count": len(parsed_neuron_reasonings_for_insertion),
                    },
                )
