import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import numpy as np
import pytest
import torch
from bittensor.core.chain_data import AxonInfo
from bittensor.core.dendrite import DendriteMixin
from bittensor.core.metagraph import AsyncMetagraph

from neurons.protocol import EventPrediction, EventPredictionSynapse
from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.models.reasoning import ReasoningModel
from neurons.validator.tasks.query_miners import (
    REASONING_LENGTH_LIMIT,
    ExtendedAxonInfo,
    QueryMiners,
)
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class TestQueryMiners:
    @pytest.fixture
    def db_operations(self, db_client: DatabaseClient):
        logger = MagicMock(spec=InfiniteGamesLogger)

        return DatabaseOperations(db_client=db_client, logger=logger)

    @pytest.fixture
    def query_miners_task(self, db_operations: DatabaseOperations):
        dendrite = MagicMock(spec=DendriteMixin)
        metagraph = MagicMock(spec=AsyncMetagraph)
        metagraph.sync = AsyncMock()
        logger = MagicMock(spec=InfiniteGamesLogger)

        return QueryMiners(
            interval_seconds=60.0,
            db_operations=db_operations,
            dendrite=dendrite,
            metagraph=metagraph,
            # We set it to 'prod' to test prod miner filtering logic
            env="prod",
            logger=logger,
        )

    def test_get_axons(
        self,
        query_miners_task: QueryMiners,
    ):
        # Set up the mock attributes
        query_miners_task.metagraph.uids = np.array([0, 1, 2, 3])  # UIDs in the metagraph
        query_miners_task.metagraph.axons = (
            AxonInfo(
                hotkey="hotkey1", coldkey="coldkey1", version=1, ip="ip1", port=1, ip_type=1
            ),  # Serving axon
            AxonInfo(
                hotkey="hotkey2", coldkey="coldkey2", version=1, ip="ip2", port=1, ip_type=1
            ),  # Serving axon
            AxonInfo(
                hotkey="hotkey3", coldkey="coldkey3", version=1, ip="0.0.0.0", port=1, ip_type=1
            ),  # Not serving axon
            None,  # No axon at this UID
        )
        query_miners_task.metagraph.validator_trust = torch.nn.Parameter(
            torch.tensor([0.5, 0.0, 0.0, 0.0])
        )
        query_miners_task.metagraph.validator_permit = torch.nn.Parameter(
            torch.tensor([1.0, 1.0, 0.0, 0.0])
        )

        # Call the method
        all_axons, filtered_axons = query_miners_task.get_axons()

        # Assertions
        # All non none axons should be included
        assert all_axons == {
            0: ExtendedAxonInfo(
                version=1,
                ip="ip1",
                port=1,
                ip_type=1,
                hotkey="hotkey1",
                coldkey="coldkey1",
                protocol=4,
                placeholder1=0,
                placeholder2=0,
                is_validating=True,
                validator_permit=True,
            ),
            1: ExtendedAxonInfo(
                version=1,
                ip="ip2",
                port=1,
                ip_type=1,
                hotkey="hotkey2",
                coldkey="coldkey2",
                protocol=4,
                placeholder1=0,
                placeholder2=0,
                is_validating=False,
                validator_permit=True,
            ),
            2: ExtendedAxonInfo(
                version=1,
                ip="0.0.0.0",
                port=1,
                ip_type=1,
                hotkey="hotkey3",
                coldkey="coldkey3",
                protocol=4,
                placeholder1=0,
                placeholder2=0,
                is_validating=False,
                validator_permit=False,
            ),
        }

        # Only UIDs 0 & 1 should be included because they are serving
        assert filtered_axons == {
            0: ExtendedAxonInfo(
                version=1,
                ip="ip1",
                port=1,
                ip_type=1,
                hotkey="hotkey1",
                coldkey="coldkey1",
                protocol=4,
                placeholder1=0,
                placeholder2=0,
                is_validating=True,
                validator_permit=True,
            ),
            1: ExtendedAxonInfo(
                version=1,
                ip="ip2",
                port=1,
                ip_type=1,
                hotkey="hotkey2",
                coldkey="coldkey2",
                protocol=4,
                placeholder1=0,
                placeholder2=0,
                is_validating=False,
                validator_permit=True,
            ),
        }

    def test_get_axons_unique_cold_keys(
        self,
        query_miners_task: QueryMiners,
    ):
        # Set up the mock attributes
        query_miners_task.metagraph.uids = np.array([0, 1])  # UIDs in the metagraph
        query_miners_task.metagraph.axons = (
            AxonInfo(
                hotkey="hotkey1", coldkey="coldkey1", version=1, ip="ip1", port=1, ip_type=1
            ),  # Serving axon
            AxonInfo(
                hotkey="hotkey2", coldkey="coldkey1", version=1, ip="ip2", port=1, ip_type=1
            ),  # Serving axon but duplicate cold key
        )
        query_miners_task.metagraph.validator_trust = torch.nn.Parameter(torch.tensor([0.5, 0.0]))
        query_miners_task.metagraph.validator_permit = torch.nn.Parameter(torch.tensor([1.0, 1.0]))

        # Call the method
        all_axons, filtered_axons = query_miners_task.get_axons()

        # Assertions

        assert all_axons == {
            0: ExtendedAxonInfo(
                version=1,
                ip="ip1",
                port=1,
                ip_type=1,
                hotkey="hotkey1",
                coldkey="coldkey1",
                protocol=4,
                placeholder1=0,
                placeholder2=0,
                is_validating=True,
                validator_permit=True,
            ),
            1: ExtendedAxonInfo(
                version=1,
                ip="ip2",
                port=1,
                ip_type=1,
                hotkey="hotkey2",
                coldkey="coldkey1",
                protocol=4,
                placeholder1=0,
                placeholder2=0,
                is_validating=False,
                validator_permit=True,
            ),
        }

        # Only UID 0 should be included because it is serving and has unique cold key
        assert filtered_axons == {
            0: ExtendedAxonInfo(
                version=1,
                ip="ip1",
                port=1,
                ip_type=1,
                hotkey="hotkey1",
                coldkey="coldkey1",
                protocol=4,
                placeholder1=0,
                placeholder2=0,
                is_validating=True,
                validator_permit=True,
            )
        }

    def test_get_axons_empty_metagraph(
        self,
        query_miners_task: QueryMiners,
    ):
        # Set up the mock attributes
        query_miners_task.metagraph.uids = np.array([])  # No UIDs in the metagraph

        # Call the method
        all_axons, filtered_axons = query_miners_task.get_axons()

        # Assertions
        assert all_axons == {}

        # No axons should be included
        assert filtered_axons == {}

    def test_make_predictions_synapse(self, query_miners_task: QueryMiners):
        events_from_db = [
            (
                "event1",  # event_id
                "ifgames",  # market_type
                "LLM",  # event_type
                "Test match",  # description
                "2012-12-02T14:30:00+00:00",  # cutoff
                json.dumps(
                    {"topics": ["topic_1", "topic_2"], "trigger_name": "trigger"}
                ),  # metadata
            ),
            (
                "event2",  # event_id
                "ifgames",  # market_type
                "aZuro",  # event_type
                "Test match 2",  # description
                "2012-12-02T14:30:00+00:00",  # cutoff
                json.dumps({"topics": [], "trigger_name": None}),  # metadata
            ),
        ]

        synapse = query_miners_task.make_predictions_synapse(events=events_from_db)

        assert isinstance(synapse, EventPredictionSynapse)
        assert len(synapse.events) == 2

        # Assert event 1
        expected_key = "ifgames-event1"

        assert expected_key in synapse.events

        event_data = synapse.events[expected_key]

        assert event_data == EventPrediction(
            event_id="event1",
            market_type="LLM",
            probability=None,
            reasoning=None,
            miner_answered=False,
            description="Test match",
            cutoff=1354458600,
            metadata={"topics": ["topic_1", "topic_2"], "trigger_name": "trigger"},
        )

        # Assert event 2
        expected_key = "ifgames-event2"

        assert expected_key in synapse.events

        event_data = synapse.events[expected_key]

        assert event_data == EventPrediction(
            event_id="event2",
            market_type="aZuro",
            probability=None,
            reasoning=None,
            miner_answered=False,
            description="Test match 2",
            cutoff=1354458600,
            metadata={"topics": [], "trigger_name": None},
        )

    def test_make_predictions_synapse_empty_events(self, query_miners_task: QueryMiners):
        """Test handling of empty events"""
        events = []

        result = query_miners_task.make_predictions_synapse(events=events)

        assert isinstance(result, EventPredictionSynapse)
        assert len(result.events) == 0

    def test_parse_neuron_predictions(self, query_miners_task: QueryMiners):
        # Set up the mock attributes
        query_miners_task.metagraph.uids = np.array([1, 2])
        query_miners_task.metagraph.axons = {
            1: MagicMock(hotkey="hotkey1"),
            2: MagicMock(hotkey="hotkey2"),
        }

        interval_start_minutes = 12345
        uid = 2

        neuron_predictions = EventPredictionSynapse(
            events={
                "ifgames-event1": EventPrediction(
                    event_id="event1",
                    market_type="acled",
                    probability=0.5,
                    reasoning=None,
                    miner_answered=True,
                    description="Test match",
                    cutoff=1354458600,
                    metadata={"topics": ["topic_1", "topic_2"], "trigger_name": "trigger"},
                ),
                "ifgames-event2": EventPrediction(
                    event_id="event2",
                    market_type="azuro",
                    probability=0.75,
                    reasoning="s" * (REASONING_LENGTH_LIMIT * 2),
                    miner_answered=False,
                    description="Test match 2",
                    cutoff=1354458600,
                    metadata={
                        "topics": [
                            "topic_10",
                        ],
                        "trigger_name": None,
                    },
                ),
                "ifgames-event3": EventPrediction(
                    event_id="event3",
                    market_type="azuro",
                    # None predictions are dropped
                    probability=None,
                    reasoning="Reasoning event 3",
                    miner_answered=False,
                    description="Test match 2",
                    cutoff=1354458600,
                    metadata={},
                ),
            }
        )

        predictions, reasonings = query_miners_task.parse_neuron_predictions(
            interval_start_minutes=interval_start_minutes,
            uid=uid,
            neuron_predictions=neuron_predictions,
        )

        assert predictions == [
            (
                "ifgames-event1",
                query_miners_task.metagraph.axons[uid].hotkey,
                uid,
                0.5,
                interval_start_minutes,
                0.5,
            ),
            (
                "ifgames-event2",
                query_miners_task.metagraph.axons[uid].hotkey,
                uid,
                0.75,
                interval_start_minutes,
                0.75,
            ),
        ]

        assert reasonings == [
            ReasoningModel(
                event_id="ifgames-event2",
                miner_uid=2,
                miner_hotkey="hotkey2",
                reasoning="s" * REASONING_LENGTH_LIMIT + "--TRUNCATED--",
                created_at=None,
                updated_at=None,
                exported=False,
            )
        ]

    async def test_query_neurons(self, query_miners_task: QueryMiners, monkeypatch):
        synapse = MagicMock(model_copy=MagicMock())

        axons_by_uid = {
            "1": MagicMock(spec=AxonInfo, hotkey="hotkey_1"),
            "50": MagicMock(spec=AxonInfo, hotkey="hotkey_50"),
        }

        query_miners_task.dendrite.call = AsyncMock(return_value=synapse)

        iterator_called = False

        async def fake_async_iterator(items):
            nonlocal iterator_called
            iterator_called = True

            for item in items:
                yield item

        monkeypatch.setattr(
            "neurons.validator.tasks.query_miners.async_iterator", fake_async_iterator
        )

        response = await query_miners_task.query_neurons(axons_by_uid=axons_by_uid, synapse=synapse)

        # Assertions
        assert len(response) == 2
        assert response["1"] == synapse
        assert response["50"] == synapse

        assert iterator_called is True
        assert synapse.model_copy.call_count == 2
        assert query_miners_task.dendrite.call.await_count == 2

    async def test_store_miners(self, db_client: DatabaseClient, query_miners_task: QueryMiners):
        block = 12345
        axons = {
            "uid_1": MagicMock(
                hotkey="hotkey_1", ip="ip_1", is_validating=False, validator_permit=True
            )
        }

        await query_miners_task.store_miners(block=block, axons=axons)

        # Assertions
        result = await db_client.many(
            """
                SELECT
                    miner_uid,
                    miner_hotkey,
                    node_ip,
                    registered_date,
                    last_updated,
                    blocktime,
                    is_validating,
                    validator_permit
                FROM
                    miners
            """
        )

        assert len(result) == 1
        assert result[0] == (
            "uid_1",
            "hotkey_1",
            "ip_1",
            "2024-01-01T00:00:00",
            ANY,
            block,
            False,
            True,
        )

        # Conflicting insert
        new_block = 23456
        new_axons = {
            "uid_1": MagicMock(
                hotkey="hotkey_1", ip="ip_2", is_validating=True, validator_permit=False
            ),
            "uid_2": MagicMock(
                hotkey="hotkey_2", ip="ip_3", is_validating=True, validator_permit=False
            ),
        }

        # Wait 1s so last updated is different
        await asyncio.sleep(1)

        await query_miners_task.store_miners(block=new_block, axons=new_axons)

        # Assertions
        result_2 = await db_client.many(
            """
                SELECT
                    miner_uid,
                    miner_hotkey,
                    node_ip,
                    registered_date,
                    last_updated,
                    blocktime,
                    is_validating,
                    validator_permit
                FROM
                    miners
            """
        )

        assert len(result_2) == 2
        assert result_2[0] == (
            "uid_1",
            "hotkey_1",
            "ip_2",
            "2024-01-01T00:00:00",
            ANY,
            new_block,
            True,
            False,
        )
        assert result_2[1] == (
            "uid_2",
            "hotkey_2",
            "ip_3",
            ANY,
            ANY,
            new_block,
            True,
            False,
        )

        # Check that last update has been updated
        assert result[0][4] != result_2[0][4]
        # Check that the registered_date is not anymore 2024-01-01T00:00:00
        assert result_2[1][4] != "2024-01-01T00:00:00"

    async def test_store_predictions_and_reasonings(self, query_miners_task: QueryMiners):
        # Set up mocks
        query_miners_task.db_operations.upsert_predictions = AsyncMock()
        query_miners_task.db_operations.upsert_reasonings = AsyncMock()

        query_miners_task.metagraph.axons = {
            "1": MagicMock(hotkey="hotkey_1"),
            "2": MagicMock(hotkey="hotkey_2"),
        }

        interval_start_minutes = 12345

        synapse = EventPredictionSynapse(
            events={
                "acled-event1": EventPrediction(
                    event_id="event1",
                    market_type="acled",
                    probability=0.5,
                    reasoning="Reasoning 1",
                    miner_answered=True,
                    description="Test match",
                    cutoff=1354458600,
                    metadata={},
                ),
                "azuro-event2": EventPrediction(
                    event_id="event2",
                    market_type="azuro",
                    probability=0.75,
                    reasoning="Reasoning 2",
                    miner_answered=False,
                    description="Test match 2",
                    cutoff=1354458600,
                    metadata={},
                ),
            }
        )

        neurons_predictions = {"1": synapse, "2": synapse}

        await query_miners_task.store_predictions_and_reasonings(
            interval_start_minutes=interval_start_minutes,
            neurons_predictions=neurons_predictions,
        )

        # Assertions
        assert query_miners_task.db_operations.upsert_predictions.await_count == 2

        assert query_miners_task.db_operations.upsert_predictions.await_args_list == [
            call(
                predictions=[
                    (
                        "acled-event1",
                        "hotkey_1",
                        "1",
                        0.5,
                        interval_start_minutes,
                        0.5,
                    ),
                    (
                        "azuro-event2",
                        "hotkey_1",
                        "1",
                        0.75,
                        interval_start_minutes,
                        0.75,
                    ),
                ]
            ),
            call(
                predictions=[
                    (
                        "acled-event1",
                        "hotkey_2",
                        "2",
                        0.5,
                        interval_start_minutes,
                        0.5,
                    ),
                    (
                        "azuro-event2",
                        "hotkey_2",
                        "2",
                        0.75,
                        interval_start_minutes,
                        0.75,
                    ),
                ]
            ),
        ]

        assert query_miners_task.db_operations.upsert_reasonings.await_count == 2

        assert query_miners_task.db_operations.upsert_reasonings.await_args_list == [
            call(
                reasonings=[
                    ReasoningModel(
                        event_id="acled-event1",
                        miner_uid=1,
                        miner_hotkey="hotkey_1",
                        reasoning="Reasoning 1",
                        created_at=None,
                        updated_at=None,
                        exported=False,
                    ),
                    ReasoningModel(
                        event_id="azuro-event2",
                        miner_uid=1,
                        miner_hotkey="hotkey_1",
                        reasoning="Reasoning 2",
                        created_at=None,
                        updated_at=None,
                        exported=False,
                    ),
                ]
            ),
            call(
                reasonings=[
                    ReasoningModel(
                        event_id="acled-event1",
                        miner_uid=2,
                        miner_hotkey="hotkey_2",
                        reasoning="Reasoning 1",
                        created_at=None,
                        updated_at=None,
                        exported=False,
                    ),
                    ReasoningModel(
                        event_id="azuro-event2",
                        miner_uid=2,
                        miner_hotkey="hotkey_2",
                        reasoning="Reasoning 2",
                        created_at=None,
                        updated_at=None,
                        exported=False,
                    ),
                ]
            ),
        ]

    async def test_run(
        self,
        db_client: DatabaseClient,
        db_operations: DatabaseOperations,
        query_miners_task: QueryMiners,
    ):
        # Set events to query & predict
        cutoff_future = (datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat()

        events = [
            EventsModel(
                unique_event_id="ifgames-event1",
                event_id="event1",
                market_type="ifgames",
                event_type="sports",
                description="desc1",
                outcome="outcome2",
                status=EventStatus.PENDING,
                metadata=json.dumps({"market_type": "sports"}),
                created_at="2012-12-02T14:30:00+00:00",
                cutoff=cutoff_future,
            ),
            EventsModel(
                unique_event_id="ifgames-event2",
                event_id="event2",
                market_type="ifgames",
                event_type="acled",
                description="desc2",
                outcome="outcome2",
                status=EventStatus.PENDING,
                metadata=json.dumps({"market_type": "acled"}),
                created_at="2012-12-02T14:30:00+00:00",
                cutoff=cutoff_future,
            ),
        ]

        await db_operations.upsert_events(events=events)

        # Set up the bittensor mocks
        block = 101.0

        query_miners_task.metagraph.uids = np.array([0, 1, 2])
        query_miners_task.metagraph.block = torch.nn.Parameter(torch.tensor(block))
        query_miners_task.metagraph.axons = (
            AxonInfo(
                hotkey="hotkey_1", coldkey="coldkey1", version=1, ip="ip_1", port=1, ip_type=1
            ),  # Serving axon
            AxonInfo(
                hotkey="hotkey_2", coldkey="coldkey2", version=1, ip="ip_2", port=1, ip_type=1
            ),  # Serving axon
            AxonInfo(
                hotkey="hotkey_3", coldkey="coldkey1", version=1, ip="ip_2", port=1, ip_type=1
            ),  # Serving axon duplicate cold key
        )
        query_miners_task.metagraph.validator_trust = torch.nn.Parameter(
            torch.tensor(
                [
                    0.5,
                    0.0,
                    0.0,
                ]
            )
        )
        query_miners_task.metagraph.validator_permit = torch.nn.Parameter(
            torch.tensor(
                [
                    1.0,
                    0.0,
                    0.0,
                ]
            )
        )

        async def call(
            target_axon: AxonInfo, synapse: EventPredictionSynapse, deserialize: bool, timeout: int
        ):
            # Add a fake probability to each event in synapse.events
            for _, event in synapse.events.items():
                event.probability = 0.8
                event.reasoning = "Test reasoning"

            return synapse

        query_miners_task.dendrite.call = call

        # Configure mock to return our test time when now() is called
        mocked_interval_start_minutes = 530160

        with patch(
            "neurons.validator.tasks.query_miners.get_interval_start_minutes"
        ) as mock_get_interval_start_minutes:
            mock_get_interval_start_minutes.return_value = mocked_interval_start_minutes

            # Run the task
            await query_miners_task.run()

        # Assertions
        predictions = await db_client.many(
            """
                SELECT * FROM predictions
            """
        )

        assert len(predictions) == 4

        assert predictions == [
            (
                # unique_event_id
                "ifgames-event1",
                # miner_uid
                0,
                # miner_hotkey
                "hotkey_1",
                # prediction
                0.8,
                # interval_start_minutes
                mocked_interval_start_minutes,
                # interval_agg_prediction
                0.8,
                # interval_count
                1,
                # submitted
                ANY,
                # updated_at
                ANY,
                # exported
                0,
            ),
            (
                "ifgames-event2",
                0,
                "hotkey_1",
                0.8,
                mocked_interval_start_minutes,
                0.8,
                1,
                ANY,
                ANY,
                0,
            ),
            (
                "ifgames-event1",
                1,
                "hotkey_2",
                0.8,
                mocked_interval_start_minutes,
                0.8,
                1,
                ANY,
                ANY,
                0,
            ),
            (
                "ifgames-event2",
                1,
                "hotkey_2",
                0.8,
                mocked_interval_start_minutes,
                0.8,
                1,
                ANY,
                ANY,
                0,
            ),
        ]

        miners = await db_client.many(
            """
                SELECT
                    miner_uid,
                    miner_hotkey,
                    is_validating,
                    validator_permit
                FROM
                    miners
                """
        )

        # All axons should be stored, including the axon uid 2 that is filtered out
        assert miners == [("0", "hotkey_1", 1, 1), ("1", "hotkey_2", 0, 0), ("2", "hotkey_3", 0, 0)]

        # Assert reasonings
        reasonings = await db_client.many(
            """
                SELECT
                    event_id,
                    miner_uid,
                    miner_hotkey,
                    reasoning,
                    exported
                FROM
                    reasoning
                """
        )

        assert len(reasonings) == 4
        assert reasonings[0] == ("ifgames-event1", 0, "hotkey_1", "Test reasoning", 0)
        assert reasonings[1] == ("ifgames-event2", 0, "hotkey_1", "Test reasoning", 0)
        assert reasonings[2] == ("ifgames-event1", 1, "hotkey_2", "Test reasoning", 0)
        assert reasonings[3] == ("ifgames-event2", 1, "hotkey_2", "Test reasoning", 0)

    async def test_run_no_events_to_predict(self, query_miners_task: QueryMiners):
        # Set mocks
        query_miners_task.make_predictions_synapse = MagicMock()

        # Run the task
        await query_miners_task.run()

        # Assertions
        query_miners_task.make_predictions_synapse.assert_not_called()

    async def test_run_no_axons_to_query(
        self, db_operations: DatabaseOperations, query_miners_task: QueryMiners
    ):
        # Set up the mock attributes
        query_miners_task.metagraph.uids = torch.nn.Parameter(torch.tensor([]))
        query_miners_task.metagraph.block = torch.nn.Parameter(torch.tensor(99.0))

        query_miners_task.query_neurons = MagicMock()
        query_miners_task.make_predictions_synapse = MagicMock()

        # Set events to query & predict
        events = [
            EventsModel(
                unique_event_id="unique1",
                event_id="event1",
                market_type="ifgames",
                event_type="sports",
                description="desc1",
                outcome="outcome2",
                status=EventStatus.PENDING,
                metadata=json.dumps({"market_type": "sports"}),
                created_at="2012-12-02T14:30:00+00:00",
                cutoff=(datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat(),
            ),
        ]

        await db_operations.upsert_events(events=events)

        # Run the task
        await query_miners_task.run()

        # Assertions
        query_miners_task.make_predictions_synapse.assert_called_once()
        query_miners_task.query_neurons.assert_not_called()
