import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from freezegun import freeze_time

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.models.score import SCORE_FIELDS, ScoresModel
from neurons.validator.tasks.metagraph_scoring import MOVING_AVERAGE_EVENTS, MetagraphScoring
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class TestMetagraphScoring:
    @pytest.fixture
    def db_operations(self, db_client: DatabaseClient):
        logger = MagicMock(spec=InfiniteGamesLogger)

        return DatabaseOperations(db_client=db_client, logger=logger)

    @pytest.fixture
    def metagraph_scoring_task(
        self,
        db_operations: DatabaseOperations,
    ):
        logger = MagicMock(spec=InfiniteGamesLogger)

        with freeze_time("2025-01-02 03:00:00"):
            return MetagraphScoring(
                interval_seconds=60.0,
                page_size=100,
                db_operations=db_operations,
                logger=logger,
            )

    def test_init(self, metagraph_scoring_task: MetagraphScoring):
        unit = metagraph_scoring_task

        assert isinstance(unit, MetagraphScoring)
        assert unit.interval_seconds == 60.0
        assert unit.page_size == 100
        assert unit.errors_count == 0

    @pytest.mark.parametrize(
        "scores_list, expected_result, log_calls",
        [
            # Case 0: No scores.
            (
                [],
                [],
                {
                    "debug": [
                        ("No events to calculate metagraph scores.", {}),
                    ]
                },
            ),
            # Case 1: All scores are processed.
            (
                [
                    ScoresModel(
                        event_id="processed_event_id_1",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.80,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=True,
                    ),
                    ScoresModel(
                        event_id="processed_event_id_2",
                        miner_uid=1,
                        miner_hotkey="hk1",
                        prediction=0.85,
                        event_score=0.90,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=True,
                    ),
                ],
                [],
                {
                    "debug": [
                        ("No events to calculate metagraph scores.", {}),
                    ]
                },
            ),
            # Case 2: Single miner single event.
            (
                [
                    ScoresModel(
                        event_id="expected_event_id",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.80,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                ],
                [
                    {
                        "event_id": "expected_event_id",
                        "processed": 1,
                        "metagraph_score": 1.0,
                        "other_data": {
                            "sum_weighted_peer_score": 1.6,
                            "sum_weight": 2.0,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 1,
                            "avg_peer_score": 0.8,
                            "sqmax_avg_peer_score": 0.64,
                        },
                    },
                ],
                {
                    "debug": [
                        (
                            "Found events to calculate metagraph scores.",
                            {"n_events": 1},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id"},
                        ),
                        (
                            "Metagraph scores calculated successfully.",
                            {"event_id": "expected_event_id"},
                        ),
                        ("Metagraph scoring task completed.", {"errors_count": 0}),
                    ]
                },
            ),
            # Case 3: Two miners single event, one miner processed -> reprocess.
            (
                [
                    ScoresModel(
                        event_id="expected_event_id",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.80,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=True,
                    ),
                    ScoresModel(
                        event_id="expected_event_id",
                        miner_uid=4,
                        miner_hotkey="hk4",
                        prediction=0.75,
                        event_score=0.40,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                ],
                [
                    {
                        "event_id": "expected_event_id",
                        "processed": 1,
                        "metagraph_score": 0.8,
                        "other_data": {
                            "sum_weighted_peer_score": 1.60,
                            "sum_weight": 2.0,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 1,
                            "avg_peer_score": 0.8,
                            "sqmax_avg_peer_score": 0.64,
                        },
                    },
                    {
                        "event_id": "expected_event_id",
                        "processed": 1,
                        "metagraph_score": 0.2,
                        "other_data": {
                            "sum_weighted_peer_score": 0.80,
                            "sum_weight": 2.0,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 1,
                            "avg_peer_score": 0.4,
                            "sqmax_avg_peer_score": 0.16,
                        },
                    },
                ],
                {
                    "debug": [
                        (
                            "Found events to calculate metagraph scores.",
                            {"n_events": 1},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id"},
                        ),
                        (
                            "Metagraph scores calculated successfully.",
                            {"event_id": "expected_event_id"},
                        ),
                        ("Metagraph scoring task completed.", {"errors_count": 0}),
                    ]
                },
            ),
            # Case 4: Exception during set metagraph peer scores.
            (
                [
                    ScoresModel(
                        event_id="expected_event_id",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.80,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                ],
                [],
                {
                    "debug": [
                        (
                            "Found events to calculate metagraph scores.",
                            {"n_events": 1},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id"},
                        ),
                    ],
                    "exception": [
                        (
                            "Error calculating metagraph scores.",
                            {"event_id": "expected_event_id"},
                        ),
                    ],
                },
            ),
            # Case 5: Three miners single event, one has negative peer score.
            (
                [
                    ScoresModel(
                        event_id="expected_event_id",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.80,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id",
                        miner_uid=4,
                        miner_hotkey="hk4",
                        prediction=0.75,
                        event_score=0.40,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id",
                        miner_uid=5,
                        miner_hotkey="hk5",
                        prediction=0.75,
                        event_score=-0.40,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                ],
                [
                    {
                        "event_id": "expected_event_id",
                        "processed": 1,
                        "metagraph_score": 0.8,
                        "other_data": {
                            "sum_weighted_peer_score": 1.6,
                            "sum_weight": 2.0,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 1,
                            "avg_peer_score": 0.8,
                            "sqmax_avg_peer_score": 0.64,
                        },
                    },
                    {
                        "event_id": "expected_event_id",
                        "processed": 1,
                        "metagraph_score": 0.2,
                        "other_data": {
                            "sum_weighted_peer_score": 0.8,
                            "sum_weight": 2.0,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 1,
                            "avg_peer_score": 0.4,
                            "sqmax_avg_peer_score": 0.16,
                        },
                    },
                    {
                        "event_id": "expected_event_id",
                        "processed": 1,
                        "metagraph_score": 0.0,
                        "other_data": {
                            "sum_weighted_peer_score": -0.80,
                            "sum_weight": 2.0,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 1,
                            "avg_peer_score": -0.4,
                            "sqmax_avg_peer_score": 0.0,
                        },
                    },
                ],
                {
                    "debug": [
                        (
                            "Found events to calculate metagraph scores.",
                            {"n_events": 1},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id"},
                        ),
                        (
                            "Metagraph scores calculated successfully.",
                            {"event_id": "expected_event_id"},
                        ),
                        ("Metagraph scoring task completed.", {"errors_count": 0}),
                    ]
                },
            ),
            # Case 6: One miner for 2 events, one miner for 1 event.
            (
                [
                    ScoresModel(
                        event_id="expected_event_id_1",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.80,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_2",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.40,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_2",
                        miner_uid=4,
                        miner_hotkey="hk4",
                        prediction=0.75,
                        event_score=0.40,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                ],
                [
                    {
                        "event_id": "expected_event_id_1",
                        "processed": 1,
                        "metagraph_score": 1.0,
                        "other_data": {
                            "sum_weighted_peer_score": 1.6,
                            "sum_weight": 2.0,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 1,
                            "avg_peer_score": 0.8,
                            "sqmax_avg_peer_score": 0.64,
                        },
                    },
                    {
                        "event_id": "expected_event_id_2",
                        "processed": 1,
                        "metagraph_score": 0.9,
                        "other_data": {
                            "sum_weighted_peer_score": 1.8,
                            "sum_weight": 3.0,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 2,
                            "avg_peer_score": 0.6,
                            "sqmax_avg_peer_score": 0.36,
                        },
                    },
                    {
                        "event_id": "expected_event_id_2",
                        "processed": 1,
                        "metagraph_score": 0.1,
                        "other_data": {
                            "sum_weighted_peer_score": 0.6,
                            "sum_weight": 3.0,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 1,
                            "avg_peer_score": 0.2,
                            "sqmax_avg_peer_score": 0.04,
                        },
                    },
                ],
                {
                    "debug": [
                        (
                            "Found events to calculate metagraph scores.",
                            {"n_events": 2},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id_1"},
                        ),
                        (
                            "Metagraph scores calculated successfully.",
                            {"event_id": "expected_event_id_1"},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id_2"},
                        ),
                        (
                            "Metagraph scores calculated successfully.",
                            {"event_id": "expected_event_id_2"},
                        ),
                        ("Metagraph scoring task completed.", {"errors_count": 0}),
                    ]
                },
            ),
            # Case 7: 1 miner 3 events, 1 miner 2 events, 1 miner 1 event with negative peer score.
            (
                [
                    ScoresModel(
                        event_id="expected_event_id_1",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.80,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_2",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.40,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_3",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.60,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_2",
                        miner_uid=4,
                        miner_hotkey="hk4",
                        prediction=0.75,
                        event_score=0.40,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_3",
                        miner_uid=4,
                        miner_hotkey="hk4",
                        prediction=0.75,
                        event_score=0.40,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_2",
                        miner_uid=5,
                        miner_hotkey="hk5",
                        prediction=0.75,
                        event_score=-0.40,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                ],
                [
                    {
                        "event_id": "expected_event_id_1",
                        "processed": 1,
                        "metagraph_score": 1.0,
                        "other_data": {
                            "sum_weighted_peer_score": 1.6,
                            "sum_weight": 2.0,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 1,
                            "avg_peer_score": 0.8,
                            "sqmax_avg_peer_score": 0.64,
                        },
                    },
                    {
                        "event_id": "expected_event_id_2",
                        "processed": 1,
                        "metagraph_score": 0.9,
                        "other_data": {
                            "sum_weighted_peer_score": 1.8,
                            "sum_weight": 3.0,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 2,
                            "avg_peer_score": 0.6,
                            "sqmax_avg_peer_score": 0.36,
                        },
                    },
                    {
                        "event_id": "expected_event_id_3",
                        "processed": 1,
                        "metagraph_score": 0.835,
                        "other_data": {
                            "sum_weighted_peer_score": 2.4,
                            "sum_weight": 4.0,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 3,
                            "avg_peer_score": 0.6,
                            "sqmax_avg_peer_score": 0.36,
                        },
                    },
                    {
                        "event_id": "expected_event_id_2",
                        "processed": 1,
                        "metagraph_score": 0.1,
                        "other_data": {
                            "sum_weighted_peer_score": 0.6,
                            "sum_weight": 3.0,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 1,
                            "avg_peer_score": 0.2,
                            "sqmax_avg_peer_score": 0.04,
                        },
                    },
                    {
                        "event_id": "expected_event_id_3",
                        "processed": 1,
                        "metagraph_score": 0.165,
                        "other_data": {
                            "sum_weighted_peer_score": 1.0667,
                            "sum_weight": 4.0,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 2,
                            "avg_peer_score": 0.26667,
                            "sqmax_avg_peer_score": 0.0711,
                        },
                    },
                    {
                        "event_id": "expected_event_id_2",
                        "processed": 1,
                        "metagraph_score": 0.0,
                        "other_data": {
                            "sum_weighted_peer_score": -0.6,
                            "sum_weight": 3.0,
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": 1,
                            "avg_peer_score": -0.2,
                            "sqmax_avg_peer_score": 0.0,
                        },
                    },
                ],
                {
                    "debug": [
                        (
                            "Found events to calculate metagraph scores.",
                            {"n_events": 3},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id_1"},
                        ),
                        (
                            "Metagraph scores calculated successfully.",
                            {"event_id": "expected_event_id_1"},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id_2"},
                        ),
                        (
                            "Metagraph scores calculated successfully.",
                            {"event_id": "expected_event_id_2"},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id_3"},
                        ),
                        (
                            "Metagraph scores calculated successfully.",
                            {"event_id": "expected_event_id_3"},
                        ),
                        ("Metagraph scoring task completed.", {"errors_count": 0}),
                    ]
                },
            ),
            # Case 8: 2 miners 100 events each
            (
                [
                    ScoresModel(
                        event_id=f"expected_event_id_{i}",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.1 if i % 2 == 0 else 0.9,
                        event_score=0.80,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    )
                    for i in range(100)
                ]
                + [
                    ScoresModel(
                        event_id=f"expected_event_id_{i}",
                        miner_uid=4,
                        miner_hotkey="hk4",
                        prediction=0.1 if i % 2 == 0 else 0.9,
                        event_score=0.40,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    )
                    for i in range(100)
                ],
                [
                    {
                        "event_id": f"expected_event_id_{i}",
                        "processed": 1,
                        "metagraph_score": 0.8,
                        "other_data": {
                            "sum_weighted_peer_score": 1.6 * (i + 1),
                            "sum_weight": 2.0 * (i + 1),
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": i + 1,
                            "avg_peer_score": 0.8,
                            "sqmax_avg_peer_score": 0.64,
                        },
                    }
                    for i in range(100)
                ]
                + [
                    {
                        "event_id": f"expected_event_id_{i}",
                        "processed": 1,
                        "metagraph_score": 0.2,
                        "other_data": {
                            "sum_weighted_peer_score": 0.8 * (i + 1),
                            "sum_weight": 2.0 * (i + 1),
                            "count_peer_score": MOVING_AVERAGE_EVENTS + 1,
                            "true_count_peer_score": i + 1,
                            "avg_peer_score": 0.4,
                            "sqmax_avg_peer_score": 0.16,
                        },
                    }
                    for i in range(100)
                ],
                {
                    "debug": [
                        (
                            "Found events to calculate metagraph scores.",
                            {"n_events": 100},
                        ),
                    ],
                },
            ),
        ],
    )
    async def test_run(
        self,
        metagraph_scoring_task: MetagraphScoring,
        scores_list,
        expected_result,
        log_calls,
        db_client,
        db_operations,
    ):
        # insert EVENTS that match the score rows
        if scores_list:
            uniq_event_ids = {s.event_id for s in scores_list}

            stub_events = []
            for eid in uniq_event_ids:
                # simple heuristic: prediction > 0.5 -> outcome "1", else "0"
                some_score = next(s for s in scores_list if s.event_id == eid)
                outcome = "1" if some_score.prediction >= 0.5 else "0"

                stub_events.append(
                    EventsModel(
                        unique_event_id=f"stub_{eid}",
                        event_id=eid,
                        market_type="unit_test",
                        event_type="unit_test",
                        description=f"stub for {eid}",
                        outcome=outcome,
                        status=EventStatus.SETTLED,
                        metadata="{}",
                        created_at="2100-01-01T00:00:00+00:00",
                        cutoff="2100-01-01T00:00:00+00:00",
                    )
                )

            await db_operations.upsert_events(stub_events)

        # insert scores
        sql = f"""
            INSERT INTO scores ({', '.join(SCORE_FIELDS)})
            VALUES ({', '.join(['?'] * len(SCORE_FIELDS))})
        """
        if scores_list:
            score_tuples = [
                tuple(getattr(score, field) for field in SCORE_FIELDS) for score in scores_list
            ]
            await db_client.insert_many(sql, score_tuples)

            # confirm setup
            inserted_scores = await db_client.many("SELECT * FROM scores")
            assert len(inserted_scores) == len(scores_list)

        # run the task
        if "exception" in log_calls:
            # Mock the set_metagraph_peer_scores method to return non empty list
            db_operations.set_metagraph_peer_scores = AsyncMock(return_value=[100])

        await metagraph_scoring_task.run()
        updated_scores = await db_client.many("SELECT * FROM scores", use_row_factory=True)
        debug_calls = metagraph_scoring_task.logger.debug.call_args_list
        exception_calls = metagraph_scoring_task.logger.exception.call_args_list

        # confirm logs
        for log_type, calls in log_calls.items():
            for i, (args, kwargs) in enumerate(calls):
                if log_type == "debug":
                    assert debug_calls[i][0][0] == args
                    if kwargs:
                        assert debug_calls[i][1]["extra"] == kwargs
                elif log_type == "exception":
                    assert exception_calls[i][0][0] == args
                    assert exception_calls[i][1]["extra"] == kwargs

        # confirm results
        if expected_result:
            assert len(updated_scores) == len(expected_result)
            for i, expected in enumerate(expected_result):
                updated = updated_scores[i]
                assert updated["event_id"] == expected["event_id"]
                assert updated["processed"] == expected["processed"]
                assert updated["metagraph_score"] == pytest.approx(
                    expected["metagraph_score"], abs=1e-3
                )

                other_data = json.loads(updated["other_data"])
                # quite complex to calculate weighted values for all i
                if len(updated_scores) < 20 or (
                    len(updated_scores) >= 20 and (i > 30 and i < 100 or i > 130)
                ):
                    assert other_data["sum_weighted_peer_score"] == pytest.approx(
                        expected["other_data"]["sum_weighted_peer_score"], rel=1e-3
                    )
                    assert other_data["sum_weight"] == pytest.approx(
                        expected["other_data"]["sum_weight"], rel=1e-3
                    )
                assert other_data["count_peer_score"] == expected["other_data"]["count_peer_score"]
                assert (
                    other_data["true_count_peer_score"]
                    == expected["other_data"]["true_count_peer_score"]
                )
                assert other_data["avg_peer_score"] == pytest.approx(
                    expected["other_data"]["avg_peer_score"], abs=1e-3
                )
                assert other_data["sqmax_avg_peer_score"] == pytest.approx(
                    expected["other_data"]["sqmax_avg_peer_score"], abs=1e-3
                )
        else:
            assert len(updated_scores) == len(scores_list)
            for i, updated in enumerate(updated_scores):
                assert updated["metagraph_score"] is None
                assert updated["other_data"] is None
