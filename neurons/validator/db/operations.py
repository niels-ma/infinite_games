from pathlib import Path
from typing import Iterable, Type, TypeVar

from pydantic import BaseModel

from neurons.validator.db.client import DatabaseClient
from neurons.validator.models.event import EVENTS_FIELDS, EventsModel, EventStatus
from neurons.validator.models.miner import MINERS_FIELDS, MinersModel
from neurons.validator.models.prediction import (
    PREDICTION_FIELDS,
    PredictionExportedStatus,
    PredictionsModel,
)
from neurons.validator.models.reasoning import REASONING_FIELDS, ReasoningModel
from neurons.validator.models.score import (
    SCORE_FIELDS,
    AlternativeMetagraphScore,
    ScoresExportedStatus,
    ScoresModel,
)
from neurons.validator.utils.logger.logger import InfiniteGamesLogger

SQL_FOLDER = Path(Path(__file__).parent, "sql")


GenericModel = TypeVar("GenericModel", bound=BaseModel)


class DatabaseOperations:
    __db_client: DatabaseClient
    logger: InfiniteGamesLogger

    def __init__(self, db_client: DatabaseClient, logger: InfiniteGamesLogger):
        if not isinstance(db_client, DatabaseClient):
            raise ValueError("Invalid db_client arg")

        if not isinstance(logger, InfiniteGamesLogger):
            raise TypeError("logger must be an instance of InfiniteGamesLogger.")

        self.__db_client = db_client
        self.logger = logger

    def _parse_rows(
        self, model: Type[GenericModel], rows: Iterable[tuple], throw_on_error: bool = False
    ) -> list[GenericModel]:
        parsed_rows = []

        for row in rows:
            try:
                parsed_rows.append(model(**dict(row)))
            except Exception as e:
                if throw_on_error:
                    raise e

                self.logger.exception(
                    "Error parsing model", extra={"row": row, "model": model.__name__}
                )

        return parsed_rows

    async def delete_event(self, event_id: str, deleted_at: str) -> Iterable[tuple[str]]:
        return await self.__db_client.update(
            """
                UPDATE
                    events
                SET
                    status = ?,
                    deleted_at = ?,
                    local_updated_at = CURRENT_TIMESTAMP
                WHERE event_id = ?
                RETURNING event_id
            """,
            [EventStatus.DELETED, deleted_at, event_id],
        )

    async def delete_events_hard_delete(self, batch_size: int) -> Iterable[tuple[str]]:
        return await self.__db_client.delete(
            """
            WITH events_to_delete AS (
                SELECT
                    ROWID
                FROM
                    events
                WHERE
                    status = ?
                    AND datetime(deleted_at) < datetime(CURRENT_TIMESTAMP, '-14 day')
                ORDER BY
                    ROWID ASC
                LIMIT ?
            )
            DELETE FROM
                events
            WHERE
                ROWID IN (
                    SELECT
                        ROWID
                    FROM
                        events_to_delete
                )
            RETURNING
                ROWID
            """,
            [EventStatus.DELETED, batch_size],
        )

    async def delete_predictions(self, batch_size: int) -> Iterable[tuple[int]]:
        return await self.__db_client.delete(
            """
                WITH predictions_to_delete AS (
                    SELECT
                        p.ROWID
                    FROM
                        predictions p
                    JOIN
                        events e ON e.unique_event_id = p.unique_event_id
                    WHERE
                        p.exported = ?

                    AND (
                        -- Predictions for processed events and older than X
                        (
                            e.processed = TRUE
                            AND datetime(e.resolved_at) < datetime(CURRENT_TIMESTAMP, '-4 day')
                        )

                        -- Predictions for discarded or deleted events
                        OR  (
                            e.status IN (?, ?)
                        )
                    )
                    ORDER BY
                        p.ROWID ASC
                    LIMIT ?
                )
                DELETE FROM
                    predictions
                WHERE
                    ROWID IN (
                        SELECT
                            ROWID
                        FROM
                            predictions_to_delete
                    )
                RETURNING
                    ROWID
            """,
            [
                PredictionExportedStatus.EXPORTED,
                EventStatus.DISCARDED,
                EventStatus.DELETED,
                batch_size,
            ],
        )

    async def delete_scores(self, batch_size: int) -> Iterable[tuple[int]]:
        return await self.__db_client.delete(
            """
                WITH scores_to_delete AS (
                    SELECT
                        s.ROWID
                    FROM
                        scores s
                    LEFT JOIN
                        events e ON s.event_id = e.event_id
                    WHERE
                        -- Orphan scores
                        e.event_id IS NULL

                        -- Scores for processed events older than X
                        OR (
                            e.processed = TRUE
                            AND datetime(e.resolved_at) < datetime(CURRENT_TIMESTAMP, '-15 day')
                            AND s.exported = ?
                        )

                        -- Scores for discarded events
                        OR (
                            e.status = ?
                            AND s.exported = ?
                        )

                        -- Scores for deleted events
                        OR (
                            e.status = ?
                        )
                    ORDER BY
                        s.ROWID ASC
                    LIMIT ?
                )
                DELETE FROM
                    scores
                WHERE
                    ROWID IN (
                        SELECT
                            ROWID
                        FROM
                            scores_to_delete
                    )
                RETURNING
                    ROWID
            """,
            [
                ScoresExportedStatus.EXPORTED,
                EventStatus.DISCARDED,
                ScoresExportedStatus.EXPORTED,
                EventStatus.DELETED,
                batch_size,
            ],
        )

    async def delete_reasonings(self, batch_size: int) -> Iterable[tuple[int]]:
        return await self.__db_client.delete(
            """
                WITH reasonings_to_delete AS (
                    SELECT
                        r.ROWID
                    FROM
                        reasoning r
                    LEFT JOIN
                        events e ON r.event_id = e.event_id
                    WHERE
                        -- Orphan reasonings
                        e.event_id IS NULL

                        -- Reasonings for resolved events older than 7 days
                        OR (
                            e.processed = TRUE
                            AND datetime(e.resolved_at) < datetime(CURRENT_TIMESTAMP, '-7 day')
                        )

                        -- Reasonings for discarded or deleted events
                        OR (
                            e.status IN (?, ?)
                        )
                    ORDER BY
                        r.ROWID ASC
                    LIMIT ?
                )
                DELETE FROM
                    reasoning
                WHERE
                    ROWID IN (
                        SELECT
                            ROWID
                        FROM
                            reasonings_to_delete
                    )
                RETURNING
                    ROWID
            """,
            [
                EventStatus.DISCARDED,
                EventStatus.DELETED,
                batch_size,
            ],
        )

    async def get_event(self, unique_event_id: str) -> None | EventsModel:
        result = await self.__db_client.one(
            f"""
                SELECT
                    {', '.join(EVENTS_FIELDS)}
                FROM events
                WHERE
                    unique_event_id = ?
            """,
            parameters=[unique_event_id],
            use_row_factory=True,
        )

        if result is None:
            return None

        return EventsModel(**dict(result))

    async def get_events(self, unique_event_ids) -> list[EventsModel]:
        placeholders = ", ".join(["?"] * len(unique_event_ids))

        rows = await self.__db_client.many(
            f"""
                SELECT
                    {', '.join(EVENTS_FIELDS)}
                FROM
                    events
                WHERE
                    unique_event_id IN ({placeholders})
            """,
            parameters=unique_event_ids,
            use_row_factory=True,
        )

        events = self._parse_rows(model=EventsModel, rows=rows)

        return events

    async def get_events_last_deleted_at(self) -> str | None:
        row = await self.__db_client.one(
            """
                SELECT MAX(deleted_at) FROM events
            """
        )

        if row is not None:
            return row[0]

    async def get_events_last_resolved_at(self) -> str | None:
        row = await self.__db_client.one(
            """
                SELECT MAX(resolved_at) FROM events
            """
        )

        if row is not None:
            return row[0]

    async def get_events_pending_first_created_at(self) -> str | None:
        row = await self.__db_client.one(
            """
                SELECT MIN(created_at) FROM events WHERE status = ?
            """,
            [EventStatus.PENDING],
        )

        if row is not None:
            return row[0]

    async def get_events_to_predict(self) -> Iterable[tuple[str]]:
        return await self.__db_client.many(
            """
                SELECT
                    event_id,
                    market_type,
                    event_type,
                    description,
                    cutoff,
                    metadata
                FROM
                    events
                WHERE
                    status = ?
                    AND datetime(CURRENT_TIMESTAMP) < datetime(cutoff)
            """,
            parameters=[EventStatus.PENDING],
        )

    async def get_last_event_from(self) -> str | None:
        row = await self.__db_client.one(
            """
                SELECT MAX(created_at) FROM events
            """
        )

        if row is not None:
            return row[0]

    async def get_miners_count(self) -> int:
        row = await self.__db_client.one(
            """
                SELECT COUNT(*) FROM miners
            """
        )

        return row[0]

    async def get_predictions_to_export(self, current_interval_minutes: int, batch_size: int):
        return await self.__db_client.many(
            """
                SELECT
                    p.ROWID,
                    p.unique_event_id,
                    p.miner_uid,
                    p.miner_hotkey,
                    e.event_type,
                    p.latest_prediction,
                    p.interval_start_minutes,
                    p.interval_agg_prediction,
                    p.interval_count,
                    p.submitted
                FROM
                    predictions p
                JOIN
                    events e ON e.unique_event_id = p.unique_event_id
                WHERE
                    p.exported = ?
                    AND p.interval_start_minutes < ?
                ORDER BY
                    p.ROWID ASC
                LIMIT
                    ?
            """,
            [PredictionExportedStatus.NOT_EXPORTED, current_interval_minutes, batch_size],
        )

    async def mark_predictions_as_exported(self, ids: list[str]):
        placeholders = ", ".join(["?"] * len(ids))

        return await self.__db_client.update(
            f"""
                UPDATE
                    predictions
                SET
                    exported = ?
                WHERE
                    ROWID IN ({placeholders})
                RETURNING
                    ROWID
            """,
            [PredictionExportedStatus.EXPORTED] + ids,
        )

    async def get_predictions_ranked(self, moving_window: int):
        return await self.__db_client.many(
            """
                WITH events_pool AS (
                    SELECT
                        event_id,
                        MIN(ROWID) AS event_min_row
                    FROM
                        scores
                    GROUP BY
                        event_id
                ),
                ranked_events AS (
                    SELECT
                        event_id,
                        RANK() OVER (ORDER BY event_min_row DESC) AS event_rank
                    FROM
                        events_pool
                )
                SELECT
                    rev.event_id,
                    rev.event_rank,
                    ev.outcome,
                    sc.miner_uid,
                    sc.miner_hotkey,
                    sc.prediction
                FROM
                    ranked_events rev
                JOIN scores sc USING(event_id)
                JOIN events ev USING(event_id)
                WHERE
                    rev.event_rank <= ?
                ORDER BY
                    rev.event_rank ASC, sc.miner_uid ASC, sc.miner_hotkey ASC
            """,
            [moving_window],
        )

    async def resolve_event(
        self, event_id: str, outcome: str, resolved_at: str, forecasts: str
    ) -> Iterable[tuple[str]]:
        return await self.__db_client.update(
            """
                UPDATE
                    events
                SET
                    status = ?,
                    outcome = ?,
                    resolved_at = ?,
                    forecasts = ?,
                    local_updated_at = CURRENT_TIMESTAMP
                WHERE
                    event_id = ?
                    AND status = ?
                RETURNING
                    event_id
            """,
            [EventStatus.SETTLED, outcome, resolved_at, forecasts, event_id, EventStatus.PENDING],
        )

    async def update_alternative_metagraph_scores(
        self, alternative_metagraph_scores: list[AlternativeMetagraphScore]
    ):
        values = ",\n ".join(
            f"({row.miner_uid}, '{row.miner_hotkey}', {row.alternative_metagraph_score if row.alternative_metagraph_score is not None else 'NULL'}, '{row.alternative_other_data}')"
            for row in alternative_metagraph_scores
        )

        sql = f"""
            WITH mapping(miner_uid, miner_hotkey, alternative_metagraph_score, alternative_other_data) AS (
                VALUES {values}
            )
            UPDATE
                scores AS s
            SET
                alternative_metagraph_score = m.alternative_metagraph_score,
                alternative_other_data  = m.alternative_other_data
            FROM
                mapping AS m
            WHERE
                m.miner_uid = s.miner_uid
                AND m.miner_hotkey = s.miner_hotkey
                AND s.alternative_processed = FALSE
        """

        return await self.__db_client.update(sql=sql)

    async def upsert_miners(self, miners: list[list[any]]) -> None:
        return await self.__db_client.insert_many(
            """
                INSERT INTO miners
                    (
                        miner_uid,
                        miner_hotkey,
                        node_ip,
                        registered_date,
                        last_updated,
                        blocktime,
                        blocklisted,
                        is_validating,
                        validator_permit
                    )
                VALUES
                    (
                        ?,
                        ?,
                        ?,
                        ?,
                        CURRENT_TIMESTAMP,
                        ?,
                        FALSE,
                        ?,
                        ?
                    )
                ON CONFLICT
                    (miner_hotkey, miner_uid)
                DO UPDATE
                    set node_ip = ?,
                    last_updated = CURRENT_TIMESTAMP,
                    blocktime = ?,
                    is_validating = excluded.is_validating,
                    validator_permit = excluded.validator_permit
            """,
            miners,
        )

    async def upsert_predictions(self, predictions: list[list[any]]):
        await self.__db_client.insert_many(
            """
                    INSERT INTO predictions (
                        unique_event_id,
                        miner_hotkey,
                        miner_uid,
                        latest_prediction,
                        interval_start_minutes,
                        interval_agg_prediction,
                        interval_count
                    )
                    VALUES (
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        1
                    )
                    ON CONFLICT(unique_event_id, miner_uid, miner_hotkey, interval_start_minutes)
                    DO UPDATE SET
                        latest_prediction = excluded.latest_prediction,
                        interval_agg_prediction = (interval_agg_prediction * interval_count + excluded.interval_agg_prediction) / (interval_count + 1),
                        interval_count = interval_count + 1,
                        updated_at = CURRENT_TIMESTAMP
                """,
            predictions,
        )

    async def upsert_events(self, events: list[EventsModel]) -> None:
        """Upsert a list of EventsModel objects into the database"""

        fields_to_insert = [
            field_name
            for field_name in EVENTS_FIELDS
            if field_name not in ("registered_date", "local_updated_at")
        ]
        placeholders = ", ".join(["?"] * len(fields_to_insert))
        columns = ", ".join(fields_to_insert)

        # Convert each event into a tuple of values in the same order as fields_to_insert
        event_tuples = [
            tuple(getattr(event, field_name) for field_name in fields_to_insert) for event in events
        ]

        sql = f"""
                INSERT INTO events
                    ({columns}, registered_date, local_updated_at)
                VALUES
                    ({placeholders}, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT
                    (unique_event_id)
                DO NOTHING
        """
        return await self.__db_client.insert_many(
            sql=sql,
            parameters=event_tuples,
        )

    async def upsert_reasonings(self, reasonings: list[ReasoningModel]) -> None:
        """Upsert a list of ReasoningModel objects into the database"""

        fields_to_insert = [
            field_name
            for field_name in REASONING_FIELDS
            if field_name not in ("created_at", "updated_at")
        ]
        placeholders = ", ".join(["?"] * len(fields_to_insert))
        columns = ", ".join(fields_to_insert)

        # Convert each reasoning into a tuple of values in the same order as fields_to_insert
        reasoning_tuples = [
            tuple(getattr(reasoning, field_name) for field_name in fields_to_insert)
            for reasoning in reasonings
        ]

        sql = f"""
                INSERT INTO reasoning
                    ({columns}, created_at, updated_at)
                VALUES
                    ({placeholders}, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT
                    (event_id, miner_uid, miner_hotkey)
                DO UPDATE SET
                    reasoning = excluded.reasoning,
                    updated_at = CURRENT_TIMESTAMP
        """
        return await self.__db_client.insert_many(
            sql=sql,
            parameters=reasoning_tuples,
        )

    async def get_events_for_scoring(self, max_events=1000) -> list[EventsModel]:
        """
        Returns all events that were recently resolved and need to be scored
        """

        rows = await self.__db_client.many(
            f"""
                SELECT
                    {', '.join(EVENTS_FIELDS)}
                FROM events
                WHERE status = ?
                    AND outcome IS NOT NULL
                    AND processed = false
                ORDER BY resolved_at ASC
                LIMIT ?
            """,
            parameters=[EventStatus.SETTLED, max_events],
            use_row_factory=True,
        )

        events = self._parse_rows(model=EventsModel, rows=rows)

        return events

    async def get_predictions_for_event(
        self, unique_event_id: str, interval_start_minutes: int
    ) -> list[PredictionsModel]:
        rows = await self.__db_client.many(
            f"""
                SELECT
                    {', '.join(PREDICTION_FIELDS)}
                FROM
                    predictions
                WHERE
                    unique_event_id = ?
                    AND interval_start_minutes = ?
                ORDER BY
                    miner_uid ASC,
                    miner_hotkey ASC
            """,
            parameters=[unique_event_id, interval_start_minutes],
            use_row_factory=True,
        )

        predictions = self._parse_rows(model=PredictionsModel, rows=rows)

        return predictions

    async def get_predictions_for_scoring(self, unique_event_id: str) -> list[PredictionsModel]:
        rows = await self.__db_client.many(
            f"""
                SELECT
                    {', '.join(PREDICTION_FIELDS)}
                FROM predictions
                WHERE unique_event_id = ?
            """,
            parameters=(unique_event_id,),
            use_row_factory=True,
        )

        predictions = self._parse_rows(model=PredictionsModel, rows=rows)

        return predictions

    async def get_miners_last_registration(self) -> list:
        rows = await self.__db_client.many(
            f"""
                WITH ranked AS (
                    SELECT
                        {', '.join(MINERS_FIELDS)},
                        ROW_NUMBER() OVER (
                            PARTITION BY miner_uid
                            ORDER BY registered_date DESC
                        ) AS rn
                    FROM miners t
                )
                SELECT
                    {', '.join(MINERS_FIELDS)}
                FROM ranked
                WHERE rn = 1
                ORDER BY miner_uid
            """,
            use_row_factory=True,
        )

        miners = self._parse_rows(model=MinersModel, rows=rows)

        return miners

    async def mark_event_as_processed(self, unique_event_id: str) -> None:
        return await self.__db_client.update(
            """
                UPDATE events
                SET processed = true
                WHERE unique_event_id = ?
            """,
            parameters=(unique_event_id,),
        )

    async def mark_event_as_exported(self, unique_event_id: str) -> None:
        return await self.__db_client.update(
            """
                UPDATE events
                SET exported = true
                WHERE unique_event_id = ?
            """,
            parameters=(unique_event_id,),
        )

    async def mark_event_as_discarded(self, unique_event_id: str) -> None:
        """For resolved events which cannot be scored"""
        return await self.__db_client.update(
            """
                UPDATE events
                SET status = ?
                WHERE unique_event_id = ?
            """,
            parameters=[EventStatus.DISCARDED, unique_event_id],
        )

    async def insert_peer_scores(self, scores: list[ScoresModel]) -> None:
        """Insert raw peer scores into the scores table"""

        fields_to_insert = [
            "event_id",
            "miner_uid",
            "miner_hotkey",
            "prediction",
            "event_score",
            "spec_version",
        ]
        placeholders = ", ".join("?" for _ in fields_to_insert)
        columns = ", ".join(fields_to_insert)

        # Convert each event into a tuple of values in the same order as fields_to_insert
        score_tuples = [
            tuple(getattr(score, field_name) for field_name in fields_to_insert) for score in scores
        ]

        sql = f"""
                INSERT INTO scores ({columns})
                VALUES ({placeholders})
                ON CONFLICT
                    (event_id, miner_uid, miner_hotkey)
                DO UPDATE SET
                    prediction = excluded.prediction,
                    event_score = excluded.event_score,
                    spec_version = excluded.spec_version
        """
        return await self.__db_client.insert_many(
            sql=sql,
            parameters=score_tuples,
        )

    async def get_events_for_metagraph_scoring(self, max_events: int = 1000) -> list[dict]:
        """
        Returns all events that were recently peer scored and not processed.
        These events need to be ordered by row_id for the moving average calculation.
        """

        rows = await self.__db_client.many(
            """
                SELECT
                    event_id,
                    MIN(ROWID) AS min_row_id
                FROM scores
                WHERE processed = false
                GROUP BY event_id
                ORDER BY min_row_id ASC
                LIMIT ?
            """,
            use_row_factory=True,
            parameters=[
                max_events,
            ],
        )

        events = []
        for row in rows:
            try:
                event = dict(row)
                events.append(event)
            except Exception:
                self.logger.exception("Error parsing event", extra={"row": row})
        return events

    async def count_events_for_alternative_metagraph_scoring(self) -> int:
        row = await self.__db_client.one(
            """
                SELECT
                    COUNT(DISTINCT event_id)
                FROM
                    scores
                WHERE
                    alternative_processed = False
            """
        )

        return row[0] if row is not None else 0

    async def set_metagraph_peer_scores(self, event_id: str, n_events: int) -> list:
        """
        Calculate the moving average of peer scores for a given event
        """
        raw_sql = Path(SQL_FOLDER, "metagraph_peer_score.sql").read_text()
        updated = await self.__db_client.update(
            raw_sql,
            parameters={"event_id": event_id, "n_events": n_events},
        )

        return updated

    async def get_peer_scored_events_for_export(self, max_events: int = 1000) -> list[EventsModel]:
        """
        Get peer scored events that have not been exported
        """
        ev_event_fields = ["ev." + field for field in EVENTS_FIELDS]
        rows = await self.__db_client.many(
            f"""
                WITH events_to_export AS (
                    SELECT
                        event_id,
                        MIN(ROWID) AS min_row_id
                    FROM scores
                    WHERE processed = 1
                        AND exported = ?
                    GROUP BY event_id
                    ORDER BY min_row_id ASC
                    LIMIT ?
                )
                SELECT
                    {', '.join(ev_event_fields)}
                FROM events ev
                JOIN events_to_export ete ON ev.event_id = ete.event_id
                ORDER BY ete.min_row_id ASC
            """,
            use_row_factory=True,
            parameters=[
                ScoresExportedStatus.NOT_EXPORTED,
                max_events,
            ],
        )

        events = self._parse_rows(model=EventsModel, rows=rows)

        return events

    async def get_peer_scores_for_export(self, event_id: str) -> list:
        """
        Get peer scores for a given event
        Processed has to be true, to guarantee that metagraph score is set
        """
        rows = await self.__db_client.many(
            f"""
                SELECT
                    {', '.join(SCORE_FIELDS)}
                FROM scores
                WHERE event_id = ?
                    AND processed = 1
            """,
            parameters=[event_id],
            use_row_factory=True,
        )

        scores = self._parse_rows(model=ScoresModel, rows=rows)

        return scores

    async def mark_peer_scores_as_exported(self, event_id: str) -> list:
        """
        Mark peer scores from event_id as exported
        """
        return await self.__db_client.update(
            """
                UPDATE scores
                SET exported = ?
                WHERE event_id = ?
            """,
            parameters=(
                ScoresExportedStatus.EXPORTED,
                event_id,
            ),
        )

    async def mark_scores_as_alternative_processed_where_not_processed(self):
        """
        Mark all scores as alternative_processed where not processed
        """

        return await self.__db_client.update(
            """
                UPDATE
                    scores
                SET
                    alternative_processed = TRUE
                WHERE
                    alternative_processed = FALSE
            """,
        )

    async def get_last_metagraph_scores(self) -> list:
        """
        Returns the last known metagraph_score for each miner_uid, miner_hotkey;
        We cannot simply take from the last event - could be an old event scored now, so
        if the miner registered after the event cutoff, we will have no metagraph_score
        """
        rows = await self.__db_client.many(
            f"""
                WITH grouped AS (
                    SELECT miner_uid AS g_miner_uid,
                        miner_hotkey AS g_miner_hotkey,
                        MAX(ROWID) AS max_rowid
                    FROM scores
                    WHERE processed = 1
                        AND created_at > datetime(CURRENT_TIMESTAMP, '-10 day')
                    GROUP BY miner_uid, miner_hotkey
                )
                SELECT
                    {', '.join(SCORE_FIELDS)}
                FROM scores s
                JOIN grouped
                    ON s.miner_uid = grouped.g_miner_uid
                    AND s.miner_hotkey = grouped.g_miner_hotkey
                    AND s.ROWID = grouped.max_rowid
            """,
            use_row_factory=True,
        )

        scores = self._parse_rows(model=ScoresModel, rows=rows)

        return scores

    async def get_last_alternative_metagraph_scores(self) -> list:
        """
        Returns the last known metagraph_score for each miner_uid, miner_hotkey;
        We cannot simply take from the last event - could be an old event scored now, so
        if the miner registered after the event cutoff, we will have no metagraph_score
        """
        rows = await self.__db_client.many(
            f"""
                WITH grouped AS (
                    SELECT miner_uid AS g_miner_uid,
                        miner_hotkey AS g_miner_hotkey,
                        MAX(ROWID) AS max_rowid
                    FROM scores
                    WHERE alternative_processed = 1
                        AND created_at > datetime(CURRENT_TIMESTAMP, '-10 day')
                    GROUP BY miner_uid, miner_hotkey
                )
                SELECT
                    {', '.join(SCORE_FIELDS)}
                FROM scores s
                JOIN grouped
                    ON s.miner_uid = grouped.g_miner_uid
                    AND s.miner_hotkey = grouped.g_miner_hotkey
                    AND s.ROWID = grouped.max_rowid
            """,
            use_row_factory=True,
        )

        scores = self._parse_rows(model=ScoresModel, rows=rows)

        return scores

    async def vacuum_database(self, pages: int):
        await self.__db_client.script(f"PRAGMA incremental_vacuum({pages})")

    async def get_wa_predictions_events(
        self, unique_event_ids: list[str], interval_start_minutes: int
    ) -> dict[str, None | float]:
        """
        Retrieve the weighted average of the latest predictions for the given events
        """
        raw_sql_template = Path(SQL_FOLDER, "latest_predictions_events.sql").read_text()

        # Dynamically create named parameters for the IN clause
        unique_event_params = {
            f"unique_event_id_{i}": uid for i, uid in enumerate(unique_event_ids)
        }
        in_clause = ", ".join(f":unique_event_id_{i}" for i in range(len(unique_event_ids)))
        raw_sql = raw_sql_template.replace(":unique_event_ids", in_clause)

        rows = await self.__db_client.many(
            raw_sql,
            parameters={
                **unique_event_params,
                "interval_start_minutes": interval_start_minutes,
            },
        )

        return {
            unique_event_id: (
                float(weighted_avg_prediction) if weighted_avg_prediction is not None else None
            )
            for unique_event_id, weighted_avg_prediction in rows
        }

    async def get_community_train_dataset(self) -> list:
        """
        Retrieve the community train dataset
        """
        raw_sql = Path(SQL_FOLDER, "community_train_dataset.sql").read_text()

        rows = await self.__db_client.many(
            raw_sql,
            use_row_factory=True,
        )

        return rows

    async def save_community_predictions_model(
        self, name: str, model_blob: str, other_data_json: str | None = None
    ):
        """
        Save the community predictions model to the database
        """
        return await self.__db_client.insert(
            """
                INSERT INTO models (name, model_blob, other_data)
                VALUES (?, ?, ?)
            """,
            parameters=[name, model_blob, other_data_json],
        )

    async def get_community_predictions_model(self, name: str) -> None | str:
        """
        Retrieve the last community predictions model from the database
        """
        row = await self.__db_client.one(
            """
                SELECT model_blob
                FROM models
                WHERE name = ?
                ORDER BY created_at DESC
                LIMIT 1
            """,
            parameters=[name],
        )

        if row is not None:
            return row[0]

    async def get_community_inference_dataset(
        self, unique_event_ids: list, interval_start_minutes: int, top_n_ranks: int
    ) -> list:
        """
        Retrieve the community inference dataset
        """
        raw_sql_template = Path(SQL_FOLDER, "community_predict_events.sql").read_text()

        # Dynamically create named parameters for the IN clause
        unique_event_params = {
            f"unique_event_id_{i}": uid for i, uid in enumerate(unique_event_ids)
        }
        in_clause = ", ".join(f":unique_event_id_{i}" for i in range(len(unique_event_ids)))
        raw_sql = raw_sql_template.replace(":unique_event_ids", in_clause)

        rows = await self.__db_client.many(
            raw_sql,
            parameters={
                **unique_event_params,
                "interval_start_minutes": interval_start_minutes,
                "top_n_ranks": top_n_ranks,
            },
            use_row_factory=True,
        )

        return rows
