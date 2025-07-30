from datetime import datetime, timedelta

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.if_games.client import IfGamesClient
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class ResolveEvents(AbstractTask):
    interval: float
    api_client: IfGamesClient
    db_operations: DatabaseOperations
    page_size: int
    logger: InfiniteGamesLogger

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        api_client: IfGamesClient,
        page_size: int,
        logger: InfiniteGamesLogger,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        # Validate api_client
        if not isinstance(api_client, IfGamesClient):
            raise TypeError("api_client must be an instance of IfGamesClient.")

        # Validate page_size
        if not isinstance(page_size, int) or page_size <= 0 or page_size > 500:
            raise ValueError("page_size must be a positive integer.")

        # Validate logger
        if not isinstance(logger, InfiniteGamesLogger):
            raise TypeError("logger must be an instance of InfiniteGamesLogger.")

        self.interval = interval_seconds
        self.db_operations = db_operations
        self.api_client = api_client
        self.page_size = page_size
        self.logger = logger

    @property
    def name(self):
        return "resolve-events"

    @property
    def interval_seconds(self):
        return self.interval

    async def run(self):
        # Read last resolved from db
        resolved_since = await self.db_operations.get_events_last_resolved_at()

        if resolved_since is None:
            # Fall back to oldest pending event
            resolved_since = await self.db_operations.get_events_pending_first_created_at()

        if resolved_since is None:
            self.logger.debug("No events to resolve")

            return

        # Back track 1 hour for safety, avoid missing any backdated delete
        resolved_since = (
            (datetime.fromisoformat(resolved_since) - timedelta(hours=1))
            .isoformat()
            .replace("+00:00", "Z")
        )

        offset = 0

        while True:
            # Query resolved events in batches
            response = await self.api_client.get_resolved_events(
                resolved_since=resolved_since,
                offset=offset,
                limit=self.page_size,
            )

            resolved_events = response.items

            for event in resolved_events:
                event_id = event.event_id

                db_resolved_event = await self.db_operations.resolve_event(
                    event_id=event_id,
                    outcome=event.answer,
                    resolved_at=event.resolved_at,
                )

                if len(db_resolved_event) > 0:
                    self.logger.debug("Event resolved", extra={"event_id": event_id})

            if len(resolved_events) < self.page_size:
                # Break if no more events
                break

            offset += self.page_size
