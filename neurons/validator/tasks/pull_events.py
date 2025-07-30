import json
import math
from datetime import datetime, timedelta

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.if_games.client import IfGamesClient
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.models.if_games_client import IfGamesEvent
from neurons.validator.scheduler.task import AbstractTask

TITLE_SEPARATOR = " ==Further Information==: "


class PullEvents(AbstractTask):
    interval: float
    page_size: int
    api_client: IfGamesClient
    db_operations: DatabaseOperations

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        api_client: IfGamesClient,
        page_size: int,
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

        self.interval = interval_seconds
        self.db_operations = db_operations
        self.api_client = api_client
        self.page_size = page_size

    @property
    def name(self):
        return "pull-events"

    @property
    def interval_seconds(self):
        return self.interval

    async def run(self):
        # Pick up from where it left, get 'from' from latest db events
        events_from = await self.db_operations.get_last_event_from()

        if events_from:
            # Back track 1 second
            events_from = (datetime.fromisoformat(events_from) - timedelta(seconds=1)).timestamp()
            events_from = math.floor(events_from)
        else:
            events_from = 0

        offset = 0

        while True:
            # Query events in batches
            response = await self.api_client.get_events(events_from, offset, self.page_size)

            # Parse events
            events = response.items

            parsed_events_for_insertion = [(self.parse_event(e)) for e in events]

            # Batch insert in the db
            if len(parsed_events_for_insertion) > 0:
                await self.db_operations.upsert_events(events=parsed_events_for_insertion)

            if len(events) < self.page_size:
                # Break if no more events
                break

            offset += self.page_size

    def parse_event(self, event: IfGamesEvent):
        status = EventStatus.PENDING
        truncated_market_type = "ifgames"

        event_type = event.market_type.lower()
        metadata = event.event_metadata or {}

        return EventsModel(
            unique_event_id=f"{truncated_market_type}-{event.event_id}",
            event_id=event.event_id,
            market_type=truncated_market_type,
            event_type=event_type,
            description=event.title + TITLE_SEPARATOR + event.description,
            outcome=None,
            status=status,
            metadata=json.dumps(metadata),
            created_at=event.created_at,
            cutoff=event.cutoff,
        )
