from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.logger.logger import InfiniteGamesLogger

# how many previous events to consider for the moving average
MOVING_AVERAGE_EVENTS = 149


class MetagraphScoringAlternative(AbstractTask):
    interval: float
    page_size: int
    db_operations: DatabaseOperations
    logger: InfiniteGamesLogger

    def __init__(
        self,
        interval_seconds: float,
        page_size: int,
        db_operations: DatabaseOperations,
        logger: InfiniteGamesLogger,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        # Validate page_size
        if not isinstance(page_size, int) or page_size <= 0 or page_size > 1000:
            raise ValueError("page_size must be a positive integer.")

        # Validate logger
        if not isinstance(logger, InfiniteGamesLogger):
            raise TypeError("logger must be an instance of InfiniteGamesLogger.")

        self.interval = interval_seconds
        self.page_size = page_size
        self.db_operations = db_operations
        self.logger = logger

    @property
    def name(self):
        return "metagraph-scoring-alternative"

    @property
    def interval_seconds(self):
        return self.interval

    async def run(self):
        self.logger.add_context({"source_task": self.name})

        events_to_score = await self.db_operations.get_events_for_alternative_metagraph_scoring(
            max_events=self.page_size
        )
        if not events_to_score:
            self.logger.debug("No events to calculate metagraph scores.")

            return

        self.logger.debug(
            "Found events to calculate metagraph scores.",
            extra={"n_events": len(events_to_score)},
        )

        for event in events_to_score:
            self.logger.debug(
                "Processing event for metagraph scoring.",
                extra={"event_id": event["event_id"]},
            )

            #  Calculate metagraph score

            self.logger.debug(
                "Metagraph scores calculated successfully.",
                extra={"event_id": event["event_id"]},
            )

            # Mark peer score as processed
