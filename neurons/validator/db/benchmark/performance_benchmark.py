import asyncio
import logging

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.utils.logger.logger import create_logger


async def main():
    logger = create_logger("performance_test")
    logger.setLevel(logging.ERROR)

    db_client = DatabaseClient(db_path="performance_test.db", logger=logger)
    db_operations = DatabaseOperations(db_client=db_client, logger=logger)

    await db_client.migrate()
    await db_client.delete("DELETE FROM events")

    i = 0
    try:
        while True:
            tasks = []

            for _ in range(100):
                i += 1

                event_id = f"event_id_{i}"

                event = EventsModel(
                    unique_event_id=event_id,
                    event_id=event_id,
                    market_type="truncated_market1",
                    event_type="market_1",
                    description="desc1",
                    outcome="outcome1",
                    status=EventStatus.PENDING,
                    metadata='{"key": "value"}',
                    created_at="2000-12-02T14:30:00+00:00",
                    cutoff="2000-01-01T14:30:00+00:00",
                )

                tasks.append(asyncio.create_task(db_operations.upsert_events(events=[event])))

            await asyncio.gather(*tasks)

    except BaseException as e:
        result = await db_client.one("SELECT COUNT(*) FROM events")
        logger.setLevel(logging.INFO)

        logger.info(f"Events inserted: {result[0]}", extra={"error_message": str(e)})

    finally:
        if callable(getattr(db_client, "close", None)):
            await db_client.close()


if __name__ == "__main__":
    asyncio.run(main())
