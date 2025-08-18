import asyncio

from neurons.validator.utils.logger.logger import InfiniteGamesLogger


async def measure_event_loop_lag(
    measuring_frequency_seconds: float, lag_threshold_seconds: float, logger: InfiniteGamesLogger
):
    while True:
        started_time = asyncio.get_event_loop().time()

        await asyncio.sleep(measuring_frequency_seconds)

        end_time = asyncio.get_event_loop().time()

        lag = end_time - started_time - measuring_frequency_seconds

        if lag >= lag_threshold_seconds:
            logger.warning(
                "Event loop lag breached threshold", extra={"lag_ms": round(lag * 1000, 0)}
            )
