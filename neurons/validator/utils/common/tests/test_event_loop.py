import asyncio
from time import sleep

from neurons.validator.utils.common.event_loop import measure_event_loop_lag
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class TestEventLoop:
    async def test_no_lag_under_normal_conditions(self, mocked_if_logger: InfiniteGamesLogger):
        measuring_frequency_seconds = 0.01
        lag_threshold_seconds = 0.2

        task = asyncio.create_task(
            measure_event_loop_lag(
                measuring_frequency_seconds=measuring_frequency_seconds,
                lag_threshold_seconds=lag_threshold_seconds,
                logger=mocked_if_logger,
            )
        )

        # Let it run for a second
        await asyncio.sleep(0.5)
        task.cancel()
        await asyncio.sleep(0.1)

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Under normal conditions, we shouldn't see warnings
        assert mocked_if_logger.warning.call_count == 0

    async def test_lag_detected_with_blocking_operation(
        self, mocked_if_logger: InfiniteGamesLogger
    ):
        measuring_frequency_seconds = 0.01
        lag_threshold_seconds = 0.2

        def blocking_coroutine():
            sleep(1.1)

        task = asyncio.create_task(
            measure_event_loop_lag(
                measuring_frequency_seconds=measuring_frequency_seconds,
                lag_threshold_seconds=lag_threshold_seconds,
                logger=mocked_if_logger,
            )
        )
        # Let measure_event_loop_lag start
        await asyncio.sleep(0.1)

        blocking_coroutine()

        # Let measure_event_loop_lag finish measurement
        await asyncio.sleep(0.1)

        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have detected some lag
        warning_calls = mocked_if_logger.warning.call_args_list

        for call in warning_calls:
            args, kwargs = call
            assert args[0] == "Event loop lag breached threshold"
            assert "extra" in kwargs
            assert "lag_ms" in kwargs["extra"]
            assert isinstance(kwargs["extra"]["lag_ms"], (float))
            assert kwargs["extra"]["lag_ms"] > 0
