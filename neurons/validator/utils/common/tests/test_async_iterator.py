import asyncio

from neurons.validator.utils.common.async_iterator import async_iterator


class TestAsyncIterator:
    async def test_async_iterator_yields_all_items(self):
        items = [1, 2, 3, 4]

        result = []

        async for x in async_iterator(items):
            result.append(x)

        assert result == items

    async def test_async_iterator_empty_list(self):
        items = []

        result = []

        async for x in async_iterator(items):
            result.append(x)

        assert result == []

    async def test_async_iterator_allows_other_tasks(self):
        # Track if another task got a chance to run
        ran = asyncio.Event()

        async def other_task():
            await asyncio.sleep(0)
            ran.set()

        # Start the other task in background
        asyncio.create_task(other_task())

        items = {"key": "value"}

        result = []

        async for x in async_iterator(items.values()):
            result.append(x)

        # Wait a tick so event has a chance to be set
        await asyncio.sleep(0)

        assert result == ["value"]

        assert ran.is_set()

    async def test_async_iterator_sleeps_zero(self, monkeypatch):
        called = {}

        async def fake_sleep(delay: float):
            called["delay"] = delay

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        items = ["a"]

        result = []

        async for x in async_iterator(items):
            result.append(x)

        assert result == items
        # Ensure asyncio.sleep was called with the exact delay
        assert called["delay"] == 0.001
