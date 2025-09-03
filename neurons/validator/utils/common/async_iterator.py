import asyncio
from typing import AsyncIterator, Iterable, TypeVar

T = TypeVar("T")


async def async_iterator(items: Iterable[T]) -> AsyncIterator[T]:
    for item in items:
        yield item

        await asyncio.sleep(0.001)
