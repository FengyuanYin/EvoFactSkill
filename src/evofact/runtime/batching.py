from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from math import ceil
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")
_MISSING = object()


async def _notify(callback, *args) -> None:
    if callback is None:
        return
    result = callback(*args)
    if inspect.isawaitable(result):
        await result


async def ordered_batched_map(
    items: Sequence[T],
    worker: Callable[[T], Awaitable[R]],
    *,
    batch_size: int,
    concurrency: int,
    on_item_done: Callable[[int, int], object] | None = None,
    on_batch_start: Callable[[int, int, int, int], object] | None = None,
    on_batch_done: Callable[[int, int, int, int], object] | None = None,
) -> list[R]:
    """Run ordered batches concurrently while preserving input result order."""
    if batch_size < 1 or concurrency < 1:
        raise ValueError("batch_size and concurrency must be positive")
    if not items:
        return []

    total = len(items)
    batch_count = ceil(total / batch_size)
    completed = 0
    output: list[R | object] = [_MISSING] * total

    for batch_number, start in enumerate(range(0, total, batch_size), start=1):
        stop = min(total, start + batch_size)
        await _notify(on_batch_start, batch_number, batch_count, start, stop)
        semaphore = asyncio.Semaphore(min(concurrency, stop - start))

        async def run_one(index: int) -> None:
            nonlocal completed
            async with semaphore:
                result = await worker(items[index])
            output[index] = result
            completed += 1
            await _notify(on_item_done, completed, total)

        tasks = [asyncio.create_task(run_one(index)) for index in range(start, stop)]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        await _notify(on_batch_done, batch_number, batch_count, start, stop)

    if any(item is _MISSING for item in output):
        raise RuntimeError("batched map completed without producing every result")
    return list(output)  # type: ignore[return-value]
