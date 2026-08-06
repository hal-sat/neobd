"""Small process-parallel execution helper."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

Input = TypeVar("Input")
Output = TypeVar("Output")


def ordered_parallel_map(
    function: Callable[[Input], Output],
    tasks: Iterable[Input],
    n_para: int,
) -> list[Output]:
    """Map tasks in input order, using processes only when useful."""
    task_list = list(tasks)
    if n_para < 1:
        raise ValueError("n_para must be at least one")
    if n_para == 1 or len(task_list) <= 1:
        return [function(task) for task in task_list]
    with ProcessPoolExecutor(max_workers=n_para) as executor:
        return list(executor.map(function, task_list, chunksize=1))
