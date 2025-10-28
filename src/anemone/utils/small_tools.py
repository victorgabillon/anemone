import os
import typing
from dataclasses import dataclass
from itertools import islice
from typing import Any

path = typing.Annotated[str | os.PathLike[str], "path"]


def nth_key[_T](dct: dict[_T, Any], n: int) -> _T:
    """
    Get the nth key from a dictionary.

    Args:
        dct: The dictionary.
        n: The index of the key to retrieve.

    Returns:
        The nth key from the dictionary.

    """
    it = iter(dct)
    # Consume n elements.
    next(islice(it, n, n), None)
    # Return the value at the current position.
    # This raises StopIteration if n is beyond the limits.
    # Use next(it, None) to suppress that exception.
    return next(it)


@dataclass
class Interval:
    """
    Represents an interval with a minimum and maximum value.
    """

    min_value: float | None = None
    max_value: float | None = None


def intersect_intervals(interval_1: Interval, interval_2: Interval) -> Interval | None:
    """
    Find the intersection of two intervals.

    Args:
        interval_1: The first interval.
        interval_2: The second interval.

    Returns:
        The intersection of the two intervals, or None if there is no intersection.

    Raises:
        AssertionError: If any of the intervals have missing values.

    """
    assert interval_1.max_value is not None and interval_1.min_value is not None
    assert interval_2.max_value is not None and interval_2.min_value is not None
    min_value: float = max(interval_1.min_value, interval_2.min_value)
    max_value: float = min(interval_1.max_value, interval_2.max_value)
    if max_value < min_value:
        return None
    else:
        interval_res = Interval(max_value=max_value, min_value=min_value)
        return interval_res


def distance_number_to_interval(value: float, interval: Interval) -> float:
    """
    Calculate the distance between a number and an interval.

    Args:
        value: The number.
        interval: The interval.

    Returns:
        The distance between the number and the interval.

    Raises:
        AssertionError: If the interval has missing values.

    """
    assert interval.max_value is not None and interval.min_value is not None
    if value < interval.min_value:
        return interval.min_value - value
    elif value > interval.max_value:
        return value - interval.max_value
    else:
        return 0
