"""
Module for sorting a dictionary by ascending order
"""

from typing import Any, Protocol


class _Sortable(Protocol):
        def __lt__(self, other: Any, /) -> bool: ...


def sort_dic[K, V: _Sortable](dic: dict[K, V]) -> dict[K, V]:
    """
    Sorts a dictionary by ascending order of values.

    Args:
            dic (dict[K, V]): The dictionary to be sorted.

    Returns:
            dict[K, V]: The sorted dictionary.
    """
    z = dic.items()
    a = sorted(z, key=lambda item: item[1])
    sorted_dic = dict(a)
    return sorted_dic
