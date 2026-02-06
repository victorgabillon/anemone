"""
This module defines the NodeSelectorType enumeration, which represents the types of node selectors.
"""

from enum import StrEnum


class NodeSelectorType(StrEnum):
    """
    Enumeration representing the types of node selectors.
    """

    RECUR_ZIPF_BASE = "RecurZipfBase"
    SEQUOOL = "Sequool"
    UNIFORM = "Uniform"
