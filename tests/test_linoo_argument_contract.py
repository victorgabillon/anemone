"""Public Linoo arguments remain usable by dataclass command-line parsers."""

import sys
from dataclasses import dataclass, field
from typing import get_args

import pytest
from parsley import create_parsley

from anemone.node_selector.linoo import LinooArgs, LinooDepthSelectionPolicy
from anemone.node_selector.node_selector_types import NodeSelectorType


@dataclass
class _SearchConfig:
    """Represent the nested defaulted selector config used by application parsers."""

    selector: LinooArgs = field(
        default_factory=lambda: LinooArgs(type=NodeSelectorType.LINOO)
    )


@pytest.mark.parametrize(
    "policy", ["inverse_depth", "opened_count_depth_index", "alternating_by_step"]
)
def test_linoo_policies_parse_from_command_line(
    policy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Supported policies parse without treating a type alias as a converter."""
    monkeypatch.setattr(
        sys, "argv", ["anemone", "--selector.depth_selection_policy", policy]
    )
    parser = create_parsley(_SearchConfig)
    result = parser.parse_arguments()
    assert result.selector.depth_selection_policy == policy
    assert result.selector.type is NodeSelectorType.LINOO


def test_linoo_default_remains_unchanged_with_optional_alternating_policy() -> None:
    """The optional policy extends the vocabulary without changing defaults."""
    args = LinooArgs(type=NodeSelectorType.LINOO)
    assert args.depth_selection_policy == "inverse_depth"
    assert get_args(LinooDepthSelectionPolicy.__value__) == (
        "inverse_depth",
        "opened_count_depth_index",
        "alternating_by_step",
    )
    parser = create_parsley(_SearchConfig, should_parse_command_line_arguments=False)
    assert parser.parse_arguments().selector == args
