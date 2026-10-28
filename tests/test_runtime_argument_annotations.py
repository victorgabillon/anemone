"""Public Anemone argument dataclasses must support runtime configuration parsing."""

from dataclasses import is_dataclass
from typing import get_type_hints

import pytest
from parsley import make_partial_dataclass_with_optional_paths

from anemone import TreeAndValuePlayerArgs
from anemone.node_selector.composed.args import ComposedNodeSelectorArgs
from anemone.node_selector.priority_check.noop_args import NoPriorityCheckArgs
from anemone.node_selector.priority_check.registered_args import (
    RegisteredPriorityCheckArgs,
)
from anemone.node_selector.recurzipf.recur_zipf_base import RecurZipfBaseArgs
from anemone.node_selector.sequool.factory import SequoolArgs
from anemone.node_selector.uniform.uniform import UniformArgs


@pytest.mark.parametrize(
    "args_type",
    [
        ComposedNodeSelectorArgs,
        NoPriorityCheckArgs,
        RegisteredPriorityCheckArgs,
        RecurZipfBaseArgs,
        SequoolArgs,
        UniformArgs,
    ],
)
def test_public_argument_annotations_resolve(args_type: type) -> None:
    """Fail when an annotation name is available only during type checking."""
    assert get_type_hints(args_type)


def test_parsley_can_build_partial_player_configuration() -> None:
    """Exercise the public configuration parser through every nested argument union."""
    assert is_dataclass(
        make_partial_dataclass_with_optional_paths(TreeAndValuePlayerArgs)
    )
