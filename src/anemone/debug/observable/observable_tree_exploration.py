"""Observable facade for tree exploration built from wrapped collaborators."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, is_dataclass, replace
from typing import TYPE_CHECKING, Any

from anemone.debug.events import SearchIterationCompleted, SearchIterationStarted
from anemone.debug.observable.observable_node_selector import ObservableNodeSelector
from anemone.debug.observable.observable_tree_manager import (
    ObservableAlgorithmNodeTreeManager,
)
from anemone.debug.sink import NullSearchDebugSink, SearchDebugSink

if TYPE_CHECKING:
    from random import Random


@dataclass
class _IterationState:
    """Track the latest observed iteration when delegating to core exploration."""

    started_iteration: int = 0
    completed_iteration: int = 0


class _IterationObservingStoppingCriterion:
    """Wrap a stopping criterion to emit iteration-start events."""

    def __init__(
        self,
        base: Any,
        *,
        debug_sink: SearchDebugSink,
        iteration_state: _IterationState,
    ) -> None:
        self._base = base
        self._debug_sink = debug_sink
        self._iteration_state = iteration_state

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped stopping criterion."""
        return getattr(self._base, name)

    def should_we_continue(self, *, tree: Any) -> bool:
        """Delegate the continuation check and emit an iteration-start event."""
        should_continue = bool(self._base.should_we_continue(tree=tree))
        if should_continue:
            self._iteration_state.started_iteration += 1
            self._debug_sink.emit(
                SearchIterationStarted(
                    iteration_index=self._iteration_state.started_iteration
                )
            )

        return should_continue


class _IterationObservingTreeManager:
    """Wrap a tree manager to infer iteration completion after index updates.

    This is a best-effort boundary aligned with the current exploration flow,
    where index updates happen once per completed iteration.
    """

    def __init__(
        self,
        base: Any,
        *,
        debug_sink: SearchDebugSink,
        iteration_state: _IterationState,
    ) -> None:
        self._base = base
        self._debug_sink = debug_sink
        self._iteration_state = iteration_state

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped tree manager."""
        return getattr(self._base, name)

    def update_indices(self, *, tree: Any) -> None:
        """Delegate index updates and emit one completion event per started iteration."""
        self._base.update_indices(tree=tree)

        if (
            self._iteration_state.completed_iteration
            >= self._iteration_state.started_iteration
        ):
            return

        self._iteration_state.completed_iteration += 1
        self._debug_sink.emit(
            SearchIterationCompleted(
                iteration_index=self._iteration_state.completed_iteration
            )
        )


class ObservableTreeExploration:
    """Facade over a core exploration object with debug-aware wrapped collaborators."""

    def __init__(
        self,
        base_exploration: Any,
        *,
        debug_sink: SearchDebugSink | None = None,
    ) -> None:
        """Store the wrapped exploration facade."""
        self._base_exploration = base_exploration
        self._debug_sink = debug_sink or NullSearchDebugSink()

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped exploration."""
        return getattr(self._base_exploration, name)

    @classmethod
    def from_tree_exploration(
        cls,
        tree_exploration: Any,
        *,
        debug_sink: SearchDebugSink | None = None,
    ) -> ObservableTreeExploration:
        """Build a debug facade by shallow-cloning ``tree_exploration``.

        This helper is intended for debug observation and assumes replacing the
        wrapped collaborators is sufficient for delegated execution.
        """
        sink = debug_sink or NullSearchDebugSink()
        iteration_state = _IterationState()

        observed_tree_manager = _IterationObservingTreeManager(
            _ensure_observable_tree_manager(
                tree_exploration.tree_manager,
                debug_sink=sink,
            ),
            debug_sink=sink,
            iteration_state=iteration_state,
        )
        observed_node_selector = _ensure_observable_node_selector(
            tree_exploration.node_selector,
            debug_sink=sink,
        )
        observed_stopping_criterion = _IterationObservingStoppingCriterion(
            tree_exploration.stopping_criterion,
            debug_sink=sink,
            iteration_state=iteration_state,
        )

        observed_exploration = _clone_exploration(
            tree_exploration,
            tree_manager=observed_tree_manager,
            node_selector=observed_node_selector,
            stopping_criterion=observed_stopping_criterion,
        )
        return cls(observed_exploration, debug_sink=sink)

    def explore(self, random_generator: Random) -> Any:
        """Delegate exploration to the wrapped core exploration object."""
        return self._base_exploration.explore(random_generator=random_generator)


def _clone_exploration(tree_exploration: Any, **updates: Any) -> Any:
    """Return a shallow clone of ``tree_exploration`` with updated collaborators.

    This helper is intended for debug observation and assumes collaborator
    replacement is sufficient for safe delegated execution.
    """
    if is_dataclass(tree_exploration) and not isinstance(tree_exploration, type):
        return replace(tree_exploration, **updates)

    cloned_exploration = copy(tree_exploration)
    for attribute_name, value in updates.items():
        setattr(cloned_exploration, attribute_name, value)
    return cloned_exploration


def _ensure_observable_tree_manager(base: Any, *, debug_sink: SearchDebugSink) -> Any:
    """Return ``base`` wrapped in ``ObservableAlgorithmNodeTreeManager`` when needed."""
    if isinstance(base, ObservableAlgorithmNodeTreeManager):
        return base
    return ObservableAlgorithmNodeTreeManager(base, debug_sink=debug_sink)


def _ensure_observable_node_selector(base: Any, *, debug_sink: SearchDebugSink) -> Any:
    """Return ``base`` wrapped in ``ObservableNodeSelector`` when needed."""
    if isinstance(base, ObservableNodeSelector):
        return base
    return ObservableNodeSelector(base, debug_sink=debug_sink)
