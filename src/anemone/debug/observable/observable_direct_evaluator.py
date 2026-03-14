"""Direct-evaluator wrapper emitting best-effort value-assignment events."""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Any

from anemone.debug.events import DirectValueAssigned
from anemone.debug.formatting import safe_getattr
from anemone.debug.observable.state_diff import summarize_node_evaluation
from anemone.debug.sink import NullSearchDebugSink, SearchDebugSink

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


class ObservableDirectEvaluator:
    """Wrap a direct evaluator and emit events from before/after value diffs."""

    def __init__(self, base: Any, debug_sink: SearchDebugSink | None = None) -> None:
        """Store the wrapped evaluator and the sink receiving inferred events."""
        self._base = base
        self._debug_sink = debug_sink or NullSearchDebugSink()

    @property
    def debug_sink(self) -> SearchDebugSink:
        """Return the sink currently receiving inferred direct-value events."""
        return self._debug_sink

    def set_debug_sink(self, debug_sink: SearchDebugSink) -> None:
        """Replace the sink used for inferred direct-value events."""
        self._debug_sink = debug_sink

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped evaluator."""
        return getattr(self._base, name)

    def add_evaluation_query(self, node: Any, evaluation_queries: Any) -> None:
        """Observe terminal direct-value assignment during query enqueue."""
        self._observe_nodes(
            (node,),
            lambda: self._base.add_evaluation_query(
                node=node,
                evaluation_queries=evaluation_queries,
            ),
        )

    def evaluate_all_queried_nodes(self, evaluation_queries: Any) -> None:
        """Observe direct-value assignment across the queued nodes."""
        queried_nodes = tuple(self._iter_queried_nodes(evaluation_queries))
        self._observe_nodes(
            queried_nodes,
            lambda: self._base.evaluate_all_queried_nodes(
                evaluation_queries=evaluation_queries
            ),
        )

    def evaluate_all_not_over(self, not_over_nodes: list[Any]) -> None:
        """Observe direct-value assignment for explicit batch evaluation calls."""
        self._observe_nodes(
            tuple(not_over_nodes),
            lambda: self._base.evaluate_all_not_over(not_over_nodes),
        )

    def _observe_nodes(
        self,
        nodes: Iterable[Any],
        action: Callable[[], Any],
    ) -> Any:
        """Observe ``nodes`` before and after ``action`` and emit value events."""
        observed_nodes = tuple(nodes)
        before = {id(node): summarize_node_evaluation(node) for node in observed_nodes}
        result = action()

        for node in observed_nodes:
            self._emit_direct_value_event_if_needed(node=node, before=before[id(node)])

        return result

    def _emit_direct_value_event_if_needed(self, *, node: Any, before: Any) -> None:
        """Emit a direct-value event when the formatted value changed."""
        after = summarize_node_evaluation(node)
        if after.direct_value_repr is None:
            return

        if after.direct_value_repr == before.direct_value_repr:
            return

        self._debug_sink.emit(
            DirectValueAssigned(
                node_id=str(node.id),
                value_repr=after.direct_value_repr,
            )
        )

    def _iter_queried_nodes(self, evaluation_queries: Any) -> tuple[Any, ...]:
        """Return nodes referenced by common evaluation-query containers."""
        return tuple(
            chain(
                safe_getattr(evaluation_queries, "over_nodes") or (),
                safe_getattr(evaluation_queries, "not_over_nodes") or (),
            )
        )
