"""Small-fixture max oracle independent of Anemone's value/backup helpers."""

from collections.abc import Iterator, Sequence
from random import Random
from typing import Any

from valanga import Outcome, OverEvent
from valanga.evaluations import Certainty, Value

from anemone import create_search_with_tree_eval_factory
from anemone.node_evaluation.tree.single_agent.factory import NodeMaxEvaluationFactory
from tests.fake_yaml_game import FakeYamlDynamics, MasterStateValueEvaluatorFromYaml
from tests.test_refresh_scopes import _build_args, _ConcreteFakeYamlState


def nodes(search: Any, reverse: bool = False) -> Iterator[Any]:
    """Walk the resident registry in dependency order."""
    depths = search.tree.descendants.range()
    for depth in reversed(depths) if reverse else depths:
        yield from search.tree.descendants[depth].values()


def exact(value: Value | None) -> bool:
    """Inspect each Value certainty independently."""
    return value is not None and value.certainty in (
        Certainty.TERMINAL,
        Certainty.FORCED,
    )


def reference(search: Any, evaluator: Any = None) -> dict[int, dict[str, Any]]:
    """Compute direct/tree/effective/source, independent certainties, PV and order.

    No Anemone selection, ordering, proof, backup or propagation helper is used.
    The dictionary exists only for small test fixtures, never the refresh path.
    """
    result = {}
    for node in nodes(search, reverse=True):
        old = node.tree_evaluation
        if evaluator is None or exact(old.direct_value) or node.state.is_game_over():
            direct = old.direct_value
        else:
            raw = evaluator.evaluate_batch_items([node])[0]
            assert raw.certainty == Certainty.ESTIMATE
            direct = Value(
                score=(1 / 0.99999999) ** node.tree_depth * raw.score,
                certainty=Certainty.ESTIMATE,
            )
        children = [(b, c) for b, c in node.iter_child_links() if c is not None]
        ordered = sorted(
            children,
            key=lambda bc: (
                -result[bc[1].id]["effective"].score,
                -int(exact(result[bc[1].id]["effective"])),
                len(result[bc[1].id]["pv"]),
                bc[1].id,
                str(bc[0]),
            ),
        )
        tree, pv = None, ()
        if ordered:
            branch, child = ordered[0]
            selected = result[child.id]["effective"]
            proven = (
                node.tree_node.all_branches_generated
                and len(children) == node.child_link_count()
                and all(exact(result[c.id]["effective"]) for _, c in children)
            )
            tree = Value(
                score=selected.score,
                certainty=Certainty.FORCED if proven else Certainty.ESTIMATE,
                over_event=selected.over_event if proven else None,
            )
            pv = (branch, *result[child.id]["pv"])
        if tree is None:
            effective, source = direct, "direct_self" if direct is not None else "none"
        elif (
            direct is None
            or node.tree_node.all_branches_generated
            or (direct.score, exact(direct)) <= (tree.score, exact(tree))
        ):
            effective, source = tree, "tree_child"
        else:
            effective, source = direct, "direct_self"
        result[node.id] = dict(
            direct=direct,
            tree=tree,
            effective=effective,
            source=source,
            pv=pv,
            order=tuple(b for b, _ in ordered),
            terminal=effective is not None
            and effective.certainty == Certainty.TERMINAL,
            exact=exact(effective),
        )
    return result


def snapshot(search: Any) -> dict[int, dict[str, Any]]:
    """Capture the complete set of scientific fields compared by the oracle."""
    return {
        n.id: dict(
            direct=n.tree_evaluation.direct_value,
            tree=n.tree_evaluation.tree_value,
            effective=n.tree_evaluation.effective_value,
            source=n.tree_evaluation.effective_value_source.value,
            pv=tuple(n.tree_evaluation.best_branch_sequence),
            order=tuple(n.tree_evaluation.decision_ordered_branches()),
            terminal=n.tree_evaluation.is_terminal(),
            exact=n.tree_evaluation.has_exact_value(),
        )
        for n in nodes(search)
    }


class Evaluator(MasterStateValueEvaluatorFromYaml):
    """Deterministic predictions plus genuine node-local terminal detection."""

    def __init__(self, values: dict[int, float]) -> None:
        """Bind deterministic raw estimates and genuine terminal scores."""
        super().__init__(values)
        self.over = self
        self.batches = []

    def check_obvious_over_events(self, state: Any) -> tuple[Any, float | None]:
        """Recognize the fixture state's own terminality."""
        if state.is_game_over():
            return OverEvent(
                outcome=Outcome.WIN, winner=None, termination="fixture"
            ), self._value_by_id[state.tag]
        return None, None

    def evaluate_batch_items(self, items: Sequence[Any]) -> list[Value]:
        """Record batches while retaining the YAML evaluator numeric policy."""
        self.batches.append(tuple(n.id for n in items))
        return super().evaluate_batch_items(items)


def runtime(
    children: dict[int, list[int]], values: dict[int, float], steps: int = 0
) -> Any:
    """Build a small generic max search without a game-specific adapter."""
    result = create_search_with_tree_eval_factory(
        state_type=_ConcreteFakeYamlState,
        dynamics=FakeYamlDynamics(),
        starting_state=_ConcreteFakeYamlState(0, children),
        args=_build_args(),
        random_generator=Random(4),
        master_state_evaluator=Evaluator(values),
        state_representation_factory=None,
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
    )
    for _ in range(steps):
        result.step()
    return result


def certifies(node: Any, terminal_scores: dict[int, float]) -> bool:
    """Solve the entire tiny fixture independently, including unopened actions.

    This is explicit external proof evidence for tests, separate from current
    search Values and from the model. It is never used by production refresh.
    """
    children = node.state.children_by_id

    def optimum(tag: int) -> float:
        continuations = children.get(tag, [])
        return (
            max(map(optimum, continuations)) if continuations else terminal_scores[tag]
        )

    direct = node.tree_evaluation.direct_value
    return direct is not None and direct.score == optimum(node.tag)
