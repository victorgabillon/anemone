"""Generic six-node counterexamples, without Morpion state/evaluator code."""

from __future__ import annotations
import json
import logging
import argparse
from pathlib import Path
from typing import Any
from anemone.checkpoints import (
    build_search_checkpoint_payload,
    checkpoint_payload_to_jsonable,
)
from tests.test_checkpoint_load import (
    _build_runtime,
    _load_runtime_from_payload,
    _SelectiveDeltaCheckpointCodec,
    _FakeIncrementalStateCheckpointCodec,
)

GRAPH = {0: [1, 2, 6], 1: [3, 3], 2: [3], 3: [4], 4: [5], 5: [], 6: [7], 7: []}


def build(codec: Any, *, steps: int = 2) -> Any:
    runtime = _build_runtime(children_by_id=GRAPH)
    for _ in range(steps):
        runtime.step()
    return runtime


def snapshot(runtime: Any, codec: Any) -> Any:
    return build_search_checkpoint_payload(runtime, state_codec=codec)


def restore(payload: Any, codec: Any) -> Any:
    return _load_runtime_from_payload(payload, state_codec=codec, children_by_id=GRAPH)


def join(runtime: Any) -> Any:
    return next(n for n in runtime._all_nodes_in_tree_order() if n.parent_count() > 1)


def reverse(runtime: Any) -> None:
    n = join(runtime)
    links = dict(
        sorted(n.iter_parent_items(), key=lambda item: item[0].id, reverse=True)
    )
    n.tree_node.parent_nodes_ = n.tree_node._normalize_parent_nodes(links)


def differences(a: Any, b: Any, path: str = "") -> list[dict[str, Any]]:
    if a == b:
        return []
    if isinstance(a, dict) and isinstance(b, dict):
        return [
            d
            for k in sorted(a.keys() | b.keys())
            for d in differences(a.get(k), b.get(k), path + "/" + k)
        ]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [
            d
            for i, (x, y) in enumerate(zip(a, b))
            for d in differences(x, y, path + "/" + str(i))
        ]
    return [{"field": path, "live": a, "restored": b}]


def main() -> None:
    logging.disable(logging.CRITICAL)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    results = []
    for kind, rejected in [
        ("anchor", {(1, 3), (2, 3)}),
        ("delta", set()),
        ("alternate_delta", set()),
    ]:
        codec = _SelectiveDeltaCheckpointCodec(GRAPH, rejected)
        live = build(codec)
        reverse(live)
        if kind == "alternate_delta":
            first_parent = next(join(live).iter_parent_items())[0]
            codec.invalid_pairs = {(first_parent.state.node_id, 3)}
        a = snapshot(live, codec)
        b = snapshot(live, codec)
        cold = restore(a, codec)
        c = snapshot(cold, codec)
        fixed_order = restore(a, codec)
        reverse(fixed_order)
        d = snapshot(fixed_order, codec)
        if kind == "anchor":
            n = join(live)
            n.tree_node.parent_nodes_ = n.tree_node._normalize_parent_nodes(
                dict(sorted(n.iter_parent_items(), key=lambda item: item[0].id))
            )
            changed_order = snapshot(live, codec)
            reverse(live)
        else:
            changed_order = a
        result = {
            "E_changed_insertion_order": differences(
                checkpoint_payload_to_jsonable(a),
                checkpoint_payload_to_jsonable(changed_order),
            ),
            "kind": kind,
            "node_count": len(a.tree.nodes),
            "live_parent_order": [p.id for p, _ in join(live).iter_parent_items()],
            "cold_parent_order": [p.id for p, _ in join(cold).iter_parent_items()],
            "live_handle": type(join(live).state_handle).__name__,
            "cold_handle": type(join(cold).state_handle).__name__,
            "A_live_save_twice": differences(
                checkpoint_payload_to_jsonable(a), checkpoint_payload_to_jsonable(b)
            ),
            "B_save_restore_save": differences(
                checkpoint_payload_to_jsonable(a), checkpoint_payload_to_jsonable(c)
            ),
            "F_same_parent_order_different_handle": differences(
                checkpoint_payload_to_jsonable(a), checkpoint_payload_to_jsonable(d)
            ),
        }
        for _ in range(2):
            live.step()
            cold.step()
        result["C_D_continuation"] = differences(
            checkpoint_payload_to_jsonable(snapshot(live, codec)),
            checkpoint_payload_to_jsonable(snapshot(cold, codec)),
        )
        results.append(result)
    output.write_text(json.dumps(results, indent=2))
    print(output.read_text())


if __name__ == "__main__":
    main()
