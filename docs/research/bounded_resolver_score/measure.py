"""Fresh real Morpion graphs; matched serial resolver comparisons, no activation."""

import argparse
import hashlib
import inspect
import json
import logging
import resource
from pathlib import Path
from time import perf_counter
from typing import Any
from unittest.mock import patch

import orjson

from anemone.checkpoints import (
    CheckpointStateResolver,
    build_search_checkpoint_payload,
    serialize_value,
    write_sharded_search_checkpoint,
)
from tests.morpion_checkpoint_cache_helpers import Codec, load
from tests.morpion_refresh_helpers import Evaluator, runtime
from tests.scientific_refresh_helpers import nodes


def fingerprint(search: Any, codec: Codec) -> str:
    """Stream canonical states, edges and every scientific field into one digest."""
    digest = hashlib.sha256()
    for node in nodes(search):
        evaluation = node.tree_evaluation
        row = dict(
            id=node.id,
            depth=node.tree_depth,
            state=codec.dump_anchor_ref(node.state),
            terminal=node.state.is_game_over(),
            direct=serialize_value(evaluation.direct_value)
            if evaluation.direct_value is not None
            else None,
            tree=serialize_value(evaluation.tree_value)
            if evaluation.tree_value is not None
            else None,
            effective=serialize_value(evaluation.effective_value),
            source=evaluation.effective_value_source.value,
            version=evaluation.direct_evaluation_version,
            pv=evaluation.best_branch_sequence,
            order=evaluation.decision_ordered_branches(),
            edges=[
                (b, c.id if c is not None else None) for b, c in node.iter_child_links()
            ],
            parents=sorted((p.id, sorted(bs)) for p, bs in node.parent_nodes.items()),
            complete=node.tree_node.all_branches_generated,
        )
        digest.update(orjson.dumps(row, option=orjson.OPT_SORT_KEYS))
    return digest.hexdigest()


def prepare(root: Path) -> None:
    """Generate new legal 1k/5k checkpoints once, from an initial board."""
    search = runtime(branch_limit=100000, steps=0)
    codec = Codec()
    for target in (1000, 5000):
        started = perf_counter()
        while search.tree.nodes_count < target:
            search.step()
        destination = root / str(target)
        payload = build_search_checkpoint_payload(search, state_codec=codec)
        write_sharded_search_checkpoint(payload, destination, node_count_per_shard=1024)
        record = dict(
            nodes=search.tree.nodes_count,
            edges=search.tree.branch_count,
            max_depth=max(n.tree_depth for n in nodes(search)),
            multi_parent_nodes=sum(len(n.parent_nodes) > 1 for n in nodes(search)),
            prepare_seconds=perf_counter() - started,
        )
        (root / f"{target}-source.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)


def measure(checkpoint: Path, capacity: int | None, output: Path) -> None:
    """Measure complete refresh with aggregate instrumentation only."""
    codec = Codec()
    stats = dict(hits=0, misses=0, max_retained=0)
    original = CheckpointStateResolver.resolve

    def counted(resolver: Any, node_id: int) -> Any:
        stats["hits" if node_id in resolver._resolved_states else "misses"] += 1
        result = original(resolver, node_id)
        stats["max_retained"] = max(
            stats["max_retained"], len(resolver._resolved_states)
        )
        return result

    with patch.object(CheckpointStateResolver, "resolve", counted):
        search = load(checkpoint, codec, capacity)
        resolver = search.tree.root_node.tree_node.state_handle.resolver
        selector = search.node_selector.base
        rng, steps = (
            selector.random_generator.getstate(),
            selector._selection_step_count,
        )
        phases = []
        for score in (42.0, 7.0):
            before = dict(stats)
            retained_before = len(resolver._resolved_states)
            anchors, deltas = codec.anchor_decodes, codec.delta_decodes
            usage_before = resource.getrusage(resource.RUSAGE_SELF)
            started = perf_counter()
            search.refresh_with_evaluator(
                Evaluator({n.id: score for n in nodes(search)}),
                scope="all",
                batch_size=64,
            )
            seconds = perf_counter() - started
            usage_after = resource.getrusage(resource.RUSAGE_SELF)
            assert selector.random_generator.getstate() == rng
            assert selector._selection_step_count == steps
            retained = len(resolver._resolved_states)
            if capacity is not None:
                assert stats["max_retained"] <= capacity
            decodes = codec.anchor_decodes + codec.delta_decodes - anchors - deltas
            phases.append(
                dict(
                    score=score,
                    refresh_seconds=seconds,
                    cpu_seconds=(
                        usage_after.ru_utime
                        + usage_after.ru_stime
                        - usage_before.ru_utime
                        - usage_before.ru_stime
                    ),
                    retained=retained,
                    max_retained=stats["max_retained"],
                    evictions=(
                        decodes - retained + retained_before if capacity != 0 else 0
                    ),
                    cache_hits=stats["hits"] - before["hits"],
                    cache_misses=stats["misses"] - before["misses"],
                    anchor_decodes=codec.anchor_decodes - anchors,
                    delta_decodes=codec.delta_decodes - deltas,
                    peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    / 1024,
                )
            )
            phases[-1]["scientific_sha256"] = fingerprint(search, codec)
        # All nodes were resolved during each full pass; extra decodes are repeats.
        total_decodes = codec.anchor_decodes + codec.delta_decodes
        result = dict(
            resolver_source=inspect.getfile(CheckpointStateResolver),
            resolver_source_sha256=hashlib.sha256(
                Path(inspect.getfile(CheckpointStateResolver)).read_bytes()
            ).hexdigest(),
            source=str(checkpoint),
            capacity=capacity,
            nodes=search.tree.nodes_count,
            edges=search.tree.branch_count,
            phases=phases,
            total_decodes_including_parity=total_decodes,
            repeated_decodes_including_parity=total_decodes - search.tree.nodes_count,
            cache_hits_including_parity=stats["hits"],
            evictions_including_parity=(
                total_decodes - len(resolver._resolved_states) if capacity != 0 else 0
            ),
            retained_after_parity=len(resolver._resolved_states),
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        )
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


def main() -> None:
    """Run preparation or one fresh-process capacity measurement."""
    logging.disable(logging.CRITICAL)
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["prepare", "measure"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--capacity", default="1024")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.operation == "prepare":
        args.root.mkdir(parents=True, exist_ok=True)
        prepare(args.root)
    else:
        measure(
            args.root,
            None if args.capacity == "baseline" else int(args.capacity),
            args.output,
        )


if __name__ == "__main__":
    main()
