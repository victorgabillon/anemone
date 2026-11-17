"""Opt-in synthetic PR74 harness adapter; never accepts a historical run."""

from __future__ import annotations
import argparse
import gc
import importlib.util
import json
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import anemone
import zstandard
from anemone.checkpoints import prepare_checkpoint_adoption
from anemone.tree_exploration import TreeExploration
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.nodes.tree_node import TreeNode
from chipiron.environments.morpion.bootstrap.runtime import runner as runner_module
from chipiron.environments.morpion.bootstrap.performance import StageMeasurement

CHIPIRON = Path(
    os.environ.get(
        "CHIPIRON_LIFECYCLE_SOURCE", "/tmp/chipiron-checkpoint-lifecycle-integration"
    )
)
spec = importlib.util.spec_from_file_location(
    "pr74_benchmark", CHIPIRON / "scripts/benchmark_morpion_warm_growth.py"
)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)
CURRENT = None
CONFIG = "plain"
PROBE = False
ROTATE_EVICTION = True
EXERCISE_CACHE = False


def objects():
    tracked = gc.get_objects()
    return {
        "gc_tracked_objects": len(tracked),
        "live_runtime_graphs": sum(type(o) is TreeExploration for o in tracked),
        "live_algorithm_nodes": sum(type(o) is AlgorithmNode for o in tracked),
        "live_tree_nodes": sum(type(o) is TreeNode for o in tracked),
    }


def records(path):
    result = {}
    for shard in json.loads((path / "manifest.json").read_text())["shards"]:
        with zstandard.open(path / shard["path"], "rt") as stream:
            if shard["encoding"] == "jsonl_zst":
                result.setdefault(shard["kind"], []).extend(
                    json.loads(line) for line in stream
                )
            else:
                result[shard["kind"]] = json.load(stream)
    return result


class AdoptionRunner(harness.BenchmarkRunner):
    def __init__(self, *, cold):
        super().__init__(cold=cold)
        global CURRENT
        CURRENT = self
        self.lifecycle = []
        self.pending = None
        self.saves = []
        self.memory_probes = []
        self.eviction_boundaries = []
        self.cache_checks = []
        self.external_handles = []
        if CONFIG != "plain":
            from chipiron.environments.morpion.bootstrap.config import (
                MorpionBootstrapRolloutConfig,
            )

            self._args = replace(
                self._args,
                search_args=runner_module.default_search_args(
                    rollout=MorpionBootstrapRolloutConfig(
                        enabled="rollout" in CONFIG, max_extra_steps=2
                    )
                ),
                growth_state_eviction_policy="frontier_cold"
                if "eviction" in CONFIG
                else "none",
                growth_state_eviction_payload_mode="delta_when_safe",
                growth_state_eviction_recent_window=1,
                growth_state_eviction_scan_interval_steps=1,
                growth_state_rematerialization_cache_size=8 if EXERCISE_CACHE else 128,
            )

    def lifecycle_operation(self, phase, function):
        if PROBE:
            from anemone.checkpoints import adoption
            from contextlib import ExitStack

            with ExitStack() as stack:
                for name in [
                    "_graph_identity",
                    "_create_state_handles_from_node_ids",
                    "_create_state_resolver",
                ]:
                    original = getattr(adoption, name)

                    def measured(*args, _original=original, _name=name, **kwargs):
                        result = _original(*args, **kwargs)
                        self.memory_probes.append(
                            dict(phase=phase, boundary=_name, **objects())
                        )
                        return result

                    stack.enter_context(patch.object(adoption, name, measured))
                return self._lifecycle_operation(phase, function)
        return self._lifecycle_operation(phase, function)

    def _lifecycle_operation(self, phase, function):
        before = objects()
        loads = []
        codec_type = type(self._state_codec)

        def counted(name):
            original = getattr(codec_type, name)

            def call(*args, **kwargs):
                loads.append(name)
                return original(*args, **kwargs)

            return call

        with (
            patch.object(codec_type, "load_anchor_ref", counted("load_anchor_ref")),
            patch.object(
                codec_type, "load_child_from_delta", counted("load_child_from_delta")
            ),
        ):
            observation = StageMeasurement()
            result = function()
            record = observation.finish(phase=phase, state_decodes=len(loads))
        record.update(before=before, after=objects())
        if phase == "adopt":
            record["stats"] = asdict(result)
        self.lifecycle.append(record)
        assert not loads, record
        return result

    def save_checkpoint(self, output_path):
        self.pending = None
        if EXERCISE_CACHE:
            resolver = self._live_compact_state_resolver
            handles = [
                node.state_handle
                for node in self.profile_runtime_root()._all_nodes_in_tree_order()
                if getattr(node.state_handle, "resolver", None) is resolver
            ]
            for handle in handles:
                handle.get()
            for handle in handles[-5:]:
                state = handle.get()
                assert handle.get() is state
            assert len(resolver._decoded_state_cache) <= resolver.cache_size
            self.external_handles = [(h, h.get().tag) for h in handles[-5:]]
            self.cache_checks.append(
                dict(
                    compact_handles=len(handles),
                    cache_size=len(resolver._decoded_state_cache),
                    metrics=self.profile_state_eviction_runtime(),
                )
            )
        builder = runner_module.build_search_checkpoint_payload

        def build(*args, **kwargs):
            payload = builder(*args, **kwargs)
            self.pending = self.lifecycle_operation(
                "prepare",
                lambda: prepare_checkpoint_adoption(
                    self.profile_runtime_root(), payload, state_codec=self._state_codec
                ),
            )
            return payload

        observation = StageMeasurement()
        try:
            if self.cold:
                super().save_checkpoint(output_path)
            else:
                with patch.object(
                    runner_module, "build_search_checkpoint_payload", build
                ):
                    super().save_checkpoint(output_path)
        except BaseException:
            self.pending = None
            raise
        self.saves.append(
            observation.finish(
                checkpoint_digest=harness.checkpoint_digest(Path(output_path))
            )
        )

    def acknowledge_growth_publication(self, **kwargs):
        super().acknowledge_growth_publication(**kwargs)
        pending, self.pending = self.pending, None
        if (
            not self.cold
            and kwargs["generation_published"]
            and self._warm_growth.committed is not None
        ):
            assert pending is not None
            self.lifecycle_operation("adopt", pending.adopt)
            # Caller-owned eviction history is reset by cold load_or_create too.
            # Rotate its owner; never clear an old resolver/cache still referenced
            # by a handle outside the live graph. No state is replayed here.
            self.eviction_boundaries.append(self.profile_state_eviction_runtime())
            if ROTATE_EVICTION:
                self.lifecycle_operation(
                    "rotate_eviction", self._reset_growth_state_eviction_runtime
                )
            for handle, tag in self.external_handles:
                assert handle.get().tag == tag
            self.external_handles = []


harness.BenchmarkRunner = AdoptionRunner


def main():
    global CONFIG, PROBE, ROTATE_EVICTION, EXERCISE_CACHE
    logging.disable(logging.CRITICAL)
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--mode", choices=["prepare", "cold", "warm", "compare"], required=True
    )
    p.add_argument("--nodes", type=int, default=10000)
    p.add_argument(
        "--config",
        choices=["plain", "eviction", "rollout", "eviction-rollout"],
        default="plain",
    )
    p.add_argument("--export", action="store_true")
    p.add_argument(
        "--object-probe",
        action="store_true",
        help="Intrusive internal object snapshots; exclude from timing comparisons",
    )
    p.add_argument(
        "--retain-eviction-owner",
        action="store_true",
        help="Reproduce the independent caller-owned eviction failure; parity must fail",
    )
    p.add_argument(
        "--exercise-cache",
        action="store_true",
        help="Explicitly decode compact states before saves and test external handles after rotation; validation only",
    )
    p.add_argument("--anemone-sha", default="866560f7606eea095981ec25090c5f2c5908a643")
    p.add_argument("--chipiron-sha", default="027231820f1b2a192d1adb4ec0d90469dac19dbe")
    args = p.parse_args()
    assert args.root.resolve().is_relative_to(Path("/tmp"))
    assert 1 <= args.nodes <= 10000
    anemone_source = Path(anemone.__file__).resolve().parents[2]
    assert harness.git(anemone_source, "rev-parse", "HEAD") == args.anemone_sha
    assert harness.git(CHIPIRON, "rev-parse", "HEAD") == args.chipiron_sha
    assert not harness.git(anemone_source, "status", "--porcelain")
    assert not harness.git(CHIPIRON, "status", "--porcelain")
    CONFIG = args.config
    PROBE = args.object_probe
    ROTATE_EVICTION = not args.retain_eviction_owner
    EXERCISE_CACHE = args.exercise_cache
    args.root.mkdir(parents=True, exist_ok=True)
    if CONFIG != "plain":
        original_args = harness.MorpionBootstrapArgs

        def pipeline_args(**kwargs):
            from chipiron.environments.morpion.bootstrap.config import (
                MorpionBootstrapRolloutConfig,
                MorpionBootstrapSearchConfig,
            )

            kwargs.update(
                growth_state_eviction_policy="frontier_cold"
                if "eviction" in CONFIG
                else "none",
                growth_state_eviction_payload_mode="delta_when_safe",
                growth_state_eviction_recent_window=1,
                growth_state_eviction_scan_interval_steps=1,
                growth_state_rematerialization_cache_size=8 if EXERCISE_CACHE else 128,
                search=MorpionBootstrapSearchConfig(
                    rollout=MorpionBootstrapRolloutConfig(
                        enabled="rollout" in CONFIG, max_extra_steps=2
                    )
                ),
            )
            return original_args(**kwargs)

        harness.MorpionBootstrapArgs = pipeline_args
    if args.mode == "compare":
        cold, warm = [
            json.loads((args.root / f"{mode}.json").read_text())
            for mode in ["cold", "warm"]
        ]
        result = dict(
            checkpoint_parity=cold["checkpoint_digest"] == warm["checkpoint_digest"],
            selection_parity=cold["selected"] == warm["selected"],
            every_save_parity=[
                a["checkpoint_digest"] == b["checkpoint_digest"]
                for a, b in zip(cold["saves"], warm["saves"], strict=True)
            ],
        )
        result["retained_generations"] = []
        for path in sorted(
            (args.root / "cold/search_checkpoints").glob("generation_*.sharded")
        ):
            a, b = (
                records(path),
                records(args.root / "warm/search_checkpoints" / path.name),
            )
            assert a.keys() == b.keys()
            comparison = dict(
                generation=path.name,
                component_equal={key: a[key] == b[key] for key in a},
                different_node_shells=sum(
                    x != y
                    for x, y in zip(a["node_shells"], b["node_shells"], strict=True)
                ),
            )
            result["retained_generations"].append(comparison)
            assert a == b, comparison
        (args.root / "comparison.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
        assert all(
            [
                result["checkpoint_parity"],
                result["selection_parity"],
                all(result["every_save_parity"]),
            ]
        )
        assert cold["restores"] == 3 and warm["restores"] == 1 and warm["same_runtime"]
        return
    if args.mode == "prepare":
        assert not (args.root / "prepare").exists()
        result = harness.prepare(args.root / "prepare", args.nodes)
    else:
        marker = args.root / "prepare/pipeline/generation_000001/manifest.json"
        assert json.loads(marker.read_text())["created_at_utc"] == "synthetic-benchmark"
        shutil.copytree(args.root / "prepare", args.root / args.mode)
        result = harness.measure(
            args.root / args.mode,
            cold=args.mode == "cold",
            cycles=3,
            skip_export=not args.export,
        )
    result.update(
        lifecycle=CURRENT.lifecycle,
        saves=CURRENT.saves,
        memory_probes=CURRENT.memory_probes,
        eviction_boundaries=CURRENT.eviction_boundaries,
        cache_checks=CURRENT.cache_checks,
        eviction_owner_rotated=ROTATE_EVICTION,
        config=CONFIG,
        export=args.export,
        anemone_import=anemone.__file__,
        anemone_sha=harness.git(
            Path(anemone.__file__).resolve().parents[2], "rev-parse", "HEAD"
        ),
        chipiron_sha=harness.git(CHIPIRON, "rev-parse", "HEAD"),
        chipiron_dirty=harness.git(CHIPIRON, "status", "--porcelain"),
    )
    (args.root / f"{args.mode}.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["observations", "lifecycle", "saves"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
