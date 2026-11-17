"""Read-only instrumentation around the unchanged PR74 synthetic harness."""

from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
import anemone
import zstandard
from anemone.checkpoints import serialize_checkpoint_atom
from anemone.checkpoints.state_handles import checkpoint_payload_for_reuse_or_none

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


def records(path: Path) -> dict[str, Any]:
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


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class DiagnosticRunner(harness.BenchmarkRunner):
    def save_checkpoint(self, output_path: str | Path) -> None:
        super().save_checkpoint(output_path)
        output = Path(output_path)
        persisted = records(output)
        shells = {v["node_id"]: v for v in persisted["node_shells"]}
        states = {v["node_id"]: v["state_payload"] for v in persisted["state_payloads"]}
        context = {}
        for node in self.profile_runtime_root()._all_nodes_in_tree_order():
            handle = node.state_handle
            reusable = checkpoint_payload_for_reuse_or_none(handle)
            links = [
                {
                    "parent_id": parent.id,
                    "branches": sorted(
                        (serialize_checkpoint_atom(b) for b in branches), key=repr
                    ),
                }
                for parent, branches in node.iter_parent_items()
            ]
            state = states[node.id]
            context[str(node.id)] = {
                "node_id": node.id,
                "depth": node.tree_depth,
                "parent_count": node.parent_count(),
                "incoming_links_in_storage_order": links,
                "shell": shells[node.id],
                "saved_state_payload": state,
                "saved_state_payload_type": "anchor"
                if "anchor_ref" in state
                else "delta",
                "persisted_state_parent_id": state.get("state_parent_node_id"),
                "persisted_state_parent_branch": state.get("state_parent_branch"),
                "state_handle_type": type(handle).__name__,
                "live_reusable_payload_type": None
                if reusable is None
                else type(reusable).__name__,
                "live_reusable_state_parent_id": getattr(
                    reusable, "state_parent_node_id", None
                ),
                "live_reusable_state_parent_branch": getattr(
                    reusable, "state_parent_branch", None
                ),
            }
        output.with_suffix(".context.json").write_text(json.dumps(context, indent=2))


harness.BenchmarkRunner = DiagnosticRunner


def compare(root: Path) -> None:
    comparisons = []
    for cold_path in sorted(
        (root / "cold/search_checkpoints").glob("generation_*.sharded")
    ):
        warm_path = root / "warm/search_checkpoints" / cold_path.name
        if not warm_path.exists():
            continue
        cold, warm = records(cold_path), records(warm_path)
        cold_context = (
            json.loads(cold_path.with_suffix(".context.json").read_text())
            if cold_path.with_suffix(".context.json").exists()
            else {}
        )
        warm_context = (
            json.loads(warm_path.with_suffix(".context.json").read_text())
            if warm_path.with_suffix(".context.json").exists()
            else {}
        )
        by_mode = [
            {v["node_id"]: v for v in value["node_shells"]} for value in [cold, warm]
        ]
        differences = []
        for node_id in sorted(by_mode[0].keys() | by_mode[1].keys()):
            a, b = by_mode[0].get(node_id), by_mode[1].get(node_id)
            if a != b:
                fields = {
                    key: {"cold": a.get(key), "warm": b.get(key)}
                    for key in sorted(a.keys() | b.keys())
                    if a.get(key) != b.get(key)
                }
                differences.append(
                    {
                        "node_id": node_id,
                        "fields": fields,
                        "cold_context": cold_context.get(str(node_id)),
                        "warm_context": warm_context.get(str(node_id)),
                    }
                )
        comparisons.append(
            {
                "generation": cold_path.name,
                "different_node_count": len(differences),
                "component_equal": {key: cold[key] == warm[key] for key in cold},
                "differences": differences,
            }
        )
    result = {
        "anemone_import": anemone.__file__,
        "anemone_sha": subprocess.check_output(
            [
                "git",
                "-C",
                str(Path(anemone.__file__).resolve().parents[2]),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip(),
        "chipiron_sha": subprocess.check_output(
            ["git", "-C", str(CHIPIRON), "rev-parse", "HEAD"], text=True
        ).strip(),
        "comparisons": comparisons,
    }
    (root / "field-differences.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# Warm/cold node-shell field differences",
        "",
        f"Anemone: `{result['anemone_sha']}`",
        f"Chipiron: `{result['chipiron_sha']}`",
        "",
    ]
    for item in comparisons:
        lines += [
            f"## {item['generation']}",
            "",
            f"Differing nodes: {item['different_node_count']}",
            f"Component equality: `{item['component_equal']}`",
            "",
        ]
        for diff in item["differences"][:20]:
            lines.append(
                f"- Node {diff['node_id']}: `{json.dumps(diff['fields'], sort_keys=True)}`"
            )
    (root / "field-differences.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                v["generation"]: {
                    "count": v["different_node_count"],
                    "first_ids": [d["node_id"] for d in v["differences"][:20]],
                    "components": v["component_equal"],
                }
                for v in comparisons
            },
            indent=2,
        )
    )


def main() -> None:
    logging.disable(logging.CRITICAL)
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--mode", choices=["prepare", "cold", "warm", "compare"], required=True
    )
    p.add_argument("--nodes", type=int, default=10000)
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        result = harness.prepare(args.root / "prepare", args.nodes)
    elif args.mode == "compare":
        compare(args.root)
        return
    else:
        shutil.copytree(args.root / "prepare", args.root / args.mode)
        result = harness.measure(
            args.root / args.mode, cold=args.mode == "cold", cycles=3, skip_export=True
        )
    (args.root / (args.mode + ".json")).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
