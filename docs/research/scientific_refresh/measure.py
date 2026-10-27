"""Bounded synthetic comparison; select Anemone source with PYTHONPATH."""

import json
import statistics
import time
import tracemalloc

import anemone
from anemone import NodeValueUpdate
from tests.scientific_refresh_helpers import Evaluator, nodes, runtime


def main() -> None:
    """Measure five samples after warmup on fresh 40/364/1093-node trees."""
    rows = []
    for levels in (3, 5, 6):
        count = (3 ** (levels + 1) - 1) // 2
        children = {i: [3 * i + j for j in (1, 2, 3)] for i in range(count)}
        values = {i: float(i % 17) for i in range(3 * count + 1)}
        search = runtime(children, values)
        search.stopping_criterion.tree_branch_limit = 100000
        for _ in range(levels):
            search.step()
        graph_nodes = list(nodes(search))
        leaf = graph_nodes[-1]
        refresh, updates = [], []
        peak = 0
        for step in range(6):
            evaluator = Evaluator({i: float((i + step) % 17) for i in values})
            tracemalloc.start()
            start = time.perf_counter()
            search.refresh_with_evaluator(evaluator, scope="all")
            elapsed = time.perf_counter() - start
            peak = max(peak, tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()
            if step:
                refresh.append(elapsed)
            start = time.perf_counter()
            for value in range(100):
                search.apply_node_value_updates(
                    [NodeValueUpdate(node_id=str(leaf.id), direct_value=float(value))]
                )
            if step:
                updates.append((time.perf_counter() - start) / 100)
        rows.append(
            dict(
                nodes=len(graph_nodes),
                levels=levels,
                refresh_median_seconds=statistics.median(refresh),
                refresh_peak_allocated_bytes=peak,
                ordinary_update_median_seconds=statistics.median(updates),
                samples=5,
            )
        )
    print(json.dumps(dict(module=anemone.__file__, measurements=rows), indent=2))


if __name__ == "__main__":
    main()
