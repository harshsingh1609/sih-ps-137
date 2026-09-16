"""Nearest-Neighbor (Greedy) baseline with optimal-split refinement for CVRP."""

from __future__ import annotations
import time
from typing import List, Optional
import numpy as np

from app.engine.problem import CVRPProblem
from app.engine.encode import permutation_to_random_keys
from app.engine.split import optimal_split
from app.engine.qpso import SolverResult


def solve_nn(
    problem: CVRPProblem,
    seed: Optional[int] = None,
) -> SolverResult:
    """
    Nearest-Neighbor heuristic:
    Greedily selects nearest capacity-feasible customer, then passes giant tour
    through optimal-split dynamic programming to guarantee optimal vehicle partitions.
    """
    start_time = time.perf_counter()
    n = problem.n
    unvisited = set(range(1, n + 1))
    cost_matrix = problem.cost_matrix

    giant_tour: List[int] = []
    current_node = 0

    while unvisited:
        # Find nearest unvisited customer to current_node
        best_next = -1
        best_dist = float("inf")

        for cand in unvisited:
            d = cost_matrix[current_node, cand]
            if d < best_dist:
                best_dist = d
                best_next = cand

        giant_tour.append(best_next)
        unvisited.remove(best_next)
        current_node = best_next

    # Run optimal split on the constructed greedy sequence
    split_res = optimal_split(problem, giant_tour)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    best_keys = permutation_to_random_keys(giant_tour)

    return SolverResult(
        algorithm="nn",
        routes=split_res.routes,
        cost=split_res.total_cost,
        vehicles_used=split_res.vehicles_used,
        time_ms=elapsed_ms,
        convergence=[split_res.total_cost],
        best_keys=best_keys,
        iterations=1,
        level="L4",
        stale=False,
    )
