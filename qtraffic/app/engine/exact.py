"""Exact brute-force solver for small CVRP instances (n <= 8)."""

from __future__ import annotations
import time
import itertools
from typing import List, Optional
import numpy as np

from app.engine.problem import CVRPProblem
from app.engine.encode import permutation_to_random_keys
from app.engine.split import optimal_split
from app.engine.qpso import SolverResult


def solve_exact(
    problem: CVRPProblem,
    seed: Optional[int] = None,
) -> SolverResult:
    """
    Exact ground-truth solver for small instances (n <= 8).
    Exhaustively searches all n! giant tour permutations combined with optimal-split DP.
    """
    if problem.n > 8:
        raise ValueError(
            f"Exact solver only supports instances with n <= 8 for computational feasibility (got n={problem.n})."
        )

    start_time = time.perf_counter()
    n = problem.n
    customers = list(range(1, n + 1))

    best_cost = float("inf")
    best_routes: List[List[int]] = []
    best_perm: List[int] = []
    best_vehicles = 0

    convergence: List[float] = []

    for perm in itertools.permutations(customers):
        split_res = optimal_split(problem, perm)
        if split_res.total_cost < best_cost:
            best_cost = split_res.total_cost
            best_routes = split_res.routes
            best_perm = list(perm)
            best_vehicles = split_res.vehicles_used
            convergence.append(best_cost)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    best_keys = permutation_to_random_keys(best_perm)

    return SolverResult(
        algorithm="exact",
        routes=best_routes,
        cost=best_cost,
        vehicles_used=best_vehicles,
        time_ms=elapsed_ms,
        convergence=convergence,
        best_keys=best_keys,
        iterations=len(convergence),
        level="L0",
        stale=False,
    )
