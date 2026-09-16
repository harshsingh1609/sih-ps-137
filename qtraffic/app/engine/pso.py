"""Classical Particle Swarm Optimization (PSO) baseline for CVRP."""

from __future__ import annotations
import time
from typing import List, Optional
import numpy as np

from app.engine.problem import CVRPProblem
from app.engine.encode import random_keys_to_permutation
from app.engine.split import optimal_split
from app.engine.qpso import SolverResult


def solve_pso(
    problem: CVRPProblem,
    pop: int = 40,
    iters: int = 300,
    seed: Optional[int] = None,
    early_stop_patience: int = 60,
    time_budget_sec: Optional[float] = None,
) -> SolverResult:
    """
    Classical PSO algorithm:
    - Inertia w decreasing from 0.9 to 0.4
    - c1 = c2 = 1.4962
    - Velocity clipped to [-0.3, 0.3]
    - Positions clipped to [0, 1]
    """
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = problem.n
    M = int(pop)
    T = int(iters)

    # Initialize positions and velocities
    X = rng.uniform(0.0, 1.0, size=(M, n))
    V = rng.uniform(-0.3, 0.3, size=(M, n))

    pbest = X.copy()
    pbest_cost = np.full(M, np.inf, dtype=np.float64)

    gbest_cost = float("inf")
    gbest_keys = np.zeros(n, dtype=np.float64)
    gbest_routes: List[List[int]] = []
    gbest_vehicles = 0

    # Initial evaluation
    for i in range(M):
        perm = random_keys_to_permutation(X[i])
        res = optimal_split(problem, perm)
        pbest_cost[i] = res.total_cost

        if res.total_cost < gbest_cost:
            gbest_cost = res.total_cost
            gbest_keys = X[i].copy()
            gbest_routes = res.routes
            gbest_vehicles = res.vehicles_used

    convergence: List[float] = [gbest_cost]

    c1 = 1.4962
    c2 = 1.4962
    v_max = 0.3

    stagnation_count = 0
    executed_iters = 0
    hit_time_budget = False

    for t in range(1, T + 1):
        if time_budget_sec is not None and (time.perf_counter() - start_time) >= time_budget_sec:
            hit_time_budget = True
            break

        executed_iters = t
        w = 0.9 - 0.5 * (t / float(T))

        r1 = rng.uniform(0.0, 1.0, size=(M, n))
        r2 = rng.uniform(0.0, 1.0, size=(M, n))

        # Velocity update
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest_keys - X)
        np.clip(V, -v_max, v_max, out=V)

        # Position update
        X = X + V
        np.clip(X, 0.0, 1.0, out=X)

        for i in range(M):
            perm = random_keys_to_permutation(X[i])
            res = optimal_split(problem, perm)

            if res.total_cost < pbest_cost[i]:
                pbest[i] = X[i].copy()
                pbest_cost[i] = res.total_cost

            if res.total_cost < gbest_cost:
                gbest_cost = res.total_cost
                gbest_keys = X[i].copy()
                gbest_routes = res.routes
                gbest_vehicles = res.vehicles_used
                stagnation_count = 0

        convergence.append(gbest_cost)
        stagnation_count += 1
        if stagnation_count >= early_stop_patience:
            break

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    level = "L2" if hit_time_budget else "L3"

    return SolverResult(
        algorithm="pso",
        routes=gbest_routes,
        cost=gbest_cost,
        vehicles_used=gbest_vehicles,
        time_ms=elapsed_ms,
        convergence=convergence,
        best_keys=gbest_keys,
        iterations=executed_iters,
        level=level,
        stale=False,
    )
