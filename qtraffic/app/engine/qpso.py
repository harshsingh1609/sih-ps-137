"""Quantum-behaved Particle Swarm Optimization (QPSO) and Lamarckian Hybrid (QPSO-H) for CVRP."""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from app.engine.problem import CVRPProblem
from app.engine.encode import random_keys_to_permutation, permutation_to_random_keys
from app.engine.split import optimal_split, SplitResult
from app.engine.local_search import refine_routes


@dataclass
class SolverResult:
    algorithm: str
    routes: List[List[int]]
    cost: float
    vehicles_used: int
    time_ms: float
    convergence: List[float]
    best_keys: np.ndarray
    iterations: int
    level: str = "L1"
    stale: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "routes": self.routes,
            "cost": float(round(self.cost, 4)),
            "vehicles_used": int(self.vehicles_used),
            "level": self.level,
            "stale": bool(self.stale),
            "time_ms": float(round(self.time_ms, 2)),
            "convergence": [float(round(c, 4)) for c in self.convergence],
        }


def solve_qpso(
    problem: CVRPProblem,
    pop: int = 40,
    iters: int = 300,
    seed: Optional[int] = None,
    hybrid: bool = False,
    local_search_interval: int = 10,
    early_stop_patience: int = 60,
    initial_keys: Optional[np.ndarray] = None,
    time_budget_sec: Optional[float] = None,
) -> SolverResult:
    """
    Quantum-behaved Particle Swarm Optimization (QPSO / QPSO-H).
    
    QPSO Equations:
    mbest_j = (1 / M) * sum_i(pbest_ij)
    P_ij = phi * pbest_ij + (1 - phi) * gbest_j,  phi ~ U(0, 1)
    X_ij(t+1) = P_ij +- beta * |mbest_j - X_ij(t)| * ln(1/u),  u ~ U(0, 1)
    beta(t) = 1.0 - 0.5 * (t / T)
    """
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = problem.n
    M = int(pop)
    T = int(iters)

    # Initialize positions X in [0, 1]^(M x n)
    X = np.zeros((M, n), dtype=np.float64)
    if initial_keys is not None and len(initial_keys) == n:
        clipped = np.clip(initial_keys, 0.0, 1.0)
        # Sanitize any NaN/Inf that survive clip (e.g. np.clip(NaN)=NaN)
        bad_mask = ~np.isfinite(clipped)
        if np.any(bad_mask):
            clipped[bad_mask] = rng.uniform(0.0, 1.0, size=int(bad_mask.sum()))
        X[0] = clipped
        half_m = M // 2
        if half_m > 1:
            noise = rng.normal(0.0, 0.05, size=(half_m - 1, n))
            noisy = np.clip(initial_keys + noise, 0.0, 1.0)
            bad_noisy = ~np.isfinite(noisy)
            if np.any(bad_noisy):
                noisy[bad_noisy] = rng.uniform(0.0, 1.0, size=int(bad_noisy.sum()))
            X[1:half_m] = noisy
        X[half_m:] = rng.uniform(0.0, 1.0, size=(M - half_m, n))
    else:
        X = rng.uniform(0.0, 1.0, size=(M, n))


    # Personal bests
    pbest = X.copy()
    pbest_cost = np.full(M, np.inf, dtype=np.float64)

    # Evaluate initial swarm
    gbest_cost = float("inf")
    gbest_keys = np.zeros(n, dtype=np.float64)
    gbest_routes: List[List[int]] = []
    gbest_vehicles = 0

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

    # Annealing beta limits
    beta_start = 1.0
    beta_end = 0.5

    stagnation_count = 0
    executed_iters = 0
    hit_time_budget = False

    for t in range(1, T + 1):
        if time_budget_sec is not None:
            if (time.perf_counter() - start_time) >= time_budget_sec:
                hit_time_budget = True
                break

        executed_iters = t
        beta = beta_start - (beta_start - beta_end) * (t / float(T))

        # mbest: mean of personal bests (vector of length n)
        mbest = np.mean(pbest, axis=0)

        # Update each particle
        for i in range(M):
            phi = rng.uniform(0.0, 1.0, size=n)
            # Local attractor
            P_i = phi * pbest[i] + (1.0 - phi) * gbest_keys

            u = rng.uniform(1e-12, 1.0, size=n)
            sign = rng.choice([-1.0, 1.0], size=n)

            # Quantum position update
            X[i] = P_i + sign * beta * np.abs(mbest - X[i]) * np.log(1.0 / u)
            # Strict clipping to [0, 1]
            np.clip(X[i], 0.0, 1.0, out=X[i])

            # Evaluate candidate
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

        # Lamarckian Local Search for QPSO-H
        if hybrid and (t % local_search_interval == 0 or t == T):
            refined = refine_routes(gbest_routes, problem)
            flat_perm = [c for r in refined for c in r]
            if len(flat_perm) == n:
                ref_res = optimal_split(problem, flat_perm)
                if ref_res.total_cost < gbest_cost - 1e-6:
                    gbest_cost = ref_res.total_cost
                    gbest_routes = ref_res.routes
                    gbest_vehicles = ref_res.vehicles_used
                    gbest_keys = permutation_to_random_keys(flat_perm)
                    stagnation_count = 0

        convergence.append(gbest_cost)

        stagnation_count += 1
        if stagnation_count >= early_stop_patience:
            # Reached patience threshold
            break

    # Final refinement for QPSO-H if not already done on last iteration
    if hybrid and gbest_routes:
        refined = refine_routes(gbest_routes, problem)
        flat_perm = [c for r in refined for c in r]
        if len(flat_perm) == n:
            ref_res = optimal_split(problem, flat_perm)
            if ref_res.total_cost < gbest_cost - 1e-6:
                gbest_cost = ref_res.total_cost
                gbest_routes = ref_res.routes
                gbest_vehicles = ref_res.vehicles_used
                gbest_keys = permutation_to_random_keys(flat_perm)
                convergence.append(gbest_cost)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    algo_name = "qpso_h" if hybrid else "qpso"
    level = "L2" if hit_time_budget else "L1"

    return SolverResult(
        algorithm=algo_name,
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
