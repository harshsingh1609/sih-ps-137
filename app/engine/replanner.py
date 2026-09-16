"""Dynamic replanning engine comparing cold-start vs. warm-start optimization after congestion events."""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np

from app.engine.problem import CVRPProblem
from app.engine.qpso import solve_qpso, SolverResult
from app.engine.pso import solve_pso
from app.engine.ga import solve_ga
from app.engine.greedy import solve_nn


def compute_iterations_to_99pct(convergence: List[float]) -> int:
    """
    Compute number of iterations to reach within 1% of the final cost:
    threshold = final_cost * 1.01 (or final_cost + 0.01 * (initial_cost - final_cost)).
    """
    if not convergence:
        return 0
    final_cost = convergence[-1]
    init_cost = convergence[0]
    # Reaching 99% of total improvement
    threshold = final_cost + 0.01 * max(0.0, init_cost - final_cost)

    for it, c in enumerate(convergence):
        if c <= threshold + 1e-6:
            return it
    return len(convergence) - 1


@dataclass
class ReplanComparison:
    routes: List[List[int]]
    cost: float
    time_ms: float
    iterations_to_99pct: int
    warm: bool
    baseline_cold_iterations_to_99pct: int
    cold_cost: float
    cold_time_ms: float
    warm_faster: bool
    convergence: List[float]
    cold_convergence: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "routes": self.routes,
            "cost": float(round(self.cost, 4)),
            "time_ms": float(round(self.time_ms, 2)),
            "iterations_to_99pct": int(self.iterations_to_99pct),
            "warm": bool(self.warm),
            "baseline_cold_iterations_to_99pct": int(self.baseline_cold_iterations_to_99pct),
            "cold_cost": float(round(self.cold_cost, 4)),
            "cold_time_ms": float(round(self.cold_time_ms, 2)),
            "warm_faster": bool(self.warm_faster),
            "convergence": [float(round(c, 4)) for c in self.convergence],
            "cold_convergence": [float(round(c, 4)) for c in self.cold_convergence],
        }


def run_replan(
    problem: CVRPProblem,
    previous_keys: Optional[np.ndarray],
    algorithm: str = "qpso_h",
    warm: bool = True,
    pop: int = 40,
    iters: int = 300,
    seed: Optional[int] = None,
) -> ReplanComparison:
    """
    Execute cold and warm replanning under current (congested) problem conditions.
    Always runs both to provide honest comparative metrics.
    """
    hybrid = (algorithm == "qpso_h")
    cold_seed = seed if seed is not None else 42
    warm_seed = (seed + 1000) if seed is not None else 1042

    # 1. Cold Replanning (no prior information)
    cold_res = solve_qpso(
        problem=problem,
        pop=pop,
        iters=iters,
        seed=cold_seed,
        hybrid=hybrid,
        initial_keys=None,
    )
    cold_iters_99 = compute_iterations_to_99pct(cold_res.convergence)

    # 2. Warm Replanning (seeded with previous keys)
    warm_res = solve_qpso(
        problem=problem,
        pop=pop,
        iters=iters,
        seed=warm_seed,
        hybrid=hybrid,
        initial_keys=previous_keys,
    )
    warm_iters_99 = compute_iterations_to_99pct(warm_res.convergence)

    warm_faster = warm_res.time_ms < cold_res.time_ms or warm_iters_99 < cold_iters_99

    selected = warm_res if warm else cold_res
    selected_iters_99 = warm_iters_99 if warm else cold_iters_99

    return ReplanComparison(
        routes=selected.routes,
        cost=selected.cost,
        time_ms=selected.time_ms,
        iterations_to_99pct=selected_iters_99,
        warm=warm,
        baseline_cold_iterations_to_99pct=cold_iters_99,
        cold_cost=cold_res.cost,
        cold_time_ms=cold_res.time_ms,
        warm_faster=warm_faster,
        convergence=selected.convergence,
        cold_convergence=cold_res.convergence,
    )
