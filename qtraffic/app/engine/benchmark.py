"""Multi-seed comparative benchmarking framework for CVRP solvers."""

from __future__ import annotations
from typing import List, Dict, Any, Sequence
import numpy as np

from app.engine.city import City
from app.engine.greedy import solve_nn
from app.engine.pso import solve_pso
from app.engine.ga import solve_ga
from app.engine.qpso import solve_qpso, SolverResult
from app.engine.replanner import compute_iterations_to_99pct


def run_benchmark(
    n: int = 40,
    vehicles: int = 6,
    capacity: int = 30,
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
    pop: int = 20,
    iters: int = 100,
) -> List[Dict[str, Any]]:
    """
    Run empirical benchmarks comparing NN, PSO, GA, QPSO, and QPSO-H across multiple seeds.
    All returned data is computed from genuine executions.
    """
    algorithms = ["nn", "pso", "ga", "qpso", "qpso_h"]
    results_by_algo: Dict[str, Dict[str, List[float]]] = {
        algo: {
            "costs": [],
            "times": [],
            "iters_99": [],
        }
        for algo in algorithms
    }

    # Budget per solver call: 1.5s keeps total benchmark time reasonable on Vercel.
    BENCH_BUDGET = 1.5

    for s in seeds:
        seed_val = int(s)
        city = City(n=n, vehicles=vehicles, capacity=capacity, seed=seed_val)
        prob = city.problem

        # 1. NN
        nn_res = solve_nn(prob, seed=seed_val)
        results_by_algo["nn"]["costs"].append(nn_res.cost)
        results_by_algo["nn"]["times"].append(nn_res.time_ms)
        results_by_algo["nn"]["iters_99"].append(1.0)

        # 2. PSO
        pso_res = solve_pso(prob, pop=pop, iters=iters, seed=seed_val, time_budget_sec=BENCH_BUDGET)
        results_by_algo["pso"]["costs"].append(pso_res.cost)
        results_by_algo["pso"]["times"].append(pso_res.time_ms)
        results_by_algo["pso"]["iters_99"].append(float(compute_iterations_to_99pct(pso_res.convergence)))

        # 3. GA
        ga_res = solve_ga(prob, pop=pop, iters=iters, seed=seed_val, time_budget_sec=BENCH_BUDGET)
        results_by_algo["ga"]["costs"].append(ga_res.cost)
        results_by_algo["ga"]["times"].append(ga_res.time_ms)
        results_by_algo["ga"]["iters_99"].append(float(compute_iterations_to_99pct(ga_res.convergence)))

        # 4. QPSO
        qpso_res = solve_qpso(prob, pop=pop, iters=iters, seed=seed_val, hybrid=False, time_budget_sec=BENCH_BUDGET)
        results_by_algo["qpso"]["costs"].append(qpso_res.cost)
        results_by_algo["qpso"]["times"].append(qpso_res.time_ms)
        results_by_algo["qpso"]["iters_99"].append(float(compute_iterations_to_99pct(qpso_res.convergence)))

        # 5. QPSO-H
        qpso_h_res = solve_qpso(prob, pop=pop, iters=iters, seed=seed_val, hybrid=True, time_budget_sec=BENCH_BUDGET)
        results_by_algo["qpso_h"]["costs"].append(qpso_h_res.cost)
        results_by_algo["qpso_h"]["times"].append(qpso_h_res.time_ms)
        results_by_algo["qpso_h"]["iters_99"].append(float(compute_iterations_to_99pct(qpso_h_res.convergence)))

    rows: List[Dict[str, Any]] = []
    for algo in algorithms:
        data = results_by_algo[algo]
        costs = np.array(data["costs"], dtype=np.float64)
        times = np.array(data["times"], dtype=np.float64)
        i99 = np.array(data["iters_99"], dtype=np.float64)

        rows.append({
            "algorithm": algo,
            "mean_cost": float(round(np.mean(costs), 2)),
            "std_cost": float(round(np.std(costs), 2)),
            "best_cost": float(round(np.min(costs), 2)),
            "mean_time_ms": float(round(np.mean(times), 2)),
            "mean_iters_to_99pct": float(round(np.mean(i99), 1)),
        })

    return rows
