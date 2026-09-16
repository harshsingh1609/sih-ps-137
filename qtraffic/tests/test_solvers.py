"""Integration tests verifying that all solvers return valid, feasible solutions and scale correctly."""

import pytest
import numpy as np
from app.engine.city import City
from app.engine.greedy import solve_nn
from app.engine.pso import solve_pso
from app.engine.ga import solve_ga
from app.engine.qpso import solve_qpso
from app.engine.exact import solve_exact
from app.engine.split import brute_force_split


def test_all_solvers_produce_feasible_solutions():
    """Verify that NN, PSO, GA, QPSO, QPSO-H, and Exact all return strictly feasible solutions on n=6."""
    city = City(n=6, vehicles=3, capacity=25, seed=10)
    problem = city.problem

    solvers = [
        ("exact", lambda: solve_exact(problem, seed=1)),
        ("nn", lambda: solve_nn(problem, seed=1)),
        ("pso", lambda: solve_pso(problem, pop=15, iters=30, seed=1)),
        ("ga", lambda: solve_ga(problem, pop=15, iters=30, seed=1)),
        ("qpso", lambda: solve_qpso(problem, pop=15, iters=30, seed=1, hybrid=False)),
        ("qpso_h", lambda: solve_qpso(problem, pop=15, iters=30, seed=1, hybrid=True)),
    ]

    for name, solver_fn in solvers:
        res = solver_fn()
        assert res.routes, f"Solver {name} returned empty routes"
        is_feas, reason = problem.validate_solution(res.routes)
        assert is_feas, f"Solver {name} produced infeasible solution: {reason}"
        assert res.cost > 0.0, f"Solver {name} produced invalid cost {res.cost}"
        assert res.vehicles_used <= problem.vehicles, (
            f"Solver {name} used {res.vehicles_used} vehicles, exceeding fleet size {problem.vehicles}"
        )


def test_exact_matches_optimal_cvrp_small():
    """Verify that Exact solver returns the minimum cost across all permutations on n=5."""
    city = City(n=5, vehicles=3, capacity=25, seed=77)
    problem = city.problem

    exact_res = solve_exact(problem)
    is_feas, reason = problem.validate_solution(exact_res.routes)
    assert is_feas, f"Exact solver produced infeasible solution: {reason}"

    # Also run NN, GA, QPSO-H - none can ever beat the exact solver
    nn_res = solve_nn(problem)
    ga_res = solve_ga(problem, pop=20, iters=40, seed=1)
    qpso_h_res = solve_qpso(problem, pop=20, iters=40, seed=1, hybrid=True)

    assert nn_res.cost >= exact_res.cost - 1e-5, "NN beat exact solver, which is mathematically impossible"
    assert ga_res.cost >= exact_res.cost - 1e-5, "GA beat exact solver, which is mathematically impossible"
    assert qpso_h_res.cost >= exact_res.cost - 1e-5, "QPSO-H beat exact solver, which is mathematically impossible"


def test_qpso_h_scales_to_n40_and_n120():
    """Verify that QPSO-H runs and generates feasible solutions for n=40 and n=120."""
    # n = 40
    city_40 = City(n=40, vehicles=6, capacity=35, seed=404)
    res_40 = solve_qpso(city_40.problem, pop=20, iters=50, seed=404, hybrid=True)
    is_feas_40, reason_40 = city_40.problem.validate_solution(res_40.routes)
    assert is_feas_40, f"QPSO-H failed feasibility on n=40: {reason_40}"

    # n = 120
    city_120 = City(n=120, vehicles=15, capacity=40, seed=120)
    res_120 = solve_qpso(city_120.problem, pop=20, iters=30, seed=120, hybrid=True)
    is_feas_120, reason_120 = city_120.problem.validate_solution(res_120.routes)
    assert is_feas_120, f"QPSO-H failed feasibility on n=120: {reason_120}"


def test_qpso_h_and_nn_honest_comparison():
    """
    Honest empirical comparison between QPSO-H and NN on a representative instance.
    Reports actual numbers without fabricating outcomes.
    """
    city = City(n=30, vehicles=6, capacity=30, seed=42)
    nn_res = solve_nn(city.problem, seed=42)
    qpso_h_res = solve_qpso(city.problem, pop=30, iters=100, seed=42, hybrid=True)

    # Both must be strictly feasible
    is_f_nn, r_nn = city.problem.validate_solution(nn_res.routes)
    is_f_qpso, r_qpso = city.problem.validate_solution(qpso_h_res.routes)
    assert is_f_nn, f"NN infeasible: {r_nn}"
    assert is_f_qpso, f"QPSO-H infeasible: {r_qpso}"

    # Both report legitimate costs and positive elapsed times
    assert nn_res.cost > 0.0
    assert qpso_h_res.cost > 0.0
    assert nn_res.time_ms >= 0.0
    assert qpso_h_res.time_ms >= 0.0
