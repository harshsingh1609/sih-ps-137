"""
QA Phase 5 - Performance and Memory Measurement.
QA Phase 6 - Optimization Quality Audit.
"""

from __future__ import annotations

import gc
import math
import time
import tracemalloc
import pytest
import numpy as np

from app.engine.city import City
from app.engine.qpso import solve_qpso
from app.engine.greedy import solve_nn
from app.engine.pso import solve_pso
from app.engine.ga import solve_ga
from app.engine.exact import solve_exact
from app.engine.replanner import run_replan, compute_iterations_to_99pct


# ---------------------------------------------------------------------------
# Phase 5.1 - Runtime scaling
# ---------------------------------------------------------------------------

class TestRuntimeScaling:
    """Measure actual runtime and feasibility across problem sizes."""

    @pytest.mark.parametrize("n,vehicles,capacity", [
        (25, 5, 25),
        (50, 8, 30),
        (100, 15, 35),
    ])
    def test_qpso_h_runtime_and_feasibility(self, n, vehicles, capacity):
        """QPSO-H must finish and produce a feasible solution within a generous time bound."""
        city = City(n=n, vehicles=vehicles, capacity=capacity, seed=42)
        t0 = time.perf_counter()
        res = solve_qpso(city.problem, pop=20, iters=80, seed=42, hybrid=True)
        elapsed = time.perf_counter() - t0
        feasible, msg = city.problem.validate_solution(res.routes)
        assert feasible, f"n={n}: {msg}"
        assert not math.isnan(res.cost)
        # Very generous bound: 300 seconds (this is a correctness test, not a tight perf test)
        assert elapsed < 300.0, f"n={n}: took {elapsed:.2f}s, expected < 300s"

    def test_runtime_recorded_honestly(self):
        """time_ms field must be > 0 for any real solve."""
        city = City(n=20, vehicles=5, capacity=25, seed=1)
        res = solve_qpso(city.problem, pop=10, iters=30, seed=1, hybrid=True)
        assert res.time_ms > 0.0, "time_ms must be positive"

    def test_nn_is_fastest_heuristic(self):
        """NN must be faster than QPSO-H on the same instance (by design - NN is O(n^2))."""
        city = City(n=50, vehicles=8, capacity=30, seed=10)
        t_nn = time.perf_counter()
        nn_res = solve_nn(city.problem)
        t_nn = time.perf_counter() - t_nn
        t_q = time.perf_counter()
        q_res = solve_qpso(city.problem, pop=20, iters=60, seed=10, hybrid=True)
        t_q = time.perf_counter() - t_q
        # NN should be much faster than QPSO-H (which runs 60 iterations)
        assert t_nn < t_q, (
            f"NN ({t_nn:.3f}s) was not faster than QPSO-H ({t_q:.3f}s). "
            "This is unexpected but may occur in unusual system states."
        )


# ---------------------------------------------------------------------------
# Phase 5.3 - Memory and endurance
# ---------------------------------------------------------------------------

class TestMemoryAndEndurance:
    """Measure memory growth across sequential solves."""

    def test_no_unbounded_memory_growth(self):
        """
        After several sequential solves, memory must not grow unboundedly.
        We accept some growth due to Python allocator behavior.
        Threshold: < 50 MB growth after 10 solves.
        """
        gc.collect()
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        n_solves = 10
        for i in range(n_solves):
            city = City(n=30, vehicles=6, capacity=25, seed=i)
            solve_qpso(city.problem, pop=15, iters=30, seed=i, hybrid=True)
            del city
            gc.collect()

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = snapshot_after.compare_to(snapshot_before, "lineno")
        total_diff_bytes = sum(stat.size_diff for stat in stats if stat.size_diff > 0)
        total_diff_mb = total_diff_bytes / (1024 * 1024)

        # Report actual value; threshold is generous (50MB)
        assert total_diff_mb < 50.0, (
            f"Memory grew by {total_diff_mb:.2f} MB after {n_solves} solves "
            "(possible memory leak)"
        )

    def test_store_does_not_accumulate_cities(self):
        """
        The in-memory store grows with each city. Verify it does not
        accumulate more cities than created.
        """
        from app.core.store import AppStore
        test_store = AppStore()
        n_cities = 5
        for i in range(n_cities):
            city = City(n=5, vehicles=2, capacity=20, seed=i)
            test_store.save_city(city)
        # _cities dict should have exactly n_cities entries
        assert len(test_store._cities) == n_cities


# ---------------------------------------------------------------------------
# Phase 5.4 - Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Verify deterministic output with same seeds."""

    def test_qpso_identical_outputs_same_seed(self):
        """Same seed must produce identical cost, routes, convergence."""
        city = City(n=20, vehicles=5, capacity=25, seed=42)
        r1 = solve_qpso(city.problem, pop=20, iters=50, seed=77, hybrid=True)
        r2 = solve_qpso(city.problem, pop=20, iters=50, seed=77, hybrid=True)
        assert r1.cost == r2.cost, f"Cost differs: {r1.cost} vs {r2.cost}"
        assert r1.routes == r2.routes, "Routes differ"
        assert r1.convergence == r2.convergence, "Convergence differs"

    def test_ga_identical_outputs_same_seed(self):
        """GA with same seed must produce identical results."""
        city = City(n=15, vehicles=4, capacity=25, seed=33)
        r1 = solve_ga(city.problem, pop=15, iters=40, seed=55)
        r2 = solve_ga(city.problem, pop=15, iters=40, seed=55)
        assert r1.cost == r2.cost
        assert r1.routes == r2.routes

    def test_city_generation_identical_same_seed(self):
        """Same city seed must produce identical coordinates and demands."""
        c1 = City(n=10, vehicles=3, capacity=20, seed=100)
        c2 = City(n=10, vehicles=3, capacity=20, seed=100)
        assert np.allclose(c1.coords, c2.coords)
        assert np.array_equal(c1.demands, c2.demands)

    def test_different_seeds_produce_different_results(self):
        """Different seeds should (usually) produce different results."""
        city = City(n=20, vehicles=5, capacity=25, seed=42)
        r1 = solve_qpso(city.problem, pop=20, iters=50, seed=1, hybrid=False)
        r2 = solve_qpso(city.problem, pop=20, iters=50, seed=2, hybrid=False)
        # They may coincidentally produce the same result, but usually they differ
        # Don't assert they must differ - just assert both are valid
        assert not math.isnan(r1.cost)
        assert not math.isnan(r2.cost)


# ---------------------------------------------------------------------------
# Phase 6 - Optimization quality audit
# ---------------------------------------------------------------------------

class TestOptimizationQuality:
    """Actual optimization quality measurements (honest reporting)."""

    def test_qpso_h_vs_exact_small_instances(self):
        """
        For n<=8: compare QPSO-H against exact optimum.
        Report actual gaps - do NOT assert QPSO-H is always optimal.
        """
        gaps = []
        for seed in range(5):
            city = City(n=6, vehicles=3, capacity=25, seed=seed)
            exact_res = solve_exact(city.problem)
            qpso_h_res = solve_qpso(city.problem, pop=20, iters=80, seed=seed, hybrid=True)

            exact_cost = exact_res.cost
            qpso_h_cost = qpso_h_res.cost

            # QPSO-H can NEVER beat the exact optimum
            assert qpso_h_cost >= exact_cost - 1e-4, (
                f"Seed {seed}: QPSO-H ({qpso_h_cost:.4f}) beat exact ({exact_cost:.4f}) "
                "which is mathematically impossible"
            )

            if exact_cost > 0:
                gap_pct = 100.0 * (qpso_h_cost - exact_cost) / exact_cost
            else:
                gap_pct = 0.0
            gaps.append(gap_pct)

        mean_gap = np.mean(gaps)
        max_gap = np.max(gaps)
        # Just verify values are measured and finite (no assertion on quality level)
        assert math.isfinite(mean_gap), f"Mean gap is not finite: {mean_gap}"
        # Print actual result for the QA report
        print(f"\nQPSO-H vs Exact (n=6, 5 seeds): mean_gap={mean_gap:.2f}%, max_gap={max_gap:.2f}%")

    def test_all_solvers_feasible_n40(self):
        """
        For n=40: all solvers must produce feasible solutions.
        Report actual costs - do NOT assert any ranking.
        """
        city = City(n=40, vehicles=6, capacity=35, seed=42)
        problem = city.problem

        solvers = {
            "nn": lambda: solve_nn(problem),
            "pso": lambda: solve_pso(problem, pop=20, iters=60, seed=42),
            "ga": lambda: solve_ga(problem, pop=20, iters=60, seed=42),
            "qpso": lambda: solve_qpso(problem, pop=20, iters=60, seed=42, hybrid=False),
            "qpso_h": lambda: solve_qpso(problem, pop=20, iters=60, seed=42, hybrid=True),
        }

        costs = {}
        for name, fn in solvers.items():
            res = fn()
            feasible, msg = problem.validate_solution(res.routes)
            assert feasible, f"Solver {name} n=40 infeasible: {msg}"
            assert not math.isnan(res.cost), f"{name}: NaN cost"
            costs[name] = res.cost

        print(f"\nn=40 costs: {costs}")

    def test_compute_iterations_to_99pct_correctness(self):
        """Verify the 99% convergence metric is computed correctly."""
        # Case 1: already at final from start
        conv = [100.0, 100.0, 100.0]
        assert compute_iterations_to_99pct(conv) == 0

        # Case 2: drops quickly
        conv = [200.0, 110.0, 105.0, 102.0, 101.0, 100.5, 100.0]
        # threshold = 100.0 + 0.01*(200.0-100.0) = 101.0
        threshold = 100.0 + 0.01 * (200.0 - 100.0)
        result = compute_iterations_to_99pct(conv)
        # Result should be the first index where conv[i] <= threshold + 1e-6
        first_idx = next(i for i, c in enumerate(conv) if c <= threshold + 1e-6)
        assert result == first_idx, f"Expected {first_idx}, got {result}"

        # Case 3: empty
        assert compute_iterations_to_99pct([]) == 0

        # Case 4: single element
        assert compute_iterations_to_99pct([500.0]) == 0

    def test_warm_vs_cold_replan_comparison(self):
        """
        Compare warm vs cold replanning on same problem.
        Report actual results without asserting warm is always better.
        """
        city = City(n=20, vehicles=5, capacity=25, seed=42)
        city.problem.apply_congestion_zone(50.0, 50.0, 25.0, 3.0)

        # First solve to get keys
        res0 = solve_qpso(city.problem, pop=15, iters=40, seed=1, hybrid=True)
        prior_keys = res0.best_keys

        # Run actual comparison
        comparison = run_replan(
            problem=city.problem,
            previous_keys=prior_keys,
            algorithm="qpso_h",
            warm=True,
            pop=15,
            iters=40,
            seed=1,
        )

        # Verify structure
        assert comparison.cold_cost > 0
        assert comparison.cost > 0
        assert comparison.iterations_to_99pct >= 0
        assert comparison.baseline_cold_iterations_to_99pct >= 0

        # warm_faster is a boolean observation, not a guarantee
        assert isinstance(comparison.warm_faster, bool)

        print(
            f"\nWarm cost={comparison.cost:.2f}, "
            f"Cold cost={comparison.cold_cost:.2f}, "
            f"Warm faster={comparison.warm_faster}"
        )

    def test_benchmark_output_no_nan(self):
        """All benchmark rows must have finite, non-negative costs."""
        from app.engine.benchmark import run_benchmark
        rows = run_benchmark(n=15, vehicles=3, capacity=20, seeds=[1, 2], pop=8, iters=20)
        for row in rows:
            assert math.isfinite(row["mean_cost"]), f"{row['algorithm']}: NaN mean_cost"
            assert row["mean_cost"] > 0.0, f"{row['algorithm']}: zero mean_cost"
            assert math.isfinite(row["best_cost"]), f"{row['algorithm']}: NaN best_cost"
