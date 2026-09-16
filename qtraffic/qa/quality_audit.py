"""Phase 6 optimization quality audit script."""
import numpy as np
from app.engine.city import City
from app.engine.exact import solve_exact
from app.engine.qpso import solve_qpso
from app.engine.greedy import solve_nn
from app.engine.pso import solve_pso
from app.engine.ga import solve_ga

print("=== PHASE 6: OPTIMIZATION QUALITY AUDIT ===")
print()
print("--- n=6 (<=8): QPSO-H vs Exact (5 seeds) ---")
gaps = []
for seed in range(5):
    city = City(n=6, vehicles=3, capacity=25, seed=seed)
    ex = solve_exact(city.problem)
    qh = solve_qpso(city.problem, pop=20, iters=100, seed=seed, hybrid=True)
    gap_pct = 100.0 * (qh.cost - ex.cost) / ex.cost if ex.cost > 0 else 0.0
    gaps.append(gap_pct)
    feas, _ = city.problem.validate_solution(qh.routes)
    print(f"  seed={seed}: exact={ex.cost:.3f}  qpso_h={qh.cost:.3f}  gap={gap_pct:.2f}%  feasible={feas}")
print(f"  mean_gap={np.mean(gaps):.2f}%  max_gap={np.max(gaps):.2f}%")
print()

print("--- n=40: all solvers (5 seeds) ---")
results = {alg: [] for alg in ["nn", "pso", "ga", "qpso", "qpso_h"]}
for seed in range(1, 6):
    city = City(n=40, vehicles=6, capacity=35, seed=seed)
    p = city.problem
    results["nn"].append(solve_nn(p).cost)
    results["pso"].append(solve_pso(p, pop=20, iters=60, seed=seed).cost)
    results["ga"].append(solve_ga(p, pop=20, iters=60, seed=seed).cost)
    results["qpso"].append(solve_qpso(p, pop=20, iters=60, seed=seed, hybrid=False).cost)
    results["qpso_h"].append(solve_qpso(p, pop=20, iters=60, seed=seed, hybrid=True).cost)
    print(f"  seed={seed} done")

hdr = "  {:<10} {:>12} {:>10} {:>10}".format("Algorithm", "Mean", "Std", "Best")
print(hdr)
for alg, costs in results.items():
    arr = np.array(costs)
    print("  {:<10} {:>12.2f} {:>10.2f} {:>10.2f}".format(alg, arr.mean(), arr.std(), arr.min()))

print()
print("--- n=120: QPSO-H only (3 seeds) ---")
for seed in range(3):
    city = City(n=120, vehicles=15, capacity=40, seed=seed)
    qh = solve_qpso(city.problem, pop=20, iters=80, seed=seed, hybrid=True)
    feas, _ = city.problem.validate_solution(qh.routes)
    print(f"  seed={seed}: cost={qh.cost:.2f}  time={qh.time_ms:.1f}ms  iters={qh.iterations}  feasible={feas}")
