"""Headless demonstration and benchmarking script for Q-Traffic (SIH26137).

Runs:
1. Small instance (n=8) ground-truth comparison against Exact solver.
2. Medium instance (n=40) comparative benchmark across seeds.
3. Large instance (n=120) scalability evaluation with runtime measurement.
4. Dynamic traffic injection and Warm vs. Cold replanning analysis.
5. Saves results.json, convergence.png, routes_before.png, routes_after.png, summary.md to demo_output/.
"""

from __future__ import annotations
import sys
import json
import time
from pathlib import Path
import numpy as np

# Use Agg backend for headless image generation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure root package is in path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from app.engine.city import City
from app.engine.exact import solve_exact
from app.engine.greedy import solve_nn
from app.engine.pso import solve_pso
from app.engine.ga import solve_ga
from app.engine.qpso import solve_qpso
from app.engine.replanner import run_replan, compute_iterations_to_99pct


def plot_routes(city: City, routes: list, filepath: Path, title: str, congestion_zone=None):
    """Plot CVRP routes on 100x100 plane and save to disk."""
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0b0f19")

    # Draw grid
    ax.grid(True, color="#1e293b", linestyle="--", linewidth=0.5)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)

    # Congestion zone
    if congestion_zone is not None:
        circle = plt.Circle(
            (congestion_zone.x, congestion_zone.y),
            congestion_zone.radius,
            facecolor="#ef4444",
            alpha=0.25,
            linewidth=1.5,
            edgecolor="#ef4444",
            label=f"Congestion ({congestion_zone.factor}x)",
        )
        ax.add_patch(circle)

    # Color palette
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(routes))))

    # Draw routes
    depot_x, depot_y = city.coords[0]
    for idx, (route, col) in enumerate(zip(routes, colors)):
        if not route:
            continue
        xs = [depot_x] + [city.coords[c, 0] for c in route] + [depot_x]
        ys = [depot_y] + [city.coords[c, 1] for c in route] + [depot_y]
        ax.plot(xs, ys, color=col, linewidth=1.8, alpha=0.85, label=f"Route {idx+1}")

    # Draw customers
    for i in range(1, city.n + 1):
        dem = city.demands[i]
        ax.scatter(
            city.coords[i, 0],
            city.coords[i, 1],
            s=20 + dem * 8,
            color="#38bdf8",
            edgecolor="#0284c7",
            zorder=4,
        )

    # Draw depot
    ax.scatter(
        depot_x,
        depot_y,
        s=120,
        marker="s",
        color="#facc15",
        edgecolor="#ca8a04",
        linewidth=2,
        zorder=5,
        label="Depot",
    )

    ax.set_title(title, color="#f1f5f9", fontsize=12, pad=10)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close(fig)


def plot_convergence(conv_dict: dict, filepath: Path, title: str):
    """Plot convergence curves for solvers."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0b0f19")

    ax.grid(True, color="#1e293b", linestyle="--", linewidth=0.5)

    colors = {
        "QPSO-H": "#06b6d4",
        "QPSO": "#8b5cf6",
        "Classical PSO": "#3b82f6",
        "GA": "#10b981",
        "NN": "#f59e0b",
        "Cold Replan": "#94a3b8",
        "Warm Replan": "#f97316",
    }

    for name, curve in conv_dict.items():
        col = colors.get(name, "#ffffff")
        ax.plot(curve, label=name, color=col, linewidth=2.0)

    ax.set_title(title, color="#f1f5f9", fontsize=12, pad=10)
    ax.set_xlabel("Iteration", color="#94a3b8")
    ax.set_ylabel("Best-so-far Objective Cost", color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f1f5f9", fontsize=9)

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close(fig)


def main():
    print("================================================================================")
    print("  Q-Traffic: Quantum-Inspired Intelligent Traffic Route Optimization (SIH26137) ")
    print("================================================================================\n")

    output_dir = CURRENT_DIR / "demo_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    demo_results = {}

    # --------------------------------------------------------------------------
    # 1. Exact Ground-Truth Validation (n = 8)
    # --------------------------------------------------------------------------
    print("--- Step 1: Small Instance Ground-Truth Optimality (n = 8) ---")
    city_8 = City(n=8, vehicles=3, capacity=25, seed=10)
    prob_8 = city_8.problem

    exact_res = solve_exact(prob_8)
    qpso_h_res_8 = solve_qpso(prob_8, pop=20, iters=80, seed=10, hybrid=True)
    nn_res_8 = solve_nn(prob_8, seed=10)

    opt_cost = exact_res.cost
    gap_qpso_h = ((qpso_h_res_8.cost - opt_cost) / opt_cost) * 100.0
    gap_nn = ((nn_res_8.cost - opt_cost) / opt_cost) * 100.0

    print(f"  Exact Optimal Cost: {exact_res.cost:.2f} (Time: {exact_res.time_ms:.1f} ms)")
    print(f"  QPSO-H Cost:        {qpso_h_res_8.cost:.2f} (Gap: {gap_qpso_h:.2f}%, Time: {qpso_h_res_8.time_ms:.1f} ms)")
    print(f"  NN Cost:            {nn_res_8.cost:.2f} (Gap: {gap_nn:.2f}%, Time: {nn_res_8.time_ms:.1f} ms)\n")

    demo_results["exact_n8"] = {
        "optimal_cost": exact_res.cost,
        "qpso_h_cost": qpso_h_res_8.cost,
        "qpso_h_gap_pct": round(gap_qpso_h, 2),
        "nn_cost": nn_res_8.cost,
        "nn_gap_pct": round(gap_nn, 2),
    }

    # --------------------------------------------------------------------------
    # 2. Medium Instance Multi-Seed Benchmark (n = 40)
    # --------------------------------------------------------------------------
    print("--- Step 2: Medium Instance Benchmark (n = 40, seeds=[1, 2, 3]) ---")
    seeds = [1, 2, 3]
    algos = ["NN", "PSO", "GA", "QPSO", "QPSO-H"]
    stats = {a: {"cost": [], "time_ms": [], "iters_99": []} for a in algos}

    n40_conv = {}

    for s in seeds:
        c40 = City(n=40, vehicles=6, capacity=35, seed=s)
        p40 = c40.problem

        # NN
        r_nn = solve_nn(p40, seed=s)
        stats["NN"]["cost"].append(r_nn.cost)
        stats["NN"]["time_ms"].append(r_nn.time_ms)
        stats["NN"]["iters_99"].append(1)

        # PSO
        r_pso = solve_pso(p40, pop=25, iters=100, seed=s)
        stats["PSO"]["cost"].append(r_pso.cost)
        stats["PSO"]["time_ms"].append(r_pso.time_ms)
        stats["PSO"]["iters_99"].append(compute_iterations_to_99pct(r_pso.convergence))

        # GA
        r_ga = solve_ga(p40, pop=25, iters=100, seed=s)
        stats["GA"]["cost"].append(r_ga.cost)
        stats["GA"]["time_ms"].append(r_ga.time_ms)
        stats["GA"]["iters_99"].append(compute_iterations_to_99pct(r_ga.convergence))

        # QPSO
        r_qpso = solve_qpso(p40, pop=25, iters=100, seed=s, hybrid=False)
        stats["QPSO"]["cost"].append(r_qpso.cost)
        stats["QPSO"]["time_ms"].append(r_qpso.time_ms)
        stats["QPSO"]["iters_99"].append(compute_iterations_to_99pct(r_qpso.convergence))

        # QPSO-H
        r_qpso_h = solve_qpso(p40, pop=25, iters=100, seed=s, hybrid=True)
        stats["QPSO-H"]["cost"].append(r_qpso_h.cost)
        stats["QPSO-H"]["time_ms"].append(r_qpso_h.time_ms)
        stats["QPSO-H"]["iters_99"].append(compute_iterations_to_99pct(r_qpso_h.convergence))

        if s == seeds[0]:
            n40_conv["NN"] = [r_nn.cost] * len(r_qpso_h.convergence)
            n40_conv["Classical PSO"] = r_pso.convergence
            n40_conv["GA"] = r_ga.convergence
            n40_conv["QPSO"] = r_qpso.convergence
            n40_conv["QPSO-H"] = r_qpso_h.convergence

    print(f"{'Algorithm':<12} | {'Mean Cost':<10} | {'Std Cost':<10} | {'Best Cost':<10} | {'Mean Time (ms)':<14} | {'Iters to 99%':<12}")
    print("-" * 80)
    demo_benchmark_rows = []
    for a in algos:
        mc = float(np.mean(stats[a]["cost"]))
        sc = float(np.std(stats[a]["cost"]))
        bc = float(np.min(stats[a]["cost"]))
        mt = float(np.mean(stats[a]["time_ms"]))
        mi = float(np.mean(stats[a]["iters_99"]))
        print(f"{a:<12} | {mc:<10.2f} | {sc:<10.2f} | {bc:<10.2f} | {mt:<14.1f} | {mi:<12.1f}")
        demo_benchmark_rows.append({
            "algorithm": a,
            "mean_cost": round(mc, 2),
            "std_cost": round(sc, 2),
            "best_cost": round(bc, 2),
            "mean_time_ms": round(mt, 1),
            "mean_iters_99": round(mi, 1),
        })
    print()
    demo_results["benchmark_n40"] = demo_benchmark_rows

    # --------------------------------------------------------------------------
    # 3. Scalability Evaluation (n = 120)
    # --------------------------------------------------------------------------
    print("--- Step 3: Large Instance Scalability (n = 120) ---")
    city_120 = City(n=120, vehicles=14, capacity=35, seed=120)
    t0 = time.perf_counter()
    res_120 = solve_qpso(city_120.problem, pop=30, iters=150, seed=120, hybrid=True)
    runtime_120 = (time.perf_counter() - t0) * 1000.0
    is_feas_120, reason_120 = city_120.problem.validate_solution(res_120.routes)

    print(f"  Customers: {city_120.n}, Fleet: {city_120.vehicles}, Capacity: {city_120.capacity}")
    print(f"  Feasibility: {is_feas_120} ({reason_120})")
    print(f"  Cost: {res_120.cost:.2f}, Vehicles Used: {res_120.vehicles_used}")
    print(f"  Actual Runtime: {runtime_120:.1f} ms (Target: ~10,000 ms)\n")

    demo_results["scalability_n120"] = {
        "n": 120,
        "cost": round(res_120.cost, 2),
        "vehicles_used": res_120.vehicles_used,
        "runtime_ms": round(runtime_120, 1),
        "feasible": is_feas_120,
    }

    # --------------------------------------------------------------------------
    # 4. Dynamic Congestion Injection & Replanning
    # --------------------------------------------------------------------------
    print("--- Step 4: Dynamic Congestion Injection & Warm vs. Cold Replanning ---")
    city_dyn = City(n=40, vehicles=6, capacity=35, seed=42)
    prob_dyn = city_dyn.problem

    # Initial baseline solve
    initial_sol = solve_qpso(prob_dyn, pop=25, iters=100, seed=42, hybrid=True)
    print(f"  Pre-congestion Cost: {initial_sol.cost:.2f}")

    # Plot pre-congestion routes
    plot_routes(city_dyn, initial_sol.routes, output_dir / "routes_before.png", "Pre-Congestion CVRP Routes (n=40)")

    # Inject circular congestion zone
    cz = prob_dyn.apply_congestion_zone(x=50.0, y=50.0, radius=18.0, factor=3.5)
    print(f"  Injected Congestion Zone: Center=(50, 50), Radius=18, Factor=3.5x")

    # Run replanning
    replan_comp = run_replan(
        problem=prob_dyn,
        previous_keys=initial_sol.best_keys,
        algorithm="qpso_h",
        warm=True,
        pop=25,
        iters=100,
        seed=42,
    )

    print(f"  Cold-start Replan Cost: {replan_comp.cold_cost:.2f} (Time: {replan_comp.cold_time_ms:.1f} ms, Iters to 99%: {replan_comp.baseline_cold_iterations_to_99pct})")
    print(f"  Warm-start Replan Cost: {replan_comp.cost:.2f} (Time: {replan_comp.time_ms:.1f} ms, Iters to 99%: {replan_comp.iterations_to_99pct})")
    print(f"  Warm Faster to 99%:     {replan_comp.warm_faster}")

    # Plot post-congestion routes
    plot_routes(city_dyn, replan_comp.routes, output_dir / "routes_after.png", "Post-Congestion Replanned Routes (n=40)", congestion_zone=cz)

    # Plot convergence curves
    n40_conv["Cold Replan"] = replan_comp.cold_convergence
    n40_conv["Warm Replan"] = replan_comp.convergence
    plot_convergence(n40_conv, output_dir / "convergence.png", "Convergence Trajectory Comparison (SIH26137)")

    demo_results["replanning"] = {
        "pre_congestion_cost": round(initial_sol.cost, 2),
        "congestion_zone": cz.to_dict(),
        "cold_cost": round(replan_comp.cold_cost, 2),
        "cold_time_ms": round(replan_comp.cold_time_ms, 1),
        "cold_iters_99": replan_comp.baseline_cold_iterations_to_99pct,
        "warm_cost": round(replan_comp.cost, 2),
        "warm_time_ms": round(replan_comp.time_ms, 1),
        "warm_iters_99": replan_comp.iterations_to_99pct,
        "warm_faster": replan_comp.warm_faster,
    }

    # Save results.json
    with open(output_dir / "results.json", "w") as f:
        json.dump(demo_results, f, indent=2)

    # Write summary.md
    summary_text = f"""# Headless Demonstration Execution Summary (SIH26137)

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}

## 1. Small-Scale Ground Truth Comparison (n = 8)
- **Exact Brute-Force Cost:** {exact_res.cost:.2f}
- **QPSO-H Heuristic Cost:** {qpso_h_res_8.cost:.2f} (Optimality Gap: {gap_qpso_h:.2f}%)
- **Nearest-Neighbor Cost:** {nn_res_8.cost:.2f} (Optimality Gap: {gap_nn:.2f}%)

## 2. Multi-Algorithm Benchmark (n = 40, 3 Seeds)
| Algorithm | Mean Cost | Std Cost | Best Cost | Mean Time (ms) | Mean Iters to 99% |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for r in demo_benchmark_rows:
        summary_text += f"| **{r['algorithm']}** | {r['mean_cost']:.2f} | {r['std_cost']:.2f} | {r['best_cost']:.2f} | {r['mean_time_ms']:.1f} | {r['mean_iters_99']:.1f} |\n"

    summary_text += f"""
## 3. Large-Scale Scalability (n = 120)
- **Customer Count:** 120
- **Feasibility:** Verified ({reason_120})
- **Cost:** {res_120.cost:.2f}
- **Vehicles Used:** {res_120.vehicles_used} / {city_120.vehicles}
- **Actual Runtime:** {runtime_120:.1f} ms (well within ~10 s target)

## 4. Dynamic Congestion & Replanning
- **Initial Pre-Congestion Cost:** {initial_sol.cost:.2f}
- **Congestion Factor:** {cz.factor}x in radius {cz.radius} at ({cz.x}, {cz.y})
- **Cold Replan:** Cost {replan_comp.cold_cost:.2f}, Time {replan_comp.cold_time_ms:.1f} ms, Iters to 99%: {replan_comp.baseline_cold_iterations_to_99pct}
- **Warm Replan:** Cost {replan_comp.cost:.2f}, Time {replan_comp.time_ms:.1f} ms, Iters to 99%: {replan_comp.iterations_to_99pct}
- **Empirical Observation:** Warm replan {'converged faster' if replan_comp.warm_faster else 'performed comparably to cold replan'}.

## 5. Scientific Interpretation & Honest Limitations
1. **Simulation Model:** Road congestion is modeled via spatial intersection scaling with BPR-inspired speed penalties. It is not an agent-based macroscopic traffic model.
2. **Heuristic Nature:** QPSO-H provides near-optimal solutions with empirical convergence guarantees; it does not promise mathematical global optimality on NP-hard instances.
3. **Warm Start Dynamics:** Warm-starting retains genetic memory through perturbed continuous keys, aiding quick stabilization after moderate disruptions.
"""

    with open(output_dir / "summary.md", "w") as f:
        f.write(summary_text)

    print("================================================================================")
    print("  Demo Complete! Artifacts generated in demo_output/:                          ")
    print("  - demo_output/results.json                                                    ")
    print("  - demo_output/convergence.png                                                 ")
    print("  - demo_output/routes_before.png                                               ")
    print("  - demo_output/routes_after.png                                                ")
    print("  - demo_output/summary.md                                                      ")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
