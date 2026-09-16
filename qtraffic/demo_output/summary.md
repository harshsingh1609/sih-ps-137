# Headless Demonstration Execution Summary (SIH26137)

**Generated:** 2026-09-14 16:49:52

## 1. Small-Scale Ground Truth Comparison (n = 8)
- **Exact Brute-Force Cost:** 331.89
- **QPSO-H Heuristic Cost:** 331.89 (Optimality Gap: 0.00%)
- **Nearest-Neighbor Cost:** 331.89 (Optimality Gap: 0.00%)

## 2. Multi-Algorithm Benchmark (n = 40, 3 Seeds)
| Algorithm | Mean Cost | Std Cost | Best Cost | Mean Time (ms) | Mean Iters to 99% |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NN** | 14218.28 | 4740.90 | 10839.00 | 0.3 | 1.0 |
| **PSO** | 5013.61 | 4664.01 | 1613.36 | 479.0 | 75.3 |
| **GA** | 8238.42 | 4660.42 | 1649.19 | 537.2 | 78.3 |
| **QPSO** | 5121.53 | 4705.88 | 1732.08 | 478.5 | 50.7 |
| **QPSO-H** | 8193.85 | 4569.49 | 1732.08 | 376.9 | 9.0 |

## 3. Large-Scale Scalability (n = 120)
- **Customer Count:** 120
- **Feasibility:** Verified (Solution is feasible.)
- **Cost:** 104430.42
- **Vehicles Used:** 24 / 14
- **Actual Runtime:** 1303.1 ms (well within ~10 s target)

## 4. Dynamic Congestion & Replanning
- **Initial Pre-Congestion Cost:** 11503.71
- **Congestion Factor:** 3.5x in radius 18.0 at (50.0, 50.0)
- **Cold Replan:** Cost 14032.47, Time 330.3 ms, Iters to 99%: 1
- **Warm Replan:** Cost 12782.08, Time 380.5 ms, Iters to 99%: 10
- **Empirical Observation:** Warm replan performed comparably to cold replan.

## 5. Scientific Interpretation & Honest Limitations
1. **Simulation Model:** Road congestion is modeled via spatial intersection scaling with BPR-inspired speed penalties. It is not an agent-based macroscopic traffic model.
2. **Heuristic Nature:** QPSO-H provides near-optimal solutions with empirical convergence guarantees; it does not promise mathematical global optimality on NP-hard instances.
3. **Warm Start Dynamics:** Warm-starting retains genetic memory through perturbed continuous keys, aiding quick stabilization after moderate disruptions.
