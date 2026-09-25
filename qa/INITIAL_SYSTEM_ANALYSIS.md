# Q-Traffic (SIH26137) - Initial System Analysis

## 1. System Architecture Overview

Q-Traffic is a quantum-inspired intelligent traffic route optimization platform for Capacitated Vehicle Routing Problems (CVRP) with dynamic traffic conditions.

### Architecture Layers

1. **Frontend Layer (`public/` & `app/static/`)**:
   - Single-page application using pure vanilla HTML5, CSS3, and JavaScript (Canvas 2D rendering).
   - High-DPI 2D Canvas rendering for city network graphs, routes, and circular congestion zones.
   - 2D Canvas convergence chart tracking objective cost decay over optimization iterations.
   - REST API integration via asynchronous `fetch()` calls to `/api/city`, `/api/solve`, `/api/congestion`, `/api/replan`, `/api/benchmark`.
   - UI metrics dashboard: total travel cost, active vehicles vs. fleet capacity, computation time (ms), and fallback ladder indicator badge.

2. **Backend & API Layer (`api/index.py`, `app/main.py`, `app/api/routes.py`)**:
   - FastAPI ASGI application.
   - Entry point for serverless deployment (`api/index.py`) equipped with `VercelPathMiddleware` to decode `x-forwarded-uri` and `x-matched-path` headers due to Vercel route rewrites.
   - Pydantic models for request validation and type enforcement (`CityCreateRequest`, `SolveRequest`, `CongestionRequest`, `ReplanRequest`).
   - Global exception handling and CORS configuration.

3. **Optimization & Metaheuristic Engine (`app/engine/`)**:
   - **`CVRPProblem` (`problem.py`)**: Core problem representation, coordinate management, integer customer demands, Euclidean distance matrix, congestion overlay cost matrix, and strict CVRP feasibility validator (`validate_solution`).
   - **Random-Key Encoding (`encode.py`)**: Continuous $[0, 1]^n$ particle position encoding mapped bijectively via stable `argsort` to giant-tour customer permutations $\{1, \dots, n\}$, and reverse mapping `permutation_to_random_keys`.
   - **Optimal Split DP (`split.py`)**: Prins' dynamic programming split algorithm partitioning giant tours into capacity-feasible routes with $O(1)$ prefix-sum demand checks and $O(n^2)$ worst-case time complexity.
   - **Local Search (`local_search.py`)**: 2-opt edge-exchange and Or-opt customer block relocation (block lengths 3, 2, 1) intra-route local improvement operators.
   - **QPSO & QPSO-H (`qpso.py`)**: Quantum-behaved Particle Swarm Optimization with mean best position ($m_{\text{best}}$), stochastic local attractor $P_{ij}$, double-exponential wave-function quantum position updates with time-annealed contraction-expansion coefficient $\beta(t) = 1.0 - 0.5(t/T)$, and periodic Lamarckian local search refinement.
   - **Classical Baselines (`greedy.py`, `pso.py`, `ga.py`, `exact.py`)**: Nearest Neighbor (NN), Classical PSO with velocity clipping and inertia decay, Genetic Algorithm (GA) with Order Crossover (OX) and swap mutation, and exhaustive $O(n!)$ permutation exact solver for $n \le 8$.
   - **Dynamic Replanning (`replanner.py`)**: Side-by-side execution of cold-start (random initial swarm) vs. warm-start (swarm seeded with previous best keys perturbed with Gaussian noise) under dynamic traffic congestion.
   - **Empirical Benchmarking (`benchmark.py`)**: Multi-seed, multi-algorithm batch benchmark comparing mean cost, standard deviation, best cost, runtime, and iterations to 99% convergence.

4. **State Store & Caching Layer (`app/core/store.py`)**:
   - Thread-safe in-memory singleton `store` with tempfile disk persistence (`qtraffic_state/`) for serverless re-hydration across stateless Vercel invocations.
   - Deterministic city ID reconstruction pattern (`c_{n}_{vehicles}_{capacity}_{seed}`).

---

## 2. Request Flow

1. **City Generation**:
   - Client sends `POST /api/city` with `{n, vehicles, capacity, seed}`.
   - Server initializes `City`, constructs `CVRPProblem`, populates coordinates and customer demands, registers the city in `store`, and persists state to tempfile.
   - Server returns JSON: `{city_id, depot, customers, vehicles, capacity, seed, active_congestion}`.

2. **Route Optimization**:
   - Client sends `POST /api/solve` with `{city_id, algorithm, pop, iters, seed}`.
   - Server loads `city` from `store` (or re-hydrates from tempfile / deterministic ID).
   - Validates instance constraints (e.g., $n \le 8$ for exact solver).
   - Dispatches problem to selected solver function (`solve_qpso`, `solve_pso`, `solve_ga`, `solve_nn`, `solve_exact`).
   - Solver returns `SolverResult`.
   - Result is saved to `store` (solution and best keys).
   - Returns JSON: `{routes, cost, vehicles_used, level, stale, time_ms, convergence}`.

3. **Congestion Injection**:
   - Client sends `POST /api/congestion` with `{city_id, x, y, radius, factor}`.
   - Server fetches `city`, applies `CongestionZone` to `problem`, updates cost matrix via segment-to-point geometric distance calculations, and updates persisted state.

4. **Replanning**:
   - Client sends `POST /api/replan` with `{city_id, algorithm, warm, pop, iters, seed}`.
   - Server retrieves previous solution keys from `store`.
   - Runs `run_replan()` executing both cold and warm solvers.
   - Saves selected solution to `store` and returns comparative JSON.

5. **Benchmarking**:
   - Client sends `GET /api/benchmark?n=...&vehicles=...&capacity=...&seeds=...&pop=...&iters=...`.
   - Server executes all 5 baseline algorithms over all seeds on identical problem instances.
   - Returns aggregated statistics summary table.

---

## 3. Optimization Flow (QPSO-H Pipeline)

$$\text{Problem Initialization} \longrightarrow X \in [0, 1]^{M \times n} \longrightarrow \text{Random-Key Decode} \longrightarrow \text{Giant Tour} \longrightarrow \text{Optimal Split DP} \longrightarrow \text{Evaluate } f(X)$$

At each generation $t = 1 \dots T$:
1. Check execution time against `time_budget_sec` (if configured).
2. Calculate Mean Best Position:
   $$m_{\text{best}} = \frac{1}{M} \sum_{i=1}^M p_{\text{best}, i}$$
3. Anneal contraction-expansion parameter:
   $$\beta(t) = 1.0 - 0.5 \left(\frac{t}{T}\right)$$
4. For each particle $i = 1 \dots M$:
   - Generate $\phi \sim U(0, 1)^n$, $u \sim U(10^{-12}, 1)^n$, $s \in \{-1, +1\}^n$.
   - Calculate stochastic local attractor:
     $$P_i = \phi \odot p_{\text{best}, i} + (1 - \phi) \odot g_{\text{best}}$$
   - Update particle position via quantum state vector:
     $$X_i(t+1) = P_i + s \odot \beta(t) \odot |m_{\text{best}} - X_i(t)| \odot \ln(1/u)$$
   - Clip coordinates strictly to $[0, 1]$.
   - Decode random keys $\to$ permutation $\to$ optimal split $\to$ evaluate cost.
   - Update $p_{\text{best}, i}$ and $g_{\text{best}}$.
5. **Lamarckian Local Search**:
   - Every `local_search_interval` (default 10 iterations) and at final iteration:
   - Apply 2-opt and Or-opt operators to current $g_{\text{best}}$ routes.
   - If refined routes yield strictly lower cost:
     - Update $g_{\text{best}}$ routes and cost.
     - Inverse-encode improved customer tour back to continuous random keys ($g_{\text{best}} \leftarrow \text{keys}$).
     - Reset stagnation counter.

---

## 4. Fallback Ladder Architecture

The system implements a defined 6-tier reliability ladder:
- **L0 (Exact)**: Ground-truth exhaustive permutation solver for $n \le 8$.
- **L1 (Full Heuristic/Metaheuristic)**: Full iteration QPSO-H / QPSO / PSO / GA run completed within constraints.
- **L2 (Time-Budget Bounded)**: Metaheuristic ran out of time budget; returns best-solution-so-far with full validity.
- **L3 (Alternative Baseline)**: Standard alternative heuristic solver (e.g., PSO, GA).
- **L4 (Deterministic Greedy)**: Nearest Neighbor heuristic coupled with optimal split DP.
- **L5 (Stale Cache Fallback)**: In event of an uncaught solver exception in `POST /api/solve`, returns the previously cached solution from `store` with `stale: true`.

---

## 5. Deployment Flow (Vercel Serverless)

- **Entry Point**: `vercel.json` configures `api/index.py` using `@vercel/python` (Python runtime on AWS Lambda base) and static asset routing to `public/**`.
- **Path Routing**: `VercelPathMiddleware` strips URL rewrites and restores original path from `x-forwarded-uri` header.
- **Stateless Nature**: Each serverless invocation may run in an isolated or freshly spun-up container. State stored in memory (`store._cities`, `store._last_results`) is not guaranteed to persist across requests.
- **State Re-hydration**: `AppStore` uses temporary filesystem cache in `/tmp/qtraffic_state/` and deterministic ID decoding `c_{n}_{vehicles}_{capacity}_{seed}` to reconstruct problem instances when memory state is absent.

---

## 6. Suspected Failure Points

1. **Repository Structure Discrepancy (Root vs. `qtraffic/`)**:
   - There are two parallel code trees: `./app/` and `./qtraffic/app/`.
   - Commit `12248049` modified files in `./qtraffic/app/` (`routes.py`, `city.py`, `benchmark.py`, `replanner.py`, `main.py`) but did NOT update `./app/`.
   - `api/index.py` adds `ROOT_DIR` first to `sys.path`. When imported, Python resolves `./app/main.py`, not `./qtraffic/app/main.py`!
   - Consequently, fixes made in `qtraffic/app/` (such as `time_budget_sec`, deterministic city IDs without random UUID suffixes, etc.) were partially bypassed or inconsistent depending on execution context.

2. **L5 Fallback Trigger Conditions**:
   - `POST /api/solve` catches `Exception as exc:` and calls `store.get_last_solution(req.city_id)`.
   - If an exception occurs on initial optimization when no prior solution exists, `cached` is `None` and an HTTP 500 error is returned.
   - But if a city already had a solution cached (e.g. from an earlier run or if the user clicks "Optimize" a second time or after city re-use), any exception in `solve_qpso` drops directly into L5 fallback.
   - If an exception occurred during the initial solve on the deployed site, why did it display L5? Could a previous solve have completed, or was a default solution populated, or did a subsequent request fail?
   - What specific exception occurred in `solve_qpso`? Numerical issue? Runtime timeout? Parameter mismatch? Fleet constraint penalty?

3. **Fleet Size vs. Route Count Penalty**:
   - `CVRPProblem.evaluate_routes` and `optimal_split` add a massive fleet penalty:
     `penalty = 10000.0 * (vehicles_used - vehicles)` if `vehicles_used > vehicles`.
   - On the live server, for $n=40$, $K=6$, $Q=30$, optimal split returned 8 routes ($>6$), leading to a massive penalty in cost. Could that cause issues in convergence or numerical stability?

4. **Timeouts on Serverless Platform**:
   - Vercel hobby tier has a strict 10-second (or 15-second) serverless function timeout.
   - In `./app/api/routes.py`, `TIME_BUDGET` was NOT passed to `solve_qpso`, whereas in `./qtraffic/app/api/routes.py`, `TIME_BUDGET = 8.0` was passed.
   - If `solve_qpso(n=40, pop=40, iters=300)` takes longer than 10 seconds on Vercel's CPU, the Lambda is abruptly killed with a 504 Gateway Timeout or an uncaught exception, or local search takes too long.

5. **Dynamic Traffic Cost Matrix Calculation**:
   - `CVRPProblem._recompute_cost_matrix` checks segment distances to circular congestion zones. Does it handle zero radius or edge boundaries without division by zero?

6. **Random Key to Permutation Inversion**:
   - `permutation_to_random_keys` in `encode.py`:
     `keys[idx] = float(pos) / float(n - 1)`.
   - When keys are passed back to `random_keys_to_permutation(keys)`:
     `argsort` is used. If multiple positions have duplicate keys or floating point precision issues, does `argsort` reconstruct the exact permutation?

---

## 7. Files Responsible for Each Subsystem

| Subsystem | Workspace Files |
| :--- | :--- |
| **Frontend Application** | `public/index.html`, `public/app.js`, `public/style.css` |
| **Serverless Function Entry** | `api/index.py`, `vercel.json` |
| **FastAPI App & Endpoints** | `app/main.py`, `app/api/routes.py` (and `qtraffic/app/...`) |
| **State Storage & Cache** | `app/core/store.py` (and `qtraffic/app/core/store.py`) |
| **CVRP Problem & Constraints** | `app/engine/problem.py` |
| **City Generation** | `app/engine/city.py` |
| **Random-Key Encoding** | `app/engine/encode.py` |
| **Optimal Split DP** | `app/engine/split.py` |
| **Local Search (2-opt, Or-opt)** | `app/engine/local_search.py` |
| **QPSO / QPSO-H Optimizer** | `app/engine/qpso.py` |
| **Baseline Solvers** | `app/engine/greedy.py`, `app/engine/pso.py`, `app/engine/ga.py`, `app/engine/exact.py` |
| **Dynamic Replanner** | `app/engine/replanner.py` |
| **Benchmarking System** | `app/engine/benchmark.py` |
| **Automated Tests** | `qtraffic/tests/` |
