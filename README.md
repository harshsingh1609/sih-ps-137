# Q-Traffic: Quantum-Inspired Intelligent Traffic Route Optimization

> **Smart India Hackathon 2026 &bull; Problem ID: SIH26137**  
> **Topic:** Quantum-Inspired Intelligent Traffic Route Optimization using Quantum-behaved Particle Swarm Optimization (QPSO-H) for the Capacitated Vehicle Routing Problem (CVRP).

---

## 1. Project Overview & Problem Interpretation

Urban freight distribution and smart city logistics grapple with traffic congestion, vehicle capacity limitations, and dynamic road disruptions. The classical Capacitated Vehicle Routing Problem (CVRP) is strongly NP-hard. Real-world conditions require dynamic recalculation when congestion bottlenecks emerge.

**SIH26137: Quantum-Inspired Intelligent Traffic Route Optimization** seeks advanced optimization paradigms to rapidly deliver capacity-feasible, low-cost routing plans under evolving traffic states. This prototype implements **Quantum-behaved Particle Swarm Optimization (QPSO)** combined with **Lamarckian Local Search (QPSO-H)**, compared against classical baselines (Classical PSO, Genetic Algorithm, Nearest-Neighbor heuristic, and Exact Enumeration) on a dynamic 2D synthetic transportation network.

### Important Scientific Disclaimers
- **Simulated Traffic Model:** The traffic simulation utilizes localized spatial cost scaling and BPR-inspired travel-time penalties on a 2D Euclidean coordinate network. It is not an agent-based macroscopic physical road simulation and does not ingest live GPS/external telemetry.
- **Heuristic Nature:** QPSO and QPSO-H are stochastic metaheuristics. Solutions are near-optimal with empirical convergence guarantees; mathematical global optimality cannot be guaranteed for large NP-hard instances.
- **No Fabricated Data:** All benchmark metrics, runtimes, and warm vs. cold comparisons reported in this system are computed directly from genuine algorithm executions.

---

## 2. System Architecture

```text
+-------------------------------------------------------------------------------+
|                      Q-Traffic Full-Stack Architecture                         |
+-------------------------------------------------------------------------------+
                                        |
                 +----------------------+----------------------+
                 |                                             |
                 v                                             v
     +------------------------+                   +------------------------+
     |   Canvas 2D Frontend   |                   |    Headless Runner     |
     | (HTML5/CSS3/Vanilla JS)|                   |     (run_demo.py)      |
     +------------------------+                   +------------------------+
                 |                                             |
                 | HTTP REST Calls                             | Direct Engine
                 v                                             v
     +-------------------------------------------------------------------------+
     |                       FastAPI Application Server                        |
     |                  (/health, /api/city, /api/solve,                       |
     |                 /api/congestion, /api/replan, /api/benchmark)           |
     +-------------------------------------------------------------------------+
                                        |
                 +----------------------+----------------------+
                 |                                             |
                 v                                             v
     +------------------------+                   +------------------------+
     | Dynamic Traffic Model  |                   |  Fallback Solver Ladder|
     | (Spatial BPR Penalty)  |                   |   (L0 -> L1 -> ... L5) |
     +------------------------+                   +------------------------+
                 |                                             |
                 +----------------------+----------------------+
                                        |
                                        v
     +-------------------------------------------------------------------------+
     |                             Solvers Engine                              |
     |  - QPSO-H (Lamarckian Hybrid: 2-opt + Or-opt)                           |
     |  - Pure QPSO (Quantum Delta Potential Wavefunction)                     |
     |  - Classical PSO (Inertia Weight + Velocity Clamping)                   |
     |  - Genetic Algorithm (OX Crossover + Swap Mutation)                     |
     |  - Nearest-Neighbor (Greedy Insertion + DP Split)                      |
     |  - Exact Solver (Brute-force Permutations for n <= 8)                   |
     +-------------------------------------------------------------------------+
                                        |
                 +----------------------+----------------------+
                 |                                             |
                 v                                             v
     +------------------------+                   +------------------------+
     |  Random-Key Encoding   |                   |  Optimal Split DP      |
     |   X in [0, 1]^n <->    |                   |  (Prins' Algorithm     |
     |  Customer Permutations |                   |   with O(1) Prefix Sum)|
     +------------------------+                   +------------------------+
```

---

## 3. Mathematical CVRP Formulation

Let depot be indexed at $0$, and customer set be $C = \{1, 2, \dots, n\}$ with 2D Euclidean coordinates $(x_i, y_i) \in [0, 100]^2$.

- Customer demand: $d_i \in [1, 10]$ for $i \in C$, $d_0 = 0$.
- Homogeneous fleet: $K$ vehicles, each with maximum capacity $Q$.
- If any customer demand $d_i > Q$, the instance is rejected immediately with an unprocessable entity error.

### Constraints & Feasibility
A solution partition $S = \{R_1, \dots, R_m\}$ must strictly satisfy:
1. **Uniqueness & Coverage:** $\bigcup_{k=1}^m R_k = C$ and $R_a \cap R_b = \emptyset$ for $a \ne b$.
2. **Depot Boundaries:** Every route begins and ends at depot $0$: $R_k = (0, v_1, \dots, v_{|R_k|}, 0)$.
3. **Capacity Constraint:** $\sum_{v \in R_k} d_v \le Q$ for all $k \in \{1, \dots, m\}$.

### Objective Function
Minimize total route travel cost plus fleet excess penalty:
$$\text{Cost}(S) = \sum_{k=1}^m \text{Cost}(R_k) + 10\,000 \times \max(0, m - K)$$
where:
$$\text{Cost}(R_k) = C_{0, v_1} + \sum_{p=1}^{|R_k|-1} C_{v_p, v_{p+1}} + C_{v_{|R_k|}, 0}$$
and $C_{ij}$ is the traffic-adjusted traversal cost.

---

## 4. Algorithmic Foundation

### 4.1 Random-Key Permutation Encoding
Continuous particle swarm algorithms explore continuous spaces. We map continuous positions $X \in [0, 1]^n$ to permutations $\pi$ using stable argsort:
$$\pi = \text{argsort}(X) + 1$$
Inverse mapping (permutation to random keys):
$$X_{\pi_p - 1} = \frac{p}{n - 1} \quad (0 \le p < n)$$

### 4.2 Optimal-Split Dynamic Programming (Prins' Algorithm)
Given giant tour $\pi$, optimal partitioning into vehicle routes is solved via dynamic programming:
- Segment demand $\Delta(i, j) = \sum_{k=i}^j d_{\pi_k}$ is evaluated in $O(1)$ time using precomputed prefix sums.
- DP Recurrence:
  $$V(0) = 0, \quad V(j) = \min_{0 \le i < j, \Delta(i+1, j) \le Q} \{ V(i) + c(i+1, j) \}$$
- Optimal predecessor pointer $P(j) = i^*$ backtracks in $O(n)$ time to extract exact routes.

### 4.3 Quantum-behaved Particle Swarm Optimization (QPSO)
Unlike classical PSO, quantum particles do not maintain velocity vectors or inertia. The particles move in a quantum delta potential well:
- **Mean Best Position:**
  $$mbest_j = \frac{1}{M} \sum_{i=1}^M pbest_{ij}$$
- **Local Attractor:**
  $$P_{ij} = \phi \cdot pbest_{ij} + (1 - \phi) \cdot gbest_j, \quad \phi \sim U(0, 1)$$
- **Quantum State Update:**
  $$X_{ij}(t+1) = P_{ij} \pm \beta(t) \cdot |mbest_j - X_{ij}(t)| \cdot \ln(1/u), \quad u \sim U(0, 1)$$
- **Linear Annealing:**
  $$\beta(t) = 1.0 - 0.5 \frac{t}{T}$$
- All position components are strictly clamped to $[0, 1]$.

### 4.4 QPSO-H Lamarckian Hybridization
Every 10 iterations and at termination, QPSO-H applies local search operators to $gbest$:
1. **2-opt:** Reverses subsegments within each route until no improving move exists.
2. **Or-opt:** Relocates chains of 1, 2, or 3 consecutive customers to other positions within the route if travel cost is reduced.
3. The improved routes are flattened into a giant tour, re-encoded to random keys, and reinjected into $gbest$ (Lamarckian reinjection).

---

## 5. Dynamic Traffic & Warm-Start Replanning

### 5.1 Congestion Zone Model
A congestion event is defined by center $(x_c, y_c)$, radius $r$, and congestion factor $f \in [1, 10]$. Any network edge intersecting or falling within radius $r$ has its travel cost multiplied by $f$.

### 5.2 Warm-Start vs. Cold-Start Replanning
- **Cold Replan:** Fresh swarm randomly sampled uniformly in $[0, 1]^n$.
- **Warm Replan:**
  - Particle 0: exact prior best keys.
  - Particles $1 \dots \lfloor M/2 \rfloor$: prior keys perturbed by Gaussian noise $\mathcal{N}(0, 0.05)$, clipped to $[0, 1]$.
  - Remaining particles: uniform random in $[0, 1]^n$ to preserve swarm diversity.
- The system measures actual cost, runtime, and iterations to reach 99% of final cost.

---

## 6. Fallback Solver Ladder

| Level | Condition | Active Engine | Description |
| :--- | :--- | :--- | :--- |
| **L0** | $n \le 8$ and exact requested | Exact Solver | Exhaustive branch enumeration finding true mathematical optimum |
| **L1** | Normal operation | QPSO-H | Flagship quantum-inspired Lamarckian hybrid solver |
| **L2** | Time budget reached | Best-so-far | Gracefully returns best-so-far solution within timeout |
| **L3** | Manual baseline selection | Classical GA / PSO | Genetic Algorithm or classical velocity-based PSO |
| **L4** | Tight compute constraint | Nearest-Neighbor | Greedy insertion + DP split |
| **L5** | Unexpected solver exception | Stale Cache Plan | Returns last valid cached solution with `stale: true` |

---

## 7. Quickstart & How to Run

### 7.1 Setup Environment
```bash
# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 7.2 Run Prototype Locally (Single Command)
```bash
python dev.py
```
This launches the full FastAPI + Canvas 2D interactive prototype at `http://127.0.0.1:8000` with hot-reloading enabled. Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

### 7.3 Deploy to Vercel via GitHub (1-Click Working Prototype)

This repository is pre-configured with `vercel.json`, `api/index.py`, and `public/` static assets for direct zero-config Vercel deployment.

#### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "feat: Q-Traffic ready for Vercel deployment"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
git push -u origin main
```

#### Step 2: Import into Vercel
1. Go to [vercel.com](https://vercel.com) and log in.
2. Click **"Add New..."** &rarr; **"Project"**.
3. Select your GitHub repository and click **"Import"**.
4. Keep all default settings (Framework Preset: **Other**, Root Directory: `./`).
5. Click **"Deploy"**.

Within 60 seconds, your live interactive Q-Traffic prototype will be accessible worldwide on your custom `.vercel.app` domain!

### 7.4 Run Automated Tests
```bash
pytest qtraffic/tests -q
```
Runs 141 mathematical equation, adversarial red-team, and HTTP API integration tests.

### 7.5 Run Headless Demonstration
```bash
python run_demo.py
```
Executes the headless evaluation:
- $n = 8$ exact comparison.
- $n = 40$ multi-seed benchmark table.
- $n = 120$ scalability test measuring runtime against the ~10s target.
- Dynamic congestion injection with Warm vs. Cold replanning.
- Generates artifacts in `demo_output/` (`results.json`, `convergence.png`, `routes_before.png`, `routes_after.png`, `summary.md`).

---

## 8. API Reference

### Health Check
- `GET /health` -> `{"status": "ok", "version": "1.0.0"}`

### Generate City
- `POST /api/city`
  - Body: `{"n": 40, "vehicles": 6, "capacity": 30, "seed": 42}`
  - Returns depot coords and customer list with demands.

### Solve CVRP
- `POST /api/solve`
  - Body: `{"city_id": "...", "algorithm": "qpso_h", "pop": 40, "iters": 150, "seed": 42}`
  - Supported: `qpso_h`, `qpso`, `pso`, `ga`, `nn`, `exact` (for $n \le 8$).
  - Returns `routes`, `cost`, `vehicles_used`, `level`, `stale`, `time_ms`, `convergence`.

### Inject Dynamic Congestion
- `POST /api/congestion`
  - Body: `{"city_id": "...", "x": 50.0, "y": 50.0, "radius": 15.0, "factor": 3.0}`
  - Returns injected zone details.

### Replan Routes
- `POST /api/replan`
  - Body: `{"city_id": "...", "algorithm": "qpso_h", "warm": true, "pop": 40, "iters": 150, "seed": 42}`
  - Returns replanned routes, cost, `iterations_to_99pct`, `baseline_cold_iterations_to_99pct`, `warm_faster`.

### Benchmark
- `GET /api/benchmark?n=40&vehicles=6&capacity=30&seeds=1,2,3&pop=20&iters=80`
  - Returns comparative statistics across NN, PSO, GA, QPSO, QPSO-H.

---

## 9. Known Limitations & Roadmap
1. **Distance Metric:** Uses 2D Euclidean distances with continuous circular congestion masks. Future work will integrate OpenStreetMap graph topology and road hierarchies.
2. **Homogeneous Fleet:** Vehicles currently share identical capacity $Q$. Extension to heterogeneous fleets and multi-depot configurations is planned.
3. **Time Windows:** CVRP does not enforce delivery time windows (VRPTW). Future versions will incorporate service time windows and driver rest constraints.
