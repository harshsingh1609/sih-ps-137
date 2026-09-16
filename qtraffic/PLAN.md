# Engineering Plan: Quantum-Inspired Intelligent Traffic Route Optimization (SIH26137)

## 1. Executive Summary & Problem Scope
This document specifies the architecture, algorithmic design, and verification plan for the Smart India Hackathon 2026 problem **SIH26137: Quantum-Inspired Intelligent Traffic Route Optimization**.

The objective is to optimize delivery routes for a fleet of capacity-constrained vehicles departing from and returning to a central depot in a 2D Euclidean plane subject to dynamic, localized traffic congestion. The flagship solution methodology is **Quantum-behaved Particle Swarm Optimization with Lamarckian Local Search (QPSO-H)** compared against classical baselines (Classical PSO, Genetic Algorithm, Nearest-Neighbor heuristic, and Brute-Force Exact search).

---

## 2. Mathematical CVRP Formulation
Let $V = \{0, 1, \dots, n\}$ be the set of nodes, where $0$ represents the central depot and $C = \{1, \dots, n\}$ represents customer nodes with coordinates $(x_i, y_i) \in [0, 100]^2$.

- Each customer $i \in C$ has an integer demand $d_i \in [1, 10]$.
- Depot demand is $d_0 = 0$.
- Fleet has $K$ homogeneous vehicles, each with maximum capacity $Q$.
- If any customer has $d_i > Q$, the problem instance is strictly invalid.
- Base travel cost between nodes $i$ and $j$ is Euclidean distance $D_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$.

### Feasibility Conditions
A solution $S = \{R_1, R_2, \dots, R_m\}$ is feasible if and only if:
1. Every customer $i \in C$ appears in exactly one route $R_k$.
2. No customer is duplicated or omitted ($\bigcup_{k=1}^m R_k = C$ and $R_a \cap R_b = \emptyset$ for $a \ne b$).
3. Each route starts at 0 and ends at 0: $R_k = (0, v_{k,1}, v_{k,2}, \dots, v_{k,len}, 0)$.
4. Total demand on any route does not exceed capacity:
   $$\sum_{v \in R_k} d_v \le Q \quad \forall k \in \{1, \dots, m\}$$

### Objective Function
Minimize total route cost including a penalty for fleet size violation:
$$\text{Cost}(S) = \sum_{k=1}^m \text{Cost}(R_k) + 10\,000 \times \max(0, m - K)$$
where:
$$\text{Cost}(R_k) = C_{0, v_{k,1}} + \sum_{p=1}^{|R_k|-1} C_{v_{k,p}, v_{k,p+1}} + C_{v_{k,|R_k|}, 0}$$
and $C_{ij}$ is the traffic-adjusted edge traversal cost.

---

## 3. Algorithmic Components

### 3.1 Random-Key Permutation Encoding
- Each particle's continuous position is a vector $X = (x_1, x_2, \dots, x_n) \in [0, 1]^n$.
- Decoding into a customer permutation (giant tour) $\pi$:
  $$\pi = \text{argsort}(X) + 1$$
- Inverse mapping (permutation to random keys):
  $$x_i = \frac{\text{position of customer } i \text{ in } \pi}{n - 1} \quad (\text{or } 0.5 \text{ if } n=1)$$

### 3.2 Optimal-Split Dynamic Programming (Prins' Algorithm)
Given giant tour $\pi = (\pi_1, \pi_2, \dots, \pi_n)$:
- Segment $(\pi_i, \dots, \pi_j)$ demand: $\Delta(i, j) = \sum_{k=i}^j d_{\pi_k}$, calculated in $O(1)$ via prefix sums.
- Segment cost: $c(i, j) = C_{0, \pi_i} + \sum_{k=i}^{j-1} C_{\pi_k, \pi_{k+1}} + C_{\pi_j, 0}$.
- DP state: $V(0) = 0$; for $j = 1 \dots n$:
  $$V(j) = \min_{0 \le i < j, \Delta(i+1, j) \le Q} \{ V(i) + c(i+1, j) \}$$
- Optimal predecessor pointer $P(j) = i^*$ allows backtracking to retrieve exact vehicle assignments in $O(n)$ time after $O(n^2)$ evaluation.

### 3.3 Quantum-behaved Particle Swarm Optimization (QPSO)
Unlike classical PSO, quantum particles do not maintain velocity vectors. Instead, their state is governed by a delta potential well wave-function:
- Mean personal best position across swarm of $M$ particles:
  $$mbest_j = \frac{1}{M} \sum_{i=1}^M pbest_{ij}$$
- Stochastic local attractor:
  $$P_{ij} = \phi \cdot pbest_{ij} + (1 - \phi) \cdot gbest_j, \quad \phi \sim U(0, 1)$$
- Quantum position update:
  $$X_{ij}(t+1) = P_{ij} \pm \beta(t) \cdot |mbest_j - X_{ij}(t)| \cdot \ln(1/u), \quad u \sim U(0, 1)$$
  where the sign $\pm$ is selected with equal probability $0.5$.
- Contraction-expansion coefficient annealing:
  $$\beta(t) = \beta_0 - (\beta_0 - \beta_1) \frac{t}{T} = 1.0 - 0.5 \frac{t}{T}$$
- Position clipping strictly to $[0, 1]$.

### 3.4 Hybrid Local Search (QPSO-H)
Every 10 iterations and at the end of optimization, Lamarckian local search is invoked:
1. Decode $gbest$ into routes via optimal split.
2. For each route, perform 2-opt moves until no further cost reduction is possible:
   - Reverse subsequence $(p \dots q)$ if $C_{p-1, q} + C_{p, q+1} < C_{p-1, p} + C_{q, q+1}$.
3. For each route, perform Or-opt moves:
   - Relocate consecutive chains of 1, 2, or 3 customers into other positions within the route if cost improves.
4. Flatten improved routes back into a giant tour permutation.
5. Re-encode giant tour to random keys in $[0, 1]^n$.
6. Re-evaluate fitness: if strictly superior, update $gbest$ position and fitness.

### 3.5 Classical Baselines
- **Classical PSO**: Inertia $w$ decreasing from $0.9$ to $0.4$, acceleration $c_1 = c_2 = 1.4962$, velocity clamped to $[-0.3, 0.3]$.
- **Genetic Algorithm (GA)**: Population size 40, tournament selection ($k=3$), Order Crossover (OX), swap mutation ($p=0.02$), elitism count 2.
- **Nearest-Neighbor (NN)**: Greedy insertion starting from depot subject to capacity constraint + optimal split.
- **Exact Solver**: Enumerate customer permutations and optimal splits for $n \le 8$ to establish ground-truth global optimality.

---

## 4. Dynamic Congestion & Replanning Model

### 4.1 Congestion Zone
- Congestion region defined by center $(x_c, y_c)$, radius $r$, and congestion factor $f \in [1, 10]$.
- A connection between node $A$ and node $B$ is affected if the line segment $AB$ intersects or either endpoint falls within distance $r$ of $(x_c, y_c)$.
- Affected edge costs are scaled:
  $$C_{AB} = D_{AB} \times f$$

### 4.2 Warm-Start Replanning
Upon congestion change:
- **Cold Replan**: Fresh swarm initialized with uniform random positions in $[0, 1]^n$.
- **Warm Replan**:
  - Particle 0: exact random keys of pre-congestion best solution.
  - Particles $1 \dots \lfloor M/2 \rfloor$: pre-congestion keys perturbed by Gaussian noise $\mathcal{N}(0, 0.05)$, clipped to $[0, 1]$.
  - Remaining particles: uniform random in $[0, 1]^n$ for diversity.
- Metrics recorded: Final cost, execution time, iterations to 99% of final cost, and empirical speedup flag.

---

## 5. Fallback Solver Ladder
To ensure industrial resilience, the system implements a tiered fallback ladder:
- **L0 (Exact)**: Small instances ($n \le 8$) when exact solution is requested.
- **L1 (Primary)**: QPSO-H full heuristic.
- **L2 (Timeout)**: Best-so-far solution if iteration/time budget expires.
- **L3 (Baseline)**: GA alternative if QPSO-H encounters unresolvable stagnation.
- **L4 (Fast heuristic)**: Nearest-Neighbor + split under tight computational constraints.
- **L5 (Cache)**: Stale valid cached plan (`stale: true`) if an unexpected solver exception occurs.

---

## 6. Implementation Roadmap & Milestones
1. Core data structures & problem representation (`problem.py`, `city.py`)
2. Permutation & random-key encoder/decoder (`encode.py`)
3. Dynamic programming split decoder (`split.py`) & unit tests
4. Pure QPSO solver (`qpso.py`) & numerical property tests
5. Baseline solvers (`pso.py`, `ga.py`, `greedy.py`, `exact.py`)
6. Local search operators (2-opt, Or-opt) & QPSO-H Lamarckian integration (`local_search.py`)
7. Traffic congestion simulator & warm/cold replanner (`replanner.py`)
8. Benchmarking engine (`benchmark.py`)
9. FastAPI backend & endpoints (`app/main.py`, `app/api/routes.py`, `app/core/store.py`)
10. Vanilla HTML5/CSS3 Canvas frontend (`app/static/`)
11. Test suite (`tests/`) & headless demonstration script (`run_demo.py`)
12. Comprehensive README and verification report
