"""Optimal-split dynamic programming (Prins' algorithm) for CVRP decoding."""

from __future__ import annotations
from typing import List, Tuple, Sequence
import itertools
import numpy as np
from app.engine.problem import CVRPProblem


class SplitResult:
    """Encapsulates the result of optimal split dynamic programming."""

    def __init__(
        self,
        routes: List[List[int]],
        travel_cost: float,
        penalty: float,
        total_cost: float,
        vehicles_used: int,
    ):
        self.routes = routes
        self.travel_cost = travel_cost
        self.penalty = penalty
        self.total_cost = total_cost
        self.vehicles_used = vehicles_used


def optimal_split(problem: CVRPProblem, giant_tour: Sequence[int]) -> SplitResult:
    """
    Partition giant tour permutation into capacity-feasible routes minimizing total travel cost.
    Uses O(1) prefix-sum demand checks and dynamic programming.
    
    DP Formulation:
    V[0] = 0
    V[j] = min_{0 <= i < j, demand(i, j) <= Q} (V[i] + segment_cost(i, j-1))
    """
    n = len(giant_tour)
    if n == 0:
        return SplitResult(routes=[], travel_cost=0.0, penalty=0.0, total_cost=0.0, vehicles_used=0)

    # Prefix demands for O(1) demand calculation
    prefix_demand = np.zeros(n + 1, dtype=np.int64)
    for idx, c in enumerate(giant_tour):
        prefix_demand[idx + 1] = prefix_demand[idx] + problem.demands[c]

    # V[j] stores optimal travel cost to serve giant_tour[0..j-1]
    V = np.full(n + 1, np.inf, dtype=np.float64)
    pred = np.full(n + 1, -1, dtype=np.int64)
    V[0] = 0.0

    cost_matrix = problem.cost_matrix

    for i in range(n):
        if np.isinf(V[i]):
            continue

        c_first = giant_tour[i]
        depot_to_first = cost_matrix[0, c_first]
        cum_interior = 0.0

        for j in range(i + 1, n + 1):
            seg_demand = prefix_demand[j] - prefix_demand[i]
            if seg_demand > problem.capacity:
                # Demands are positive; extending further will only exceed capacity
                break

            c_last = giant_tour[j - 1]
            if j > i + 1:
                c_prev = giant_tour[j - 2]
                cum_interior += cost_matrix[c_prev, c_last]

            seg_cost = depot_to_first + cum_interior + cost_matrix[c_last, 0]
            cand_cost = V[i] + seg_cost

            if cand_cost < V[j]:
                V[j] = cand_cost
                pred[j] = i

    if np.isinf(V[n]):
        raise RuntimeError("No feasible split found; instance capacity constraints cannot be satisfied.")

    # Backtrack to reconstruct routes
    routes: List[List[int]] = []
    curr = n
    while curr > 0:
        p = pred[curr]
        route = [int(c) for c in giant_tour[p:curr]]
        routes.append(route)
        curr = int(p)

    routes.reverse()
    vehicles_used = len(routes)
    travel_cost = float(V[n])

    penalty = 0.0
    if vehicles_used > problem.vehicles:
        penalty = problem.fleet_penalty_per_vehicle * (vehicles_used - problem.vehicles)

    total_cost = travel_cost + penalty

    return SplitResult(
        routes=routes,
        travel_cost=travel_cost,
        penalty=penalty,
        total_cost=total_cost,
        vehicles_used=vehicles_used,
    )


def brute_force_split(problem: CVRPProblem, giant_tour: Sequence[int]) -> SplitResult:
    """
    Brute-force partition of giant tour into all valid contiguous cuts.
    Used for small n (n <= 8) to independently verify optimal_split correctness.
    """
    n = len(giant_tour)
    if n == 0:
        return SplitResult(routes=[], travel_cost=0.0, penalty=0.0, total_cost=0.0, vehicles_used=0)

    best_cost = float("inf")
    best_routes: List[List[int]] = []

    # Number of cuts can range from 0 to n-1 (1 to n routes)
    indices = list(range(1, n))
    for num_cuts in range(n):
        for cut_points in itertools.combinations(indices, num_cuts):
            splits = [0] + list(cut_points) + [n]
            cand_routes: List[List[int]] = []
            valid = True

            for k in range(len(splits) - 1):
                start, end = splits[k], splits[k + 1]
                subroute = [int(c) for c in giant_tour[start:end]]
                if problem.route_demand(subroute) > problem.capacity:
                    valid = False
                    break
                cand_routes.append(subroute)

            if valid:
                cost = sum(problem.route_cost(r) for r in cand_routes)
                if cost < best_cost:
                    best_cost = cost
                    best_routes = cand_routes

    vehicles_used = len(best_routes)
    penalty = 0.0
    if vehicles_used > problem.vehicles:
        penalty = problem.fleet_penalty_per_vehicle * (vehicles_used - problem.vehicles)

    return SplitResult(
        routes=best_routes,
        travel_cost=best_cost,
        penalty=penalty,
        total_cost=best_cost + penalty,
        vehicles_used=vehicles_used,
    )
