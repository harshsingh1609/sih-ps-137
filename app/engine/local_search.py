"""Local search operators: 2-opt and Or-opt route improvements."""

from __future__ import annotations
from typing import List, Sequence
import numpy as np
from app.engine.problem import CVRPProblem


def two_opt_route(route: List[int], problem: CVRPProblem) -> List[int]:
    """
    Perform 2-opt local search on a single route until no improving move is found.
    Preserves customer set, demand, and depot boundaries.
    """
    if len(route) < 3:
        return list(route)

    improved = True
    best_route = list(route)
    cost_matrix = problem.cost_matrix
    best_cost = problem.route_cost(best_route)

    while improved:
        improved = False
        m = len(best_route)
        for i in range(m - 1):
            prev_node = 0 if i == 0 else best_route[i - 1]
            for j in range(i + 1, m):
                next_node = 0 if j == m - 1 else best_route[j + 1]

                # Evaluate candidate reverse of segment [i..j]
                cand = best_route[:i] + best_route[i : j + 1][::-1] + best_route[j + 1 :]
                cand_cost = problem.route_cost(cand)

                if cand_cost < best_cost - 1e-7:
                    best_route = cand
                    best_cost = cand_cost
                    improved = True
                    break
            if improved:
                break

    return best_route


def or_opt_route(route: List[int], problem: CVRPProblem) -> List[int]:
    """
    Perform Or-opt local search on a single route by relocating blocks of 1, 2, or 3
    consecutive customers to other positions within the route.
    """
    if len(route) < 3:
        return list(route)

    improved = True
    best_route = list(route)
    best_cost = problem.route_cost(best_route)

    while improved:
        improved = False
        m = len(best_route)

        # Block sizes 3, 2, 1
        for block_len in (3, 2, 1):
            if block_len >= m:
                continue

            for i in range(m - block_len + 1):
                block = best_route[i : i + block_len]
                rem = best_route[:i] + best_route[i + block_len :]

                # Try inserting block into every possible position in rem
                for insert_pos in range(len(rem) + 1):
                    # Skip trivial no-op relocation
                    if insert_pos == i:
                        continue

                    cand = rem[:insert_pos] + block + rem[insert_pos:]
                    cand_cost = problem.route_cost(cand)

                    if cand_cost < best_cost - 1e-7:
                        best_route = cand
                        best_cost = cand_cost
                        improved = True
                        break

                if improved:
                    break
            if improved:
                break

    return best_route


def refine_routes(routes: List[List[int]], problem: CVRPProblem) -> List[List[int]]:
    """
    Apply 2-opt and Or-opt refinement to each individual route in a solution.
    Guarantees no customers are lost or added, and capacities remain strictly valid.
    """
    refined = []
    for r in routes:
        if len(r) <= 1:
            refined.append(list(r))
            continue
        r2 = two_opt_route(r, problem)
        ro = or_opt_route(r2, problem)
        refined.append(ro)
    return refined
