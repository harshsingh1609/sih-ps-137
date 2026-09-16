"""Genetic Algorithm (GA) baseline for CVRP using Order Crossover (OX) and optimal split."""

from __future__ import annotations
import time
from typing import List, Optional, Tuple
import numpy as np

from app.engine.problem import CVRPProblem
from app.engine.encode import permutation_to_random_keys
from app.engine.split import optimal_split
from app.engine.qpso import SolverResult


def order_crossover(p1: List[int], p2: List[int], rng: np.random.Generator) -> List[int]:
    """Order Crossover (OX) for permutation representations."""
    n = len(p1)
    if n <= 2:
        return list(p1)

    cx1, cx2 = sorted(rng.choice(n, size=2, replace=False))
    child = [-1] * n
    # Copy segment from p1
    child[cx1 : cx2 + 1] = p1[cx1 : cx2 + 1]
    copied_set = set(child[cx1 : cx2 + 1])

    # Fill remaining from p2 in cyclic order starting from cx2 + 1
    p2_order = p2[cx2 + 1 :] + p2[: cx2 + 1]
    fill_candidates = [c for c in p2_order if c not in copied_set]

    child_indices = list(range(cx2 + 1, n)) + list(range(0, cx1))
    for idx, cand in zip(child_indices, fill_candidates):
        child[idx] = cand

    return child


def swap_mutation(perm: List[int], p_mut: float, rng: np.random.Generator) -> List[int]:
    """Apply swap mutation with probability p_mut per individual."""
    res = list(perm)
    n = len(res)
    if n <= 1:
        return res
    if rng.uniform(0.0, 1.0) < p_mut:
        i, j = rng.choice(n, size=2, replace=False)
        res[i], res[j] = res[j], res[i]
    return res


def tournament_selection(pop: List[List[int]], costs: List[float], k: int, rng: np.random.Generator) -> List[int]:
    """Select parent using k-way tournament."""
    selected_indices = rng.choice(len(pop), size=k, replace=False)
    best_idx = selected_indices[np.argmin([costs[i] for i in selected_indices])]
    return pop[best_idx]


def solve_ga(
    problem: CVRPProblem,
    pop: int = 40,
    iters: int = 300,
    seed: Optional[int] = None,
    tournament_k: int = 3,
    mutation_prob: float = 0.02,
    elitism: int = 2,
    early_stop_patience: int = 60,
    time_budget_sec: Optional[float] = None,
) -> SolverResult:
    """Genetic Algorithm baseline for CVRP."""
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = problem.n
    P = int(pop)
    G = int(iters)

    # Initial population of random permutations
    base_perm = list(range(1, n + 1))
    population: List[List[int]] = []
    for _ in range(P):
        p = list(base_perm)
        rng.shuffle(p)
        population.append(p)

    # Evaluate initial population
    costs: List[float] = []
    routes_list: List[List[List[int]]] = []
    vehicles_list: List[int] = []

    for indiv in population:
        res = optimal_split(problem, indiv)
        costs.append(res.total_cost)
        routes_list.append(res.routes)
        vehicles_list.append(res.vehicles_used)

    best_idx = int(np.argmin(costs))
    gbest_cost = costs[best_idx]
    gbest_perm = population[best_idx]
    gbest_routes = routes_list[best_idx]
    gbest_vehicles = vehicles_list[best_idx]

    convergence: List[float] = [gbest_cost]

    stagnation_count = 0
    executed_iters = 0
    hit_time_budget = False

    for gen in range(1, G + 1):
        if time_budget_sec is not None and (time.perf_counter() - start_time) >= time_budget_sec:
            hit_time_budget = True
            break

        executed_iters = gen

        # Elitism: retain top `elitism` individuals
        sorted_indices = np.argsort(costs)
        new_pop: List[List[int]] = [population[sorted_indices[i]] for i in range(min(elitism, P))]

        # Produce offspring
        while len(new_pop) < P:
            p1 = tournament_selection(population, costs, tournament_k, rng)
            p2 = tournament_selection(population, costs, tournament_k, rng)
            child = order_crossover(p1, p2, rng)
            child = swap_mutation(child, mutation_prob, rng)
            new_pop.append(child)

        population = new_pop
        costs = []
        routes_list = []
        vehicles_list = []

        for indiv in population:
            res = optimal_split(problem, indiv)
            costs.append(res.total_cost)
            routes_list.append(res.routes)
            vehicles_list.append(res.vehicles_used)

        gen_best_idx = int(np.argmin(costs))
        if costs[gen_best_idx] < gbest_cost:
            gbest_cost = costs[gen_best_idx]
            gbest_perm = population[gen_best_idx]
            gbest_routes = routes_list[gen_best_idx]
            gbest_vehicles = vehicles_list[gen_best_idx]
            stagnation_count = 0
        else:
            stagnation_count += 1

        convergence.append(gbest_cost)
        if stagnation_count >= early_stop_patience:
            break

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    best_keys = permutation_to_random_keys(gbest_perm)
    level = "L2" if hit_time_budget else "L3"

    return SolverResult(
        algorithm="ga",
        routes=gbest_routes,
        cost=gbest_cost,
        vehicles_used=gbest_vehicles,
        time_ms=elapsed_ms,
        convergence=convergence,
        best_keys=best_keys,
        iterations=executed_iters,
        level=level,
        stale=False,
    )
