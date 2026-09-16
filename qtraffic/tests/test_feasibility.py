"""Property-based feasibility tests verifying optimal split decoder satisfies all CVRP constraints."""

import pytest
import numpy as np
from app.engine.city import City
from app.engine.encode import random_keys_to_permutation
from app.engine.split import optimal_split


def test_feasibility_200_randomized_instances():
    """
    Run 200 randomized decoder tests verifying:
    - Every customer appears exactly once across all routes.
    - No customer is missing.
    - No duplicate customers exist.
    - Every route starts and ends at the depot.
    - Every route respects vehicle capacity.
    - Fleet penalty is zero when sufficient vehicles are allocated.
    """
    rng = np.random.default_rng(42)

    for i in range(200):
        n = int(rng.integers(5, 50))
        # Generously sized fleet so fleet penalty should be 0
        vehicles = n
        capacity = int(rng.integers(15, 45))
        seed = int(rng.integers(1, 1000000))

        city = City(n=n, vehicles=vehicles, capacity=capacity, seed=seed)
        problem = city.problem

        # Generate random continuous keys and giant tour
        keys = rng.uniform(0.0, 1.0, size=n)
        giant_tour = random_keys_to_permutation(keys)

        assert len(giant_tour) == n, f"Giant tour length mismatch: got {len(giant_tour)}, expected {n}"

        # Decode via optimal split
        split_res = optimal_split(problem, giant_tour)
        routes = split_res.routes

        # 1. Independent validator
        is_feasible, reason = problem.validate_solution(routes)
        assert is_feasible, f"Instance {i} failed feasibility: {reason}"

        # 2. Detailed individual assertion checks
        all_visited = []
        for r_idx, route in enumerate(routes):
            assert len(route) > 0, f"Instance {i}: Empty route produced at index {r_idx}"
            route_demand = sum(problem.demands[c] for c in route)
            assert route_demand <= capacity, (
                f"Instance {i}: Route {r_idx} demand {route_demand} exceeds capacity {capacity}"
            )
            all_visited.extend(route)

        # Exact coverage and uniqueness
        assert sorted(all_visited) == list(range(1, n + 1)), (
            f"Instance {i}: Customer visited list does not match 1..n"
        )
        assert len(all_visited) == len(set(all_visited)), (
            f"Instance {i}: Duplicate customer IDs detected"
        )

        # Zero fleet penalty since vehicles == n
        assert split_res.penalty == 0.0, (
            f"Instance {i}: Expected zero penalty when vehicles={vehicles}, got {split_res.penalty}"
        )
        assert split_res.vehicles_used <= vehicles, (
            f"Instance {i}: Vehicles used ({split_res.vehicles_used}) exceeded available ({vehicles})"
        )
        assert split_res.total_cost == split_res.travel_cost, (
            f"Instance {i}: Total cost must equal travel cost when penalty is 0"
        )
