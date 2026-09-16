"""Test comparing dynamic-programming split cost against brute-force enumeration for small instances."""

import pytest
import numpy as np
from app.engine.city import City
from app.engine.encode import random_keys_to_permutation
from app.engine.split import optimal_split, brute_force_split


def test_optimal_split_matches_brute_force_50_instances():
    """
    For 50 small random instances (n <= 7):
    - Compute optimal split using dynamic programming.
    - Compute optimal split using brute force over all cut patterns.
    - Assert that both methods agree in total cost and travel cost.
    """
    rng = np.random.default_rng(12345)

    for trial in range(50):
        n = int(rng.integers(2, 8))  # n in [2, 7]
        vehicles = int(rng.integers(2, 6))
        capacity = int(rng.integers(15, 30))
        seed = int(rng.integers(1, 100000))

        city = City(n=n, vehicles=vehicles, capacity=capacity, seed=seed)
        problem = city.problem

        # Random giant tour
        keys = rng.uniform(0.0, 1.0, size=n)
        giant_tour = random_keys_to_permutation(keys)

        # DP split
        dp_result = optimal_split(problem, giant_tour)

        # Brute-force split
        bf_result = brute_force_split(problem, giant_tour)

        # Costs must match closely
        assert np.isclose(dp_result.travel_cost, bf_result.travel_cost, atol=1e-5), (
            f"Trial {trial} (n={n}): DP travel cost {dp_result.travel_cost} != "
            f"BF travel cost {bf_result.travel_cost}"
        )
        assert np.isclose(dp_result.total_cost, bf_result.total_cost, atol=1e-5), (
            f"Trial {trial} (n={n}): DP total cost {dp_result.total_cost} != "
            f"BF total cost {bf_result.total_cost}"
        )
        assert dp_result.vehicles_used == bf_result.vehicles_used, (
            f"Trial {trial} (n={n}): DP vehicles {dp_result.vehicles_used} != "
            f"BF vehicles {bf_result.vehicles_used}"
        )
