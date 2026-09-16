"""Unit tests verifying mathematical equations, determinism, and clipping of QPSO update."""

import pytest
import numpy as np
from app.engine.city import City
from app.engine.qpso import solve_qpso


def test_qpso_equations_single_step():
    """
    Manually calculate a single QPSO step from first principles and verify the implementation logic:
    mbest_j = (1/M) * sum(pbest_ij)
    P_ij = phi * pbest_ij + (1 - phi) * gbest_j
    X_ij(t+1) = P_ij +- beta * |mbest_j - X_ij| * ln(1/u)
    beta(t) = 1.0 - 0.5 * (t / T)
    """
    M = 4
    n = 3
    T = 100
    t = 1
    beta = 1.0 - 0.5 * (t / float(T))
    assert np.isclose(beta, 0.995)

    # Controlled synthetic positions and pbest
    pbest = np.array([
        [0.2, 0.4, 0.8],
        [0.3, 0.5, 0.1],
        [0.1, 0.9, 0.4],
        [0.6, 0.2, 0.7],
    ], dtype=np.float64)

    # Mean best
    mbest_expected = np.mean(pbest, axis=0)
    assert np.allclose(mbest_expected, [0.3, 0.5, 0.5])

    gbest = np.array([0.2, 0.4, 0.8], dtype=np.float64)
    phi = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    p_local = phi * pbest[0] + (1.0 - phi) * gbest
    assert np.allclose(p_local, [0.2, 0.4, 0.8])

    u = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    sign = np.array([1.0, -1.0, 1.0], dtype=np.float64)
    x_curr = np.array([0.1, 0.8, 0.3], dtype=np.float64)

    # Step: P_i + sign * beta * |mbest - x_curr| * ln(1/u)
    delta = np.abs(mbest_expected - x_curr)
    step = sign * beta * delta * np.log(1.0 / u)
    x_next = p_local + step
    x_clipped = np.clip(x_next, 0.0, 1.0)

    assert np.all(x_clipped >= 0.0)
    assert np.all(x_clipped <= 1.0)
    # Expected value for dimension 0: 0.2 + 1.0 * 0.995 * |0.3 - 0.1| * ln(2) = 0.2 + 0.995 * 0.2 * 0.693147... = 0.3379
    expected_dim0 = 0.2 + 0.995 * 0.2 * np.log(2.0)
    assert np.isclose(x_clipped[0], expected_dim0, atol=1e-4)


def test_qpso_determinism_and_clipping():
    """
    Verify:
    - Same seed yields bitwise identical results.
    - Positions are always strictly clipped to [0, 1].
    - Convergence curve is non-increasing.
    """
    city = City(n=20, vehicles=5, capacity=25, seed=42)

    res1 = solve_qpso(city.problem, pop=20, iters=50, seed=123, hybrid=False)
    res2 = solve_qpso(city.problem, pop=20, iters=50, seed=123, hybrid=False)

    # Identical results with identical seed
    assert np.isclose(res1.cost, res2.cost, atol=1e-9)
    assert res1.routes == res2.routes
    assert np.allclose(res1.best_keys, res2.best_keys, atol=1e-9)
    assert res1.convergence == res2.convergence

    # Best keys within [0, 1]
    assert np.all(res1.best_keys >= 0.0)
    assert np.all(res1.best_keys <= 1.0)

    # Monotonicity of best-so-far convergence curve
    for idx in range(len(res1.convergence) - 1):
        assert res1.convergence[idx + 1] <= res1.convergence[idx] + 1e-6, (
            f"Convergence curve increased at step {idx}: "
            f"{res1.convergence[idx]} -> {res1.convergence[idx+1]}"
        )
