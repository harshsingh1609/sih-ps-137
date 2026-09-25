"""Regression tests verifying that initial optimization executes nominal solvers (L1/L2)
and never improperly degrades to L5 stale cache fallback.
"""

from __future__ import annotations
import math
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.store import store
from app.engine.city import City
from app.engine.qpso import solve_qpso
from app.engine.problem import CVRPProblem

client = TestClient(app)


class TestInitialOptimizationRegression:
    """Regression suite for deployed solver failure diagnosis and L5 fallback prevention."""

    def setup_method(self):
        """Clear state store between test runs."""
        store.clear()

    def test_initial_qpso_h_does_not_fallback_to_l5_under_default_configuration(self):
        """
        Verify that fresh city optimization under default UI settings:
        n=40, vehicles=6, capacity=30, seed=42, algo=qpso_h
        executes successfully via QPSO-H, returning level 'L1' and stale=False.
        It must NEVER return L5.
        """
        # 1. Create city
        cr = client.post("/api/city", json={"n": 40, "vehicles": 6, "capacity": 30, "seed": 42})
        assert cr.status_code == 200, f"City creation failed: {cr.text}"
        city_data = cr.json()
        cid = city_data["city_id"]
        assert cid.startswith("c_40_6_30_42")

        # 2. Solve with QPSO-H
        sr = client.post(
            "/api/solve",
            json={
                "city_id": cid,
                "algorithm": "qpso_h",
                "pop": 20,
                "iters": 50,
                "seed": 42,
            },
        )
        assert sr.status_code == 200, f"Solve failed: {sr.text}"
        data = sr.json()

        # Primary assertions: MUST NOT be L5 fallback
        assert data["level"] == "L1", f"Expected L1 nominal solver, got {data['level']}"
        assert data["stale"] is False, "Stale cache flag was True on initial solve!"
        assert data["cost"] > 0.0
        assert data["time_ms"] > 0.0
        assert len(data["convergence"]) >= 1

        # Customer completeness & feasibility check
        flat_customers = [c for route in data["routes"] for c in route]
        assert len(flat_customers) == 40, f"Expected 40 customers visited, got {len(flat_customers)}"
        assert set(flat_customers) == set(range(1, 41)), "Customer visitation set mismatch"

    def test_initial_qpso_h_with_default_request_parameters_runs_cleanly(self):
        """
        Verify solve with default SolveRequest parameters (pop=40, iters=300).
        Even if time budget expires, it must return L2 (time-budget bounded), never L5.
        """
        cr = client.post("/api/city", json={"n": 40, "vehicles": 6, "capacity": 30, "seed": 42})
        assert cr.status_code == 200
        cid = cr.json()["city_id"]

        sr = client.post(
            "/api/solve",
            json={
                "city_id": cid,
                "algorithm": "qpso_h",
                "pop": 40,
                "iters": 300,
                "seed": 42,
            },
        )
        assert sr.status_code == 200
        data = sr.json()
        assert data["level"] in ("L1", "L2"), f"Expected L1 or L2, got {data['level']}"
        assert data["stale"] is False

    @pytest.mark.parametrize("algo", ["qpso_h", "qpso", "pso", "ga", "nn"])
    def test_all_algorithms_nominal_level_no_l5_on_initial_solve(self, algo):
        """Verify none of the supported heuristic algorithms drop into L5 on initial solve."""
        cr = client.post("/api/city", json={"n": 25, "vehicles": 5, "capacity": 30, "seed": 123})
        assert cr.status_code == 200
        cid = cr.json()["city_id"]

        sr = client.post(
            "/api/solve",
            json={
                "city_id": cid,
                "algorithm": algo,
                "pop": 10,
                "iters": 20,
                "seed": 123,
            },
        )
        assert sr.status_code == 200
        data = sr.json()
        assert data["level"] != "L5", f"Algorithm {algo} dropped to L5!"
        assert data["stale"] is False

    def test_deterministic_solver_behavior_across_repeated_invocations(self):
        """Identical inputs must yield identical numerical results and routes."""
        cr = client.post("/api/city", json={"n": 20, "vehicles": 4, "capacity": 30, "seed": 999})
        cid = cr.json()["city_id"]

        sr1 = client.post(
            "/api/solve",
            json={"city_id": cid, "algorithm": "qpso_h", "pop": 15, "iters": 30, "seed": 999},
        )
        sr2 = client.post(
            "/api/solve",
            json={"city_id": cid, "algorithm": "qpso_h", "pop": 15, "iters": 30, "seed": 999},
        )
        assert sr1.status_code == 200
        assert sr2.status_code == 200
        d1 = sr1.json()
        d2 = sr2.json()
        assert d1["cost"] == d2["cost"]
        assert d1["routes"] == d2["routes"]
        assert d1["level"] == "L1"
        assert d2["level"] == "L1"
