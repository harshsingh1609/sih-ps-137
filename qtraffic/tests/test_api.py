"""Integration tests for FastAPI endpoints: health, city generation, solve, congestion, replan, benchmark, and 422 validation."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /health returns status ok and version 1.0.0."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"


def test_complete_api_lifecycle():
    """
    Test full lifecycle:
    health -> city generation -> solve -> congestion -> warm replan -> cold replan -> benchmark
    """
    # 1. City Generation
    city_payload = {
        "n": 20,
        "vehicles": 5,
        "capacity": 30,
        "seed": 42,
    }
    city_res = client.post("/api/city", json=city_payload)
    assert city_res.status_code == 200
    city_data = city_res.json()
    assert "city_id" in city_data
    assert len(city_data["customers"]) == 20
    city_id = city_data["city_id"]

    # 2. Solve with QPSO-H
    solve_payload = {
        "city_id": city_id,
        "algorithm": "qpso_h",
        "pop": 20,
        "iters": 40,
        "seed": 42,
    }
    solve_res = client.post("/api/solve", json=solve_payload)
    assert solve_res.status_code == 200
    solve_data = solve_res.json()
    assert "routes" in solve_data
    assert solve_data["cost"] > 0.0
    assert solve_data["vehicles_used"] <= 5
    assert solve_data["level"] in ("L1", "L2")
    assert solve_data["stale"] is False

    # 3. Inject Congestion Zone
    cong_payload = {
        "city_id": city_id,
        "x": 50.0,
        "y": 50.0,
        "radius": 15.0,
        "factor": 3.0,
    }
    cong_res = client.post("/api/congestion", json=cong_payload)
    assert cong_res.status_code == 200
    cong_data = cong_res.json()
    assert cong_data["zone"]["factor"] == 3.0

    # 4. Warm Replan
    warm_payload = {
        "city_id": city_id,
        "algorithm": "qpso_h",
        "warm": True,
        "pop": 20,
        "iters": 40,
        "seed": 42,
    }
    warm_res = client.post("/api/replan", json=warm_payload)
    assert warm_res.status_code == 200
    warm_data = warm_res.json()
    assert warm_data["warm"] is True
    assert "iterations_to_99pct" in warm_data
    assert "baseline_cold_iterations_to_99pct" in warm_data

    # 5. Cold Replan
    cold_payload = {
        "city_id": city_id,
        "algorithm": "qpso_h",
        "warm": False,
        "pop": 20,
        "iters": 40,
        "seed": 42,
    }
    cold_res = client.post("/api/replan", json=cold_payload)
    assert cold_res.status_code == 200
    cold_data = cold_res.json()
    assert cold_data["warm"] is False

    # 6. Benchmark
    bench_res = client.get("/api/benchmark?n=15&vehicles=4&capacity=25&seeds=1,2&pop=10&iters=20")
    assert bench_res.status_code == 200
    bench_data = bench_res.json()
    assert "rows" in bench_data
    algos = [r["algorithm"] for r in bench_data["rows"]]
    assert "nn" in algos
    assert "qpso_h" in algos


def test_api_422_validations():
    """Verify HTTP 422 responses for illegal inputs."""
    # 1. n outside [5, 300]
    r1 = client.post("/api/city", json={"n": 2, "vehicles": 2, "capacity": 20})
    assert r1.status_code == 422

    r2 = client.post("/api/city", json={"n": 500, "vehicles": 2, "capacity": 20})
    assert r2.status_code == 422

    # 2. Invalid vehicle count
    r3 = client.post("/api/city", json={"n": 10, "vehicles": 0, "capacity": 20})
    assert r3.status_code == 422

    # 3. Invalid capacity
    r4 = client.post("/api/city", json={"n": 10, "vehicles": 2, "capacity": 0})
    assert r4.status_code == 422

    # 4. Invalid congestion factor (< 1 or > 10)
    # First create a valid city
    c_res = client.post("/api/city", json={"n": 10, "vehicles": 3, "capacity": 20})
    cid = c_res.json()["city_id"]

    r5 = client.post("/api/congestion", json={"city_id": cid, "x": 50, "y": 50, "radius": 10, "factor": 0.5})
    assert r5.status_code == 422

    r6 = client.post("/api/congestion", json={"city_id": cid, "x": 50, "y": 50, "radius": 10, "factor": 15.0})
    assert r6.status_code == 422

    # 5. Unsupported algorithm
    r7 = client.post("/api/solve", json={"city_id": cid, "algorithm": "quantum_annealer_magic"})
    assert r7.status_code == 422

    # 6. Exact solver requested for n > 8
    r8 = client.post("/api/solve", json={"city_id": cid, "algorithm": "exact"})
    assert r8.status_code == 422
