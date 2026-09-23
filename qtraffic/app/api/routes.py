"""FastAPI routing endpoints for CVRP solving, congestion injection, replanning, and benchmarks."""

from __future__ import annotations
from typing import List, Optional, Dict, Any
import numpy as np
from pydantic import BaseModel, Field, field_validator
from fastapi import APIRouter, HTTPException, Query, status

from app.core.store import store
from app.engine.city import City
from app.engine.exact import solve_exact
from app.engine.greedy import solve_nn
from app.engine.pso import solve_pso
from app.engine.ga import solve_ga
from app.engine.qpso import solve_qpso, SolverResult
from app.engine.replanner import run_replan
from app.engine.benchmark import run_benchmark

router = APIRouter()


class CityCreateRequest(BaseModel):
    n: int = Field(..., ge=5, le=300, description="Customer count in [5, 300]")
    vehicles: int = Field(..., ge=1, description="Number of vehicles K >= 1")
    capacity: int = Field(..., ge=1, description="Vehicle capacity Q >= 1")
    seed: Optional[int] = Field(default=1, description="Random seed")


class SolveRequest(BaseModel):
    city_id: str
    algorithm: str = Field(default="qpso_h")
    pop: int = Field(default=40, ge=4, le=500)
    iters: int = Field(default=300, ge=1, le=2000)
    seed: Optional[int] = Field(default=1)

    @field_validator("algorithm")
    @classmethod
    def validate_algo(cls, v: str) -> str:
        allowed = {"qpso_h", "qpso", "pso", "ga", "nn", "exact"}
        if v.lower() not in allowed:
            raise ValueError(f"Unsupported algorithm '{v}'. Allowed: {sorted(list(allowed))}")
        return v.lower()


class CongestionRequest(BaseModel):
    city_id: str
    x: float
    y: float
    radius: float = Field(..., gt=0)
    factor: float = Field(..., ge=1.0, le=10.0)


class ReplanRequest(BaseModel):
    city_id: str
    algorithm: str = Field(default="qpso_h")
    warm: bool = Field(default=True)
    pop: int = Field(default=40, ge=4, le=500)
    iters: int = Field(default=300, ge=1, le=2000)
    seed: Optional[int] = Field(default=1)

    @field_validator("algorithm")
    @classmethod
    def validate_algo(cls, v: str) -> str:
        allowed = {"qpso_h", "qpso", "pso", "ga", "nn"}
        if v.lower() not in allowed:
            raise ValueError(f"Unsupported replan algorithm '{v}'.")
        return v.lower()


@router.post("/api/city", status_code=status.HTTP_200_OK)
def create_city(req: CityCreateRequest):
    """Generate a new CVRP city instance."""
    try:
        city = City(
            n=req.n,
            vehicles=req.vehicles,
            capacity=req.capacity,
            seed=req.seed,
        )
        store.save_city(city)
        return city.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))


@router.post("/api/solve", status_code=status.HTTP_200_OK)
def solve_cvrp(req: SolveRequest):
    """Solve CVRP on the specified city with automated fallback ladder."""
    city = store.get_city(req.city_id)
    if not city:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"City {req.city_id} not found.")

    algo = req.algorithm.lower()
    problem = city.problem

    # Validation: exact solver only for n <= 8
    if algo == "exact" and problem.n > 8:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Exact solver only supported for n <= 8 (got n={problem.n}).",
        )

    try:
        # time_budget_sec keeps us within Vercel's 10-second function timeout.
        # Solvers return the best solution found so far when the budget expires.
        TIME_BUDGET = 8.0
        if algo == "exact":
            res = solve_exact(problem, seed=req.seed)
        elif algo == "nn":
            res = solve_nn(problem, seed=req.seed)
        elif algo == "pso":
            res = solve_pso(problem, pop=req.pop, iters=req.iters, seed=req.seed, time_budget_sec=TIME_BUDGET)
        elif algo == "ga":
            res = solve_ga(problem, pop=req.pop, iters=req.iters, seed=req.seed, time_budget_sec=TIME_BUDGET)
        elif algo == "qpso":
            res = solve_qpso(problem, pop=req.pop, iters=req.iters, seed=req.seed, hybrid=False, time_budget_sec=TIME_BUDGET)
        elif algo == "qpso_h":
            res = solve_qpso(problem, pop=req.pop, iters=req.iters, seed=req.seed, hybrid=True, time_budget_sec=TIME_BUDGET)
        else:
            raise ValueError(f"Unknown algorithm {algo}")

        store.save_solution(req.city_id, res)
        return {
            "routes": res.routes,
            "cost": float(round(res.cost, 2)),
            "vehicles_used": int(res.vehicles_used),
            "level": res.level,
            "stale": bool(res.stale),
            "time_ms": float(round(res.time_ms, 2)),
            "convergence": [float(round(c, 2)) for c in res.convergence],
        }

    except Exception as exc:
        # Fallback Level L5: Cached valid plan with stale: true
        cached = store.get_last_solution(req.city_id)
        if cached is not None:
            return {
                "routes": cached.routes,
                "cost": float(round(cached.cost, 2)),
                "vehicles_used": int(cached.vehicles_used),
                "level": "L5",
                "stale": True,
                "time_ms": 0.0,
                "convergence": cached.convergence,
            }
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Solver execution failed: {str(exc)}",
        )


@router.post("/api/congestion", status_code=status.HTTP_200_OK)
def inject_congestion(req: CongestionRequest):
    """Inject dynamic circular congestion zone into transportation network."""
    city = store.get_city(req.city_id)
    if not city:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"City {req.city_id} not found.")

    try:
        zone = city.problem.apply_congestion_zone(
            x=req.x,
            y=req.y,
            radius=req.radius,
            factor=req.factor,
        )
        return {
            "city_id": req.city_id,
            "zone": zone.to_dict(),
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))


@router.post("/api/replan", status_code=status.HTTP_200_OK)
def replan(req: ReplanRequest):
    """Execute warm or cold replanning after congestion event."""
    city = store.get_city(req.city_id)
    if not city:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"City {req.city_id} not found.")

    prev_keys = store.get_last_keys(req.city_id)

    comparison = run_replan(
        problem=city.problem,
        previous_keys=prev_keys,
        algorithm=req.algorithm,
        warm=req.warm,
        pop=req.pop,
        iters=req.iters,
        seed=req.seed,
    )

    # Save selected solution in cache
    res = SolverResult(
        algorithm=f"{req.algorithm}_{'warm' if req.warm else 'cold'}",
        routes=comparison.routes,
        cost=comparison.cost,
        vehicles_used=len(comparison.routes),
        time_ms=comparison.time_ms,
        convergence=comparison.convergence,
        best_keys=prev_keys if prev_keys is not None else np.empty(0),
        iterations=len(comparison.convergence),
        level="L1",
        stale=False,
    )
    store.save_solution(req.city_id, res)

    return {
        "routes": comparison.routes,
        "cost": float(round(comparison.cost, 2)),
        "time_ms": float(round(comparison.time_ms, 2)),
        "iterations_to_99pct": int(comparison.iterations_to_99pct),
        "warm": bool(comparison.warm),
        "baseline_cold_iterations_to_99pct": int(comparison.baseline_cold_iterations_to_99pct),
        "cold_cost": float(round(comparison.cold_cost, 2)),
        "cold_time_ms": float(round(comparison.cold_time_ms, 2)),
        "warm_faster": bool(comparison.warm_faster),
        "convergence": [float(round(c, 2)) for c in comparison.convergence],
        "cold_convergence": [float(round(c, 2)) for c in comparison.cold_convergence],
    }


@router.get("/api/benchmark", status_code=status.HTTP_200_OK)
def benchmark_endpoint(
    n: int = Query(default=30, ge=5, le=100),
    vehicles: int = Query(default=6, ge=1),
    capacity: int = Query(default=30, ge=1),
    seeds: str = Query(default="1,2"),
    pop: int = Query(default=10, ge=4, le=100),
    iters: int = Query(default=50, ge=10, le=500),
):
    """Execute multi-algorithm multi-seed benchmark and return honest summary table."""
    try:
        seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
        if not seed_list:
            seed_list = [1, 2, 3, 4, 5]
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid seeds query parameter. Must be comma-separated integers.",
        )

    rows = run_benchmark(
        n=n,
        vehicles=vehicles,
        capacity=capacity,
        seeds=seed_list,
        pop=pop,
        iters=iters,
    )

    return {"rows": rows}
