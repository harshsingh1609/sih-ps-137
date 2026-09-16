"""City generation, customer coordinate and demand initialization."""

from __future__ import annotations
import uuid
from typing import Dict, Any, Optional
import numpy as np
from app.engine.problem import CVRPProblem


class City:
    """Represents a generated city instance with depot and customer locations."""

    def __init__(
        self,
        n: int,
        vehicles: int,
        capacity: int,
        seed: Optional[int] = None,
        city_id: Optional[str] = None,
        depot_coords: tuple[float, float] = (50.0, 50.0),
    ):
        seed_tag = seed if seed is not None else 1
        self.city_id = city_id or f"c_{int(n)}_{int(vehicles)}_{int(capacity)}_{seed_tag}_{uuid.uuid4().hex[:8]}"
        self.n = int(n)
        self.vehicles = int(vehicles)
        self.capacity = int(capacity)
        self.seed = seed
        self.depot_coords = depot_coords

        rng = np.random.default_rng(seed)

        # Coordinates: index 0 is depot, indices 1..n are customers
        coords = np.zeros((self.n + 1, 2), dtype=np.float64)
        coords[0] = [depot_coords[0], depot_coords[1]]
        coords[1:] = rng.uniform(0.0, 100.0, size=(self.n, 2))
        self.coords = coords

        # Demands: depot has 0 demand, customers have demand in [1, 10]
        demands = np.zeros(self.n + 1, dtype=np.int64)
        demands[0] = 0
        demands[1:] = rng.integers(1, 11, size=self.n)
        self.demands = demands

        # CVRP problem representation
        self.problem = CVRPProblem(
            n=self.n,
            vehicles=self.vehicles,
            capacity=self.capacity,
            coords=self.coords,
            demands=self.demands,
            depot_coords=depot_coords,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert city configuration to API-friendly dictionary."""
        return {
            "city_id": self.city_id,
            "depot": {
                "x": float(self.coords[0, 0]),
                "y": float(self.coords[0, 1]),
            },
            "customers": [
                {
                    "id": i,
                    "x": float(self.coords[i, 0]),
                    "y": float(self.coords[i, 1]),
                    "demand": int(self.demands[i]),
                }
                for i in range(1, self.n + 1)
            ],
            "vehicles": self.vehicles,
            "capacity": self.capacity,
            "seed": self.seed,
            "active_congestion": [z.to_dict() for z in self.problem.congestion_zones],
        }
