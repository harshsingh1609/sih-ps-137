"""CVRP problem representation, instance data structures, distance matrices, and feasibility validation."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class CongestionZone:
    x: float
    y: float
    radius: float
    factor: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "radius": self.radius,
            "factor": self.factor,
        }


def segment_distance_to_point(p1: np.ndarray, p2: np.ndarray, p0: np.ndarray) -> float:
    """Compute the minimum distance from point p0 to segment p1-p2."""
    v = p2 - p1
    w = p0 - p1
    c1 = np.dot(w, v)
    if c1 <= 0:
        return float(np.linalg.norm(p0 - p1))
    c2 = np.dot(v, v)
    if c2 <= c1:
        return float(np.linalg.norm(p0 - p2))
    b = c1 / c2
    pb = p1 + b * v
    return float(np.linalg.norm(p0 - pb))


class CVRPProblem:
    """Capacitated Vehicle Routing Problem model with dynamic traffic cost support."""

    def __init__(
        self,
        n: int,
        vehicles: int,
        capacity: int,
        coords: np.ndarray,
        demands: np.ndarray,
        depot_coords: Optional[Tuple[float, float]] = None,
    ):
        """
        Initialize CVRP instance.
        - n: Number of customers (indices 1..n).
        - vehicles: Fleet size K.
        - capacity: Vehicle capacity Q.
        - coords: (n+1, 2) numpy array (index 0 is depot, 1..n are customers).
        - demands: (n+1,) array of integer demands (demands[0] == 0).
        """
        if n < 1:
            raise ValueError("Customer count must be at least 1.")
        if vehicles < 1:
            raise ValueError("Vehicle count must be at least 1.")
        if capacity < 1:
            raise ValueError("Vehicle capacity must be positive.")

        self.n = int(n)
        self.vehicles = int(vehicles)
        self.capacity = int(capacity)
        self.coords = np.array(coords, dtype=np.float64)
        self.demands = np.array(demands, dtype=np.int64)

        if self.demands[0] != 0:
            self.demands[0] = 0

        # Strict validation: any individual customer demand > capacity is invalid
        for i in range(1, self.n + 1):
            if self.demands[i] > self.capacity:
                raise ValueError(
                    f"Customer {i} demand ({self.demands[i]}) exceeds vehicle capacity ({self.capacity})."
                )
            if self.demands[i] < 1:
                raise ValueError(f"Customer {i} demand must be at least 1, got {self.demands[i]}.")

        # Base Euclidean distance matrix (n+1 x n+1)
        diff = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
        self.base_dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

        # Traffic-aware cost matrix (initially matches base distance)
        self.cost_matrix = self.base_dist_matrix.copy()
        self.congestion_zones: List[CongestionZone] = []

    @property
    def fleet_penalty_per_vehicle(self) -> float:
        return 10000.0

    def apply_congestion_zone(self, x: float, y: float, radius: float, factor: float) -> CongestionZone:
        """Inject a circular congestion zone affecting passing edges."""
        if factor < 1.0 or factor > 10.0:
            raise ValueError("Congestion factor must be in [1.0, 10.0].")
        if radius <= 0:
            raise ValueError("Congestion radius must be positive.")

        zone = CongestionZone(x=float(x), y=float(y), radius=float(radius), factor=float(factor))
        self.congestion_zones.append(zone)
        self._recompute_cost_matrix()
        return zone

    def reset_congestion(self) -> None:
        """Clear all active congestion zones and restore base Euclidean cost matrix."""
        self.congestion_zones.clear()
        self.cost_matrix = self.base_dist_matrix.copy()

    def _recompute_cost_matrix(self) -> None:
        """Recompute the cost matrix from base distances and all active congestion zones."""
        self.cost_matrix = self.base_dist_matrix.copy()
        if not self.congestion_zones:
            return

        total_nodes = self.n + 1
        for i in range(total_nodes):
            p1 = self.coords[i]
            for j in range(i + 1, total_nodes):
                p2 = self.coords[j]
                max_factor = 1.0
                for zone in self.congestion_zones:
                    p0 = np.array([zone.x, zone.y], dtype=np.float64)
                    dist = segment_distance_to_point(p1, p2, p0)
                    if dist <= zone.radius:
                        if zone.factor > max_factor:
                            max_factor = zone.factor

                if max_factor > 1.0:
                    scaled = self.base_dist_matrix[i, j] * max_factor
                    self.cost_matrix[i, j] = scaled
                    self.cost_matrix[j, i] = scaled

    def route_demand(self, route: List[int]) -> int:
        """Calculate total demand served on a single route."""
        return int(sum(self.demands[c] for c in route))

    def route_cost(self, route: List[int]) -> float:
        """
        Calculate total travel cost of a single route:
        depot(0) -> c1 -> c2 -> ... -> cm -> depot(0).
        """
        if not route:
            return 0.0
        cost = self.cost_matrix[0, route[0]]
        for idx in range(len(route) - 1):
            cost += self.cost_matrix[route[idx], route[idx + 1]]
        cost += self.cost_matrix[route[-1], 0]
        return float(cost)

    def evaluate_routes(self, routes: List[List[int]]) -> float:
        """
        Evaluate full CVRP solution:
        sum of route travel costs + fleet penalty if routes_used > K.
        """
        valid_routes = [r for r in routes if r]
        total_travel_cost = sum(self.route_cost(r) for r in valid_routes)
        routes_used = len(valid_routes)

        fleet_penalty = 0.0
        if routes_used > self.vehicles:
            fleet_penalty = self.fleet_penalty_per_vehicle * (routes_used - self.vehicles)

        return float(total_travel_cost + fleet_penalty)

    def validate_solution(self, routes: List[List[int]]) -> Tuple[bool, str]:
        """
        Independent feasibility checker:
        - Every customer 1..n visited exactly once.
        - No missing customers.
        - No duplicate customers.
        - Every route respects capacity Q.
        """
        visited = []
        for idx, route in enumerate(routes):
            if not route:
                continue
            # Capacity check
            dem = self.route_demand(route)
            if dem > self.capacity:
                return (
                    False,
                    f"Route {idx} exceeds capacity: demand {dem} > capacity {self.capacity}.",
                )
            for c in route:
                if c < 1 or c > self.n:
                    return False, f"Invalid customer id {c} in route {idx}."
                visited.append(c)

        visited_set = set(visited)
        if len(visited) != len(visited_set):
            return False, f"Duplicate customer IDs detected across routes (total: {len(visited)}, unique: {len(visited_set)})."

        expected_set = set(range(1, self.n + 1))
        missing = expected_set - visited_set
        if missing:
            return False, f"Missing {len(missing)} customer(s) in solution: {sorted(list(missing))[:5]}..."

        extra = visited_set - expected_set
        if extra:
            return False, f"Unexpected customer(s) not in instance: {sorted(list(extra))}."

        return True, "Solution is feasible."
