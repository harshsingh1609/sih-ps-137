"""In-memory thread-safe state store with serverless re-hydration for Vercel deployment."""

from __future__ import annotations
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np

from app.engine.city import City
from app.engine.qpso import SolverResult


class AppStore:
    """Central state store managing active city instances and cached routing solutions."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cities: Dict[str, City] = {}
        self._last_results: Dict[str, SolverResult] = {}
        self._last_keys: Dict[str, np.ndarray] = {}
        self._cache_dir = Path(tempfile.gettempdir()) / "qtraffic_state"
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def _city_file(self, city_id: str) -> Path:
        safe_id = "".join(c for c in city_id if c.isalnum() or c in ("-", "_"))
        return self._cache_dir / f"city_{safe_id}.json"

    def _keys_file(self, city_id: str) -> Path:
        safe_id = "".join(c for c in city_id if c.isalnum() or c in ("-", "_"))
        return self._cache_dir / f"keys_{safe_id}.npy"

    def save_city(self, city: City) -> None:
        with self._lock:
            self._cities[city.city_id] = city
            try:
                data = {
                    "city_id": city.city_id,
                    "n": city.n,
                    "vehicles": city.vehicles,
                    "capacity": city.capacity,
                    "seed": city.seed,
                    "congestion_zones": [z.to_dict() for z in city.problem.congestion_zones],
                }
                self._city_file(city.city_id).write_text(json.dumps(data), encoding="utf-8")
            except Exception:
                pass

    def get_city(self, city_id: str) -> Optional[City]:
        with self._lock:
            if city_id in self._cities:
                return self._cities[city_id]

            # Try loading from temp cache
            c_file = self._city_file(city_id)
            if c_file.exists():
                try:
                    data = json.loads(c_file.read_text(encoding="utf-8"))
                    city = City(
                        n=data["n"],
                        vehicles=data["vehicles"],
                        capacity=data["capacity"],
                        seed=data.get("seed"),
                        city_id=city_id,
                    )
                    for z in data.get("congestion_zones", []):
                        city.problem.apply_congestion_zone(
                            x=float(z["x"]),
                            y=float(z["y"]),
                            radius=float(z["radius"]),
                            factor=float(z["factor"]),
                        )
                    self._cities[city_id] = city
                    return city
                except Exception:
                    pass

            # If not in cache, check if city_id follows deterministic pattern: c_{n}_{vehicles}_{capacity}_{seed}_{tag}
            if city_id and city_id.startswith("c_"):
                parts = city_id.split("_")
                if len(parts) >= 5:
                    try:
                        n = int(parts[1])
                        vehicles = int(parts[2])
                        capacity = int(parts[3])
                        seed_val = int(parts[4]) if parts[4] != "None" else None
                        city = City(
                            n=n,
                            vehicles=vehicles,
                            capacity=capacity,
                            seed=seed_val,
                            city_id=city_id,
                        )
                        self._cities[city_id] = city
                        return city
                    except (ValueError, IndexError):
                        pass

            return None

    def save_solution(self, city_id: str, result: SolverResult) -> None:
        with self._lock:
            self._last_results[city_id] = result
            if result.best_keys is not None:
                self._last_keys[city_id] = result.best_keys.copy()
                try:
                    np.save(self._keys_file(city_id), result.best_keys)
                except Exception:
                    pass

    def get_last_solution(self, city_id: str) -> Optional[SolverResult]:
        with self._lock:
            return self._last_results.get(city_id)

    def get_last_keys(self, city_id: str) -> Optional[np.ndarray]:
        with self._lock:
            if city_id in self._last_keys:
                return self._last_keys[city_id].copy()
            k_file = self._keys_file(city_id)
            if k_file.exists():
                try:
                    keys = np.load(k_file)
                    self._last_keys[city_id] = keys
                    return keys.copy()
                except Exception:
                    pass
            return None

    def clear(self) -> None:
        with self._lock:
            self._cities.clear()
            self._last_results.clear()
            self._last_keys.clear()
            try:
                for f in self._cache_dir.glob("*"):
                    try:
                        f.unlink()
                    except Exception:
                        pass
            except Exception:
                pass


# Global singleton instance
store = AppStore()
