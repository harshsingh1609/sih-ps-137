"""Profile script for cProfile analysis."""
from app.engine.city import City
from app.engine.qpso import solve_qpso

city = City(n=120, vehicles=15, capacity=40, seed=42)
res = solve_qpso(city.problem, pop=30, iters=100, seed=42, hybrid=True)
print(f"n=120 result: cost={res.cost:.2f}, iters={res.iterations}, time={res.time_ms:.1f}ms")
