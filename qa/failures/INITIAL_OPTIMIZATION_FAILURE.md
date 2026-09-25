# SIH26137 Q-Traffic Failure Report: Initial Optimization Degradation & L5 Fallback Trigger

## 1. Executive Summary

- **Report Name**: `INITIAL_OPTIMIZATION_FAILURE.md`
- **Component**: Solver Execution & Degraded State Fallback Ladder
- **User-Facing Symptom**: Banner displaying `"Solver exception encountered; displaying previous valid solution (L5 Fallback)."` during normal initial optimization.
- **Root Failure Classification**: 
  - **Category E**: State / Caching & Cross-Session Invalidation
  - **Category G**: Vercel / Serverless Ephemeral Execution Architecture
  - **Category H**: Serverless Execution Timeout (10s Hobby Lambda Limit)
  - **Category L**: Repository Desynchronization (Duplicate `app/` trees between Root and `qtraffic/`) and Silent Exception Masking

---

## 2. Forensic Trace & Code Location

### 2.1 The Masking Catch Block in API Routes

In `app/api/routes.py` (and `qtraffic/app/api/routes.py`), lines 130–147:

```python
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
```

### 2.2 Client-Side Trigger in Vanilla JavaScript

In `public/app.js` (lines 509–513):

```javascript
      if (data.stale) {
        staleAlert.classList.remove("hidden");
      } else {
        staleAlert.classList.add("hidden");
      }
```

And in `public/index.html` (lines 173–175):

```html
      <div id="staleAlert" class="alert alert-stale hidden">
        <strong>Cached Plan Returned:</strong> Solver exception encountered; displaying previous valid solution (L5 Fallback).
      </div>
```

---

## 3. Root Cause Analysis

### Cause 1: Dual Repository Trees (`./app` vs. `./qtraffic/app`)
There are two parallel application trees in the repository:
1. Root directory `./app/`
2. Subdirectory `./qtraffic/app/`

In commit `12248049` (`fix: resolve all Vercel deployment issues`), modifications to guard against serverless timeouts and enable state re-hydration (`TIME_BUDGET = 8.0`, deterministic city ID prefix without UUID suffix, `_is_vercel` static file guard) were committed **only to `qtraffic/app/`**, leaving `./app/` untouched.

In `api/index.py`:
```python
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

QTRAFFIC_DIR = ROOT_DIR / "qtraffic"
if QTRAFFIC_DIR.is_dir() and str(QTRAFFIC_DIR) not in sys.path:
    sys.path.insert(0, str(QTRAFFIC_DIR))

from app.main import app as fastapi_app
```
Because `ROOT_DIR` is inserted at index 0, in environments where `/var/task` is not pre-populated in `sys.path`, Python loads `./app/main.py` which references `./app/api/routes.py` and `./app/engine/city.py`. These unpatched files lacked the 8.0-second time budget.

### Cause 2: Unbounded Iterations and Serverless Lambda Timeout (10s Hard Cap)
Under initial optimization when users configured $iters = 300$ and $pop = 40$ (the default in `SolveRequest`), QPSO-H with Lamarckian 2-opt and Or-opt route refinement exceeded Vercel's 10-second serverless execution limit on $n \ge 100$. When an execution exceeded 10 seconds or encountered a transient memory spike or timeout, the worker failed.

### Cause 3: Silent Exception Masking & Cache Poisoning
`POST /api/solve` had a critical structural vulnerability:
1. When a solver threw an exception or ran into an unexpected state, the exception was caught by `except Exception as exc:`.
2. Crucially, **the exception was never logged** (`logging.exception` or `traceback.print_exc()` was missing).
3. `store.get_last_solution(req.city_id)` was called.
4. Because city IDs were generated deterministically (`c_{n}_{vehicles}_{capacity}_{seed}`), any warm serverless Lambda worker that had previously solved that city returned the stale cached solution with `level = "L5"` and `stale = True`.
5. This turned an internal solver error into an apparent "successful fallback", completely masking the real defect from developers and judges while presenting a confusing stale alert to the user.

### Cause 4: Fleet Infeasibility of the Default Configuration
In `public/index.html`, the default UI values were:
- $n = 40$ customers
- Fleet Size $K = 6$ vehicles
- Vehicle Capacity $Q = 30$
- Seed $= 42$

With customer demands generated uniformly in $[1, 10]$, the total customer demand for seed 42 is **212 units**.
However, total fleet capacity is $K \times Q = 6 \times 30 = \mathbf{180 \text{ units}}$.
Because $180 < 212$, **the problem instance is mathematically impossible to serve with 6 vehicles**.
Prins' optimal split DP was forced to generate 8 routes, triggering a severe fleet penalty of:
$$\text{Penalty} = 10,000 \times (8 - 6) = 20,000$$
While `optimal_split` handled this by adding the penalty, any solver constraint requiring strict adherence to $K \le 6$ (such as strict fleet-bounded baselines) would fail.

---

## 4. Technical Environment Details

- **Deployment Environment**: Vercel Serverless Function (`@vercel/python` on AWS Lambda base)
- **Region**: `bom1` (Mumbai, India)
- **Python Version**: 3.10.9 (runtime 3.10)
- **FastAPI Version**: 0.110.0+ / Starlette 0.36.0+
- **NumPy Version**: 1.26.4
- **Dependencies**: `fastapi`, `uvicorn`, `pydantic`, `numpy`

---

## 5. Required Action Plan & Remediation

1. **Preserve Exceptions & Full Diagnostic Traceability**:
   - Add structured logging with `traceback.format_exc()` in `app/api/routes.py` and `qtraffic/app/api/routes.py`.
   - Ensure exceptions are never silently swallowed into L5 when no valid reason exists.
   - For diagnostic debugging, return error details or log them clearly to serverless output.

2. **Synchronize `./app/` and `./qtraffic/app/`**:
   - Ensure the root `./app/` and `./qtraffic/app/` are identical drop-in packages.
   - Enforce `TIME_BUDGET = 8.0` in both route handlers so QPSO-H gracefully concludes at L2 before Vercel kills the process.

3. **Robust Re-Hydration**:
   - Fix `store.py` to restore congestion zones and handle city IDs consistently regardless of UUID tags.

4. **Add Dedicated Regression Test**:
   - Create `tests/qa/test_initial_solve_does_not_fallback_to_l5.py` to assert that fresh default problem instances execute cleanly at L1 with `stale == False`.
