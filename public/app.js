/**
 * Q-Traffic: Quantum-Inspired CVRP Route Optimizer (SIH26137)
 * Pure Vanilla JavaScript Client Application with Canvas 2D Rendering
 */

(function () {
  "use strict";

  // Application State
  const state = {
    cityId: null,
    depot: { x: 50, y: 50 },
    customers: [],
    vehicles: 6,
    capacity: 30,
    routes: [],
    congestionZones: [],
    convergence: [],
    isProcessing: false,
    hoveredCustomer: null,
  };

  // Color palette for vehicle routes (vibrant, distinguishable HSL)
  const ROUTE_COLORS = [
    "#38bdf8", // Sky blue
    "#a855f7", // Purple
    "#22c55e", // Emerald green
    "#f97316", // Orange
    "#ec4899", // Pink
    "#eab308", // Yellow
    "#06b6d4", // Cyan
    "#84cc16", // Lime
    "#6366f1", // Indigo
    "#14b8a6", // Teal
    "#f43f5e", // Rose
    "#8b5cf6", // Violet
    "#10b981", // Mint
    "#d946ef", // Fuchsia
    "#0ea5e9", // Ocean
  ];

  // DOM Elements
  const inputN = document.getElementById("inputN");
  const valN = document.getElementById("valN");
  const inputVehicles = document.getElementById("inputVehicles");
  const inputCapacity = document.getElementById("inputCapacity");
  const inputSeed = document.getElementById("inputSeed");
  const selectAlgo = document.getElementById("selectAlgo");
  const inputPop = document.getElementById("inputPop");
  const inputIters = document.getElementById("inputIters");

  const btnGenerateCity = document.getElementById("btnGenerateCity");
  const btnOptimize = document.getElementById("btnOptimize");
  const btnInjectCongestion = document.getElementById("btnInjectCongestion");
  const btnReplanWarm = document.getElementById("btnReplanWarm");
  const btnReplanCold = document.getElementById("btnReplanCold");
  const btnBenchmark = document.getElementById("btnBenchmark");

  const inputCongX = document.getElementById("inputCongX");
  const inputCongY = document.getElementById("inputCongY");
  const inputCongRadius = document.getElementById("inputCongRadius");
  const inputCongFactor = document.getElementById("inputCongFactor");
  const valCongFactor = document.getElementById("valCongFactor");

  const statusToast = document.getElementById("statusToast");
  const systemStatusBadge = document.getElementById("systemStatusBadge");
  const routeLegend = document.getElementById("routeLegend");

  const metricCost = document.getElementById("metricCost");
  const metricVehicles = document.getElementById("metricVehicles");
  const metricTime = document.getElementById("metricTime");
  const metricLevel = document.getElementById("metricLevel");
  const staleAlert = document.getElementById("staleAlert");

  const mapCanvas = document.getElementById("mapCanvas");
  const mapCtx = mapCanvas.getContext("2d");
  const mapTooltip = document.getElementById("mapTooltip");

  const chartCanvas = document.getElementById("chartCanvas");
  const chartCtx = chartCanvas.getContext("2d");

  const replanStatsArea = document.getElementById("replanStatsArea");
  const statWarmCost = document.getElementById("statWarmCost");
  const statColdCost = document.getElementById("statColdCost");
  const statWarmTime = document.getElementById("statWarmTime");
  const statColdTime = document.getElementById("statColdTime");
  const statWarmIters = document.getElementById("statWarmIters");
  const statColdIters = document.getElementById("statColdIters");
  const warmFasterBanner = document.getElementById("warmFasterBanner");

  const benchmarkArea = document.getElementById("benchmarkArea");
  const benchmarkTableBody = document.getElementById("benchmarkTableBody");

  // High-DPI Canvas Setup
  function setupCanvas(canvas, ctx) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = rect.width || canvas.width;
    const h = rect.height || canvas.height;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.scale(dpr, dpr);
  }

  // Toast notifications
  function showToast(message, type = "info") {
    statusToast.textContent = message;
    statusToast.className = `status-toast ${type}`;
    statusToast.classList.remove("hidden");
    if (type === "success" || type === "info") {
      setTimeout(() => {
        statusToast.classList.add("hidden");
      }, 5000);
    }
  }

  function setBusy(isBusy, statusText = "Ready") {
    state.isProcessing = isBusy;
    systemStatusBadge.textContent = statusText;
    if (isBusy) {
      systemStatusBadge.classList.add("working");
    } else {
      systemStatusBadge.classList.remove("working");
    }

    btnGenerateCity.disabled = isBusy;
    btnOptimize.disabled = isBusy || !state.cityId;
    btnInjectCongestion.disabled = isBusy || !state.cityId;
    btnReplanWarm.disabled = isBusy || !state.cityId || state.routes.length === 0;
    btnReplanCold.disabled = isBusy || !state.cityId || state.routes.length === 0;
    btnBenchmark.disabled = isBusy;
  }

  // Coordinate Conversion (Virtual 100x100 space with margin)
  const MAP_PADDING = 35;
  function toCanvasCoords(x, y) {
    const rect = mapCanvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    const usableW = w - 2 * MAP_PADDING;
    const usableH = h - 2 * MAP_PADDING;
    return {
      cx: MAP_PADDING + (x / 100) * usableW,
      cy: h - MAP_PADDING - (y / 100) * usableH,
    };
  }

  function fromCanvasCoords(cx, cy) {
    const rect = mapCanvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    const usableW = w - 2 * MAP_PADDING;
    const usableH = h - 2 * MAP_PADDING;
    const x = ((cx - MAP_PADDING) / usableW) * 100;
    const y = ((h - MAP_PADDING - cy) / usableH) * 100;
    return { x, y };
  }

  // Canvas 2D Map Rendering
  function renderMap() {
    const rect = mapCanvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    mapCtx.clearRect(0, 0, w, h);

    // 1. Draw Grid Lines
    mapCtx.strokeStyle = "#162030";
    mapCtx.lineWidth = 1;
    mapCtx.font = "10px monospace";
    mapCtx.fillStyle = "#475569";

    for (let v = 0; v <= 100; v += 20) {
      const p1 = toCanvasCoords(v, 0);
      const p2 = toCanvasCoords(v, 100);
      mapCtx.beginPath();
      mapCtx.moveTo(p1.cx, p1.cy);
      mapCtx.lineTo(p2.cx, p2.cy);
      mapCtx.stroke();

      const q1 = toCanvasCoords(0, v);
      const q2 = toCanvasCoords(100, v);
      mapCtx.beginPath();
      mapCtx.moveTo(q1.cx, q1.cy);
      mapCtx.lineTo(q2.cx, q2.cy);
      mapCtx.stroke();

      // Axis labels
      mapCtx.fillText(v.toString(), p1.cx - 6, h - 12);
      mapCtx.fillText(v.toString(), 8, q1.cy + 4);
    }

    // 2. Draw Active Congestion Zones
    state.congestionZones.forEach((zone) => {
      const center = toCanvasCoords(zone.x, zone.y);
      const edgePoint = toCanvasCoords(zone.x + zone.radius, zone.y);
      const rPx = Math.abs(edgePoint.cx - center.cx);

      mapCtx.beginPath();
      mapCtx.arc(center.cx, center.cy, rPx, 0, 2 * Math.PI);
      mapCtx.fillStyle = "rgba(239, 68, 68, 0.18)";
      mapCtx.fill();
      mapCtx.lineWidth = 1.5;
      mapCtx.strokeStyle = "rgba(239, 68, 68, 0.6)";
      mapCtx.stroke();

      // Label
      mapCtx.fillStyle = "rgba(252, 165, 165, 0.8)";
      mapCtx.font = "10px sans-serif";
      mapCtx.fillText(`Congested (${zone.factor}x)`, center.cx - 28, center.cy);
    });

    // 3. Draw Vehicle Routes (Polylines)
    if (state.routes && state.routes.length > 0) {
      const depotPos = toCanvasCoords(state.depot.x, state.depot.y);
      const custMap = new Map();
      state.customers.forEach((c) => custMap.set(c.id, c));

      state.routes.forEach((route, rIdx) => {
        if (!route || route.length === 0) return;
        const color = ROUTE_COLORS[rIdx % ROUTE_COLORS.length];
        mapCtx.strokeStyle = color;
        mapCtx.lineWidth = 2.2;
        mapCtx.beginPath();

        // Depot to first
        mapCtx.moveTo(depotPos.cx, depotPos.cy);
        route.forEach((cid) => {
          const cust = custMap.get(cid);
          if (cust) {
            const cp = toCanvasCoords(cust.x, cust.y);
            mapCtx.lineTo(cp.cx, cp.cy);
          }
        });
        // Back to depot
        mapCtx.lineTo(depotPos.cx, depotPos.cy);
        mapCtx.stroke();
      });
    }

    // 4. Draw Customer Nodes
    state.customers.forEach((c) => {
      const cp = toCanvasCoords(c.x, c.y);
      const radius = 3 + c.demand * 0.6;

      mapCtx.beginPath();
      mapCtx.arc(cp.cx, cp.cy, radius, 0, 2 * Math.PI);
      mapCtx.fillStyle = "#38bdf8";
      mapCtx.fill();
      mapCtx.lineWidth = 1;
      mapCtx.strokeStyle = "#0284c7";
      mapCtx.stroke();

      // Show ID if reasonable density
      if (state.customers.length <= 60) {
        mapCtx.fillStyle = "#cbd5e1";
        mapCtx.font = "9px sans-serif";
        mapCtx.fillText(c.id.toString(), cp.cx + radius + 2, cp.cy + 3);
      }
    });

    // 5. Draw Depot Node (Gold Square)
    const dp = toCanvasCoords(state.depot.x, state.depot.y);
    const dSize = 14;
    mapCtx.fillStyle = "#facc15";
    mapCtx.fillRect(dp.cx - dSize / 2, dp.cy - dSize / 2, dSize, dSize);
    mapCtx.strokeStyle = "#ca8a04";
    mapCtx.lineWidth = 2;
    mapCtx.strokeRect(dp.cx - dSize / 2, dp.cy - dSize / 2, dSize, dSize);

    mapCtx.fillStyle = "#0f172a";
    mapCtx.font = "bold 9px sans-serif";
    mapCtx.fillText("D", dp.cx - 3, dp.cy + 3);
  }

  // Update Route Legend Chips
  function updateRouteLegend() {
    routeLegend.innerHTML = "";
    if (!state.routes || state.routes.length === 0) return;

    state.routes.forEach((route, idx) => {
      const color = ROUTE_COLORS[idx % ROUTE_COLORS.length];
      const chip = document.createElement("div");
      chip.className = "legend-chip";
      const totalDemand = route.reduce((acc, cid) => {
        const c = state.customers.find((item) => item.id === cid);
        return acc + (c ? c.demand : 0);
      }, 0);
      chip.innerHTML = `<span class="chip-color" style="background:${color}"></span><strong>R${idx + 1}:</strong> ${route.length} stops (${totalDemand} dem)`;
      routeLegend.appendChild(chip);
    });
  }

  // Canvas 2D Convergence Trajectory Chart
  let chartAnimId = null;
  function drawConvergenceChart(curve, animate = true) {
    if (chartAnimId) {
      cancelAnimationFrame(chartAnimId);
      chartAnimId = null;
    }

    const rect = chartCanvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    const padL = 40;
    const padR = 15;
    const padT = 15;
    const padB = 25;

    if (!curve || curve.length === 0) {
      chartCtx.clearRect(0, 0, w, h);
      return;
    }

    const maxVal = Math.max(...curve);
    const minVal = Math.min(...curve);
    const range = maxVal - minVal > 1e-4 ? maxVal - minVal : 1.0;

    function renderFrame(pointsCount) {
      chartCtx.clearRect(0, 0, w, h);

      // Axes
      chartCtx.strokeStyle = "#263346";
      chartCtx.lineWidth = 1;
      chartCtx.beginPath();
      chartCtx.moveTo(padL, padT);
      chartCtx.lineTo(padL, h - padB);
      chartCtx.lineTo(w - padR, h - padB);
      chartCtx.stroke();

      // Y-axis labels
      chartCtx.fillStyle = "#64748b";
      chartCtx.font = "9px monospace";
      chartCtx.fillText(Math.round(maxVal).toString(), 4, padT + 8);
      chartCtx.fillText(Math.round(minVal).toString(), 4, h - padB - 2);
      chartCtx.fillText("Cost", 4, (padT + h - padB) / 2);

      // X-axis label
      chartCtx.fillText("0", padL - 4, h - 8);
      chartCtx.fillText(curve.length.toString() + " iters", w - padR - 35, h - 8);

      // Line plot
      chartCtx.strokeStyle = "#06b6d4";
      chartCtx.lineWidth = 2;
      chartCtx.beginPath();

      const usableW = w - padL - padR;
      const usableH = h - padT - padB;

      for (let i = 0; i < pointsCount; i++) {
        const x = padL + (i / (curve.length - 1 || 1)) * usableW;
        const normY = (curve[i] - minVal) / range;
        const y = h - padB - normY * usableH;
        if (i === 0) {
          chartCtx.moveTo(x, y);
        } else {
          chartCtx.lineTo(x, y);
        }
      }
      chartCtx.stroke();
    }

    if (!animate || curve.length <= 5) {
      renderFrame(curve.length);
      return;
    }

    let progress = 1;
    const step = Math.max(1, Math.floor(curve.length / 25));
    function animateLoop() {
      progress = Math.min(progress + step, curve.length);
      renderFrame(progress);
      if (progress < curve.length) {
        chartAnimId = requestAnimationFrame(animateLoop);
      }
    }
    chartAnimId = requestAnimationFrame(animateLoop);
  }

  // Tooltip Hover Handler on Map
  mapCanvas.addEventListener("mousemove", (e) => {
    const rect = mapCanvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const { x, y } = fromCanvasCoords(cx, cy);

    let found = null;
    for (const c of state.customers) {
      const dist = Math.hypot(c.x - x, c.y - y);
      if (dist < 3.0) {
        found = c;
        break;
      }
    }

    if (found) {
      mapTooltip.innerHTML = `<strong>Customer #${found.id}</strong><br>Coords: (${found.x.toFixed(1)}, ${found.y.toFixed(1)})<br>Demand: ${found.demand}`;
      mapTooltip.style.left = `${cx + 12}px`;
      mapTooltip.style.top = `${cy - 12}px`;
      mapTooltip.classList.remove("hidden");
    } else {
      mapTooltip.classList.add("hidden");
    }
  });

  mapCanvas.addEventListener("mouseleave", () => {
    mapTooltip.classList.add("hidden");
  });

  // Slider updates
  inputN.addEventListener("input", (e) => {
    valN.textContent = e.target.value;
  });

  inputCongFactor.addEventListener("input", (e) => {
    valCongFactor.textContent = `${parseFloat(e.target.value).toFixed(1)}x`;
  });

  // API Call: Generate City
  async function generateCity() {
    setBusy(true, "Generating City...");
    showToast("Generating city instances with depot and customers...", "info");

    const payload = {
      n: parseInt(inputN.value, 10),
      vehicles: parseInt(inputVehicles.value, 10),
      capacity: parseInt(inputCapacity.value, 10),
      seed: parseInt(inputSeed.value, 10),
    };

    try {
      const res = await fetch("/api/city", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "City generation failed");
      }

      const data = await res.json();
      state.cityId = data.city_id;
      state.depot = data.depot;
      state.customers = data.customers;
      state.vehicles = data.vehicles;
      state.capacity = data.capacity;
      state.routes = [];
      state.congestionZones = [];
      state.convergence = [];

      // Reset metrics UI
      metricCost.textContent = "—";
      metricVehicles.textContent = `0 / ${state.vehicles}`;
      metricTime.textContent = "—";
      metricLevel.textContent = "—";
      staleAlert.classList.add("hidden");
      replanStatsArea.classList.add("hidden");

      renderMap();
      updateRouteLegend();
      drawConvergenceChart([], false);

      showToast(`City generated: ${state.customers.length} customers, fleet ${state.vehicles}`, "success");
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      setBusy(false);
    }
  }

  // API Call: Optimize Routes
  async function optimizeRoutes() {
    if (!state.cityId) return;

    setBusy(true, "Optimizing...");
    showToast(`Running ${selectAlgo.value.toUpperCase()} optimizer...`, "info");

    const payload = {
      city_id: state.cityId,
      algorithm: selectAlgo.value,
      pop: parseInt(inputPop.value, 10),
      iters: parseInt(inputIters.value, 10),
      seed: parseInt(inputSeed.value, 10),
    };

    try {
      const res = await fetch("/api/solve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Optimization failed");
      }

      const data = await res.json();
      state.routes = data.routes;
      state.convergence = data.convergence || [];

      metricCost.textContent = data.cost.toFixed(2);
      metricVehicles.textContent = `${data.vehicles_used} / ${state.vehicles}`;
      metricTime.textContent = `${data.time_ms.toFixed(1)} ms`;
      metricLevel.textContent = data.level;

      if (data.stale) {
        staleAlert.classList.remove("hidden");
      } else {
        staleAlert.classList.add("hidden");
      }

      renderMap();
      updateRouteLegend();
      drawConvergenceChart(state.convergence, true);

      showToast(`Optimization complete: Cost ${data.cost.toFixed(2)} in ${data.time_ms.toFixed(1)} ms`, "success");
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      setBusy(false);
    }
  }

  // API Call: Inject Congestion
  async function injectCongestion() {
    if (!state.cityId) return;

    setBusy(true, "Injecting Congestion...");

    const payload = {
      city_id: state.cityId,
      x: parseFloat(inputCongX.value),
      y: parseFloat(inputCongY.value),
      radius: parseFloat(inputCongRadius.value),
      factor: parseFloat(inputCongFactor.value),
    };

    try {
      const res = await fetch("/api/congestion", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Congestion injection failed");
      }

      const data = await res.json();
      state.congestionZones.push(data.zone);
      renderMap();

      showToast(`Injected ${payload.factor}x congestion zone at (${payload.x}, ${payload.y})`, "info");
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      setBusy(false);
    }
  }

  // API Call: Replan Routes (Warm or Cold)
  async function replanRoutes(isWarm) {
    if (!state.cityId) return;

    setBusy(true, isWarm ? "Warm Replanning..." : "Cold Replanning...");
    showToast(`Executing ${isWarm ? "Warm-start" : "Cold-start"} replan...`, "info");

    const payload = {
      city_id: state.cityId,
      algorithm: selectAlgo.value,
      warm: isWarm,
      pop: parseInt(inputPop.value, 10),
      iters: parseInt(inputIters.value, 10),
      seed: parseInt(inputSeed.value, 10),
    };

    try {
      const res = await fetch("/api/replan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Replanning failed");
      }

      const data = await res.json();
      state.routes = data.routes;
      state.convergence = data.convergence || [];

      metricCost.textContent = data.cost.toFixed(2);
      metricVehicles.textContent = `${data.routes.length} / ${state.vehicles}`;
      metricTime.textContent = `${data.time_ms.toFixed(1)} ms`;
      metricLevel.textContent = isWarm ? "Warm" : "Cold";

      // Display Warm vs. Cold comparative statistics
      replanStatsArea.classList.remove("hidden");
      const warmIters = isWarm ? data.iterations_to_99pct : "—";
      const coldIters = data.baseline_cold_iterations_to_99pct;
      statWarmCost.textContent = isWarm ? data.cost.toFixed(2) : "—";
      statColdCost.textContent = data.cold_cost.toFixed(2);
      statWarmTime.textContent = isWarm ? `${data.time_ms.toFixed(1)}` : "—";
      statColdTime.textContent = `${data.cold_time_ms.toFixed(1)}`;
      statWarmIters.textContent = isWarm ? warmIters : "—";
      statColdIters.textContent = coldIters;

      if (data.warm_faster) {
        warmFasterBanner.className = "speedup-banner positive";
        warmFasterBanner.textContent = `✓ Warm start converged faster to 99% cost improvement.`;
      } else {
        warmFasterBanner.className = "speedup-banner neutral";
        warmFasterBanner.textContent = `— Warm and cold starts showed comparable convergence on this instance.`;
      }

      renderMap();
      updateRouteLegend();
      drawConvergenceChart(state.convergence, true);

      showToast(`Replanned (${isWarm ? "Warm" : "Cold"}): Cost ${data.cost.toFixed(2)}`, "success");
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      setBusy(false);
    }
  }

  // API Call: Benchmark
  async function runBenchmark() {
    setBusy(true, "Running Multi-Algorithm Benchmark...");
    showToast("Executing multi-seed benchmark across NN, PSO, GA, QPSO, QPSO-H (please wait)...", "info");

    const n = parseInt(inputN.value, 10);
    const vehicles = parseInt(inputVehicles.value, 10);
    const capacity = parseInt(inputCapacity.value, 10);

    try {
      const res = await fetch(
        `/api/benchmark?n=${n}&vehicles=${vehicles}&capacity=${capacity}&seeds=1,2,3&pop=20&iters=80`
      );

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Benchmark failed");
      }

      const data = await res.json();
      benchmarkArea.classList.remove("hidden");
      benchmarkTableBody.innerHTML = "";

      data.rows.forEach((row) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${row.algorithm.toUpperCase()}</strong></td>
          <td>${row.mean_cost.toFixed(1)} &plusmn; ${row.std_cost.toFixed(1)}</td>
          <td>${row.best_cost.toFixed(1)}</td>
          <td>${row.mean_time_ms.toFixed(1)}</td>
          <td>${row.mean_iters_to_99pct.toFixed(0)}</td>
        `;
        benchmarkTableBody.appendChild(tr);
      });

      showToast("Benchmark complete: actual performance metrics displayed.", "success");
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      setBusy(false);
    }
  }

  // Event Listeners
  btnGenerateCity.addEventListener("click", generateCity);
  btnOptimize.addEventListener("click", optimizeRoutes);
  btnInjectCongestion.addEventListener("click", injectCongestion);
  btnReplanWarm.addEventListener("click", () => replanRoutes(true));
  btnReplanCold.addEventListener("click", () => replanRoutes(false));
  btnBenchmark.addEventListener("click", runBenchmark);

  // Window Resize & Init
  function init() {
    setupCanvas(mapCanvas, mapCtx);
    setupCanvas(chartCanvas, chartCtx);
    renderMap();
    generateCity();
  }

  window.addEventListener("resize", () => {
    setupCanvas(mapCanvas, mapCtx);
    setupCanvas(chartCanvas, chartCtx);
    renderMap();
    if (state.convergence.length > 0) {
      drawConvergenceChart(state.convergence, false);
    }
  });

  init();
})();
