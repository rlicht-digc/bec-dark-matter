import { useState, useEffect, useRef, useCallback } from "react";

const W = 420;
const GRID = 100;
const CELL = W / GRID;

function makeParticles() {
  const p = [];
  const seeds = [];
  for (let s = 0; s < 18; s++) {
    seeds.push([50 + Math.random() * (W - 100), 50 + Math.random() * (W - 100)]);
  }
  for (const [cx, cy] of seeds) {
    const n = 8 + Math.floor(Math.random() * 14);
    for (let i = 0; i < n; i++) {
      const a = Math.random() * Math.PI * 2;
      const r = Math.random() * 18;
      p.push({
        x: cx + Math.cos(a) * r, y: cy + Math.sin(a) * r,
        vx: (Math.random() - 0.5) * 0.2, vy: (Math.random() - 0.5) * 0.2,
        m: 0.5 + Math.random() * 1.2,
      });
    }
  }
  return p;
}

export default function DiagnosticSim() {
  const mRef = useRef(null), dRef = useRef(null), cRef = useRef(null);
  const stateRef = useRef(null);
  const animRef = useRef(null);
  const metricsRef = useRef({});
  const [step, setStep] = useState(0);
  const [running, setRunning] = useState(false);
  const [metrics, setMetrics] = useState({});
  const [params, setParams] = useState({
    gravity: 0.099, softening: 25, damping: 1.0, dmGrow: 2.0, dmFade: 0.977,
  });
  const [showGuide, setShowGuide] = useState(true);

  const init = useCallback(() => {
    stateRef.current = { particles: makeParticles(), dm: new Float32Array(GRID * GRID) };
    setStep(0);
    metricsRef.current = {};
    setMetrics({});
  }, []);

  useEffect(() => { init(); }, [init]);

  const simulate = useCallback(() => {
    const s = stateRef.current;
    if (!s) return;
    const { particles: pts, dm } = s;
    const { gravity, softening, damping, dmGrow, dmFade } = params;
    const soft2 = softening * softening;
    const N = pts.length;
    const GC = 35, GW = Math.ceil(W / GC);
    const grid = new Map();
    for (let i = 0; i < N; i++) {
      const p = pts[i];
      const key = Math.floor(p.y / GC) * GW + Math.floor(p.x / GC);
      if (!grid.has(key)) grid.set(key, []);
      grid.get(key).push(i);
    }
    for (let i = 0; i < N; i++) {
      const p = pts[i];
      let ax = 0, ay = 0;
      const gx = Math.floor(p.x / GC), gy = Math.floor(p.y / GC);
      for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
        const cell = grid.get((gy + dy) * GW + (gx + dx));
        if (!cell) continue;
        for (const j of cell) {
          if (j === i) continue;
          const o = pts[j];
          const ddx = o.x - p.x, ddy = o.y - p.y;
          const d2 = ddx * ddx + ddy * ddy + soft2;
          const f = gravity * o.m / (d2 * Math.sqrt(d2));
          ax += ddx * f; ay += ddy * f;
        }
      }
      const dmx = Math.floor(p.x / CELL), dmy = Math.floor(p.y / CELL);
      if (dmx >= 1 && dmx < GRID - 1 && dmy >= 1 && dmy < GRID - 1) {
        const grdx = (dm[dmy * GRID + dmx + 1] - dm[dmy * GRID + dmx - 1]) * 0.5;
        const grdy = (dm[(dmy + 1) * GRID + dmx] - dm[(dmy - 1) * GRID + dmx]) * 0.5;
        ax += grdx * gravity * 0.12;
        ay += grdy * gravity * 0.12;
      }
      p.vx = (p.vx + ax) * damping;
      p.vy = (p.vy + ay) * damping;
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x += W; if (p.x >= W) p.x -= W;
      if (p.y < 0) p.y += W; if (p.y >= W) p.y -= W;
    }
    for (let i = 0; i < dm.length; i++) dm[i] *= dmFade;
    for (const p of pts) {
      const gx = Math.floor(p.x / CELL), gy = Math.floor(p.y / CELL);
      for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
        const nx = gx + dx, ny = gy + dy;
        if (nx < 0 || nx >= GRID || ny < 0 || ny >= GRID) continue;
        dm[ny * GRID + nx] += Math.exp(-(dx * dx + dy * dy) / 1.2) * p.m * dmGrow * 0.01;
      }
    }
  }, [params]);

  const computeMetrics = useCallback(() => {
    const s = stateRef.current;
    if (!s) return {};
    const { particles: pts, dm } = s;

    // 1. Matter density on grid
    const mDens = new Float32Array(GRID * GRID);
    for (const p of pts) {
      const gx = Math.floor(p.x / CELL), gy = Math.floor(p.y / CELL);
      if (gx >= 0 && gx < GRID && gy >= 0 && gy < GRID) mDens[gy * GRID + gx] += p.m;
    }

    // 2. Thresholds
    let maxM = 0, maxD = 0;
    for (let i = 0; i < GRID * GRID; i++) {
      if (mDens[i] > maxM) maxM = mDens[i];
      if (dm[i] > maxD) maxD = dm[i];
    }
    if (maxM < 0.001 || maxD < 0.001) return {};

    const mThresh = maxM * 0.1, dThresh = maxD * 0.08;
    let mCount = 0, dCount = 0, bothCount = 0, neitherCount = 0;
    for (let i = 0; i < GRID * GRID; i++) {
      const hasM = mDens[i] > mThresh;
      const hasD = dm[i] > dThresh;
      if (hasM) mCount++;
      if (hasD) dCount++;
      if (hasM && hasD) bothCount++;
      if (!hasM && !hasD) neitherCount++;
    }

    const total = GRID * GRID;
    const mFrac = mCount / total;
    const dFrac = dCount / total;
    const voidFrac = neitherCount / total;
    const haloRatio = dCount > 0 && mCount > 0 ? dFrac / mFrac : 0;
    const dmExcess = dCount > 0 && mCount > 0 ? (dCount - bothCount) / dCount : 0;

    // 3. Structure: count connected clusters in matter
    const visited = new Uint8Array(GRID * GRID);
    let clusterCount = 0;
    for (let i = 0; i < total; i++) {
      if (mDens[i] > mThresh && !visited[i]) {
        clusterCount++;
        const stack = [i];
        while (stack.length) {
          const c = stack.pop();
          if (visited[c]) continue;
          visited[c] = 1;
          const cx = c % GRID, cy = Math.floor(c / GRID);
          for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
            const nx = cx + dx, ny = cy + dy;
            if (nx >= 0 && nx < GRID && ny >= 0 && ny < GRID) {
              const ni = ny * GRID + nx;
              if (mDens[ni] > mThresh && !visited[ni]) stack.push(ni);
            }
          }
        }
      }
    }

    // 4. DM-only cells (filament proxy — DM present where matter isn't)
    let dmOnlyCount = 0;
    for (let i = 0; i < total; i++) {
      if (dm[i] > dThresh && mDens[i] <= mThresh) dmOnlyCount++;
    }
    const filamentProxy = dmOnlyCount / total;

    return {
      matterFrac: mFrac,
      dmFrac: dFrac,
      voidFrac,
      haloRatio,
      dmExcess,
      clusters: clusterCount,
      filamentProxy,
    };
  }, []);

  const render = useCallback(() => {
    const s = stateRef.current;
    if (!s) return;
    const { particles: pts, dm } = s;
    let maxD = 0.001;
    for (let i = 0; i < dm.length; i++) if (dm[i] > maxD) maxD = dm[i];

    const mc = mRef.current?.getContext("2d");
    if (mc) {
      mc.fillStyle = "#04040c"; mc.fillRect(0, 0, W, W);
      for (const p of pts) {
        const sz = Math.max(1.5, Math.min(3, p.m * 1.5));
        mc.fillStyle = "#e8c44a";
        mc.beginPath(); mc.arc(p.x, p.y, sz, 0, Math.PI * 2); mc.fill();
      }
    }

    const dc = dRef.current?.getContext("2d");
    if (dc) {
      const img = dc.createImageData(GRID, GRID);
      const d = img.data;
      for (let i = 0; i < GRID * GRID; i++) {
        const n = Math.pow(Math.min(1, dm[i] / maxD), 0.35);
        const pi = i * 4;
        d[pi] = Math.min(255, n * n * 220 + n * 35 | 0);
        d[pi + 1] = Math.min(255, n * n * 60 + n * 15 | 0);
        d[pi + 2] = Math.min(255, n * 255 | 0);
        d[pi + 3] = 255;
      }
      const tmp = document.createElement("canvas");
      tmp.width = GRID; tmp.height = GRID;
      tmp.getContext("2d").putImageData(img, 0, 0);
      dc.imageSmoothingEnabled = true;
      dc.imageSmoothingQuality = "high";
      dc.drawImage(tmp, 0, 0, GRID, GRID, 0, 0, W, W);
    }

    const cc = cRef.current?.getContext("2d");
    if (cc) {
      const mDens = new Float32Array(GRID * GRID);
      for (const p of pts) {
        const gx = Math.floor(p.x / CELL), gy = Math.floor(p.y / CELL);
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const nx = gx + dx, ny = gy + dy;
          if (nx >= 0 && nx < GRID && ny >= 0 && ny < GRID)
            mDens[ny * GRID + nx] += p.m * Math.exp(-(dx * dx + dy * dy) / 0.8);
        }
      }
      let maxC = 0.001;
      const cBuf = new Float32Array(GRID * GRID);
      for (let i = 0; i < cBuf.length; i++) {
        cBuf[i] = mDens[i] + dm[i] * 3;
        if (cBuf[i] > maxC) maxC = cBuf[i];
      }
      const img = cc.createImageData(GRID, GRID);
      const d = img.data;
      for (let i = 0; i < GRID * GRID; i++) {
        const n = Math.pow(Math.min(1, cBuf[i] / maxC), 0.4);
        const pi = i * 4;
        if (n < 0.3) { const t = n / 0.3; d[pi] = t * 180 | 0; d[pi + 1] = 0; d[pi + 2] = t * 20 | 0; }
        else if (n < 0.6) { const t = (n - 0.3) / 0.3; d[pi] = 180 + t * 75 | 0; d[pi + 1] = t * 140 | 0; d[pi + 2] = 0; }
        else { const t = (n - 0.6) / 0.4; d[pi] = 255; d[pi + 1] = 140 + t * 115 | 0; d[pi + 2] = t * 220 | 0; }
        d[pi + 3] = 255;
      }
      const tmp = document.createElement("canvas");
      tmp.width = GRID; tmp.height = GRID;
      tmp.getContext("2d").putImageData(img, 0, 0);
      cc.imageSmoothingEnabled = true;
      cc.imageSmoothingQuality = "high";
      cc.drawImage(tmp, 0, 0, GRID, GRID, 0, 0, W, W);
    }

    const m = computeMetrics();
    metricsRef.current = m;
    setMetrics(m);
  }, [computeMetrics]);

  useEffect(() => {
    if (!running) { cancelAnimationFrame(animRef.current); return; }
    const loop = () => {
      simulate(); simulate(); render();
      setStep(s => s + 2);
      animRef.current = requestAnimationFrame(loop);
    };
    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, [running, simulate, render]);

  useEffect(() => { render(); }, [render]);

  const prerun = (n) => {
    for (let i = 0; i < n; i++) simulate();
    render();
    setStep(s => s + n);
  };

  const canvasStyle = { width: W, height: W, borderRadius: 6, border: "1px solid rgba(255,255,255,0.05)" };

  const grade = (val, target, tol) => {
    const diff = Math.abs(val - target);
    if (diff < tol * 0.5) return { symbol: "✓", color: "#22c55e" };
    if (diff < tol) return { symbol: "~", color: "#f59e0b" };
    return { symbol: "✗", color: "#ef4444" };
  };

  const Slider = ({ label, param, min, max, step: s, fmt }) => (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 2 }}>
        <span style={{ color: "#9ca3af", fontFamily: "'JetBrains Mono', monospace" }}>{label}</span>
        <span style={{ color: "#e5e7eb", fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>
          {fmt ? fmt(params[param]) : params[param].toFixed(3)}
        </span>
      </div>
      <input type="range" min={min} max={max} step={s}
        value={params[param]}
        onChange={e => setParams(p => ({ ...p, [param]: parseFloat(e.target.value) }))}
        style={{ width: "100%", accentColor: "#c084fc", height: 4 }}
      />
    </div>
  );

  const MetricRow = ({ label, val, target, targetLabel, tol, explain }) => {
    if (val == null || isNaN(val)) return null;
    const g = grade(val, target, tol);
    return (
      <div style={{
        display: "grid", gridTemplateColumns: "140px 70px 50px 90px 1fr",
        alignItems: "center", padding: "5px 0",
        borderBottom: "1px solid rgba(255,255,255,0.04)", gap: 6,
      }}>
        <span style={{ fontSize: 11, color: "#9ca3af", fontFamily: "'JetBrains Mono', monospace" }}>{label}</span>
        <span style={{ fontSize: 12, color: "#e5e7eb", fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>
          {typeof val === 'number' ? val.toFixed(2) : val}
        </span>
        <span style={{ fontSize: 14, color: g.color, fontWeight: 700 }}>{g.symbol}</span>
        <span style={{ fontSize: 10, color: "#6b7280", fontFamily: "'JetBrains Mono', monospace" }}>
          target: {targetLabel || target}
        </span>
        <span style={{ fontSize: 10, color: "#4b5563" }}>{explain}</span>
      </div>
    );
  };

  return (
    <div style={{
      minHeight: "100vh", background: "#020208", color: "#d1d5db",
      fontFamily: "'Instrument Sans', sans-serif", padding: 16,
    }}>
      <link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet" />

      <div style={{ maxWidth: 1520, margin: "0 auto" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 4 }}>
          <h1 style={{
            fontSize: 18, fontWeight: 700, margin: 0,
            background: "linear-gradient(90deg, #fbbf24, #c084fc, #22c55e)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          }}>Parameter Tuning Lab</h1>
          <span style={{ fontSize: 11, color: "#4b5563", fontFamily: "'JetBrains Mono', monospace" }}>
            Find the regime where structure matches real observations
          </span>
        </div>

        <div style={{ display: "flex", gap: 12, marginTop: 12, flexWrap: "wrap" }}>
          {/* Left column: controls */}
          <div style={{ width: 240, flexShrink: 0 }}>
            <div style={{ display: "flex", gap: 6, marginBottom: 12, flexWrap: "wrap" }}>
              <button onClick={() => setRunning(!running)} style={{
                padding: "6px 16px", borderRadius: 5, border: "none", cursor: "pointer",
                fontWeight: 700, fontSize: 12, color: "#fff",
                background: running ? "#dc2626" : "#22c55e",
                fontFamily: "'Instrument Sans', sans-serif",
              }}>{running ? "⏸ Pause" : "▶ Run"}</button>
              <button onClick={() => prerun(500)} style={{
                padding: "6px 10px", borderRadius: 5, border: "1px solid #1c1c2e",
                cursor: "pointer", background: "#0a0a14", color: "#9ca3af", fontSize: 11,
                fontFamily: "'JetBrains Mono', monospace",
              }}>+500</button>
              <button onClick={() => prerun(2000)} style={{
                padding: "6px 10px", borderRadius: 5, border: "1px solid #1c1c2e",
                cursor: "pointer", background: "#0a0a14", color: "#9ca3af", fontSize: 11,
                fontFamily: "'JetBrains Mono', monospace",
              }}>+2000</button>
              <button onClick={() => { init(); setRunning(false); }} style={{
                padding: "6px 10px", borderRadius: 5, border: "1px solid #1c1c2e",
                cursor: "pointer", background: "#0a0a14", color: "#6b7280", fontSize: 11,
                fontFamily: "'JetBrains Mono', monospace",
              }}>Reset</button>
            </div>

            <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 8,
              fontFamily: "'JetBrains Mono', monospace" }}>t = {step}</div>

            <Slider label="Gravity" param="gravity" min={0.001} max={0.15} step={0.001}
              fmt={v => v.toFixed(3)} />
            <Slider label="Softening" param="softening" min={1} max={50} step={1}
              fmt={v => v.toFixed(0)} />
            <Slider label="Damping" param="damping" min={0.9} max={1.0} step={0.001}
              fmt={v => v.toFixed(3)} />
            <Slider label="DM Growth" param="dmGrow" min={0.1} max={3.0} step={0.05}
              fmt={v => v.toFixed(2)} />
            <Slider label="DM Fade" param="dmFade" min={0.9} max={0.999} step={0.001}
              fmt={v => v.toFixed(3)} />

            {/* Presets */}
            <div style={{ marginTop: 12, fontSize: 11, color: "#6b7280", fontFamily: "'JetBrains Mono', monospace", marginBottom: 6 }}>
              PRESETS
            </div>
            <button onClick={() => setParams({ gravity: 0.099, softening: 25, damping: 1.0, dmGrow: 2.0, dmFade: 0.977 })} style={{
              display: "block", width: "100%", padding: "6px 10px", marginBottom: 4,
              borderRadius: 5, border: "1px solid rgba(192,132,252,0.2)",
              cursor: "pointer", background: "rgba(192,132,252,0.06)", color: "#c084fc",
              fontSize: 11, fontFamily: "'JetBrains Mono', monospace", textAlign: "left",
            }}>Russell Sonogram</button>
            <button onClick={() => setParams({ gravity: 0.06, softening: 15, damping: 0.998, dmGrow: 1.2, dmFade: 0.99 })} style={{
              display: "block", width: "100%", padding: "6px 10px", marginBottom: 4,
              borderRadius: 5, border: "1px solid rgba(251,191,36,0.2)",
              cursor: "pointer", background: "rgba(251,191,36,0.06)", color: "#fbbf24",
              fontSize: 11, fontFamily: "'JetBrains Mono', monospace", textAlign: "left",
            }}>Moderate Growth</button>
            <button onClick={() => setParams({ gravity: 0.045, softening: 20, damping: 0.999, dmGrow: 0.8, dmFade: 0.993 })} style={{
              display: "block", width: "100%", padding: "6px 10px", marginBottom: 4,
              borderRadius: 5, border: "1px solid rgba(34,197,94,0.2)",
              cursor: "pointer", background: "rgba(34,197,94,0.06)", color: "#22c55e",
              fontSize: 11, fontFamily: "'JetBrains Mono', monospace", textAlign: "left",
            }}>Slow Superfluid</button>
            <button onClick={() => setParams({ gravity: 0.08, softening: 30, damping: 1.0, dmGrow: 1.5, dmFade: 0.985 })} style={{
              display: "block", width: "100%", padding: "6px 10px", marginBottom: 4,
              borderRadius: 5, border: "1px solid rgba(96,165,250,0.2)",
              cursor: "pointer", background: "rgba(96,165,250,0.06)", color: "#60a5fa",
              fontSize: 11, fontFamily: "'JetBrains Mono', monospace", textAlign: "left",
            }}>Extended Halos</button>
          </div>

          {/* Three panels */}
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <div>
              <div style={{ fontSize: 10, fontWeight: 700, color: "#fbbf24", fontFamily: "'JetBrains Mono', monospace", marginBottom: 4, letterSpacing: 1 }}>
                MATTER
              </div>
              <canvas ref={mRef} width={W} height={W} style={canvasStyle} />
            </div>
            <div>
              <div style={{ fontSize: 10, fontWeight: 700, color: "#c084fc", fontFamily: "'JetBrains Mono', monospace", marginBottom: 4, letterSpacing: 1 }}>
                DARK MATTER
              </div>
              <canvas ref={dRef} width={W} height={W} style={canvasStyle} />
            </div>
            <div>
              <div style={{ fontSize: 10, fontWeight: 700, color: "#fb923c", fontFamily: "'JetBrains Mono', monospace", marginBottom: 4, letterSpacing: 1 }}>
                CURVATURE
              </div>
              <canvas ref={cRef} width={W} height={W} style={canvasStyle} />
            </div>
          </div>
        </div>

        {/* Metrics dashboard */}
        <div style={{
          marginTop: 14, padding: 14, borderRadius: 8,
          background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)",
          maxWidth: 1500,
        }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: "#22c55e", marginBottom: 8,
            fontFamily: "'JetBrains Mono', monospace", letterSpacing: 1 }}>
            DIAGNOSTIC SCORECARD — Does This Match Reality?
          </div>

          <MetricRow label="Halo/Matter ratio" val={metrics.haloRatio} target={3.0} tol={2.0}
            targetLabel="2-5×" explain="DM halos should cover 2-5× more area than matter" />
          <MetricRow label="Void fraction" val={metrics.voidFrac} target={0.75} tol={0.15}
            targetLabel="60-90%" explain="Real universe is ~80% voids by volume" />
          <MetricRow label="Matter fraction" val={metrics.matterFrac} target={0.06} tol={0.06}
            targetLabel="3-12%" explain="Galaxies occupy ~5% of cosmic volume" />
          <MetricRow label="DM fraction" val={metrics.dmFrac} target={0.20} tol={0.12}
            targetLabel="8-32%" explain="DM fills ~25% of volume (halos + filaments)" />
          <MetricRow label="DM excess" val={metrics.dmExcess} target={0.50} tol={0.25}
            targetLabel="25-75%" explain="Fraction of DM area beyond matter locations (filaments + extended halos)" />
          <MetricRow label="Matter clusters" val={metrics.clusters} target={8} tol={6}
            targetLabel="3-14" explain="Distinct gravitationally-bound groups" />
          <MetricRow label="Filament proxy" val={metrics.filamentProxy} target={0.12} tol={0.09}
            targetLabel="3-20%" explain="DM-only cells = filaments + extended halos without matter" />

          <div style={{
            marginTop: 10, padding: 10, borderRadius: 6,
            background: "rgba(34,197,94,0.06)", border: "1px solid rgba(34,197,94,0.1)",
          }}>
            <div style={{ fontSize: 11, color: "#22c55e", fontWeight: 600, marginBottom: 4,
              fontFamily: "'JetBrains Mono', monospace" }}>
              HOW TO READ THIS
            </div>
            <div style={{ fontSize: 11, color: "#9ca3af", lineHeight: 1.6 }}>
              <span style={{ color: "#22c55e", fontWeight: 700 }}>✓</span> = matches real universe &nbsp;&nbsp;
              <span style={{ color: "#f59e0b", fontWeight: 700 }}>~</span> = close &nbsp;&nbsp;
              <span style={{ color: "#ef4444", fontWeight: 700 }}>✗</span> = off target. 
              Get all green checks and you've found the parameter regime that reproduces observed cosmic structure.
              Run at least 1000+ steps before judging — structure needs time to form.
              The sweet spot: DM halos visibly larger than matter clumps, with filamentary bridges between them, and wide empty voids.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
