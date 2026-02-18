import { useState, useEffect, useRef, useCallback } from "react";

const W = 400;
const GRID = 100;
const CELL = W / GRID;

function makeParticles() {
  const p = [];
  const seeds = [];
  for (let s = 0; s < 18; s++) {
    seeds.push([45 + Math.random() * (W - 90), 45 + Math.random() * (W - 90)]);
  }
  for (const [cx, cy] of seeds) {
    const n = 6 + Math.floor(Math.random() * 14);
    for (let i = 0; i < n; i++) {
      const a = Math.random() * Math.PI * 2;
      const r = Math.random() * 16;
      p.push({
        x: cx + Math.cos(a) * r, y: cy + Math.sin(a) * r,
        vx: (Math.random() - 0.5) * 0.15, vy: (Math.random() - 0.5) * 0.15,
        m: 0.4 + Math.random() * 1.2,
      });
    }
  }
  return p;
}

function analyzeStructure(particles, dmBuf, W, GRID, CELL) {
  const scores = {};

  // 1. Matter clumpiness - are particles in distinct groups or spread out?
  let clumpCount = 0;
  const visited = new Uint8Array(particles.length);
  const thresh = 12;
  for (let i = 0; i < particles.length; i++) {
    if (visited[i]) continue;
    let size = 0;
    const stack = [i];
    while (stack.length) {
      const ci = stack.pop();
      if (visited[ci]) continue;
      visited[ci] = 1; size++;
      for (let j = 0; j < particles.length; j++) {
        if (visited[j]) continue;
        const dx = particles[ci].x - particles[j].x;
        const dy = particles[ci].y - particles[j].y;
        if (dx*dx + dy*dy < thresh*thresh) stack.push(j);
      }
    }
    if (size >= 3) clumpCount++;
  }
  scores.clumps = Math.min(1, clumpCount / 8);

  // 2. Void fraction - how much of the matter space is empty?
  const mGrid = new Float32Array(GRID * GRID);
  for (const p of particles) {
    const gx = Math.floor(p.x / CELL), gy = Math.floor(p.y / CELL);
    if (gx >= 0 && gx < GRID && gy >= 0 && gy < GRID)
      mGrid[gy * GRID + gx] += p.m;
  }
  let emptyM = 0;
  for (let i = 0; i < mGrid.length; i++) if (mGrid[i] < 0.01) emptyM++;
  scores.voids = emptyM / mGrid.length;

  // 3. DM halo extent - does DM extend beyond matter?
  let dmOnlyPixels = 0, dmTotal = 0;
  let maxDM = 0.001;
  for (let i = 0; i < dmBuf.length; i++) if (dmBuf[i] > maxDM) maxDM = dmBuf[i];
  const dmThresh = maxDM * 0.05;
  const mThresh = 0.01;
  for (let i = 0; i < GRID * GRID; i++) {
    if (dmBuf[i] > dmThresh) {
      dmTotal++;
      if (mGrid[i] < mThresh) dmOnlyPixels++;
    }
  }
  scores.haloExtent = dmTotal > 0 ? dmOnlyPixels / dmTotal : 0;

  // 4. Sonogram texture - variance in DM field (grainy vs smooth)
  let mean = 0, count = 0;
  for (let i = 0; i < dmBuf.length; i++) if (dmBuf[i] > dmThresh) { mean += dmBuf[i]; count++; }
  mean = count > 0 ? mean / count : 0;
  let variance = 0;
  for (let i = 0; i < dmBuf.length; i++) if (dmBuf[i] > dmThresh) variance += (dmBuf[i] - mean) ** 2;
  variance = count > 0 ? variance / count : 0;
  const cv = mean > 0 ? Math.sqrt(variance) / mean : 0;
  scores.texture = Math.min(1, cv / 0.8);

  // 5. Filamentary structure - chains between clumps
  let filPixels = 0;
  const filThreshLow = maxDM * 0.03;
  const filThreshHigh = maxDM * 0.25;
  for (let i = 0; i < dmBuf.length; i++) {
    if (dmBuf[i] > filThreshLow && dmBuf[i] < filThreshHigh && mGrid[i] < mThresh) {
      filPixels++;
    }
  }
  scores.filaments = Math.min(1, filPixels / (GRID * GRID * 0.06));

  // 6. DM void darkness - are voids truly dark in DM?
  let voidDM = 0, voidCount = 0;
  for (let i = 0; i < GRID * GRID; i++) {
    if (mGrid[i] < mThresh) {
      voidDM += dmBuf[i]; voidCount++;
    }
  }
  const avgVoidDM = voidCount > 0 ? voidDM / voidCount : 0;
  scores.voidDarkness = 1 - Math.min(1, avgVoidDM / (maxDM * 0.1));

  // Overall
  scores.overall = (
    scores.clumps * 0.15 + scores.voids * 0.15 + scores.haloExtent * 0.25 +
    scores.texture * 0.15 + scores.filaments * 0.15 + scores.voidDarkness * 0.15
  );

  return scores;
}

export default function ParameterExplorer() {
  const mRef = useRef(null);
  const dRef = useRef(null);
  const cRef = useRef(null);
  const stateRef = useRef(null);
  const animRef = useRef(null);
  const paramsRef = useRef(null);

  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(0);
  const [scores, setScores] = useState({});
  const [params, setParams] = useState({
    gravity: 0.099, softening: 25, damping: 1.0,
    dmGrow: 2.0, dmFade: 0.977, dmRadius: 2,
  });
  const [bestParams, setBestParams] = useState(null);
  const [bestScore, setBestScore] = useState(0);

  paramsRef.current = params;

  const init = useCallback(() => {
    stateRef.current = { particles: makeParticles(), dmBuf: new Float32Array(GRID * GRID) };
    setStep(0); setScores({});
  }, []);

  useEffect(() => { init(); }, [init]);

  const simulate = useCallback(() => {
    const s = stateRef.current;
    if (!s) return;
    const { particles, dmBuf } = s;
    const P = paramsRef.current;
    const soft2 = P.softening * P.softening;
    const N = particles.length;
    const GC = 30;
    const GW = Math.ceil(W / GC);
    const grid = new Map();

    for (let i = 0; i < N; i++) {
      const p = particles[i];
      const key = (Math.floor(p.y / GC)) * GW + Math.floor(p.x / GC);
      if (!grid.has(key)) grid.set(key, []);
      grid.get(key).push(i);
    }

    for (let i = 0; i < N; i++) {
      const p = particles[i];
      let ax = 0, ay = 0;
      const gx = Math.floor(p.x / GC), gy = Math.floor(p.y / GC);
      for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          const cell = grid.get((gy + dy) * GW + (gx + dx));
          if (!cell) continue;
          for (const j of cell) {
            if (j === i) continue;
            const o = particles[j];
            const ddx = o.x - p.x, ddy = o.y - p.y;
            const d2 = ddx * ddx + ddy * ddy + soft2;
            const f = P.gravity * o.m / (d2 * Math.sqrt(d2));
            ax += ddx * f; ay += ddy * f;
          }
        }
      }

      const dmx = Math.floor(p.x / CELL), dmy = Math.floor(p.y / CELL);
      if (dmx >= 1 && dmx < GRID - 1 && dmy >= 1 && dmy < GRID - 1) {
        const grdx = (dmBuf[dmy * GRID + dmx + 1] - dmBuf[dmy * GRID + dmx - 1]) * 0.4;
        const grdy = (dmBuf[(dmy + 1) * GRID + dmx] - dmBuf[(dmy - 1) * GRID + dmx]) * 0.4;
        ax += grdx * P.gravity * 0.1;
        ay += grdy * P.gravity * 0.1;
      }

      p.vx = (p.vx + ax) * P.damping;
      p.vy = (p.vy + ay) * P.damping;
      p.x += p.vx; p.y += p.vy;
      if (p.x < 2) { p.x = 2; p.vx *= -0.3; }
      if (p.x > W-2) { p.x = W-2; p.vx *= -0.3; }
      if (p.y < 2) { p.y = 2; p.vy *= -0.3; }
      if (p.y > W-2) { p.y = W-2; p.vy *= -0.3; }
    }

    for (let i = 0; i < dmBuf.length; i++) dmBuf[i] *= P.dmFade;
    const R = P.dmRadius;
    for (const p of particles) {
      const gx = Math.floor(p.x / CELL), gy = Math.floor(p.y / CELL);
      for (let dy = -R; dy <= R; dy++) {
        for (let dx = -R; dx <= R; dx++) {
          const nx = gx + dx, ny = gy + dy;
          if (nx < 0 || nx >= GRID || ny < 0 || ny >= GRID) continue;
          const r = Math.sqrt(dx * dx + dy * dy);
          if (r > R + 0.5) continue;
          dmBuf[ny * GRID + nx] += Math.exp(-r * r / (R * 0.6 + 0.3)) * p.m * P.dmGrow * 0.008;
        }
      }
    }
  }, []);

  const render = useCallback(() => {
    const s = stateRef.current;
    if (!s) return;
    const { particles, dmBuf } = s;

    let maxDM = 0.001;
    for (let i = 0; i < dmBuf.length; i++) if (dmBuf[i] > maxDM) maxDM = dmBuf[i];

    const mc = mRef.current?.getContext("2d");
    if (mc) {
      mc.fillStyle = "#04040c"; mc.fillRect(0, 0, W, W);
      for (const p of particles) {
        const sz = Math.max(1.2, Math.min(2.8, p.m * 1.3));
        mc.fillStyle = "#e8c840";
        mc.beginPath(); mc.arc(p.x, p.y, sz, 0, Math.PI * 2); mc.fill();
      }
    }

    const dc = dRef.current?.getContext("2d");
    if (dc) {
      const img = dc.createImageData(GRID, GRID);
      const d = img.data;
      for (let i = 0; i < GRID * GRID; i++) {
        const n = Math.pow(Math.min(1, dmBuf[i] / maxDM), 0.35);
        const pi = i * 4;
        d[pi] = Math.min(255, n * n * 200 + n * 40 | 0);
        d[pi + 1] = Math.min(255, n * n * 50 + n * 10 | 0);
        d[pi + 2] = Math.min(255, n * 255 | 0);
        d[pi + 3] = 255;
      }
      const tmp = document.createElement("canvas");
      tmp.width = GRID; tmp.height = GRID;
      tmp.getContext("2d").putImageData(img, 0, 0);
      dc.imageSmoothingEnabled = true;
      dc.drawImage(tmp, 0, 0, GRID, GRID, 0, 0, W, W);
    }

    const cc = cRef.current?.getContext("2d");
    if (cc) {
      const mDens = new Float32Array(GRID * GRID);
      for (const p of particles) {
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
        cBuf[i] = mDens[i] + dmBuf[i] * 2.5;
        if (cBuf[i] > maxC) maxC = cBuf[i];
      }
      const img = cc.createImageData(GRID, GRID);
      const d = img.data;
      for (let i = 0; i < GRID * GRID; i++) {
        const n = Math.pow(Math.min(1, cBuf[i] / maxC), 0.4);
        const pi = i * 4;
        if (n < 0.25) { const t = n / 0.25; d[pi] = t * 160 | 0; d[pi + 1] = 0; d[pi + 2] = t * 15 | 0; }
        else if (n < 0.55) { const t = (n - 0.25) / 0.3; d[pi] = 160 + t * 95 | 0; d[pi + 1] = t * 130 | 0; d[pi + 2] = 15; }
        else { const t = (n - 0.55) / 0.45; d[pi] = 255; d[pi + 1] = 130 + t * 125 | 0; d[pi + 2] = 15 + t * 240 | 0; }
        d[pi + 3] = 255;
      }
      const tmp = document.createElement("canvas");
      tmp.width = GRID; tmp.height = GRID;
      tmp.getContext("2d").putImageData(img, 0, 0);
      cc.imageSmoothingEnabled = true;
      cc.drawImage(tmp, 0, 0, GRID, GRID, 0, 0, W, W);
    }
  }, []);

  useEffect(() => {
    if (!running) { cancelAnimationFrame(animRef.current); return; }
    const loop = () => {
      simulate(); simulate();
      render();
      setStep(s => {
        const ns = s + 2;
        if (ns % 40 === 0 && stateRef.current) {
          const sc = analyzeStructure(stateRef.current.particles, stateRef.current.dmBuf, W, GRID, CELL);
          setScores(sc);
          if (sc.overall > bestScore) {
            setBestScore(sc.overall);
            setBestParams({ ...paramsRef.current, step: ns });
          }
        }
        return ns;
      });
      animRef.current = requestAnimationFrame(loop);
    };
    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, [running, simulate, render, bestScore]);

  useEffect(() => { render(); }, [render]);

  const skip = (n) => {
    for (let i = 0; i < n; i++) simulate();
    render();
    setStep(s => s + n);
    if (stateRef.current) {
      const sc = analyzeStructure(stateRef.current.particles, stateRef.current.dmBuf, W, GRID, CELL);
      setScores(sc);
      if (sc.overall > bestScore) {
        setBestScore(sc.overall); setBestParams({ ...params, step: step + n });
      }
    }
  };

  const updateParam = (key, val) => setParams(p => ({ ...p, [key]: val }));

  const scoreBar = (val, label, desc) => (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 2 }}>
        <span style={{ fontSize: 10, color: "#9ca3af", fontFamily: "mono" }}>{label}</span>
        <span style={{ fontSize: 10, color: val > 0.6 ? "#4ade80" : val > 0.3 ? "#fbbf24" : "#f87171",
          fontWeight: 700, fontFamily: "mono" }}>{(val * 100 | 0)}%</span>
      </div>
      <div style={{ height: 4, background: "#1a1a2e", borderRadius: 2, overflow: "hidden" }}>
        <div style={{
          height: "100%", borderRadius: 2, transition: "width 0.3s",
          width: `${val * 100}%`,
          background: val > 0.6 ? "#4ade80" : val > 0.3 ? "#fbbf24" : "#f87171",
        }} />
      </div>
      <div style={{ fontSize: 9, color: "#4b5563", marginTop: 1 }}>{desc}</div>
    </div>
  );

  const slider = (label, key, min, max, stepSize, unit = "") => (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
        <span style={{ fontSize: 10, color: "#9ca3af" }}>{label}</span>
        <span style={{ fontSize: 10, color: "#c084fc", fontFamily: "mono", fontWeight: 600 }}>
          {params[key].toFixed(stepSize < 0.01 ? 3 : stepSize < 1 ? 2 : 1)}{unit}
        </span>
      </div>
      <input type="range" min={min} max={max} step={stepSize} value={params[key]}
        onChange={e => updateParam(key, parseFloat(e.target.value))}
        style={{ width: "100%", accentColor: "#c084fc", height: 3, cursor: "pointer" }} />
    </div>
  );

  const cs = { width: W, height: W, borderRadius: 6, border: "1px solid rgba(255,255,255,0.04)" };
  const overall = scores.overall || 0;

  return (
    <div style={{
      minHeight: "100vh", background: "#020208", color: "#d1d5db",
      fontFamily: "'Segoe UI', system-ui, sans-serif", padding: 16, fontSize: 12,
    }}>
      <div style={{ maxWidth: 1580, margin: "0 auto" }}>
        <h1 style={{
          fontSize: 18, fontWeight: 700, margin: "0 0 4px",
          background: "linear-gradient(90deg, #fbbf24, #c084fc)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
        }}>Parameter Explorer — Find the Right Physics</h1>
        <p style={{ fontSize: 11, color: "#4b5563", margin: "0 0 12px" }}>
          Adjust sliders while running. The scorer checks if the structure matches real cosmology.
        </p>

        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          {/* LEFT: Controls + Scoring */}
          <div style={{ width: 210, flexShrink: 0 }}>
            {/* Controls */}
            <div style={{ display: "flex", gap: 6, marginBottom: 12, flexWrap: "wrap" }}>
              <button onClick={() => setRunning(!running)} style={{
                padding: "6px 14px", borderRadius: 5, border: "none", cursor: "pointer",
                fontWeight: 700, fontSize: 11, color: "#fff",
                background: running ? "#dc2626" : "#22c55e",
              }}>{running ? "⏸" : "▶"} {running ? "Pause" : "Run"}</button>
              <button onClick={() => skip(500)} style={{
                padding: "6px 10px", borderRadius: 5, border: "1px solid #1c1c2e",
                cursor: "pointer", background: "#0a0a14", color: "#6b7280", fontSize: 10,
              }}>+500</button>
              <button onClick={() => skip(2000)} style={{
                padding: "6px 10px", borderRadius: 5, border: "1px solid #1c1c2e",
                cursor: "pointer", background: "#0a0a14", color: "#6b7280", fontSize: 10,
              }}>+2000</button>
              <button onClick={() => { init(); setRunning(false); }} style={{
                padding: "6px 10px", borderRadius: 5, border: "1px solid #1c1c2e",
                cursor: "pointer", background: "#0a0a14", color: "#6b7280", fontSize: 10,
              }}>Reset</button>
            </div>

            <div style={{ fontSize: 10, color: "#4b5563", marginBottom: 10 }}>t = {step}</div>

            {/* Sliders */}
            <div style={{
              padding: 12, borderRadius: 8, background: "rgba(192,132,252,0.03)",
              border: "1px solid rgba(192,132,252,0.08)", marginBottom: 12,
            }}>
              <div style={{ fontSize: 10, fontWeight: 700, color: "#c084fc", marginBottom: 8, textTransform: "uppercase", letterSpacing: 1 }}>
                Parameters
              </div>
              {slider("Gravity", "gravity", 0.01, 0.15, 0.001)}
              {slider("Softening", "softening", 1, 50, 1)}
              {slider("Damping", "damping", 0.9, 1.0, 0.001)}
              {slider("DM Growth", "dmGrow", 0.1, 4.0, 0.1)}
              {slider("DM Fade", "dmFade", 0.95, 0.999, 0.001)}
              {slider("DM Radius", "dmRadius", 1, 5, 1, "px")}
            </div>

            {/* Scores */}
            <div style={{
              padding: 12, borderRadius: 8,
              background: overall > 0.6 ? "rgba(74,222,128,0.04)" : "rgba(251,191,36,0.04)",
              border: `1px solid ${overall > 0.6 ? "rgba(74,222,128,0.15)" : "rgba(251,191,36,0.1)"}`,
            }}>
              <div style={{
                fontSize: 10, fontWeight: 700, textTransform: "uppercase", letterSpacing: 1, marginBottom: 8,
                color: overall > 0.6 ? "#4ade80" : "#fbbf24",
              }}>
                Structure Score: {(overall * 100 | 0)}%
              </div>
              {scoreBar(scores.clumps || 0, "Clumps", "Distinct galaxy clusters")}
              {scoreBar(scores.voids || 0, "Voids", "Empty regions between structure")}
              {scoreBar(scores.haloExtent || 0, "Halo Extent", "DM extends beyond matter")}
              {scoreBar(scores.texture || 0, "Sonogram", "Grainy DM texture (not smooth)")}
              {scoreBar(scores.filaments || 0, "Filaments", "Bridges between clumps")}
              {scoreBar(scores.voidDarkness || 0, "Void Darkness", "DM absent in voids")}
            </div>

            {bestParams && (
              <div style={{
                marginTop: 10, padding: 10, borderRadius: 6,
                background: "rgba(74,222,128,0.04)", border: "1px solid rgba(74,222,128,0.1)",
              }}>
                <div style={{ fontSize: 9, fontWeight: 700, color: "#4ade80", marginBottom: 4, textTransform: "uppercase" }}>
                  Best: {(bestScore * 100 | 0)}% @ t={bestParams.step}
                </div>
                <div style={{ fontSize: 9, color: "#6b7280", fontFamily: "mono", lineHeight: 1.6 }}>
                  g={bestParams.gravity} soft={bestParams.softening}<br />
                  damp={bestParams.damping} grow={bestParams.dmGrow}<br />
                  fade={bestParams.dmFade} rad={bestParams.dmRadius}
                </div>
                <button onClick={() => setParams(p => ({
                  ...p, gravity: bestParams.gravity, softening: bestParams.softening,
                  damping: bestParams.damping, dmGrow: bestParams.dmGrow,
                  dmFade: bestParams.dmFade, dmRadius: bestParams.dmRadius,
                }))} style={{
                  marginTop: 4, padding: "4px 8px", borderRadius: 4, border: "1px solid rgba(74,222,128,0.2)",
                  background: "transparent", color: "#4ade80", fontSize: 9, cursor: "pointer",
                }}>Load Best</button>
              </div>
            )}
          </div>

          {/* RIGHT: Three panels */}
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
              <div>
                <div style={{ fontSize: 10, fontWeight: 700, color: "#fbbf24", marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>
                  Matter
                </div>
                <canvas ref={mRef} width={W} height={W} style={cs} />
              </div>
              <div>
                <div style={{ fontSize: 10, fontWeight: 700, color: "#c084fc", marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>
                  Dark Matter
                </div>
                <canvas ref={dRef} width={W} height={W} style={cs} />
              </div>
              <div>
                <div style={{ fontSize: 10, fontWeight: 700, color: "#fb923c", marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>
                  Curvature
                </div>
                <canvas ref={cRef} width={W} height={W} style={cs} />
              </div>
            </div>

            {/* Quick guide */}
            <div style={{
              marginTop: 12, padding: 12, borderRadius: 8,
              background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.04)",
              display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12,
            }}>
              <div>
                <div style={{ fontSize: 10, fontWeight: 600, color: "#fbbf24", marginBottom: 3 }}>✓ Matter should show:</div>
                <div style={{ fontSize: 10, color: "#6b7280", lineHeight: 1.5 }}>
                  Tight clumps + thin chains + big voids. Not one big blob. Not uniform scatter.
                </div>
              </div>
              <div>
                <div style={{ fontSize: 10, fontWeight: 600, color: "#c084fc", marginBottom: 3 }}>✓ DM should show:</div>
                <div style={{ fontSize: 10, color: "#6b7280", lineHeight: 1.5 }}>
                  Halos WIDER than matter. Fuzzy bridges. Grainy texture. Dark voids. Not a uniform wash.
                </div>
              </div>
              <div>
                <div style={{ fontSize: 10, fontWeight: 600, color: "#fb923c", marginBottom: 3 }}>✓ Curvature should show:</div>
                <div style={{ fontSize: 10, color: "#6b7280", lineHeight: 1.5 }}>
                  Smoothest panel. Broad orange glow beyond matter. White-hot at mass centers.
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
