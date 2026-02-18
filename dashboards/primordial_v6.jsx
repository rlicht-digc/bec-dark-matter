import { useState, useEffect, useRef, useCallback } from "react";

const SIZE = 560;
const PARTICLES_DEFAULT = 600;

function dist(x1, y1, x2, y2) {
  const dx = x1 - x2, dy = y1 - y2;
  return Math.sqrt(dx * dx + dy * dy);
}

function makeParticles(scenario) {
  const p = [];
  const S = SIZE;
  if (scenario === "bigbang") {
    for (let i = 0; i < 700; i++) {
      const a = Math.random() * Math.PI * 2;
      const r = Math.random() * 20 + 1;
      const spd = 0.3 + Math.random() * 2.2;
      p.push([
        S / 2 + Math.cos(a) * r,
        S / 2 + Math.sin(a) * r,
        Math.cos(a) * spd + (Math.random() - 0.5) * 0.3,
        Math.sin(a) * spd + (Math.random() - 0.5) * 0.3,
        0.3 + Math.random() * 1.7,
      ]);
    }
  } else if (scenario === "web") {
    for (let s = 0; s < 25; s++) {
      const cx = 60 + Math.random() * (S - 120);
      const cy = 60 + Math.random() * (S - 120);
      const n = 10 + Math.floor(Math.random() * 20);
      for (let i = 0; i < n; i++) {
        const a = Math.random() * Math.PI * 2;
        const r = Math.random() * 20;
        p.push([
          cx + Math.cos(a) * r,
          cy + Math.sin(a) * r,
          (Math.random() - 0.5) * 0.25,
          (Math.random() - 0.5) * 0.25,
          0.3 + Math.random() * 1.2,
        ]);
      }
    }
  } else if (scenario === "bullet") {
    for (let i = 0; i < 350; i++) {
      const a = Math.random() * Math.PI * 2, r = Math.random() * 35;
      p.push([S * 0.25 + Math.cos(a) * r, S * 0.5 + Math.sin(a) * r,
        0.9 + (Math.random() - 0.5) * 0.2, (Math.random() - 0.5) * 0.2,
        0.4 + Math.random() * 1.5]);
    }
    for (let i = 0; i < 350; i++) {
      const a = Math.random() * Math.PI * 2, r = Math.random() * 35;
      p.push([S * 0.75 + Math.cos(a) * r, S * 0.5 + Math.sin(a) * r,
        -0.9 + (Math.random() - 0.5) * 0.2, (Math.random() - 0.5) * 0.2,
        0.4 + Math.random() * 1.5]);
    }
  } else {
    for (let i = 0; i < PARTICLES_DEFAULT; i++) {
      p.push([
        30 + Math.random() * (S - 60),
        30 + Math.random() * (S - 60),
        (Math.random() - 0.5) * 0.5,
        (Math.random() - 0.5) * 0.5,
        0.3 + Math.random() * 1.2,
      ]);
    }
  }
  return p; // [x, y, vx, vy, mass]
}

export default function PrimordialSim() {
  const matterRef = useRef(null);   // left canvas
  const dmRef = useRef(null);       // center canvas
  const curveRef = useRef(null);    // right canvas
  const stateRef = useRef(null);
  const animRef = useRef(null);
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(0);
  const [preset, setPreset] = useState("web");
  const [params, setParams] = useState({
    gravity: 0.099,
    softening: 25,
    damping: 1.0,
    dmGrow: 2.0,
    dmFade: 0.977,
  });

  const init = useCallback((sc) => {
    const particles = makeParticles(sc);
    // DM buffer: accumulated dark matter density per pixel (downscaled)
    const DM_RES = 140;
    const dmBuffer = new Float32Array(DM_RES * DM_RES).fill(0);
    stateRef.current = { particles, dmBuffer, DM_RES };
    setStep(0);
  }, []);

  useEffect(() => { init(preset); }, [preset, init]);

  const simulate = useCallback(() => {
    const s = stateRef.current;
    if (!s) return;
    const { particles, dmBuffer, DM_RES } = s;
    const { gravity, softening, damping, dmGrow, dmFade } = params;
    const soft2 = softening * softening;
    const N = particles.length;
    const dmCell = SIZE / DM_RES;

    // Build spatial grid for efficient neighbor lookup
    const GCELL = 40;
    const GW = Math.ceil(SIZE / GCELL);
    const grid = new Map();
    for (let i = 0; i < N; i++) {
      const gx = Math.floor(particles[i][0] / GCELL);
      const gy = Math.floor(particles[i][1] / GCELL);
      const key = gy * GW + gx;
      if (!grid.has(key)) grid.set(key, []);
      grid.get(key).push(i);
    }

    // Compute forces
    for (let i = 0; i < N; i++) {
      const p = particles[i];
      let ax = 0, ay = 0;
      const gx = Math.floor(p[0] / GCELL);
      const gy = Math.floor(p[1] / GCELL);

      // Check neighboring grid cells
      for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          const key = (gy + dy) * GW + (gx + dx);
          const cell = grid.get(key);
          if (!cell) continue;
          for (const j of cell) {
            if (j === i) continue;
            const o = particles[j];
            const ddx = o[0] - p[0], ddy = o[1] - p[1];
            const d2 = ddx * ddx + ddy * ddy + soft2;
            const f = gravity * o[4] / (d2 * Math.sqrt(d2));
            ax += ddx * f;
            ay += ddy * f;
          }
        }
      }

      // Also attract toward DM density (DM feeds back into gravity)
      const dmx = Math.floor(p[0] / dmCell);
      const dmy = Math.floor(p[1] / dmCell);
      if (dmx >= 1 && dmx < DM_RES - 1 && dmy >= 1 && dmy < DM_RES - 1) {
        const dmGradX = (dmBuffer[dmy * DM_RES + dmx + 1] - dmBuffer[dmy * DM_RES + dmx - 1]) * 0.5;
        const dmGradY = (dmBuffer[(dmy + 1) * DM_RES + dmx] - dmBuffer[(dmy - 1) * DM_RES + dmx]) * 0.5;
        ax += dmGradX * gravity * 0.15;
        ay += dmGradY * gravity * 0.15;
      }

      p[2] = (p[2] + ax) * damping;
      p[3] = (p[3] + ay) * damping;
      p[0] += p[2];
      p[1] += p[3];

      // Wrap
      if (p[0] < 0) p[0] += SIZE;
      if (p[0] >= SIZE) p[0] -= SIZE;
      if (p[1] < 0) p[1] += SIZE;
      if (p[1] >= SIZE) p[1] -= SIZE;
    }

    // Update DM buffer: fade existing, deposit where particles are
    for (let i = 0; i < dmBuffer.length; i++) {
      dmBuffer[i] *= dmFade;
    }
    for (const p of particles) {
      const gx = Math.floor(p[0] / dmCell);
      const gy = Math.floor(p[1] / dmCell);
      // Deposit DM in a small radius around each particle
      for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          const nx = gx + dx, ny = gy + dy;
          if (nx < 0 || nx >= DM_RES || ny < 0 || ny >= DM_RES) continue;
          const r = Math.sqrt(dx * dx + dy * dy);
          const w = Math.exp(-r * r / 1.2) * p[4] * dmGrow * 0.01;
          dmBuffer[ny * DM_RES + nx] += w;
        }
      }
    }
  }, [params]);

  const render = useCallback(() => {
    const s = stateRef.current;
    if (!s) return;
    const { particles, dmBuffer, DM_RES } = s;
    const dmCell = SIZE / DM_RES;

    // Find DM max for normalization
    let maxDM = 0.001;
    for (let i = 0; i < dmBuffer.length; i++) {
      if (dmBuffer[i] > maxDM) maxDM = dmBuffer[i];
    }

    // --- MATTER PANEL ---
    const mc = matterRef.current;
    if (mc) {
      const ctx = mc.getContext("2d");
      ctx.fillStyle = "#04040c";
      ctx.fillRect(0, 0, SIZE, SIZE);
      for (const p of particles) {
        const sz = Math.max(1.5, Math.min(3.5, p[4] * 1.5));
        const spd = Math.sqrt(p[2] * p[2] + p[3] * p[3]);
        const heat = Math.min(1, spd / 2);
        const r = Math.floor(200 + heat * 55);
        const g = Math.floor(170 + heat * 60);
        const b = Math.floor(60 + (1 - heat) * 40);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.beginPath();
        ctx.arc(p[0], p[1], sz, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // --- DARK MATTER PANEL ---
    const dc = dmRef.current;
    if (dc) {
      const ctx = dc.getContext("2d");
      const img = ctx.createImageData(DM_RES, DM_RES);
      const d = img.data;
      for (let y = 0; y < DM_RES; y++) {
        for (let x = 0; x < DM_RES; x++) {
          const val = dmBuffer[y * DM_RES + x];
          const pi = (y * DM_RES + x) * 4;
          // Power curve for visibility
          const n = Math.pow(Math.min(1, val / maxDM), 0.35);
          if (n < 0.02) {
            d[pi] = 3; d[pi + 1] = 3; d[pi + 2] = 10; d[pi + 3] = 255;
          } else {
            // Violet-blue-pink gradient
            const n2 = n * n;
            d[pi] = Math.min(255, Math.floor(n2 * 220 + n * 35));     // R
            d[pi + 1] = Math.min(255, Math.floor(n2 * 60 + n * 15));  // G
            d[pi + 2] = Math.min(255, Math.floor(n * 255));            // B
            d[pi + 3] = 255;
          }
        }
      }
      // Draw scaled up
      const tmp = document.createElement("canvas");
      tmp.width = DM_RES; tmp.height = DM_RES;
      tmp.getContext("2d").putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(tmp, 0, 0, DM_RES, DM_RES, 0, 0, SIZE, SIZE);
    }

    // --- CURVATURE PANEL ---
    const cc = curveRef.current;
    if (cc) {
      const ctx = cc.getContext("2d");
      const img = ctx.createImageData(DM_RES, DM_RES);
      const d = img.data;
      // Curvature = matter density + DM density at each grid point
      // Build matter density on the DM grid
      const mDens = new Float32Array(DM_RES * DM_RES);
      for (const p of particles) {
        const gx = Math.floor(p[0] / dmCell);
        const gy = Math.floor(p[1] / dmCell);
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nx = gx + dx, ny = gy + dy;
            if (nx < 0 || nx >= DM_RES || ny < 0 || ny >= DM_RES) continue;
            const r = Math.sqrt(dx * dx + dy * dy);
            mDens[ny * DM_RES + nx] += p[4] * Math.exp(-r * r / 0.8);
          }
        }
      }
      let maxCurve = 0.001;
      const curveBuf = new Float32Array(DM_RES * DM_RES);
      for (let i = 0; i < curveBuf.length; i++) {
        curveBuf[i] = mDens[i] + dmBuffer[i] * 3;
        if (curveBuf[i] > maxCurve) maxCurve = curveBuf[i];
      }
      for (let y = 0; y < DM_RES; y++) {
        for (let x = 0; x < DM_RES; x++) {
          const pi = (y * DM_RES + x) * 4;
          const val = curveBuf[y * DM_RES + x];
          const n = Math.pow(Math.min(1, val / maxCurve), 0.4);
          if (n < 0.02) {
            d[pi] = 4; d[pi + 1] = 2; d[pi + 2] = 2; d[pi + 3] = 255;
          } else if (n < 0.3) {
            const t = n / 0.3;
            d[pi] = Math.floor(t * 180);
            d[pi + 1] = 0;
            d[pi + 2] = Math.floor(t * 20);
            d[pi + 3] = 255;
          } else if (n < 0.6) {
            const t = (n - 0.3) / 0.3;
            d[pi] = Math.floor(180 + t * 75);
            d[pi + 1] = Math.floor(t * 140);
            d[pi + 2] = Math.floor(20 - t * 20);
            d[pi + 3] = 255;
          } else {
            const t = (n - 0.6) / 0.4;
            d[pi] = 255;
            d[pi + 1] = Math.floor(140 + t * 115);
            d[pi + 2] = Math.floor(t * 220);
            d[pi + 3] = 255;
          }
        }
      }
      const tmp = document.createElement("canvas");
      tmp.width = DM_RES; tmp.height = DM_RES;
      tmp.getContext("2d").putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(tmp, 0, 0, DM_RES, DM_RES, 0, 0, SIZE, SIZE);
    }
  }, []);

  useEffect(() => {
    if (!running) { cancelAnimationFrame(animRef.current); return; }
    const loop = () => {
      simulate();
      simulate();
      render();
      setStep(p => p + 2);
      animRef.current = requestAnimationFrame(loop);
    };
    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, [running, simulate, render]);

  useEffect(() => { render(); }, [render]);

  const addMatter = (e, canvasEl) => {
    const rect = canvasEl.getBoundingClientRect();
    const scale = SIZE / rect.width;
    const cx = (e.clientX - rect.left) * scale;
    const cy = (e.clientY - rect.top) * scale;
    if (!stateRef.current) return;
    for (let i = 0; i < 30; i++) {
      const a = Math.random() * Math.PI * 2, r = Math.random() * 15;
      stateRef.current.particles.push([
        cx + Math.cos(a) * r, cy + Math.sin(a) * r,
        (Math.random() - 0.5) * 0.3, (Math.random() - 0.5) * 0.3,
        0.4 + Math.random() * 1.6,
      ]);
    }
    render();
  };

  const canvasStyle = {
    width: SIZE, height: SIZE, borderRadius: 6,
    border: "1px solid #1a1a2e", cursor: "crosshair",
  };

  return (
    <div style={{
      minHeight: "100vh", background: "#04040a", color: "#d1d5db",
      fontFamily: "'DM Sans', sans-serif", padding: 16,
    }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet" />
      <div style={{ maxWidth: 1800, margin: "0 auto" }}>
        <h1 style={{
          fontSize: 24, fontWeight: 700, marginBottom: 6,
          background: "linear-gradient(90deg, #fbbf24, #c084fc, #22d3ee)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
        }}>Primordial Fluid Model</h1>
        <p style={{
          fontSize: 12, color: "#6b7280", fontFamily: "'DM Mono', monospace",
          lineHeight: 1.6, marginBottom: 14, maxWidth: 700,
        }}>
          Three views of one universe: <span style={{ color: "#fbbf24" }}>matter</span> clumps under gravity →
          warps <span style={{ color: "#fb923c" }}>spacetime curvature</span> →
          <span style={{ color: "#c084fc" }}> dark matter medium</span> accumulates along the warps →
          feeds back into gravity
        </p>

        {/* Controls */}
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14, alignItems: "center" }}>
          {[["bigbang", "Big Bang"], ["web", "Cosmic Web"], ["bullet", "Bullet Cluster"], ["soup", "Random"]].map(([k, l]) => (
            <button key={k} onClick={() => { setPreset(k); setRunning(false); }} style={{
              padding: "6px 14px", fontSize: 12, borderRadius: 6, cursor: "pointer",
              border: preset === k ? "1px solid #fbbf24" : "1px solid #1c1c2e",
              background: preset === k ? "rgba(251,191,36,0.12)" : "rgba(8,8,16,0.95)",
              color: preset === k ? "#fbbf24" : "#9ca3af",
              fontFamily: "'DM Mono', monospace", fontWeight: preset === k ? 600 : 400,
            }}>{l}</button>
          ))}
          <div style={{ width: 16 }} />
          <button onClick={() => setRunning(!running)} style={{
            padding: "8px 22px", fontSize: 13, borderRadius: 6, border: "none", cursor: "pointer",
            fontWeight: 700, background: running ? "#dc2626" : "#f59e0b", color: "#fff",
            fontFamily: "'DM Sans', sans-serif",
            boxShadow: running ? "0 0 14px rgba(220,38,38,0.3)" : "0 0 14px rgba(245,158,11,0.3)",
          }}>{running ? "⏸ Pause" : "▶ Simulate"}</button>
          <button onClick={() => { simulate(); simulate(); render(); setStep(s => s + 2); }} style={{
            padding: "8px 14px", fontSize: 12, borderRadius: 6, cursor: "pointer",
            border: "1px solid #2a2a3e", background: "rgba(8,8,16,0.95)", color: "#9ca3af",
            fontFamily: "'DM Mono', monospace",
          }}>Step</button>
          <button onClick={() => { init(preset); setRunning(false); }} style={{
            padding: "8px 14px", fontSize: 12, borderRadius: 6, cursor: "pointer",
            border: "1px solid #2a2a3e", background: "rgba(8,8,16,0.95)", color: "#9ca3af",
            fontFamily: "'DM Mono', monospace",
          }}>Reset</button>
          <span style={{
            fontSize: 11, color: "#4b5563", fontFamily: "'DM Mono', monospace", marginLeft: 8,
          }}>t={step} · {stateRef.current?.particles.length || 0} particles</span>
        </div>

        {/* Sliders */}
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 16 }}>
          {[
            ["gravity", "Gravity", 0.005, 0.1, 0.002],
            ["softening", "Softening", 5, 30, 1],
            ["damping", "Damping", 0.98, 1.0, 0.001],
            ["dmGrow", "DM Growth", 0.1, 2.0, 0.05],
            ["dmFade", "DM Fade", 0.97, 1.0, 0.001],
          ].map(([k, l, mn, mx, st]) => (
            <div key={k} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ fontSize: 11, color: "#6b7280", fontFamily: "'DM Mono', monospace", width: 75 }}>{l}</span>
              <input type="range" min={mn} max={mx} step={st} value={params[k]}
                onChange={e => setParams(p => ({ ...p, [k]: parseFloat(e.target.value) }))}
                style={{ width: 80, accentColor: "#c084fc" }} />
              <span style={{ fontSize: 11, color: "#c084fc", fontFamily: "'DM Mono', monospace", width: 40, textAlign: "right" }}>
                {params[k] < 1 && params[k] > 0.1 ? params[k].toFixed(3) : params[k] < 0.1 ? params[k].toFixed(3) : params[k].toFixed(1)}
              </span>
            </div>
          ))}
        </div>

        {/* Three panels */}
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          <div style={{ textAlign: "center" }}>
            <div style={{
              fontSize: 12, fontWeight: 600, color: "#fbbf24",
              fontFamily: "'DM Mono', monospace", marginBottom: 6,
              padding: "4px 12px", background: "rgba(251,191,36,0.08)",
              borderRadius: 4, display: "inline-block",
            }}>MATTER / ENERGY</div>
            <br />
            <canvas ref={matterRef} width={SIZE} height={SIZE}
              onClick={e => addMatter(e, matterRef.current)}
              style={canvasStyle} />
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{
              fontSize: 12, fontWeight: 600, color: "#c084fc",
              fontFamily: "'DM Mono', monospace", marginBottom: 6,
              padding: "4px 12px", background: "rgba(192,132,252,0.08)",
              borderRadius: 4, display: "inline-block",
            }}>DARK MATTER MEDIUM</div>
            <br />
            <canvas ref={dmRef} width={SIZE} height={SIZE}
              onClick={e => addMatter(e, dmRef.current)}
              style={canvasStyle} />
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{
              fontSize: 12, fontWeight: 600, color: "#fb923c",
              fontFamily: "'DM Mono', monospace", marginBottom: 6,
              padding: "4px 12px", background: "rgba(251,146,60,0.08)",
              borderRadius: 4, display: "inline-block",
            }}>SPACETIME CURVATURE</div>
            <br />
            <canvas ref={curveRef} width={SIZE} height={SIZE}
              onClick={e => addMatter(e, curveRef.current)}
              style={canvasStyle} />
          </div>
        </div>

        <p style={{
          marginTop: 14, fontSize: 11, color: "#4b5563",
          fontFamily: "'DM Mono', monospace", lineHeight: 1.6, maxWidth: 600,
        }}>
          Click any panel to inject matter. Watch how the dark matter medium (center)
          forms halos and trails around matter concentrations — not as particles,
          but as the primordial fluid responding to spacetime geometry.
          The curvature panel (right) shows total gravitational warping from both matter + DM.
        </p>
      </div>
    </div>
  );
}
