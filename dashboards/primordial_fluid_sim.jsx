import { useState, useEffect, useRef, useCallback } from "react";

const W = 256;
const H = 256;
const SCALE = 2;

function createGrid(w, h, val = 0) {
  return new Float32Array(w * h).fill(val);
}

function idx(x, y) {
  return ((y + H) % H) * W + ((x + W) % W);
}

export default function PrimordialFluidSim() {
  const canvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const stateRef = useRef(null);
  const animRef = useRef(null);
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(0);
  const [view, setView] = useState("fluid");
  const [showMatter, setShowMatter] = useState(true);
  const [params, setParams] = useState({
    gravity: 0.15,
    fluidViscosity: 0.02,
    darkMatterCoupling: 0.08,
    matterFeedback: 0.05,
    diffusion: 0.12,
  });
  const [preset, setPreset] = useState("bigbang");

  const initState = useCallback((scenario) => {
    const fluid = createGrid(W, H, 1.0);
    const fluidVx = createGrid(W, H);
    const fluidVy = createGrid(W, H);
    const matter = createGrid(W, H);
    const spacetime = createGrid(W, H);
    const darkMatter = createGrid(W, H, 0.5);

    if (scenario === "bigbang") {
      const cx = W / 2, cy = H / 2;
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const dx = x - cx, dy = y - cy;
          const r = Math.sqrt(dx * dx + dy * dy);
          const i = idx(x, y);
          if (r < 12) {
            matter[i] = 3.0 * Math.exp(-r * r / 40);
            fluid[i] = 2.5 * Math.exp(-r * r / 50);
            darkMatter[i] = 2.0 * Math.exp(-r * r / 60);
          }
          const angle = Math.atan2(dy, dx);
          const speed = r < 30 ? 0.8 * Math.exp(-r * r / 200) : 0;
          fluidVx[i] = speed * Math.cos(angle) + (Math.random() - 0.5) * 0.05;
          fluidVy[i] = speed * Math.sin(angle) + (Math.random() - 0.5) * 0.05;
        }
      }
    } else if (scenario === "clusters") {
      const centers = [
        [W * 0.3, H * 0.3], [W * 0.7, H * 0.35],
        [W * 0.5, H * 0.7], [W * 0.2, H * 0.65], [W * 0.8, H * 0.7]
      ];
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const i = idx(x, y);
          darkMatter[i] = 0.3 + Math.random() * 0.1;
          for (const [cx, cy] of centers) {
            const dx = x - cx, dy = y - cy;
            const r = Math.sqrt(dx * dx + dy * dy);
            matter[i] += 1.5 * Math.exp(-r * r / 150);
            darkMatter[i] += 0.8 * Math.exp(-r * r / 300);
            fluid[i] += 0.5 * Math.exp(-r * r / 200);
          }
        }
      }
    } else if (scenario === "collision") {
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const i = idx(x, y);
          const r1 = Math.sqrt((x - W * 0.3) ** 2 + (y - H * 0.5) ** 2);
          const r2 = Math.sqrt((x - W * 0.7) ** 2 + (y - H * 0.5) ** 2);
          matter[i] = 2.0 * Math.exp(-r1 * r1 / 200) + 2.0 * Math.exp(-r2 * r2 / 200);
          darkMatter[i] = 0.4 + 1.2 * Math.exp(-r1 * r1 / 400) + 1.2 * Math.exp(-r2 * r2 / 400);
          fluid[i] = 1.0 + 0.5 * Math.exp(-r1 * r1 / 250) + 0.5 * Math.exp(-r2 * r2 / 250);
          if (r1 < 40) { fluidVx[i] = 0.4; }
          if (r2 < 40) { fluidVx[i] = -0.4; }
        }
      }
    } else {
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const i = idx(x, y);
          darkMatter[i] = 0.5 + (Math.random() - 0.5) * 0.3;
          fluid[i] = 1.0 + (Math.random() - 0.5) * 0.2;
          matter[i] = Math.random() < 0.001 ? 1.0 + Math.random() * 2 : 0;
        }
      }
    }

    stateRef.current = { fluid, fluidVx, fluidVy, matter, spacetime, darkMatter };
    setStep(0);
  }, []);

  useEffect(() => { initState(preset); }, [preset, initState]);

  const simulate = useCallback(() => {
    const s = stateRef.current;
    if (!s) return;
    const { fluid, fluidVx, fluidVy, matter, spacetime, darkMatter } = s;
    const { gravity, fluidViscosity, darkMatterCoupling, matterFeedback, diffusion } = params;

    const newFluid = new Float32Array(fluid);
    const newVx = new Float32Array(fluidVx);
    const newVy = new Float32Array(fluidVy);
    const newMatter = new Float32Array(matter);
    const newST = new Float32Array(spacetime);
    const newDM = new Float32Array(darkMatter);

    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const i = idx(x, y);
        const iL = idx(x - 1, y), iR = idx(x + 1, y);
        const iU = idx(x, y - 1), iD = idx(x, y + 1);

        // 1. Spacetime curvature from matter + dark matter
        const totalMass = matter[i] + darkMatter[i] * darkMatterCoupling;
        newST[i] = totalMass * gravity;

        // 2. Gravitational potential gradient -> fluid velocity
        const gradX = (matter[iR] + darkMatter[iR] * darkMatterCoupling)
                    - (matter[iL] + darkMatter[iL] * darkMatterCoupling);
        const gradY = (matter[iD] + darkMatter[iD] * darkMatterCoupling)
                    - (matter[iU] + darkMatter[iU] * darkMatterCoupling);

        newVx[i] = fluidVx[i] * (1 - fluidViscosity) - gradX * gravity * 0.5;
        newVy[i] = fluidVy[i] * (1 - fluidViscosity) - gradY * gravity * 0.5;

        // 3. Advect fluid density
        const srcX = x - fluidVx[i];
        const srcY = y - fluidVy[i];
        const sx = Math.floor(srcX), sy = Math.floor(srcY);
        const fx = srcX - sx, fy = srcY - sy;
        newFluid[i] = (1 - fx) * (1 - fy) * fluid[idx(sx, sy)]
                    + fx * (1 - fy) * fluid[idx(sx + 1, sy)]
                    + (1 - fx) * fy * fluid[idx(sx, sy + 1)]
                    + fx * fy * fluid[idx(sx + 1, sy + 1)];

        // 4. Dark matter responds to spacetime curvature (clumps around matter)
        const dmGradX = (spacetime[iR] - spacetime[iL]) * 0.5;
        const dmGradY = (spacetime[iD] - spacetime[iU]) * 0.5;
        const dmDiffuse = (darkMatter[iL] + darkMatter[iR] + darkMatter[iU] + darkMatter[iD]) * 0.25 - darkMatter[i];
        newDM[i] = darkMatter[i]
                 + dmGradX * darkMatterCoupling * 0.1
                 + dmGradY * darkMatterCoupling * 0.1
                 + dmDiffuse * diffusion
                 + newST[i] * darkMatterCoupling * 0.02;

        // 5. Matter feedback: dark matter density enhances gravitational attraction
        const mDiffuse = (matter[iL] + matter[iR] + matter[iU] + matter[iD]) * 0.25 - matter[i];
        const dmAttraction = (darkMatter[i] - 0.5) * matterFeedback;
        newMatter[i] = matter[i]
                     + mDiffuse * diffusion * 0.5
                     + dmAttraction * 0.01;

        // Clamp
        newFluid[i] = Math.max(0, newFluid[i]);
        newDM[i] = Math.max(0.01, newDM[i]);
        newMatter[i] = Math.max(0, newMatter[i]);
      }
    }

    s.fluid = newFluid;
    s.fluidVx = newVx;
    s.fluidVy = newVy;
    s.matter = newMatter;
    s.spacetime = newST;
    s.darkMatter = newDM;
  }, [params]);

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    const overlay = overlayCanvasRef.current;
    if (!canvas || !stateRef.current) return;
    const ctx = canvas.getContext("2d");
    const octx = overlay.getContext("2d");
    const imgData = ctx.createImageData(W, H);
    const d = imgData.data;
    const s = stateRef.current;

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = idx(x, y);
        const pi = (y * W + x) * 4;
        let r = 0, g = 0, b = 0;

        if (view === "fluid") {
          const fv = Math.min(1, s.fluid[i] / 2.5);
          const dm = Math.min(1, s.darkMatter[i] / 2.0);
          const st = Math.min(1, s.spacetime[i] / 1.5);
          // Deep space palette: dark matter = deep indigo, fluid = teal, spacetime warp = amber
          r = Math.floor(dm * 60 + st * 200 + fv * 20);
          g = Math.floor(dm * 30 + st * 120 + fv * 180);
          b = Math.floor(dm * 140 + st * 40 + fv * 200);
        } else if (view === "darkmatter") {
          const dm = Math.min(1, s.darkMatter[i] / 2.0);
          const dm2 = dm * dm;
          r = Math.floor(dm2 * 180 + dm * 40);
          g = Math.floor(dm2 * 60 + dm * 20);
          b = Math.floor(dm * 220 + dm2 * 35);
        } else if (view === "spacetime") {
          const st = Math.min(1, s.spacetime[i] / 1.5);
          const curve = st * st;
          r = Math.floor(curve * 255);
          g = Math.floor(st * 160 - curve * 80);
          b = Math.floor((1 - st) * 60);
        } else if (view === "velocity") {
          const vMag = Math.sqrt(s.fluidVx[i] ** 2 + s.fluidVy[i] ** 2);
          const v = Math.min(1, vMag * 5);
          const angle = (Math.atan2(s.fluidVy[i], s.fluidVx[i]) / Math.PI + 1) * 0.5;
          r = Math.floor(v * (120 + angle * 135));
          g = Math.floor(v * (60 + (1 - angle) * 120));
          b = Math.floor(v * 200 + (1 - v) * 15);
        }

        if (showMatter && s.matter[i] > 0.05) {
          const m = Math.min(1, s.matter[i] / 3.0);
          r = Math.floor(r * (1 - m * 0.7) + 255 * m * 0.9);
          g = Math.floor(g * (1 - m * 0.7) + 240 * m * 0.8);
          b = Math.floor(b * (1 - m * 0.5) + 200 * m * 0.3);
        }

        d[pi] = Math.min(255, Math.max(0, r));
        d[pi + 1] = Math.min(255, Math.max(0, g));
        d[pi + 2] = Math.min(255, Math.max(0, b));
        d[pi + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);
    octx.clearRect(0, 0, W * SCALE, H * SCALE);
    octx.drawImage(canvas, 0, 0, W, H, 0, 0, W * SCALE, H * SCALE);
    octx.imageSmoothingEnabled = false;
  }, [view, showMatter]);

  useEffect(() => {
    if (!running) {
      cancelAnimationFrame(animRef.current);
      return;
    }
    let frameCount = 0;
    const loop = () => {
      for (let i = 0; i < 3; i++) simulate();
      render();
      frameCount += 3;
      setStep(s => s + 3);
      animRef.current = requestAnimationFrame(loop);
    };
    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, [running, simulate, render]);

  useEffect(() => { render(); }, [view, showMatter, render]);

  const handleCanvasClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / SCALE);
    const y = Math.floor((e.clientY - rect.top) / SCALE);
    if (!stateRef.current) return;
    const s = stateRef.current;
    for (let dy = -6; dy <= 6; dy++) {
      for (let dx = -6; dx <= 6; dx++) {
        const r = Math.sqrt(dx * dx + dy * dy);
        if (r < 6) {
          const i = idx(x + dx, y + dy);
          const strength = Math.exp(-r * r / 8);
          s.matter[i] += 2.0 * strength;
          s.fluid[i] += 1.0 * strength;
        }
      }
    }
    render();
  };

  const paramSlider = (label, key, min, max, step2) => (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
      <span style={{ width: 140, fontSize: 11, color: "#9ca3af", fontFamily: "'DM Mono', monospace" }}>{label}</span>
      <input
        type="range" min={min} max={max} step={step2}
        value={params[key]}
        onChange={e => setParams(p => ({ ...p, [key]: parseFloat(e.target.value) }))}
        style={{ flex: 1, accentColor: "#f59e0b" }}
      />
      <span style={{ width: 40, fontSize: 11, color: "#f59e0b", fontFamily: "'DM Mono', monospace", textAlign: "right" }}>{params[key].toFixed(2)}</span>
    </div>
  );

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #0a0a12 0%, #0d0d1a 40%, #0a0f14 100%)",
      color: "#e5e7eb",
      fontFamily: "'DM Sans', 'Helvetica Neue', sans-serif",
      padding: 24,
    }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@700&display=swap" rel="stylesheet" />

      <div style={{ maxWidth: 900, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 28 }}>
          <h1 style={{
            fontSize: 28,
            fontWeight: 700,
            fontFamily: "'DM Sans', sans-serif",
            background: "linear-gradient(90deg, #f59e0b, #8b5cf6, #06b6d4)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            marginBottom: 6,
            letterSpacing: "-0.5px",
          }}>
            Primordial Fluid Cosmological Model
          </h1>
          <p style={{ fontSize: 13, color: "#6b7280", fontFamily: "'DM Mono', monospace", lineHeight: 1.6 }}>
            Three co-evolving substrates: <span style={{ color: "#fbbf24" }}>matter/energy</span> + <span style={{ color: "#06b6d4" }}>spacetime fabric</span> + <span style={{ color: "#8b5cf6" }}>dark matter medium</span><br/>
            Hypothesis: dark matter is the primordial fluid substrate of spacetime itself — clumping where geometry warps
          </p>
        </div>

        <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
          {/* Canvas */}
          <div style={{ position: "relative" }}>
            <canvas ref={canvasRef} width={W} height={H} style={{ display: "none" }} />
            <canvas
              ref={overlayCanvasRef}
              width={W * SCALE}
              height={H * SCALE}
              onClick={handleCanvasClick}
              style={{
                borderRadius: 8,
                border: "1px solid #1e1e30",
                cursor: "crosshair",
                boxShadow: "0 0 60px rgba(139, 92, 246, 0.1), 0 0 120px rgba(6, 182, 212, 0.05)",
                imageRendering: "pixelated",
              }}
            />
            <div style={{
              position: "absolute", bottom: 8, left: 8,
              background: "rgba(0,0,0,0.7)", borderRadius: 6, padding: "4px 10px",
              fontSize: 11, color: "#6b7280", fontFamily: "'DM Mono', monospace",
            }}>
              t={step} · click to add matter
            </div>
          </div>

          {/* Controls */}
          <div style={{ flex: 1, minWidth: 260 }}>
            {/* Scenario */}
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 11, color: "#6b7280", fontFamily: "'DM Mono', monospace", marginBottom: 8, textTransform: "uppercase", letterSpacing: 1 }}>Initial Conditions</div>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {[
                  ["bigbang", "Big Bang"],
                  ["clusters", "Galaxy Clusters"],
                  ["collision", "Bullet Cluster"],
                  ["random", "Random Soup"],
                ].map(([key, label]) => (
                  <button
                    key={key}
                    onClick={() => { setPreset(key); setRunning(false); }}
                    style={{
                      padding: "6px 12px", fontSize: 11, borderRadius: 6,
                      border: preset === key ? "1px solid #f59e0b" : "1px solid #2a2a3e",
                      background: preset === key ? "rgba(245,158,11,0.15)" : "rgba(20,20,35,0.8)",
                      color: preset === key ? "#fbbf24" : "#9ca3af",
                      cursor: "pointer", fontFamily: "'DM Mono', monospace",
                    }}
                  >{label}</button>
                ))}
              </div>
            </div>

            {/* View mode */}
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 11, color: "#6b7280", fontFamily: "'DM Mono', monospace", marginBottom: 8, textTransform: "uppercase", letterSpacing: 1 }}>Visualization</div>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {[
                  ["fluid", "Composite", "#06b6d4"],
                  ["darkmatter", "Dark Matter", "#8b5cf6"],
                  ["spacetime", "Spacetime Curvature", "#f59e0b"],
                  ["velocity", "Flow Field", "#10b981"],
                ].map(([key, label, color]) => (
                  <button
                    key={key}
                    onClick={() => setView(key)}
                    style={{
                      padding: "6px 12px", fontSize: 11, borderRadius: 6,
                      border: view === key ? `1px solid ${color}` : "1px solid #2a2a3e",
                      background: view === key ? `${color}22` : "rgba(20,20,35,0.8)",
                      color: view === key ? color : "#9ca3af",
                      cursor: "pointer", fontFamily: "'DM Mono', monospace",
                    }}
                  >{label}</button>
                ))}
              </div>
              <label style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 10, cursor: "pointer" }}>
                <input type="checkbox" checked={showMatter} onChange={e => setShowMatter(e.target.checked)} style={{ accentColor: "#fbbf24" }} />
                <span style={{ fontSize: 11, color: "#9ca3af", fontFamily: "'DM Mono', monospace" }}>Show matter overlay</span>
              </label>
            </div>

            {/* Physics params */}
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 11, color: "#6b7280", fontFamily: "'DM Mono', monospace", marginBottom: 8, textTransform: "uppercase", letterSpacing: 1 }}>Physics Parameters</div>
              {paramSlider("Gravity", "gravity", 0, 0.5, 0.01)}
              {paramSlider("Viscosity", "fluidViscosity", 0, 0.2, 0.005)}
              {paramSlider("DM Coupling", "darkMatterCoupling", 0, 0.3, 0.005)}
              {paramSlider("Matter ↔ DM", "matterFeedback", 0, 0.2, 0.005)}
              {paramSlider("Diffusion", "diffusion", 0, 0.3, 0.005)}
            </div>

            {/* Play controls */}
            <div style={{ display: "flex", gap: 8 }}>
              <button
                onClick={() => setRunning(!running)}
                style={{
                  padding: "10px 24px", fontSize: 13, borderRadius: 8,
                  border: "none",
                  background: running ? "linear-gradient(135deg, #dc2626, #b91c1c)" : "linear-gradient(135deg, #f59e0b, #d97706)",
                  color: "#fff", cursor: "pointer", fontWeight: 600,
                  fontFamily: "'DM Sans', sans-serif",
                  boxShadow: running ? "0 0 20px rgba(220,38,38,0.3)" : "0 0 20px rgba(245,158,11,0.3)",
                }}
              >{running ? "⏸ Pause" : "▶ Simulate"}</button>
              <button
                onClick={() => { initState(preset); setRunning(false); }}
                style={{
                  padding: "10px 18px", fontSize: 13, borderRadius: 8,
                  border: "1px solid #2a2a3e",
                  background: "rgba(20,20,35,0.8)",
                  color: "#9ca3af", cursor: "pointer",
                  fontFamily: "'DM Sans', sans-serif",
                }}
              >↺ Reset</button>
            </div>

            {/* Legend */}
            <div style={{
              marginTop: 20, padding: 14, borderRadius: 8,
              background: "rgba(15,15,28,0.8)", border: "1px solid #1e1e30",
            }}>
              <div style={{ fontSize: 11, color: "#6b7280", fontFamily: "'DM Mono', monospace", marginBottom: 8, textTransform: "uppercase", letterSpacing: 1 }}>Model Dynamics</div>
              <div style={{ fontSize: 12, color: "#9ca3af", lineHeight: 1.8, fontFamily: "'DM Mono', monospace" }}>
                <div><span style={{ color: "#fbbf24" }}>●</span> Matter warps spacetime curvature</div>
                <div><span style={{ color: "#8b5cf6" }}>●</span> Dark medium clumps along curvature</div>
                <div><span style={{ color: "#06b6d4" }}>●</span> Fluid carries matter via advection</div>
                <div><span style={{ color: "#10b981" }}>●</span> DM density feeds back into gravity</div>
                <div style={{ marginTop: 6, color: "#6b7280", fontSize: 11 }}>
                  The three-way cycle: matter → curvature → DM clumping → enhanced gravity → matter attraction
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
