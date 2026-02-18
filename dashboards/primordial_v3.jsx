import { useState, useEffect, useRef, useCallback } from "react";

/*
  Hybrid approach:
  - Matter = discrete particles with gravitational N-body dynamics (softened)
  - Dark Matter = continuous density field that responds to matter's gravity wells
  - Spacetime Curvature = computed from both matter + DM density
  
  This way matter actually CLUMPS (particles fall toward each other)
  and dark matter forms halos/filaments around the matter concentrations.
*/

const CW = 560; // canvas display
const CH = 560;
const GRID = 80; // field resolution
const CELL = CW / GRID;

function createField(val = 0) {
  return new Float32Array(GRID * GRID).fill(val);
}

function gi(gx, gy) {
  return ((gy + GRID) % GRID) * GRID + ((gx + GRID) % GRID);
}

function initParticles(scenario) {
  const particles = [];
  const W = CW, H = CH;

  if (scenario === "bigbang") {
    // Dense central cluster that explodes outward
    for (let i = 0; i < 600; i++) {
      const angle = Math.random() * Math.PI * 2;
      const r = Math.random() * 30 + 2;
      const speed = 0.6 + Math.random() * 1.8;
      particles.push({
        x: W/2 + Math.cos(angle) * r,
        y: H/2 + Math.sin(angle) * r,
        vx: Math.cos(angle) * speed + (Math.random() - 0.5) * 0.3,
        vy: Math.sin(angle) * speed + (Math.random() - 0.5) * 0.3,
        mass: 0.5 + Math.random() * 1.5,
      });
    }
  } else if (scenario === "web") {
    // Many small clusters scattered - should form cosmic web
    const seeds = [];
    for (let i = 0; i < 20; i++) {
      seeds.push([80 + Math.random() * (W - 160), 80 + Math.random() * (H - 160)]);
    }
    for (const [sx, sy] of seeds) {
      const count = 15 + Math.floor(Math.random() * 25);
      for (let i = 0; i < count; i++) {
        const angle = Math.random() * Math.PI * 2;
        const r = Math.random() * 25;
        particles.push({
          x: sx + Math.cos(angle) * r,
          y: sy + Math.sin(angle) * r,
          vx: (Math.random() - 0.5) * 0.4,
          vy: (Math.random() - 0.5) * 0.4,
          mass: 0.5 + Math.random() * 1.0,
        });
      }
    }
  } else if (scenario === "bullet") {
    // Two clusters heading toward each other
    for (let i = 0; i < 300; i++) {
      const angle = Math.random() * Math.PI * 2;
      const r = Math.random() * 40;
      particles.push({
        x: W * 0.25 + Math.cos(angle) * r,
        y: H * 0.5 + Math.sin(angle) * r,
        vx: 0.8 + (Math.random() - 0.5) * 0.2,
        vy: (Math.random() - 0.5) * 0.3,
        mass: 0.5 + Math.random() * 1.5,
      });
    }
    for (let i = 0; i < 300; i++) {
      const angle = Math.random() * Math.PI * 2;
      const r = Math.random() * 40;
      particles.push({
        x: W * 0.75 + Math.cos(angle) * r,
        y: H * 0.5 + Math.sin(angle) * r,
        vx: -0.8 + (Math.random() - 0.5) * 0.2,
        vy: (Math.random() - 0.5) * 0.3,
        mass: 0.5 + Math.random() * 1.5,
      });
    }
  } else {
    // Random uniform
    for (let i = 0; i < 500; i++) {
      particles.push({
        x: 40 + Math.random() * (W - 80),
        y: 40 + Math.random() * (H - 80),
        vx: (Math.random() - 0.5) * 0.6,
        vy: (Math.random() - 0.5) * 0.6,
        mass: 0.3 + Math.random() * 1.2,
      });
    }
  }

  return particles;
}

export default function PrimordialSim() {
  const canvasRef = useRef(null);
  const stateRef = useRef(null);
  const animRef = useRef(null);
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(0);
  const [view, setView] = useState("composite");
  const [preset, setPreset] = useState("bigbang");
  const [showParticles, setShowParticles] = useState(true);
  const [params, setParams] = useState({
    gravity: 0.035,
    dmResponse: 0.12,
    dmDecay: 0.04,
    softening: 15,
    damping: 0.998,
  });

  const init = useCallback((scenario) => {
    const particles = initParticles(scenario);
    const darkMatter = createField(0.15);
    const curvature = createField(0);
    const matterField = createField(0);
    stateRef.current = { particles, darkMatter, curvature, matterField };
    setStep(0);
  }, []);

  useEffect(() => { init(preset); }, [preset, init]);

  const simulate = useCallback(() => {
    const s = stateRef.current;
    if (!s) return;
    const { particles, darkMatter, curvature, matterField } = s;
    const { gravity, dmResponse, dmDecay, softening, damping } = params;
    const soft2 = softening * softening;

    // Clear matter field
    matterField.fill(0);

    // Deposit matter onto grid
    for (const p of particles) {
      const gx = Math.floor(p.x / CELL);
      const gy = Math.floor(p.y / CELL);
      if (gx >= 0 && gx < GRID && gy >= 0 && gy < GRID) {
        matterField[gi(gx, gy)] += p.mass;
        // Spread to neighbors for smoother field
        if (gx > 0) matterField[gi(gx-1, gy)] += p.mass * 0.3;
        if (gx < GRID-1) matterField[gi(gx+1, gy)] += p.mass * 0.3;
        if (gy > 0) matterField[gi(gx, gy-1)] += p.mass * 0.3;
        if (gy < GRID-1) matterField[gi(gx, gy+1)] += p.mass * 0.3;
      }
    }

    // Update curvature from matter + dark matter
    for (let gy = 0; gy < GRID; gy++) {
      for (let gx = 0; gx < GRID; gx++) {
        const i = gi(gx, gy);
        curvature[i] = (matterField[i] + darkMatter[i] * 2.0) * gravity;
      }
    }

    // Dark matter responds: grows where curvature is high, slowly decays elsewhere
    const newDM = new Float32Array(darkMatter);
    for (let gy = 0; gy < GRID; gy++) {
      for (let gx = 0; gx < GRID; gx++) {
        const i = gi(gx, gy);
        const curveHere = curvature[i];
        // DM accumulates where curvature exists (matter warps space, DM follows)
        const growth = curveHere * dmResponse;
        // Slight diffusion for smoothness
        const l = gi(gx-1, gy), r = gi(gx+1, gy);
        const u = gi(gx, gy-1), d = gi(gx, gy+1);
        const lap = (darkMatter[l] + darkMatter[r] + darkMatter[u] + darkMatter[d]) * 0.25 - darkMatter[i];
        newDM[i] = darkMatter[i] + growth + lap * 0.05 - darkMatter[i] * dmDecay * 0.1;
        newDM[i] = Math.max(0, Math.min(8, newDM[i]));
      }
    }
    s.darkMatter = newDM;

    // N-body gravity for particles (use grid-based force for speed)
    // Compute gravitational potential gradient on grid
    const forceX = createField(0);
    const forceY = createField(0);
    for (let gy = 1; gy < GRID-1; gy++) {
      for (let gx = 1; gx < GRID-1; gx++) {
        const i = gi(gx, gy);
        // Force from matter + dark matter gradient
        const totalR = matterField[gi(gx+1, gy)] + newDM[gi(gx+1, gy)] * 0.5;
        const totalL = matterField[gi(gx-1, gy)] + newDM[gi(gx-1, gy)] * 0.5;
        const totalD = matterField[gi(gx, gy+1)] + newDM[gi(gx, gy+1)] * 0.5;
        const totalU = matterField[gi(gx, gy-1)] + newDM[gi(gx, gy-1)] * 0.5;
        forceX[i] = (totalR - totalL) * 0.5 * gravity;
        forceY[i] = (totalD - totalU) * 0.5 * gravity;
      }
    }

    // Move particles
    for (const p of particles) {
      const gx = Math.floor(p.x / CELL);
      const gy = Math.floor(p.y / CELL);
      if (gx >= 1 && gx < GRID-1 && gy >= 1 && gy < GRID-1) {
        const i = gi(gx, gy);
        p.vx += forceX[i] * 0.8;
        p.vy += forceY[i] * 0.8;
      }

      // Also direct particle-to-nearby-particle gravity for tight interactions
      // (sample a few neighbors for performance)
      const nearCount = Math.min(particles.length, 20);
      for (let k = 0; k < nearCount; k++) {
        const other = particles[Math.floor(Math.random() * particles.length)];
        if (other === p) continue;
        const dx = other.x - p.x;
        const dy = other.y - p.y;
        const dist2 = dx * dx + dy * dy + soft2;
        const f = gravity * other.mass / (dist2 * Math.sqrt(dist2)) * 8;
        p.vx += dx * f;
        p.vy += dy * f;
      }

      p.vx *= damping;
      p.vy *= damping;
      p.x += p.vx;
      p.y += p.vy;

      // Wrap
      if (p.x < 0) p.x += CW;
      if (p.x >= CW) p.x -= CW;
      if (p.y < 0) p.y += CH;
      if (p.y >= CH) p.y -= CH;
    }
  }, [params]);

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    const s = stateRef.current;
    if (!canvas || !s) return;
    const ctx = canvas.getContext("2d");

    if (view === "sidebyside") {
      canvas.width = CW * 3 + 4;
      canvas.height = CH;
      ctx.fillStyle = "#06060c";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const panels = [
        { offset: 0, field: "matterField", label: "MATTER / ENERGY", color: "#fbbf24", drawP: true,
          colorFn: (v) => {
            const n = Math.min(1, v / 6);
            return `rgb(${Math.floor(n*255)},${Math.floor(n*190)},${Math.floor(n*50)})`;
          }},
        { offset: CW + 2, field: "darkMatter", label: "DARK MATTER MEDIUM", color: "#a78bfa", drawP: false,
          colorFn: (v) => {
            const n = Math.min(1, v / 3);
            const n2 = n * n;
            return `rgb(${Math.floor(n2*140+n*30)},${Math.floor(n2*40+n*15)},${Math.floor(n*220+20)})`;
          }},
        { offset: (CW + 2) * 2, field: "curvature", label: "SPACETIME CURVATURE", color: "#f97316", drawP: false,
          colorFn: (v) => {
            const n = Math.min(1, v / 0.5);
            if (n < 0.33) return `rgb(${Math.floor(n*3*180)},0,0)`;
            if (n < 0.66) return `rgb(${Math.floor(180+(n-0.33)*3*75)},${Math.floor((n-0.33)*3*130)},0)`;
            return `rgb(255,${Math.floor(130+(n-0.66)*3*125)},${Math.floor((n-0.66)*3*200)})`;
          }},
      ];

      for (const panel of panels) {
        // Draw field
        for (let gy = 0; gy < GRID; gy++) {
          for (let gx = 0; gx < GRID; gx++) {
            const v = s[panel.field][gi(gx, gy)];
            if (v > 0.01) {
              ctx.fillStyle = panel.colorFn(v);
              ctx.fillRect(panel.offset + gx * CELL, gy * CELL, CELL, CELL);
            }
          }
        }
        // Draw particles on matter panel
        if (panel.drawP) {
          ctx.fillStyle = "rgba(255,240,180,0.9)";
          for (const p of s.particles) {
            const sz = Math.max(1.5, Math.min(3.5, p.mass * 1.5));
            ctx.beginPath();
            ctx.arc(panel.offset + p.x, p.y, sz, 0, Math.PI * 2);
            ctx.fill();
          }
        }
        // Label
        ctx.fillStyle = "rgba(0,0,0,0.65)";
        ctx.fillRect(panel.offset + CW/2 - 80, 8, 160, 22);
        ctx.fillStyle = panel.color;
        ctx.font = "bold 11px 'DM Mono', monospace";
        ctx.textAlign = "center";
        ctx.fillText(panel.label, panel.offset + CW/2, 23);
      }

      // Dividers
      ctx.fillStyle = "rgba(255,255,255,0.08)";
      ctx.fillRect(CW, 0, 2, CH);
      ctx.fillRect(CW * 2 + 2, 0, 2, CH);
      return;
    }

    // Single view
    canvas.width = CW;
    canvas.height = CH;
    ctx.fillStyle = "#06060c";
    ctx.fillRect(0, 0, CW, CH);

    // Draw fields
    for (let gy = 0; gy < GRID; gy++) {
      for (let gx = 0; gx < GRID; gx++) {
        const i = gi(gx, gy);
        const m = s.matterField[i];
        const dm = s.darkMatter[i];
        const cv = s.curvature[i];
        const px = gx * CELL, py = gy * CELL;

        if (view === "composite") {
          const dmN = Math.min(1, dm / 3);
          const cvN = Math.min(1, cv / 0.5);
          const r = Math.floor(dmN * 50 + cvN * 120);
          const g = Math.floor(dmN * 20 + cvN * 60);
          const b = Math.floor(dmN * 180 + cvN * 20);
          if (r > 2 || g > 2 || b > 2) {
            ctx.fillStyle = `rgb(${r},${g},${b})`;
            ctx.fillRect(px, py, CELL, CELL);
          }
        } else if (view === "darkmatter") {
          const n = Math.min(1, dm / 3);
          if (n > 0.02) {
            const n2 = n * n;
            ctx.fillStyle = `rgb(${Math.floor(n2*140+n*30)},${Math.floor(n2*40+n*15)},${Math.floor(n*220+20)})`;
            ctx.fillRect(px, py, CELL, CELL);
          }
        } else if (view === "curvature") {
          const n = Math.min(1, cv / 0.5);
          if (n > 0.02) {
            let r2, g2, b2;
            if (n < 0.33) { r2 = n*3*180; g2 = 0; b2 = 0; }
            else if (n < 0.66) { r2 = 180+(n-0.33)*3*75; g2 = (n-0.33)*3*130; b2 = 0; }
            else { r2 = 255; g2 = 130+(n-0.66)*3*125; b2 = (n-0.66)*3*200; }
            ctx.fillStyle = `rgb(${Math.floor(r2)},${Math.floor(g2)},${Math.floor(b2)})`;
            ctx.fillRect(px, py, CELL, CELL);
          }
        }
      }
    }

    // Draw particles
    if (showParticles) {
      for (const p of s.particles) {
        const speed = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
        const heat = Math.min(1, speed / 3);
        const sz = Math.max(1.5, Math.min(4, p.mass * 1.8));
        const r = Math.floor(200 + heat * 55);
        const g2 = Math.floor(180 + heat * 40 - (1-heat) * 60);
        const b = Math.floor(100 - heat * 80 + (1-heat) * 60);
        ctx.fillStyle = `rgba(${r},${g2},${b},0.85)`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, sz, 0, Math.PI * 2);
        ctx.fill();
        // glow
        if (p.mass > 1.0) {
          ctx.fillStyle = `rgba(${r},${g2},${b},0.15)`;
          ctx.beginPath();
          ctx.arc(p.x, p.y, sz * 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }, [view, showParticles]);

  useEffect(() => {
    if (!running) { cancelAnimationFrame(animRef.current); return; }
    const loop = () => {
      for (let i = 0; i < 2; i++) simulate();
      render();
      setStep(prev => prev + 2);
      animRef.current = requestAnimationFrame(loop);
    };
    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, [running, simulate, render]);

  useEffect(() => { render(); }, [view, showParticles, render]);

  const handleClick = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;
    let cx = (e.clientX - rect.left) * scaleX;
    let cy = (e.clientY - rect.top) * scaleY;
    if (view === "sidebyside") cx = cx % (CW + 2);
    if (!stateRef.current) return;
    // Add cluster of particles
    for (let i = 0; i < 20; i++) {
      const angle = Math.random() * Math.PI * 2;
      const r = Math.random() * 15;
      stateRef.current.particles.push({
        x: cx + Math.cos(angle) * r,
        y: cy + Math.sin(angle) * r,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        mass: 0.5 + Math.random() * 1.5,
      });
    }
    render();
  };

  const viewBtns = [
    ["composite", "All Layers", "#06b6d4"],
    ["sidebyside", "Side by Side", "#e879f9"],
    ["darkmatter", "Dark Matter", "#a78bfa"],
    ["curvature", "Curvature", "#f97316"],
  ];

  const presetBtns = [
    ["bigbang", "Big Bang"],
    ["web", "Cosmic Web"],
    ["bullet", "Bullet Cluster"],
    ["soup", "Random Soup"],
  ];

  const sliders = [
    ["gravity", "Gravity", 0.005, 0.1, 0.001],
    ["dmResponse", "DM Response", 0.02, 0.3, 0.005],
    ["dmDecay", "DM Decay", 0, 0.15, 0.005],
    ["softening", "Softening", 5, 40, 1],
    ["damping", "Damping", 0.98, 1.0, 0.001],
  ];

  return (
    <div style={{
      minHeight: "100vh",
      background: "#08080f",
      color: "#d1d5db",
      fontFamily: "'DM Sans', sans-serif",
      padding: "16px",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet" />
      <div style={{ maxWidth: 1750, margin: "0 auto" }}>
        <div style={{ marginBottom: 16 }}>
          <h1 style={{
            fontSize: 22, fontWeight: 700, marginBottom: 4,
            background: "linear-gradient(90deg, #fbbf24, #a78bfa, #06b6d4)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          }}>Primordial Fluid Model</h1>
          <p style={{ fontSize: 11, color: "#6b7280", fontFamily: "'DM Mono', monospace", lineHeight: 1.5 }}>
            Particles = matter/energy · Violet field = dark matter medium · Thermal = spacetime curvature<br/>
            Matter clumps gravitationally → curvature forms → dark medium accumulates along warps → amplifies gravity
          </p>
        </div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 10 }}>
          {presetBtns.map(([k, l]) => (
            <button key={k} onClick={() => { setPreset(k); setRunning(false); }}
              style={{
                padding: "6px 14px", fontSize: 11, borderRadius: 5, cursor: "pointer",
                border: preset === k ? "1px solid #fbbf24" : "1px solid #1c1c2e",
                background: preset === k ? "rgba(251,191,36,0.1)" : "rgba(12,12,20,0.9)",
                color: preset === k ? "#fbbf24" : "#9ca3af",
                fontFamily: "'DM Mono', monospace",
              }}>{l}</button>
          ))}
          <div style={{ width: 1 }}/>
          {viewBtns.map(([k, l, c]) => (
            <button key={k} onClick={() => setView(k)}
              style={{
                padding: "6px 12px", fontSize: 11, borderRadius: 5, cursor: "pointer",
                border: view === k ? `1px solid ${c}` : "1px solid #1c1c2e",
                background: view === k ? `${c}18` : "rgba(12,12,20,0.9)",
                color: view === k ? c : "#6b7280",
                fontFamily: "'DM Mono', monospace",
              }}>{l}</button>
          ))}
        </div>

        <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
          <div style={{ position: "relative" }}>
            <canvas ref={canvasRef}
              onClick={handleClick}
              style={{
                borderRadius: 6, border: "1px solid #1a1a2e", cursor: "crosshair",
                maxWidth: "100%", height: "auto",
                boxShadow: "0 0 30px rgba(100,60,200,0.06)",
              }}
            />
            <div style={{
              position: "absolute", bottom: 6, left: 8,
              background: "rgba(0,0,0,0.7)", borderRadius: 4, padding: "3px 8px",
              fontSize: 10, color: "#6b7280", fontFamily: "'DM Mono', monospace",
            }}>t={step} · {stateRef.current?.particles.length || 0} particles · click to add matter</div>
          </div>

          <div style={{ minWidth: 220, maxWidth: 280 }}>
            <div style={{ display: "flex", gap: 6, marginBottom: 12, flexWrap: "wrap" }}>
              <button onClick={() => setRunning(!running)}
                style={{
                  padding: "8px 20px", fontSize: 12, borderRadius: 6, border: "none", cursor: "pointer",
                  fontWeight: 600, background: running ? "#dc2626" : "#f59e0b", color: "#fff",
                }}>{running ? "⏸ Pause" : "▶ Run"}</button>
              <button onClick={() => { simulate(); render(); setStep(s2 => s2 + 1); }}
                style={{
                  padding: "8px 12px", fontSize: 11, borderRadius: 6, cursor: "pointer",
                  border: "1px solid #2a2a3e", background: "rgba(12,12,20,0.9)", color: "#9ca3af",
                  fontFamily: "'DM Mono', monospace",
                }}>Step</button>
              <button onClick={() => { init(preset); setRunning(false); }}
                style={{
                  padding: "8px 12px", fontSize: 11, borderRadius: 6, cursor: "pointer",
                  border: "1px solid #2a2a3e", background: "rgba(12,12,20,0.9)", color: "#9ca3af",
                  fontFamily: "'DM Mono', monospace",
                }}>Reset</button>
            </div>

            <label style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12, cursor: "pointer" }}>
              <input type="checkbox" checked={showParticles} onChange={e => setShowParticles(e.target.checked)} style={{ accentColor: "#fbbf24" }}/>
              <span style={{ fontSize: 11, color: "#9ca3af", fontFamily: "'DM Mono', monospace" }}>Show matter particles</span>
            </label>

            <div style={{ fontSize: 10, color: "#6b7280", fontFamily: "'DM Mono', monospace", marginBottom: 6, textTransform: "uppercase", letterSpacing: 1 }}>Physics</div>
            {sliders.map(([k, l, mn, mx, st]) => (
              <div key={k} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                <span style={{ width: 80, fontSize: 10, color: "#6b7280", fontFamily: "'DM Mono', monospace" }}>{l}</span>
                <input type="range" min={mn} max={mx} step={st} value={params[k]}
                  onChange={e => setParams(p => ({...p, [k]: parseFloat(e.target.value)}))}
                  style={{ flex: 1, accentColor: "#a78bfa" }}/>
                <span style={{ width: 36, fontSize: 10, color: "#a78bfa", fontFamily: "'DM Mono', monospace", textAlign: "right" }}>
                  {params[k] < 1 ? params[k].toFixed(3) : params[k].toFixed(1)}
                </span>
              </div>
            ))}

            <div style={{
              marginTop: 14, padding: 10, borderRadius: 6,
              background: "rgba(10,10,18,0.9)", border: "1px solid #1a1a2e",
              fontSize: 11, fontFamily: "'DM Mono', monospace", lineHeight: 1.8,
            }}>
              <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>Your Model</div>
              <div><span style={{color:"#fbbf24"}}>●</span> Matter particles clump via gravity</div>
              <div><span style={{color:"#f97316"}}>●</span> Curvature forms at mass concentrations</div>
              <div><span style={{color:"#a78bfa"}}>●</span> Dark medium accumulates along warps</div>
              <div><span style={{color:"#06b6d4"}}>●</span> DM density amplifies gravitational pull</div>
              <div style={{ marginTop: 6, color: "#4b5563", fontSize: 10 }}>
                The dark matter isn't particles — it's the<br/>
                primordial fluid responding to geometry
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
