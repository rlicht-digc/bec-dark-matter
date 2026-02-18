import { useState, useEffect, useRef, useCallback } from "react";

const W = 200;
const H = 200;
const SCALE = 2.8;

function idx(x, y) {
  return ((y + H) % H) * W + ((x + W) % W);
}

export default function PrimordialFluidSim() {
  const canvasRef = useRef(null);
  const dispRef = useRef(null);
  const stateRef = useRef(null);
  const animRef = useRef(null);
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(0);
  const [view, setView] = useState("composite");
  const [preset, setPreset] = useState("bigbang");
  const [speed, setSpeed] = useState(2);
  const [params, setParams] = useState({
    gravity: 0.12,
    dmCoupling: 0.06,
    feedback: 0.04,
    diffusion: 0.1,
    viscosity: 0.015,
  });

  const initState = useCallback((scenario) => {
    const matter = new Float32Array(W * H);
    const darkMatter = new Float32Array(W * H).fill(0.5);
    const curvature = new Float32Array(W * H);
    const vx = new Float32Array(W * H);
    const vy = new Float32Array(W * H);
    const fluidDensity = new Float32Array(W * H).fill(1.0);

    if (scenario === "bigbang") {
      const cx = W / 2, cy = H / 2;
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const dx = x - cx, dy = y - cy;
          const r = Math.sqrt(dx * dx + dy * dy);
          const i = idx(x, y);
          matter[i] = 4.0 * Math.exp(-r * r / 30);
          darkMatter[i] = 0.5 + 2.5 * Math.exp(-r * r / 50);
          fluidDensity[i] = 1.0 + 2.0 * Math.exp(-r * r / 40);
          const angle = Math.atan2(dy, dx);
          const spd = r < 50 ? 1.2 * Math.exp(-r * r / 150) : 0;
          vx[i] = spd * Math.cos(angle) + (Math.random() - 0.5) * 0.15;
          vy[i] = spd * Math.sin(angle) + (Math.random() - 0.5) * 0.15;
        }
      }
    } else if (scenario === "bullet") {
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const i = idx(x, y);
          const r1 = Math.sqrt((x - W * 0.28) ** 2 + (y - H * 0.5) ** 2);
          const r2 = Math.sqrt((x - W * 0.72) ** 2 + (y - H * 0.5) ** 2);
          matter[i] = 2.5 * Math.exp(-r1 * r1 / 120) + 2.5 * Math.exp(-r2 * r2 / 120);
          darkMatter[i] = 0.4 + 1.5 * Math.exp(-r1 * r1 / 250) + 1.5 * Math.exp(-r2 * r2 / 250);
          fluidDensity[i] = 1.0 + Math.exp(-r1 * r1 / 150) + Math.exp(-r2 * r2 / 150);
          if (r1 < 35) vx[i] = 0.6;
          if (r2 < 35) vx[i] = -0.6;
        }
      }
    } else if (scenario === "web") {
      // seed many small clusters to watch cosmic web form
      const seeds = [];
      for (let i = 0; i < 30; i++) {
        seeds.push([Math.random() * W, Math.random() * H, 0.5 + Math.random() * 2]);
      }
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const i2 = idx(x, y);
          darkMatter[i2] = 0.35 + Math.random() * 0.15;
          fluidDensity[i2] = 0.9 + Math.random() * 0.15;
          for (const [sx, sy, sm] of seeds) {
            const r = Math.sqrt((x - sx) ** 2 + (y - sy) ** 2);
            matter[i2] += sm * Math.exp(-r * r / (30 + sm * 20));
            darkMatter[i2] += sm * 0.3 * Math.exp(-r * r / (60 + sm * 30));
          }
          vx[i2] = (Math.random() - 0.5) * 0.1;
          vy[i2] = (Math.random() - 0.5) * 0.1;
        }
      }
    } else {
      // random primordial soup
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const i2 = idx(x, y);
          darkMatter[i2] = 0.4 + Math.random() * 0.25;
          fluidDensity[i2] = 0.8 + Math.random() * 0.3;
          matter[i2] = Math.random() < 0.003 ? 1 + Math.random() * 3 : Math.random() * 0.05;
          vx[i2] = (Math.random() - 0.5) * 0.2;
          vy[i2] = (Math.random() - 0.5) * 0.2;
        }
      }
    }

    stateRef.current = { matter, darkMatter, curvature, vx, vy, fluidDensity };
    setStep(0);
  }, []);

  useEffect(() => { initState(preset); }, [preset, initState]);

  const simulate = useCallback(() => {
    const s = stateRef.current;
    if (!s) return;
    const { matter, darkMatter, curvature, vx, vy, fluidDensity } = s;
    const { gravity, dmCoupling, feedback, diffusion, viscosity } = params;

    const nMatter = new Float32Array(matter);
    const nDM = new Float32Array(darkMatter);
    const nCurve = new Float32Array(curvature);
    const nVx = new Float32Array(vx);
    const nVy = new Float32Array(vy);
    const nFluid = new Float32Array(fluidDensity);

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const c = idx(x, y);
        const l = idx(x - 1, y), r = idx(x + 1, y);
        const u = idx(x, y - 1), d = idx(x, y + 1);

        // Spacetime curvature = f(matter, darkMatter)
        const totalMass = matter[c] + darkMatter[c] * dmCoupling;
        nCurve[c] = totalMass * gravity;

        // Gravitational gradients
        const gx = ((matter[r] + darkMatter[r] * dmCoupling) - (matter[l] + darkMatter[l] * dmCoupling)) * 0.5;
        const gy = ((matter[d] + darkMatter[d] * dmCoupling) - (matter[u] + darkMatter[u] * dmCoupling)) * 0.5;

        // Fluid velocity responds to gravity
        nVx[c] = vx[c] * (1 - viscosity) - gx * gravity * 0.4;
        nVy[c] = vy[c] * (1 - viscosity) - gy * gravity * 0.4;

        // Advect fluid
        const sx2 = x - vx[c], sy2 = y - vy[c];
        const fx = Math.floor(sx2), fy = Math.floor(sy2);
        const tx = sx2 - fx, ty = sy2 - fy;
        nFluid[c] = (1 - tx) * (1 - ty) * fluidDensity[idx(fx, fy)]
                  + tx * (1 - ty) * fluidDensity[idx(fx + 1, fy)]
                  + (1 - tx) * ty * fluidDensity[idx(fx, fy + 1)]
                  + tx * ty * fluidDensity[idx(fx + 1, fy + 1)];

        // Dark matter clumps toward curvature + diffuses
        const dmLap = (darkMatter[l] + darkMatter[r] + darkMatter[u] + darkMatter[d]) * 0.25 - darkMatter[c];
        const curveGradX = (curvature[r] - curvature[l]) * 0.5;
        const curveGradY = (curvature[d] - curvature[u]) * 0.5;
        nDM[c] = darkMatter[c]
               + dmLap * diffusion
               + (curveGradX + curveGradY) * dmCoupling * 0.08
               + nCurve[c] * dmCoupling * 0.015;

        // Matter diffuses + attracted by dark matter density
        const mLap = (matter[l] + matter[r] + matter[u] + matter[d]) * 0.25 - matter[c];
        const dmExcess = darkMatter[c] - 0.5;
        nMatter[c] = matter[c]
                   + mLap * diffusion * 0.4
                   + dmExcess * feedback * 0.008;

        nFluid[c] = Math.max(0.01, nFluid[c]);
        nDM[c] = Math.max(0.01, nDM[c]);
        nMatter[c] = Math.max(0, nMatter[c]);
      }
    }

    s.matter = nMatter;
    s.darkMatter = nDM;
    s.curvature = nCurve;
    s.vx = nVx;
    s.vy = nVy;
    s.fluidDensity = nFluid;
  }, [params]);

  // Color mapping functions
  const colorMaps = {
    composite: (m, dm, c, fl) => {
      // Matter = hot white/gold, DM = deep violet, Curvature = warm amber glow, Fluid = teal wash
      const mN = Math.min(1, m / 3);
      const dmN = Math.min(1, dm / 2);
      const cN = Math.min(1, c / 0.8);
      const base_r = dmN * 45 + cN * 80;
      const base_g = dmN * 15 + cN * 40;
      const base_b = dmN * 120 + cN * 15;
      const r2 = base_r + mN * (255 - base_r);
      const g2 = base_g + mN * (230 - base_g);
      const b2 = base_b + mN * (140 - base_b);
      return [r2, g2, b2];
    },
    matter_only: (m) => {
      const v = Math.min(1, m / 3);
      const v2 = v * v;
      return [v2 * 255 + v * 40, v2 * 200 + v * 20, v * 60];
    },
    darkmatter_only: (m2, dm) => {
      const v = Math.min(1, dm / 2.0);
      // Violet-blue cosmic palette with high contrast
      const lo = v < 0.3;
      const hi = v > 0.8;
      if (lo) return [v * 30, v * 10, v * 80 + 8];
      if (hi) return [100 + v * 155, 40 + v * 80, 200 + v * 55];
      return [v * 80, v * 30, v * 200 + 20];
    },
    curvature_only: (m3, dm3, c) => {
      const v = Math.min(1, c / 0.8);
      // Hot thermal: black -> deep red -> orange -> white
      if (v < 0.25) return [v * 4 * 120, 0, 0];
      if (v < 0.5) return [120 + (v - 0.25) * 4 * 135, (v - 0.25) * 4 * 80, 0];
      if (v < 0.75) return [255, 80 + (v - 0.5) * 4 * 120, (v - 0.5) * 4 * 40];
      return [255, 200 + (v - 0.75) * 4 * 55, 40 + (v - 0.75) * 4 * 215];
    },
    flow: (m4, dm4, c4, fl, velX, velY) => {
      const mag = Math.sqrt(velX * velX + velY * velY);
      const v = Math.min(1, mag * 6);
      const angle = (Math.atan2(velY, velX) / Math.PI + 1) * 0.5;
      // Directional color: angle maps hue, magnitude maps brightness
      const h = angle * 360;
      const s2 = 0.8;
      const l2 = v * 0.5;
      // HSL to RGB shortcut
      const c2 = (1 - Math.abs(2 * l2 - 1)) * s2;
      const x2 = c2 * (1 - Math.abs((h / 60) % 2 - 1));
      const m5 = l2 - c2 / 2;
      let r3, g3, b3;
      if (h < 60) { r3 = c2; g3 = x2; b3 = 0; }
      else if (h < 120) { r3 = x2; g3 = c2; b3 = 0; }
      else if (h < 180) { r3 = 0; g3 = c2; b3 = x2; }
      else if (h < 240) { r3 = 0; g3 = x2; b3 = c2; }
      else if (h < 300) { r3 = x2; g3 = 0; b3 = c2; }
      else { r3 = c2; g3 = 0; b3 = x2; }
      return [(r3 + m5) * 255 + 5, (g3 + m5) * 255 + 5, (b3 + m5) * 255 + 8];
    },
    sidebyside: null, // special handling
  };

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    const disp = dispRef.current;
    if (!canvas || !disp || !stateRef.current) return;
    const s = stateRef.current;

    if (view === "sidebyside") {
      // Draw 3 panels side by side
      const pw = W, ph = H;
      canvas.width = pw * 3;
      canvas.height = ph;
      disp.width = pw * 3 * SCALE;
      disp.height = ph * SCALE;
      disp.style.width = `${pw * 3 * SCALE}px`;
      disp.style.height = `${ph * SCALE}px`;

      const ctx = canvas.getContext("2d");
      const imgData = ctx.createImageData(pw * 3, ph);
      const d = imgData.data;

      for (let y = 0; y < ph; y++) {
        for (let x = 0; x < pw; x++) {
          const gi = idx(x, y);
          const m = s.matter[gi], dm = s.darkMatter[gi], c = s.curvature[gi];

          // Panel 1: Matter
          const p1 = (y * pw * 3 + x) * 4;
          const [r1, g1, b1] = colorMaps.matter_only(m);
          d[p1] = Math.min(255, r1); d[p1+1] = Math.min(255, g1); d[p1+2] = Math.min(255, b1); d[p1+3] = 255;

          // Panel 2: Dark Matter
          const p2 = (y * pw * 3 + x + pw) * 4;
          const [r2, g2, b2] = colorMaps.darkmatter_only(m, dm);
          d[p2] = Math.min(255, r2); d[p2+1] = Math.min(255, g2); d[p2+2] = Math.min(255, b2); d[p2+3] = 255;

          // Panel 3: Curvature
          const p3 = (y * pw * 3 + x + pw * 2) * 4;
          const [r3, g3, b3] = colorMaps.curvature_only(m, dm, c);
          d[p3] = Math.min(255, r3); d[p3+1] = Math.min(255, g3); d[p3+2] = Math.min(255, b3); d[p3+3] = 255;
        }
      }

      ctx.putImageData(imgData, 0, 0);
      const dctx = disp.getContext("2d");
      dctx.imageSmoothingEnabled = false;
      dctx.drawImage(canvas, 0, 0, pw * 3, ph, 0, 0, pw * 3 * SCALE, ph * SCALE);

      // Labels
      dctx.font = "bold 13px 'DM Mono', monospace";
      dctx.textAlign = "center";
      const labels = [
        ["MATTER / ENERGY", "#fbbf24", pw * SCALE * 0.5],
        ["DARK MATTER MEDIUM", "#a78bfa", pw * SCALE * 1.5],
        ["SPACETIME CURVATURE", "#f97316", pw * SCALE * 2.5],
      ];
      for (const [text, color, xp] of labels) {
        dctx.fillStyle = "rgba(0,0,0,0.6)";
        dctx.fillRect(xp - 80, 8, 160, 22);
        dctx.fillStyle = color;
        dctx.fillText(text, xp, 24);
      }
      // Dividers
      dctx.strokeStyle = "rgba(255,255,255,0.15)";
      dctx.lineWidth = 1;
      dctx.beginPath();
      dctx.moveTo(pw * SCALE, 0); dctx.lineTo(pw * SCALE, ph * SCALE);
      dctx.moveTo(pw * 2 * SCALE, 0); dctx.lineTo(pw * 2 * SCALE, ph * SCALE);
      dctx.stroke();
      return;
    }

    canvas.width = W;
    canvas.height = H;
    disp.width = W * SCALE;
    disp.height = H * SCALE;
    disp.style.width = `${W * SCALE}px`;
    disp.style.height = `${H * SCALE}px`;

    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(W, H);
    const d = imgData.data;
    const colorFn = colorMaps[view] || colorMaps.composite;

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const gi = idx(x, y);
        const pi = (y * W + x) * 4;
        const [r, g, b] = colorFn(
          s.matter[gi], s.darkMatter[gi], s.curvature[gi],
          s.fluidDensity[gi], s.vx[gi], s.vy[gi]
        );
        d[pi] = Math.min(255, Math.max(0, r));
        d[pi+1] = Math.min(255, Math.max(0, g));
        d[pi+2] = Math.min(255, Math.max(0, b));
        d[pi+3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);
    const dctx = disp.getContext("2d");
    dctx.imageSmoothingEnabled = false;
    dctx.drawImage(canvas, 0, 0, W, H, 0, 0, W * SCALE, H * SCALE);
  }, [view]);

  useEffect(() => {
    if (!running) { cancelAnimationFrame(animRef.current); return; }
    const loop = () => {
      for (let i = 0; i < speed; i++) simulate();
      render();
      setStep(s2 => s2 + speed);
      animRef.current = requestAnimationFrame(loop);
    };
    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, [running, simulate, render, speed]);

  useEffect(() => { render(); }, [view, render]);

  const handleClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    let cx, cy;
    if (view === "sidebyside") {
      const totalW = W * 3 * SCALE;
      const rx = e.clientX - rect.left;
      // figure out which panel
      cx = Math.floor((rx % (W * SCALE)) / SCALE);
      cy = Math.floor((e.clientY - rect.top) / SCALE);
    } else {
      cx = Math.floor((e.clientX - rect.left) / SCALE);
      cy = Math.floor((e.clientY - rect.top) / SCALE);
    }
    if (!stateRef.current) return;
    const s = stateRef.current;
    for (let dy = -8; dy <= 8; dy++) {
      for (let dx = -8; dx <= 8; dx++) {
        const r = Math.sqrt(dx * dx + dy * dy);
        if (r < 8) {
          const i = idx(cx + dx, cy + dy);
          const str = Math.exp(-r * r / 12);
          s.matter[i] += 2.5 * str;
          s.fluidDensity[i] += 1.0 * str;
        }
      }
    }
    render();
  };

  const viewButtons = [
    ["composite", "All Layers", "#06b6d4"],
    ["sidebyside", "Side by Side", "#e879f9"],
    ["matter_only", "Matter", "#fbbf24"],
    ["darkmatter_only", "Dark Matter", "#a78bfa"],
    ["curvature_only", "Curvature", "#f97316"],
    ["flow", "Flow Field", "#34d399"],
  ];

  const presetButtons = [
    ["bigbang", "Big Bang", "Single primordial density → expansion"],
    ["web", "Cosmic Web", "30 seed clusters → filament formation"],
    ["bullet", "Bullet Cluster", "Two clusters colliding head-on"],
    ["soup", "Random Soup", "Uniform primordial fluctuations"],
  ];

  const paramDefs = [
    ["gravity", "Gravity Strength", 0, 0.4, 0.01],
    ["dmCoupling", "DM ↔ Spacetime", 0, 0.2, 0.005],
    ["feedback", "Matter ↔ DM", 0, 0.15, 0.005],
    ["diffusion", "Diffusion Rate", 0, 0.25, 0.005],
    ["viscosity", "Fluid Viscosity", 0, 0.1, 0.002],
  ];

  return (
    <div style={{
      minHeight: "100vh",
      background: "#08080f",
      color: "#d1d5db",
      fontFamily: "'DM Sans', sans-serif",
      padding: "20px 16px",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet" />

      <div style={{ maxWidth: 1100, margin: "0 auto" }}>
        <div style={{ marginBottom: 20 }}>
          <h1 style={{
            fontSize: 22, fontWeight: 700, marginBottom: 4,
            background: "linear-gradient(90deg, #fbbf24 0%, #a78bfa 40%, #06b6d4 80%)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          }}>
            Primordial Fluid Model
          </h1>
          <p style={{ fontSize: 12, color: "#6b7280", fontFamily: "'DM Mono', monospace", lineHeight: 1.5, maxWidth: 650 }}>
            Dark matter as the medium itself — three substrates co-evolving:<br/>
            <span style={{ color: "#fbbf24" }}>matter</span> warps → <span style={{ color: "#f97316" }}>curvature</span> forms → <span style={{ color: "#a78bfa" }}>dark medium clumps</span> → feeds back into gravity → cycle repeats
          </p>
        </div>

        {/* Scenario buttons */}
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 14 }}>
          {presetButtons.map(([key, label, desc]) => (
            <button key={key} onClick={() => { setPreset(key); setRunning(false); }}
              style={{
                padding: "7px 14px", fontSize: 12, borderRadius: 6, cursor: "pointer",
                border: preset === key ? "1px solid #fbbf24" : "1px solid #1f1f30",
                background: preset === key ? "rgba(251,191,36,0.12)" : "rgba(15,15,25,0.9)",
                color: preset === key ? "#fbbf24" : "#9ca3af",
                fontFamily: "'DM Mono', monospace",
              }}
              title={desc}
            >{label}</button>
          ))}
        </div>

        {/* View buttons */}
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 14 }}>
          {viewButtons.map(([key, label, color]) => (
            <button key={key} onClick={() => { setView(key); }}
              style={{
                padding: "6px 12px", fontSize: 11, borderRadius: 6, cursor: "pointer",
                border: view === key ? `1px solid ${color}` : "1px solid #1f1f30",
                background: view === key ? `${color}18` : "rgba(15,15,25,0.9)",
                color: view === key ? color : "#6b7280",
                fontFamily: "'DM Mono', monospace",
              }}
            >{label}</button>
          ))}
        </div>

        {/* Canvas */}
        <div style={{ marginBottom: 14, position: "relative", display: "inline-block" }}>
          <canvas ref={canvasRef} style={{ display: "none" }} />
          <canvas
            ref={dispRef}
            onClick={handleClick}
            style={{
              borderRadius: 6,
              border: "1px solid #1a1a2e",
              cursor: "crosshair",
              display: "block",
              imageRendering: "pixelated",
              boxShadow: "0 0 40px rgba(100,60,200,0.08)",
            }}
          />
          <div style={{
            position: "absolute", bottom: 6, left: 8,
            background: "rgba(0,0,0,0.7)", borderRadius: 4, padding: "3px 8px",
            fontSize: 10, color: "#6b7280", fontFamily: "'DM Mono', monospace",
          }}>
            t = {step} · click to inject matter
          </div>
        </div>

        {/* Controls row */}
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "flex-start" }}>
          {/* Playback */}
          <div>
            <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
              <button onClick={() => setRunning(!running)}
                style={{
                  padding: "9px 22px", fontSize: 13, borderRadius: 6, border: "none", cursor: "pointer",
                  fontWeight: 600, fontFamily: "'DM Sans', sans-serif",
                  background: running ? "#dc2626" : "#f59e0b",
                  color: "#fff",
                  boxShadow: running ? "0 0 16px rgba(220,38,38,0.25)" : "0 0 16px rgba(245,158,11,0.25)",
                }}
              >{running ? "⏸ Pause" : "▶ Run"}</button>
              <button onClick={() => { simulate(); render(); setStep(s2 => s2 + 1); }}
                style={{
                  padding: "9px 14px", fontSize: 12, borderRadius: 6, cursor: "pointer",
                  border: "1px solid #2a2a3e", background: "rgba(15,15,25,0.9)", color: "#9ca3af",
                  fontFamily: "'DM Mono', monospace",
                }}
              >Step</button>
              <button onClick={() => { initState(preset); setRunning(false); }}
                style={{
                  padding: "9px 14px", fontSize: 12, borderRadius: 6, cursor: "pointer",
                  border: "1px solid #2a2a3e", background: "rgba(15,15,25,0.9)", color: "#9ca3af",
                  fontFamily: "'DM Mono', monospace",
                }}
              >Reset</button>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 10, color: "#6b7280", fontFamily: "'DM Mono', monospace" }}>Speed</span>
              <input type="range" min={1} max={8} value={speed} onChange={e => setSpeed(parseInt(e.target.value))}
                style={{ width: 80, accentColor: "#f59e0b" }} />
              <span style={{ fontSize: 10, color: "#fbbf24", fontFamily: "'DM Mono', monospace" }}>{speed}x</span>
            </div>
          </div>

          {/* Physics sliders */}
          <div style={{ flex: 1, minWidth: 240, maxWidth: 340 }}>
            <div style={{ fontSize: 10, color: "#6b7280", fontFamily: "'DM Mono', monospace", marginBottom: 6, textTransform: "uppercase", letterSpacing: 1 }}>Physics</div>
            {paramDefs.map(([key, label, mn, mx, st]) => (
              <div key={key} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                <span style={{ width: 110, fontSize: 10, color: "#6b7280", fontFamily: "'DM Mono', monospace" }}>{label}</span>
                <input type="range" min={mn} max={mx} step={st} value={params[key]}
                  onChange={e => setParams(p => ({ ...p, [key]: parseFloat(e.target.value) }))}
                  style={{ flex: 1, accentColor: "#a78bfa" }} />
                <span style={{ width: 34, fontSize: 10, color: "#a78bfa", fontFamily: "'DM Mono', monospace", textAlign: "right" }}>{params[key].toFixed(3)}</span>
              </div>
            ))}
          </div>

          {/* Key */}
          <div style={{
            padding: 12, borderRadius: 6, background: "rgba(12,12,22,0.9)",
            border: "1px solid #1a1a2e", minWidth: 200,
          }}>
            <div style={{ fontSize: 10, color: "#6b7280", fontFamily: "'DM Mono', monospace", marginBottom: 6, textTransform: "uppercase", letterSpacing: 1 }}>The Cycle</div>
            <div style={{ fontSize: 11, lineHeight: 2, fontFamily: "'DM Mono', monospace" }}>
              <div><span style={{ color: "#fbbf24" }}>①</span> <span style={{ color: "#d1d5db" }}>Matter/energy exists</span></div>
              <div><span style={{ color: "#f97316" }}>②</span> <span style={{ color: "#d1d5db" }}>Warps spacetime geometry</span></div>
              <div><span style={{ color: "#a78bfa" }}>③</span> <span style={{ color: "#d1d5db" }}>Dark medium clumps along warps</span></div>
              <div><span style={{ color: "#34d399" }}>④</span> <span style={{ color: "#d1d5db" }}>Clumps amplify gravity</span></div>
              <div><span style={{ color: "#06b6d4" }}>⑤</span> <span style={{ color: "#d1d5db" }}>Pulls more matter in → repeat</span></div>
            </div>
            <div style={{ marginTop: 8, fontSize: 10, color: "#4b5563", fontFamily: "'DM Mono', monospace", lineHeight: 1.5 }}>
              Use "Side by Side" view to see<br/>all three substrates separately
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
