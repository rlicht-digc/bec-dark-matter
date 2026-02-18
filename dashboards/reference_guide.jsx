import { useState, useEffect, useRef, useCallback } from "react";

const W = 480;
const GRID = 120;
const CELL = W / GRID;

function makeParticles() {
  const p = [];
  // 20 seed clusters spread across the box — cosmic web initial conditions
  const seeds = [];
  for (let s = 0; s < 20; s++) {
    seeds.push([50 + Math.random() * (W - 100), 50 + Math.random() * (W - 100)]);
  }
  for (const [cx, cy] of seeds) {
    const n = 8 + Math.floor(Math.random() * 16);
    for (let i = 0; i < n; i++) {
      const a = Math.random() * Math.PI * 2;
      const r = Math.random() * 18;
      p.push({
        x: cx + Math.cos(a) * r,
        y: cy + Math.sin(a) * r,
        vx: (Math.random() - 0.5) * 0.2,
        vy: (Math.random() - 0.5) * 0.2,
        m: 0.4 + Math.random() * 1.4,
      });
    }
  }
  return p;
}

export default function ReferenceSimulation() {
  const matterRef = useRef(null);
  const dmRef = useRef(null);
  const curveRef = useRef(null);
  const stateRef = useRef(null);
  const animRef = useRef(null);
  const [step, setStep] = useState(0);
  const [running, setRunning] = useState(false);
  const [showGuide, setShowGuide] = useState(true);

  const PARAMS = {
    gravity: 0.099, softening: 25, damping: 1.0,
    dmGrow: 2.0, dmFade: 0.977,
  };

  const init = useCallback(() => {
    const particles = makeParticles();
    const dmBuf = new Float32Array(GRID * GRID);
    stateRef.current = { particles, dmBuf };
    setStep(0);
  }, []);

  useEffect(() => { init(); }, [init]);

  const simulate = useCallback(() => {
    const s = stateRef.current;
    if (!s) return;
    const { particles, dmBuf } = s;
    const { gravity, softening, damping, dmGrow, dmFade } = PARAMS;
    const soft2 = softening * softening;
    const N = particles.length;
    const GC = 35;
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
            const f = gravity * o.m / (d2 * Math.sqrt(d2));
            ax += ddx * f; ay += ddy * f;
          }
        }
      }

      // DM gradient feedback
      const dmx = Math.floor(p.x / CELL), dmy = Math.floor(p.y / CELL);
      if (dmx >= 1 && dmx < GRID - 1 && dmy >= 1 && dmy < GRID - 1) {
        const grdx = (dmBuf[dmy * GRID + dmx + 1] - dmBuf[dmy * GRID + dmx - 1]) * 0.5;
        const grdy = (dmBuf[(dmy + 1) * GRID + dmx] - dmBuf[(dmy - 1) * GRID + dmx]) * 0.5;
        ax += grdx * gravity * 0.12;
        ay += grdy * gravity * 0.12;
      }

      p.vx = (p.vx + ax) * damping;
      p.vy = (p.vy + ay) * damping;
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x += W; if (p.x >= W) p.x -= W;
      if (p.y < 0) p.y += W; if (p.y >= W) p.y -= W;
    }

    // DM: fade then deposit
    for (let i = 0; i < dmBuf.length; i++) dmBuf[i] *= dmFade;
    for (const p of particles) {
      const gx = Math.floor(p.x / CELL), gy = Math.floor(p.y / CELL);
      for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          const nx = gx + dx, ny = gy + dy;
          if (nx < 0 || nx >= GRID || ny < 0 || ny >= GRID) continue;
          const r = Math.sqrt(dx * dx + dy * dy);
          dmBuf[ny * GRID + nx] += Math.exp(-r * r / 1.2) * p.m * dmGrow * 0.01;
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

    // --- MATTER ---
    const mc = matterRef.current?.getContext("2d");
    if (mc) {
      mc.fillStyle = "#04040c"; mc.fillRect(0, 0, W, W);
      for (const p of particles) {
        const sz = Math.max(1.5, Math.min(3, p.m * 1.5));
        const spd = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
        const heat = Math.min(1, spd / 2);
        mc.fillStyle = `rgb(${200 + heat * 55|0},${170 + heat * 60|0},${60 + (1-heat)*40|0})`;
        mc.beginPath(); mc.arc(p.x, p.y, sz, 0, Math.PI * 2); mc.fill();
      }
    }

    // --- DARK MATTER ---
    const dc = dmRef.current?.getContext("2d");
    if (dc) {
      const img = dc.createImageData(GRID, GRID);
      const d = img.data;
      for (let i = 0; i < GRID * GRID; i++) {
        const n = Math.pow(Math.min(1, dmBuf[i] / maxDM), 0.35);
        const pi = i * 4;
        if (n < 0.02) { d[pi]=3; d[pi+1]=3; d[pi+2]=10; }
        else {
          d[pi] = Math.min(255, n*n*220 + n*35|0);
          d[pi+1] = Math.min(255, n*n*60 + n*15|0);
          d[pi+2] = Math.min(255, n*255|0);
        }
        d[pi+3] = 255;
      }
      const tmp = document.createElement("canvas");
      tmp.width = GRID; tmp.height = GRID;
      tmp.getContext("2d").putImageData(img, 0, 0);
      dc.imageSmoothingEnabled = true;
      dc.imageSmoothingQuality = "high";
      dc.drawImage(tmp, 0, 0, GRID, GRID, 0, 0, W, W);
    }

    // --- CURVATURE ---
    const cc = curveRef.current?.getContext("2d");
    if (cc) {
      const mDens = new Float32Array(GRID * GRID);
      for (const p of particles) {
        const gx = Math.floor(p.x / CELL), gy = Math.floor(p.y / CELL);
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nx = gx+dx, ny = gy+dy;
            if (nx<0||nx>=GRID||ny<0||ny>=GRID) continue;
            mDens[ny*GRID+nx] += p.m * Math.exp(-(dx*dx+dy*dy)/0.8);
          }
        }
      }
      let maxC = 0.001;
      const cBuf = new Float32Array(GRID * GRID);
      for (let i = 0; i < cBuf.length; i++) {
        cBuf[i] = mDens[i] + dmBuf[i] * 3;
        if (cBuf[i] > maxC) maxC = cBuf[i];
      }
      const img = cc.createImageData(GRID, GRID);
      const d = img.data;
      for (let i = 0; i < GRID*GRID; i++) {
        const n = Math.pow(Math.min(1, cBuf[i]/maxC), 0.4);
        const pi = i*4;
        if (n<0.02) { d[pi]=4; d[pi+1]=2; d[pi+2]=2; }
        else if (n<0.3) { const t=n/0.3; d[pi]=t*180|0; d[pi+1]=0; d[pi+2]=t*20|0; }
        else if (n<0.6) { const t=(n-0.3)/0.3; d[pi]=180+t*75|0; d[pi+1]=t*140|0; d[pi+2]=20-t*20|0; }
        else { const t=(n-0.6)/0.4; d[pi]=255; d[pi+1]=140+t*115|0; d[pi+2]=t*220|0; }
        d[pi+3]=255;
      }
      const tmp = document.createElement("canvas");
      tmp.width = GRID; tmp.height = GRID;
      tmp.getContext("2d").putImageData(img, 0, 0);
      cc.imageSmoothingEnabled = true;
      cc.imageSmoothingQuality = "high";
      cc.drawImage(tmp, 0, 0, GRID, GRID, 0, 0, W, W);
    }
  }, []);

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

  // Pre-run to mature state
  const prerun = (n) => {
    for (let i = 0; i < n; i++) simulate();
    render();
    setStep(s => s + n);
  };

  const canvasStyle = {
    width: W, height: W, borderRadius: 8,
    border: "1px solid rgba(255,255,255,0.06)",
  };

  const guideOverlay = (type) => {
    if (!showGuide) return null;
    const guides = {
      matter: [
        { text: "GALAXY CLUSTERS\n(bright clumps)", x: "20%", y: "15%", color: "#fbbf24" },
        { text: "FILAMENTS\n(chains between clumps)", x: "50%", y: "55%", color: "#fbbf24" },
        { text: "VOIDS\n(empty dark regions)", x: "75%", y: "80%", color: "#6b7280" },
      ],
      dm: [
        { text: "DM HALOS\n(bright, WIDER than\nmatter clumps)", x: "20%", y: "12%", color: "#c084fc" },
        { text: "DM FILAMENTS\n(broader, fuzzier\nthan matter chains)", x: "50%", y: "50%", color: "#c084fc" },
        { text: "FADING TRAILS\n(dim remnants where\nmatter WAS)", x: "72%", y: "78%", color: "#8b5cf6" },
        { text: "SONOGRAM\nTEXTURE\n(grainy = good)", x: "15%", y: "82%", color: "#a78bfa" },
      ],
      curve: [
        { text: "DEEP WELLS\n(white-hot = most\nmass concentrated)", x: "20%", y: "15%", color: "#fb923c" },
        { text: "SMOOTHER &\nBROADER than\nmatter panel", x: "50%", y: "50%", color: "#fb923c" },
        { text: "DM CONTRIBUTION\n(orange glow beyond\nmatter locations)", x: "70%", y: "78%", color: "#f59e0b" },
      ],
    };

    return (
      <div style={{
        position: "absolute", top: 0, left: 0, width: W, height: W,
        pointerEvents: "none", borderRadius: 8,
      }}>
        {guides[type].map((g, i) => (
          <div key={i} style={{
            position: "absolute", left: g.x, top: g.y,
            transform: "translate(-50%, -50%)",
            background: "rgba(0,0,0,0.7)", border: `1px solid ${g.color}40`,
            borderRadius: 6, padding: "5px 8px",
            color: g.color, fontSize: 10, fontWeight: 600,
            fontFamily: "'JetBrains Mono', monospace",
            whiteSpace: "pre-line", textAlign: "center", lineHeight: 1.3,
            backdropFilter: "blur(4px)",
          }}>
            {g.text}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div style={{
      minHeight: "100vh", background: "#020208", color: "#d1d5db",
      fontFamily: "'Instrument Sans', 'DM Sans', sans-serif", padding: 20,
    }}>
      <link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet" />

      <div style={{ maxWidth: 1560, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 20 }}>
          <h1 style={{
            fontSize: 22, fontWeight: 700, margin: 0,
            background: "linear-gradient(90deg, #fbbf24, #c084fc, #fb923c)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          }}>Reference Guide — What Each Panel Should Look Like</h1>
          <p style={{
            fontSize: 12, color: "#6b7280", margin: "6px 0 0",
            fontFamily: "'JetBrains Mono', monospace", maxWidth: 800, lineHeight: 1.5,
          }}>
            Your parameters: gravity=0.099  damping=1.0  dm_fade=0.977  softening=25  dm_growth=2.0
          </p>
        </div>

        {/* Controls */}
        <div style={{ display: "flex", gap: 8, marginBottom: 16, alignItems: "center", flexWrap: "wrap" }}>
          <button onClick={() => setRunning(!running)} style={{
            padding: "8px 20px", borderRadius: 6, border: "none", cursor: "pointer",
            fontWeight: 700, fontSize: 13, color: "#fff",
            background: running ? "#dc2626" : "#f59e0b",
            boxShadow: running ? "0 0 16px #dc262644" : "0 0 16px #f59e0b44",
            fontFamily: "'Instrument Sans', sans-serif",
          }}>{running ? "⏸ Pause" : "▶ Run Simulation"}</button>

          <button onClick={() => prerun(200)} style={{
            padding: "8px 16px", borderRadius: 6, border: "1px solid #1c1c2e",
            cursor: "pointer", background: "rgba(8,8,16,0.95)", color: "#9ca3af",
            fontFamily: "'JetBrains Mono', monospace", fontSize: 12,
          }}>Skip +200 steps</button>

          <button onClick={() => prerun(1000)} style={{
            padding: "8px 16px", borderRadius: 6, border: "1px solid #1c1c2e",
            cursor: "pointer", background: "rgba(8,8,16,0.95)", color: "#9ca3af",
            fontFamily: "'JetBrains Mono', monospace", fontSize: 12,
          }}>Skip +1000 steps</button>

          <button onClick={() => { init(); setRunning(false); }} style={{
            padding: "8px 16px", borderRadius: 6, border: "1px solid #1c1c2e",
            cursor: "pointer", background: "rgba(8,8,16,0.95)", color: "#9ca3af",
            fontFamily: "'JetBrains Mono', monospace", fontSize: 12,
          }}>Reset</button>

          <button onClick={() => setShowGuide(!showGuide)} style={{
            padding: "8px 16px", borderRadius: 6, cursor: "pointer",
            border: showGuide ? "1px solid #c084fc" : "1px solid #1c1c2e",
            background: showGuide ? "rgba(192,132,252,0.12)" : "rgba(8,8,16,0.95)",
            color: showGuide ? "#c084fc" : "#6b7280",
            fontFamily: "'JetBrains Mono', monospace", fontSize: 12, fontWeight: 600,
          }}>{showGuide ? "◉ Labels ON" : "○ Labels OFF"}</button>

          <span style={{
            fontSize: 11, color: "#4b5563", fontFamily: "'JetBrains Mono', monospace", marginLeft: 8,
          }}>t = {step}</span>
        </div>

        {/* Three panels */}
        <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
          {/* MATTER */}
          <div>
            <div style={{
              fontSize: 11, fontWeight: 700, color: "#fbbf24", textTransform: "uppercase",
              fontFamily: "'JetBrains Mono', monospace", marginBottom: 6,
              padding: "5px 12px", background: "rgba(251,191,36,0.06)", borderRadius: 4,
              display: "inline-block", letterSpacing: 1.5,
            }}>Matter / Energy</div>
            <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 6,
              fontFamily: "'JetBrains Mono', monospace", maxWidth: W, lineHeight: 1.4 }}>
              Corresponds to: SDSS galaxy maps, visible universe
            </div>
            <div style={{ position: "relative", display: "inline-block" }}>
              <canvas ref={matterRef} width={W} height={W} style={canvasStyle} />
              {guideOverlay("matter")}
            </div>
          </div>

          {/* DARK MATTER */}
          <div>
            <div style={{
              fontSize: 11, fontWeight: 700, color: "#c084fc", textTransform: "uppercase",
              fontFamily: "'JetBrains Mono', monospace", marginBottom: 6,
              padding: "5px 12px", background: "rgba(192,132,252,0.06)", borderRadius: 4,
              display: "inline-block", letterSpacing: 1.5,
            }}>Dark Matter Medium</div>
            <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 6,
              fontFamily: "'JetBrains Mono', monospace", maxWidth: W, lineHeight: 1.4 }}>
              Corresponds to: Gravitational lensing mass maps (DES, HSC)
            </div>
            <div style={{ position: "relative", display: "inline-block" }}>
              <canvas ref={dmRef} width={W} height={W} style={canvasStyle} />
              {guideOverlay("dm")}
            </div>
          </div>

          {/* CURVATURE */}
          <div>
            <div style={{
              fontSize: 11, fontWeight: 700, color: "#fb923c", textTransform: "uppercase",
              fontFamily: "'JetBrains Mono', monospace", marginBottom: 6,
              padding: "5px 12px", background: "rgba(251,146,60,0.06)", borderRadius: 4,
              display: "inline-block", letterSpacing: 1.5,
            }}>Spacetime Curvature</div>
            <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 6,
              fontFamily: "'JetBrains Mono', monospace", maxWidth: W, lineHeight: 1.4 }}>
              Corresponds to: Gravitational potential Φ (GR curvature)
            </div>
            <div style={{ position: "relative", display: "inline-block" }}>
              <canvas ref={curveRef} width={W} height={W} style={canvasStyle} />
              {guideOverlay("curve")}
            </div>
          </div>
        </div>

        {/* What to look for guide */}
        <div style={{
          marginTop: 20, padding: 20, borderRadius: 10,
          background: "rgba(192,132,252,0.04)", border: "1px solid rgba(192,132,252,0.1)",
          maxWidth: 1500,
        }}>
          <h3 style={{
            fontSize: 14, fontWeight: 700, color: "#c084fc", margin: "0 0 12px",
            fontFamily: "'JetBrains Mono', monospace", letterSpacing: 1,
          }}>✓ WHAT CORRECT LOOKS LIKE</h3>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#fbbf24", marginBottom: 6 }}>MATTER (Left)</div>
              <div style={{ fontSize: 11, color: "#9ca3af", lineHeight: 1.6 }}>
                ✓ Tight, discrete clumps of gold dots<br/>
                ✓ Thin chains connecting clumps<br/>
                ✓ Large empty voids between structures<br/>
                ✓ Some clumps merging over time<br/>
                ✗ NOT a uniform scatter of dots<br/>
                ✗ NOT one big central blob
              </div>
            </div>
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#c084fc", marginBottom: 6 }}>DARK MATTER (Center)</div>
              <div style={{ fontSize: 11, color: "#9ca3af", lineHeight: 1.6 }}>
                ✓ Bright violet halos LARGER than matter clumps<br/>
                ✓ Fuzzy, broader bridges between halos<br/>
                ✓ Grainy/sonogram texture (not smooth)<br/>
                ✓ Dim trails where matter has passed<br/>
                ✓ Near-black in voids<br/>
                ✗ NOT identical to matter panel<br/>
                ✗ NOT uniform purple wash
              </div>
            </div>
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#fb923c", marginBottom: 6 }}>CURVATURE (Right)</div>
              <div style={{ fontSize: 11, color: "#9ca3af", lineHeight: 1.6 }}>
                ✓ Smoothest of all three panels<br/>
                ✓ White-hot spots at matter locations<br/>
                ✓ Broad orange glow extending well beyond<br/>
                ✓ The broadest, most diffuse version<br/>
                ✗ NOT sharp or point-like<br/>
                ✗ NOT identical to matter panel
              </div>
            </div>
          </div>

          <div style={{
            marginTop: 14, padding: 12, borderRadius: 6,
            background: "rgba(251,191,36,0.06)", border: "1px solid rgba(251,191,36,0.1)",
          }}>
            <div style={{ fontSize: 11, color: "#fbbf24", fontWeight: 600, marginBottom: 4,
              fontFamily: "'JetBrains Mono', monospace" }}>
              KEY TEST: THE PROGRESSION
            </div>
            <div style={{ fontSize: 11, color: "#d1d5db", lineHeight: 1.5 }}>
              Left → Center → Right should show: <strong style={{ color: "#fbbf24" }}>discrete</strong> → <strong style={{ color: "#c084fc" }}>diffuse</strong> → <strong style={{ color: "#fb923c" }}>smooth</strong>. 
              Each panel should be progressively wider and fuzzier than the last. 
              If the DM panel has halos that visibly extend beyond the matter clumps, your model is producing 
              the correct dark matter distribution. This matches real observations: visible galaxies are compact, 
              dark matter halos (from lensing) extend 5-10× further, and the total gravitational potential is the smoothest of all.
            </div>
          </div>
        </div>

        <div style={{
          marginTop: 14, fontSize: 10, color: "#4b5563",
          fontFamily: "'JetBrains Mono', monospace", lineHeight: 1.5,
        }}>
          Tip: Click "Skip +1000 steps" a few times to reach a mature state where structure is fully formed. 
          Then compare against your v6 simulation running with the same parameters.
        </div>
      </div>
    </div>
  );
}
