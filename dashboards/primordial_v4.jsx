import { useState, useEffect, useRef, useCallback } from "react";

const CW = 560;
const CH = 560;
const GRID = 100;
const CELL = CW / GRID;

function createField(val = 0) { return new Float32Array(GRID * GRID).fill(val); }
function gi(gx, gy) { return ((gy + GRID) % GRID) * GRID + ((gx + GRID) % GRID); }

function initParticles(scenario) {
  const ps = [];
  if (scenario === "bigbang") {
    for (let i = 0; i < 800; i++) {
      const a = Math.random() * Math.PI * 2;
      const r = Math.random() * 25 + 1;
      const spd = 0.5 + Math.random() * 2.0;
      ps.push({
        x: CW/2 + Math.cos(a)*r, y: CH/2 + Math.sin(a)*r,
        vx: Math.cos(a)*spd + (Math.random()-.5)*.4,
        vy: Math.sin(a)*spd + (Math.random()-.5)*.4,
        mass: 0.4 + Math.random() * 1.6,
      });
    }
  } else if (scenario === "web") {
    const seeds = [];
    for (let i = 0; i < 25; i++) seeds.push([60+Math.random()*(CW-120), 60+Math.random()*(CH-120)]);
    for (const [sx, sy] of seeds) {
      const n = 12 + Math.floor(Math.random()*20);
      for (let i = 0; i < n; i++) {
        const a = Math.random()*Math.PI*2, r = Math.random()*20;
        ps.push({
          x: sx+Math.cos(a)*r, y: sy+Math.sin(a)*r,
          vx: (Math.random()-.5)*.3, vy: (Math.random()-.5)*.3,
          mass: 0.4 + Math.random()*1.2,
        });
      }
    }
  } else if (scenario === "bullet") {
    for (let i = 0; i < 350; i++) {
      const a = Math.random()*Math.PI*2, r = Math.random()*35;
      ps.push({
        x: CW*.25+Math.cos(a)*r, y: CH*.5+Math.sin(a)*r,
        vx: 0.9+(Math.random()-.5)*.2, vy: (Math.random()-.5)*.25,
        mass: 0.5+Math.random()*1.5,
      });
    }
    for (let i = 0; i < 350; i++) {
      const a = Math.random()*Math.PI*2, r = Math.random()*35;
      ps.push({
        x: CW*.75+Math.cos(a)*r, y: CH*.5+Math.sin(a)*r,
        vx: -0.9+(Math.random()-.5)*.2, vy: (Math.random()-.5)*.25,
        mass: 0.5+Math.random()*1.5,
      });
    }
  } else {
    for (let i = 0; i < 600; i++) {
      ps.push({
        x: 30+Math.random()*(CW-60), y: 30+Math.random()*(CH-60),
        vx: (Math.random()-.5)*.5, vy: (Math.random()-.5)*.5,
        mass: 0.3+Math.random()*1.2,
      });
    }
  }
  return ps;
}

export default function PrimordialSim() {
  const canvasRef = useRef(null);
  const stateRef = useRef(null);
  const animRef = useRef(null);
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(0);
  const [view, setView] = useState("composite");
  const [preset, setPreset] = useState("web");
  const [showP, setShowP] = useState(true);
  const [params, setParams] = useState({
    gravity: 0.04,
    dmGrow: 0.08,
    dmDecay: 0.06,
    dmThreshold: 0.02,
    softening: 14,
    damping: 0.997,
  });

  const init = useCallback((sc) => {
    stateRef.current = {
      particles: initParticles(sc),
      dm: createField(0),
      curve: createField(0),
      mField: createField(0),
    };
    setStep(0);
  }, []);

  useEffect(() => { init(preset); }, [preset, init]);

  const simulate = useCallback(() => {
    const s = stateRef.current;
    if (!s) return;
    const { particles, dm, curve, mField } = s;
    const { gravity, dmGrow, dmDecay, dmThreshold, softening, damping } = params;
    const soft2 = softening * softening;

    // Deposit matter to grid with gaussian splat
    mField.fill(0);
    for (const p of particles) {
      const gx0 = Math.floor(p.x / CELL);
      const gy0 = Math.floor(p.y / CELL);
      for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          const gx = gx0 + dx, gy = gy0 + dy;
          if (gx < 0 || gx >= GRID || gy < 0 || gy >= GRID) continue;
          const dist = Math.sqrt(dx*dx + dy*dy);
          const w = Math.exp(-dist*dist / 1.8);
          mField[gi(gx, gy)] += p.mass * w;
        }
      }
    }

    // Curvature from matter + dark matter
    for (let i = 0; i < GRID*GRID; i++) {
      curve[i] = (mField[i] + dm[i] * 0.4) * gravity;
    }

    // Dark matter: only accumulates where curvature exceeds threshold
    // Decays everywhere — so it stays sharp around matter, fades elsewhere
    const newDM = new Float32Array(dm);
    for (let gy = 1; gy < GRID-1; gy++) {
      for (let gx = 1; gx < GRID-1; gx++) {
        const i = gi(gx, gy);
        const c = curve[i];

        // Only grow where curvature is significant
        const grow = c > dmThreshold ? (c - dmThreshold) * dmGrow : 0;

        // Very mild diffusion — just enough to connect filaments
        const lap = (dm[gi(gx-1,gy)] + dm[gi(gx+1,gy)] + dm[gi(gx,gy-1)] + dm[gi(gx,gy+1)]) * 0.25 - dm[i];

        // Decay proportional to current density — prevents runaway
        const decay = dm[i] * dmDecay * 0.15;

        newDM[i] = dm[i] + grow + lap * 0.02 - decay;
        newDM[i] = Math.max(0, Math.min(5, newDM[i]));
      }
    }
    s.dm = newDM;

    // Grid force field
    const fx = createField(0), fy = createField(0);
    for (let gy = 1; gy < GRID-1; gy++) {
      for (let gx = 1; gx < GRID-1; gx++) {
        const i = gi(gx, gy);
        const potR = mField[gi(gx+1,gy)] + newDM[gi(gx+1,gy)] * 0.5;
        const potL = mField[gi(gx-1,gy)] + newDM[gi(gx-1,gy)] * 0.5;
        const potD = mField[gi(gx,gy+1)] + newDM[gi(gx,gy+1)] * 0.5;
        const potU = mField[gi(gx,gy-1)] + newDM[gi(gx,gy-1)] * 0.5;
        fx[i] = (potR - potL) * 0.5 * gravity;
        fy[i] = (potD - potU) * 0.5 * gravity;
      }
    }

    // Move particles
    for (const p of particles) {
      const gx = Math.floor(p.x / CELL);
      const gy = Math.floor(p.y / CELL);
      if (gx >= 1 && gx < GRID-1 && gy >= 1 && gy < GRID-1) {
        const i = gi(gx, gy);
        p.vx += fx[i] * 0.9;
        p.vy += fy[i] * 0.9;
      }
      // Direct N-body sample
      const sampleN = Math.min(particles.length, 15);
      for (let k = 0; k < sampleN; k++) {
        const o = particles[Math.floor(Math.random() * particles.length)];
        if (o === p) continue;
        const dx = o.x - p.x, dy = o.y - p.y;
        const d2 = dx*dx + dy*dy + soft2;
        const f = gravity * o.mass / (d2 * Math.sqrt(d2)) * 10;
        p.vx += dx * f;
        p.vy += dy * f;
      }
      p.vx *= damping; p.vy *= damping;
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x += CW; if (p.x >= CW) p.x -= CW;
      if (p.y < 0) p.y += CH; if (p.y >= CH) p.y -= CH;
    }
  }, [params]);

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    const s = stateRef.current;
    if (!canvas || !s) return;
    const ctx = canvas.getContext("2d");

    const isSBS = view === "sidebyside";
    const panelW = CW;
    canvas.width = isSBS ? panelW * 3 + 4 : CW;
    canvas.height = isSBS ? CH : CH;

    ctx.fillStyle = "#050510";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Precompute max values for normalization
    let maxM = 0.01, maxDM = 0.01, maxC = 0.001;
    for (let i = 0; i < GRID*GRID; i++) {
      if (s.mField[i] > maxM) maxM = s.mField[i];
      if (s.dm[i] > maxDM) maxDM = s.dm[i];
      if (s.curve[i] > maxC) maxC = s.curve[i];
    }

    function drawField(offX, mode) {
      for (let gy = 0; gy < GRID; gy++) {
        for (let gx = 0; gx < GRID; gx++) {
          const i = gi(gx, gy);
          const m = s.mField[i] / maxM;
          const d = s.dm[i] / maxDM;
          const c = s.curve[i] / maxC;
          let r=0, g=0, b=0, skip=false;

          if (mode === "matter") {
            if (m < 0.01) { skip = true; } else {
              const v = Math.pow(m, 0.6);
              r = v * 255; g = v * 200; b = v * 60;
            }
          } else if (mode === "dm") {
            if (d < 0.015) { skip = true; } else {
              const v = Math.pow(d, 0.5);
              // Sharp violet with bright hot spots
              r = v * v * 160 + v * 25;
              g = v * v * 50 + v * 8;
              b = v * 240 + 10;
            }
          } else if (mode === "curve") {
            if (c < 0.01) { skip = true; } else {
              const v = Math.pow(c, 0.5);
              if (v < 0.3) { r = v*3.3*160; g = 0; b = 0; }
              else if (v < 0.6) { r = 160+(v-.3)*3.3*95; g = (v-.3)*3.3*120; b = 0; }
              else { r = 255; g = 120+(v-.6)*2.5*135; b = (v-.6)*2.5*220; }
            }
          } else { // composite
            if (d < 0.01 && c < 0.01) { skip = true; } else {
              const dv = Math.pow(d, 0.5);
              const cv = Math.pow(c, 0.5);
              r = dv * 50 + cv * 140;
              g = dv * 18 + cv * 70;
              b = dv * 200 + cv * 20;
            }
          }

          if (!skip) {
            ctx.fillStyle = `rgb(${Math.min(255,Math.floor(r))},${Math.min(255,Math.floor(g))},${Math.min(255,Math.floor(b))})`;
            ctx.fillRect(offX + gx * CELL, gy * CELL, Math.ceil(CELL), Math.ceil(CELL));
          }
        }
      }
    }

    if (isSBS) {
      drawField(0, "matter");
      drawField(panelW + 2, "dm");
      drawField((panelW + 2) * 2, "curve");

      // Particles on matter panel
      ctx.fillStyle = "rgba(255,245,200,0.85)";
      for (const p of s.particles) {
        const sz = Math.max(1.2, Math.min(3, p.mass * 1.4));
        ctx.beginPath(); ctx.arc(p.x, p.y, sz, 0, Math.PI*2); ctx.fill();
      }

      // Labels
      ctx.font = "bold 11px monospace";
      ctx.textAlign = "center";
      const labels = [
        ["MATTER", "#fbbf24", panelW/2],
        ["DARK MATTER MEDIUM", "#a78bfa", panelW + 2 + panelW/2],
        ["SPACETIME CURVATURE", "#f97316", (panelW+2)*2 + panelW/2],
      ];
      for (const [t, col, xp] of labels) {
        ctx.fillStyle = "rgba(0,0,0,0.7)"; ctx.fillRect(xp-72, 6, 144, 20);
        ctx.fillStyle = col; ctx.fillText(t, xp, 20);
      }
      ctx.fillStyle = "rgba(255,255,255,0.06)";
      ctx.fillRect(panelW, 0, 2, CH);
      ctx.fillRect(panelW*2+2, 0, 2, CH);
    } else {
      drawField(0, view === "darkmatter" ? "dm" : view === "curvature" ? "curve" : view === "matter" ? "matter" : "composite");
      if (showP) {
        for (const p of s.particles) {
          const spd = Math.sqrt(p.vx*p.vx + p.vy*p.vy);
          const heat = Math.min(1, spd / 2.5);
          const sz = Math.max(1.3, Math.min(3.5, p.mass * 1.6));
          const pr = 190 + heat * 65;
          const pg = 170 + heat * 50 - (1-heat)*50;
          const pb = 90 - heat * 70 + (1-heat)*70;
          ctx.fillStyle = `rgba(${Math.floor(pr)},${Math.floor(pg)},${Math.floor(pb)},0.85)`;
          ctx.beginPath(); ctx.arc(p.x, p.y, sz, 0, Math.PI*2); ctx.fill();
          if (p.mass > 1.2) {
            ctx.fillStyle = `rgba(${Math.floor(pr)},${Math.floor(pg)},${Math.floor(pb)},0.1)`;
            ctx.beginPath(); ctx.arc(p.x, p.y, sz*3.5, 0, Math.PI*2); ctx.fill();
          }
        }
      }
    }

    // HUD
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.fillRect(4, canvas.height - 22, 210, 18);
    ctx.fillStyle = "#6b7280";
    ctx.font = "10px monospace";
    ctx.textAlign = "left";
    ctx.fillText(`t=${step}  particles=${s.particles.length}  click to add matter`, 8, canvas.height - 9);
  }, [view, showP, step]);

  useEffect(() => {
    if (!running) { cancelAnimationFrame(animRef.current); return; }
    const loop = () => {
      for (let i = 0; i < 2; i++) simulate();
      render();
      setStep(p => p + 2);
      animRef.current = requestAnimationFrame(loop);
    };
    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, [running, simulate, render]);

  useEffect(() => { render(); }, [view, showP, render]);

  const handleClick = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const sx = canvasRef.current.width / rect.width;
    const sy = canvasRef.current.height / rect.height;
    let cx = (e.clientX - rect.left) * sx;
    let cy = (e.clientY - rect.top) * sy;
    if (view === "sidebyside") cx = cx % (CW + 2);
    if (!stateRef.current) return;
    for (let i = 0; i < 25; i++) {
      const a = Math.random()*Math.PI*2, r = Math.random()*12;
      stateRef.current.particles.push({
        x: cx+Math.cos(a)*r, y: cy+Math.sin(a)*r,
        vx: (Math.random()-.5)*.4, vy: (Math.random()-.5)*.4,
        mass: 0.5+Math.random()*1.5,
      });
    }
    render();
  };

  return (
    <div style={{ minHeight:"100vh", background:"#050510", color:"#d1d5db", fontFamily:"'DM Sans',sans-serif", padding:16 }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet"/>
      <div style={{ maxWidth:1780, margin:"0 auto" }}>
        <h1 style={{
          fontSize:22, fontWeight:700, marginBottom:4,
          background:"linear-gradient(90deg,#fbbf24,#a78bfa,#06b6d4)",
          WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent",
        }}>Primordial Fluid Model</h1>
        <p style={{ fontSize:11, color:"#6b7280", fontFamily:"'DM Mono',monospace", lineHeight:1.5, marginBottom:12 }}>
          Matter clumps → curvature forms → dark medium accumulates in halos/filaments → amplifies gravity → cycle
        </p>

        <div style={{ display:"flex", gap:6, flexWrap:"wrap", marginBottom:8 }}>
          {[["bigbang","Big Bang"],["web","Cosmic Web"],["bullet","Bullet Cluster"],["soup","Random"]].map(([k,l])=>(
            <button key={k} onClick={()=>{setPreset(k);setRunning(false);}} style={{
              padding:"5px 12px", fontSize:11, borderRadius:5, cursor:"pointer",
              border:preset===k?"1px solid #fbbf24":"1px solid #1c1c2e",
              background:preset===k?"rgba(251,191,36,0.1)":"rgba(10,10,18,0.9)",
              color:preset===k?"#fbbf24":"#9ca3af", fontFamily:"'DM Mono',monospace",
            }}>{l}</button>
          ))}
          <span style={{width:8}}/>
          {[["composite","All Layers","#06b6d4"],["sidebyside","Side by Side","#e879f9"],["matter","Matter","#fbbf24"],["darkmatter","Dark Matter","#a78bfa"],["curvature","Curvature","#f97316"]].map(([k,l,c])=>(
            <button key={k} onClick={()=>setView(k)} style={{
              padding:"5px 12px", fontSize:11, borderRadius:5, cursor:"pointer",
              border:view===k?`1px solid ${c}`:"1px solid #1c1c2e",
              background:view===k?`${c}18`:"rgba(10,10,18,0.9)",
              color:view===k?c:"#6b7280", fontFamily:"'DM Mono',monospace",
            }}>{l}</button>
          ))}
        </div>

        <div style={{ display:"flex", gap:14, flexWrap:"wrap" }}>
          <canvas ref={canvasRef} onClick={handleClick} style={{
            borderRadius:6, border:"1px solid #1a1a2e", cursor:"crosshair",
            maxWidth:"100%", height:"auto",
          }}/>

          <div style={{ minWidth:210, maxWidth:260 }}>
            <div style={{ display:"flex", gap:6, marginBottom:10, flexWrap:"wrap" }}>
              <button onClick={()=>setRunning(!running)} style={{
                padding:"8px 18px", fontSize:12, borderRadius:6, border:"none", cursor:"pointer",
                fontWeight:600, background:running?"#dc2626":"#f59e0b", color:"#fff",
              }}>{running?"⏸ Pause":"▶ Run"}</button>
              <button onClick={()=>{simulate();render();setStep(s=>s+1);}} style={{
                padding:"8px 10px", fontSize:11, borderRadius:6, cursor:"pointer",
                border:"1px solid #2a2a3e", background:"rgba(10,10,18,0.9)", color:"#9ca3af",
                fontFamily:"'DM Mono',monospace",
              }}>Step</button>
              <button onClick={()=>{init(preset);setRunning(false);}} style={{
                padding:"8px 10px", fontSize:11, borderRadius:6, cursor:"pointer",
                border:"1px solid #2a2a3e", background:"rgba(10,10,18,0.9)", color:"#9ca3af",
                fontFamily:"'DM Mono',monospace",
              }}>Reset</button>
            </div>
            <label style={{ display:"flex", alignItems:"center", gap:8, marginBottom:10, cursor:"pointer" }}>
              <input type="checkbox" checked={showP} onChange={e=>setShowP(e.target.checked)} style={{accentColor:"#fbbf24"}}/>
              <span style={{fontSize:11,color:"#9ca3af",fontFamily:"'DM Mono',monospace"}}>Show particles</span>
            </label>

            <div style={{fontSize:10,color:"#6b7280",fontFamily:"'DM Mono',monospace",marginBottom:6,textTransform:"uppercase",letterSpacing:1}}>Physics</div>
            {[
              ["gravity","Gravity",0.005,0.1,0.001],
              ["dmGrow","DM Growth",0.01,0.2,0.005],
              ["dmDecay","DM Decay",0.01,0.15,0.005],
              ["dmThreshold","DM Threshold",0,0.1,0.002],
              ["softening","Softening",5,35,1],
              ["damping","Damping",0.98,1.0,0.001],
            ].map(([k,l,mn,mx,st])=>(
              <div key={k} style={{display:"flex",alignItems:"center",gap:6,marginBottom:3}}>
                <span style={{width:85,fontSize:10,color:"#6b7280",fontFamily:"'DM Mono',monospace"}}>{l}</span>
                <input type="range" min={mn} max={mx} step={st} value={params[k]}
                  onChange={e=>setParams(p=>({...p,[k]:parseFloat(e.target.value)}))}
                  style={{flex:1,accentColor:"#a78bfa"}}/>
                <span style={{width:36,fontSize:10,color:"#a78bfa",fontFamily:"'DM Mono',monospace",textAlign:"right"}}>
                  {params[k]<1?params[k].toFixed(3):params[k].toFixed(1)}
                </span>
              </div>
            ))}

            <div style={{
              marginTop:12, padding:10, borderRadius:6,
              background:"rgba(8,8,16,0.9)", border:"1px solid #1a1a2e",
              fontSize:11, fontFamily:"'DM Mono',monospace", lineHeight:1.9,
            }}>
              <div style={{color:"#6b7280",fontSize:10,marginBottom:4,textTransform:"uppercase",letterSpacing:1}}>Key Insight</div>
              <div style={{color:"#9ca3af",fontSize:10,lineHeight:1.6}}>
                Dark matter only accumulates where spacetime curvature exceeds a threshold — it forms <span style={{color:"#a78bfa"}}>halos</span> around matter concentrations and <span style={{color:"#a78bfa"}}>filaments</span> between them, not a uniform wash. This is the primordial fluid responding to geometry.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
