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
    for (let i = 0; i < 700; i++) {
      const a = Math.random() * Math.PI * 2;
      const r = Math.random() * 22 + 1;
      const spd = 0.4 + Math.random() * 1.8;
      ps.push({
        x: CW/2 + Math.cos(a)*r, y: CH/2 + Math.sin(a)*r,
        vx: Math.cos(a)*spd + (Math.random()-.5)*.3,
        vy: Math.sin(a)*spd + (Math.random()-.5)*.3,
        mass: 0.4 + Math.random() * 1.6,
      });
    }
  } else if (scenario === "web") {
    const seeds = [];
    for (let i = 0; i < 22; i++) seeds.push([70+Math.random()*(CW-140), 70+Math.random()*(CH-140)]);
    for (const [sx, sy] of seeds) {
      const n = 14 + Math.floor(Math.random()*22);
      for (let i = 0; i < n; i++) {
        const a = Math.random()*Math.PI*2, r = Math.random()*22;
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
        vx: 0.8+(Math.random()-.5)*.2, vy: (Math.random()-.5)*.25,
        mass: 0.5+Math.random()*1.5,
      });
    }
    for (let i = 0; i < 350; i++) {
      const a = Math.random()*Math.PI*2, r = Math.random()*35;
      ps.push({
        x: CW*.75+Math.cos(a)*r, y: CH*.5+Math.sin(a)*r,
        vx: -0.8+(Math.random()-.5)*.2, vy: (Math.random()-.5)*.25,
        mass: 0.5+Math.random()*1.5,
      });
    }
  } else {
    for (let i = 0; i < 500; i++) {
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
  const [view, setView] = useState("sidebyside");
  const [preset, setPreset] = useState("web");
  const [showP, setShowP] = useState(true);
  const [params, setParams] = useState({
    gravity: 0.04,
    dmGrow: 0.25,
    dmDecay: 0.03,
    softening: 14,
    damping: 0.997,
  });

  const init = useCallback((sc) => {
    stateRef.current = {
      particles: initParticles(sc),
      dm: createField(0),
      dmPeak: createField(0), // tracks peak DM ever reached — shows filament history
      curve: createField(0),
      mField: createField(0),
    };
    setStep(0);
  }, []);

  useEffect(() => { init(preset); }, [preset, init]);

  const simulate = useCallback(() => {
    const s = stateRef.current;
    if (!s) return;
    const { particles, dm, dmPeak, curve, mField } = s;
    const { gravity, dmGrow, dmDecay, softening, damping } = params;
    const soft2 = softening * softening;

    // Deposit matter to grid — gaussian splat for smoothness
    mField.fill(0);
    for (const p of particles) {
      const gx0 = Math.floor(p.x / CELL);
      const gy0 = Math.floor(p.y / CELL);
      for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          const gx = gx0+dx, gy = gy0+dy;
          if (gx<0||gx>=GRID||gy<0||gy>=GRID) continue;
          const dist = Math.sqrt(dx*dx+dy*dy);
          mField[gi(gx,gy)] += p.mass * Math.exp(-dist*dist/1.5);
        }
      }
    }

    // Curvature = matter + dark matter contribution
    for (let i = 0; i < GRID*GRID; i++) {
      curve[i] = (mField[i] + dm[i] * 0.35) * gravity;
    }

    // Dark matter evolution:
    // - Grows proportional to local matter density (not just curvature)
    // - Mild diffusion to form connecting filaments
    // - Slow constant decay so it fades where matter leaves
    const newDM = new Float32Array(dm);
    for (let gy = 1; gy < GRID-1; gy++) {
      for (let gx = 1; gx < GRID-1; gx++) {
        const i = gi(gx,gy);
        const mHere = mField[i];

        // Growth: directly proportional to matter density
        // This is the key: DM accumulates WHERE MATTER IS
        const growth = mHere * dmGrow * 0.1;

        // Mild diffusion — connects nearby DM patches into filaments
        const l=gi(gx-1,gy), r=gi(gx+1,gy), u=gi(gx,gy-1), d=gi(gx,gy+1);
        const lap = (dm[l]+dm[r]+dm[u]+dm[d])*0.25 - dm[i];

        // Decay: slow constant rate
        const decay = dm[i] * dmDecay * 0.08;

        newDM[i] = dm[i] + growth + lap * 0.035 - decay;
        newDM[i] = Math.max(0, Math.min(6, newDM[i]));

        // Track peak — shows where DM has ever been strong (filament memory)
        if (newDM[i] > dmPeak[i]) dmPeak[i] = newDM[i];
        else dmPeak[i] *= 0.999; // very slow fade of history
      }
    }
    s.dm = newDM;

    // Grid forces
    const fx = createField(0), fy = createField(0);
    for (let gy = 1; gy < GRID-1; gy++) {
      for (let gx = 1; gx < GRID-1; gx++) {
        const i = gi(gx,gy);
        const pR = mField[gi(gx+1,gy)] + newDM[gi(gx+1,gy)]*0.4;
        const pL = mField[gi(gx-1,gy)] + newDM[gi(gx-1,gy)]*0.4;
        const pD = mField[gi(gx,gy+1)] + newDM[gi(gx,gy+1)]*0.4;
        const pU = mField[gi(gx,gy-1)] + newDM[gi(gx,gy-1)]*0.4;
        fx[i] = (pR-pL)*0.5*gravity;
        fy[i] = (pD-pU)*0.5*gravity;
      }
    }

    // Move particles
    for (const p of particles) {
      const gx = Math.floor(p.x/CELL), gy = Math.floor(p.y/CELL);
      if (gx>=1&&gx<GRID-1&&gy>=1&&gy<GRID-1) {
        const i = gi(gx,gy);
        p.vx += fx[i]*0.9; p.vy += fy[i]*0.9;
      }
      const sN = Math.min(particles.length,15);
      for (let k=0;k<sN;k++) {
        const o = particles[Math.floor(Math.random()*particles.length)];
        if (o===p) continue;
        const dx=o.x-p.x, dy=o.y-p.y;
        const d2=dx*dx+dy*dy+soft2;
        const f=gravity*o.mass/(d2*Math.sqrt(d2))*10;
        p.vx+=dx*f; p.vy+=dy*f;
      }
      p.vx*=damping; p.vy*=damping;
      p.x+=p.vx; p.y+=p.vy;
      if(p.x<0)p.x+=CW; if(p.x>=CW)p.x-=CW;
      if(p.y<0)p.y+=CH; if(p.y-=CH)p.y-=CH;
      if(p.y<0)p.y+=CH; if(p.y>=CH)p.y-=CH;
    }
  }, [params]);

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    const s = stateRef.current;
    if (!canvas||!s) return;
    const ctx = canvas.getContext("2d");
    const isSBS = view==="sidebyside";

    canvas.width = isSBS ? CW*3+4 : CW;
    canvas.height = CH;
    ctx.fillStyle = "#04040a";
    ctx.fillRect(0,0,canvas.width,canvas.height);

    // Find maxes for normalization
    let maxM=0.1, maxDM=0.01, maxC=0.01, maxPeak=0.01;
    for (let i=0;i<GRID*GRID;i++) {
      if(s.mField[i]>maxM) maxM=s.mField[i];
      if(s.dm[i]>maxDM) maxDM=s.dm[i];
      if(s.curve[i]>maxC) maxC=s.curve[i];
      if(s.dmPeak[i]>maxPeak) maxPeak=s.dmPeak[i];
    }

    function drawPanel(offX, mode) {
      const imgData = ctx.createImageData(GRID, GRID);
      const d = imgData.data;

      for (let gy=0; gy<GRID; gy++) {
        for (let gx=0; gx<GRID; gx++) {
          const i = gi(gx,gy);
          const pi = (gy*GRID+gx)*4;
          const m = s.mField[i];
          const dm2 = s.dm[i];
          const pk = s.dmPeak[i];
          const cv = s.curve[i];
          let r=0,g=0,b=0;

          if (mode==="matter") {
            // Log scale for better visibility of structure
            const v = Math.log(1 + m*10) / Math.log(1 + maxM*10);
            r = v*255; g = v*210; b = v*80;
          } else if (mode==="dm") {
            // Current DM = bright violet, Peak trail = dim blue
            const vCur = maxDM > 0.01 ? Math.pow(dm2/maxDM, 0.4) : 0;
            const vPk = maxPeak > 0.01 ? Math.pow(pk/maxPeak, 0.5) * 0.25 : 0;
            const v = Math.max(vCur, vPk);
            const isCurrent = vCur >= vPk * 0.5;
            if (isCurrent) {
              // Active DM: bright violet-pink
              r = vCur*200 + vCur*vCur*55;
              g = vCur*50 + vCur*vCur*30;
              b = vCur*255;
            } else {
              // Trail: dim blue-purple
              r = vPk*40;
              g = vPk*20;
              b = vPk*120;
            }
          } else if (mode==="curve") {
            const v = maxC > 0.001 ? Math.pow(cv/maxC, 0.45) : 0;
            if (v<0.25) { r=v*4*180; g=0; b=v*4*30; }
            else if (v<0.55) { r=180+(v-.25)*3.3*75; g=(v-.25)*3.3*140; b=30-(v-.25)*3.3*30; }
            else { r=255; g=140+(v-.55)*2.2*115; b=(v-.55)*2.2*200; }
          } else {
            // Composite: violet DM base + amber curvature + white matter peaks
            const dmV = maxDM>0.01 ? Math.pow(dm2/maxDM, 0.45) : 0;
            const cvV = maxC>0.01 ? Math.pow(cv/maxC, 0.45) : 0;
            const mV = Math.log(1+m*5)/Math.log(1+maxM*5);
            r = dmV*60 + cvV*130 + mV*80;
            g = dmV*20 + cvV*65 + mV*50;
            b = dmV*180 + cvV*15 + mV*30;
          }

          d[pi] = Math.min(255,Math.max(0,Math.floor(r)));
          d[pi+1] = Math.min(255,Math.max(0,Math.floor(g)));
          d[pi+2] = Math.min(255,Math.max(0,Math.floor(b)));
          d[pi+3] = 255;
        }
      }

      // Scale up to canvas
      const tmpCanvas = document.createElement("canvas");
      tmpCanvas.width = GRID; tmpCanvas.height = GRID;
      tmpCanvas.getContext("2d").putImageData(imgData, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(tmpCanvas, 0, 0, GRID, GRID, offX, 0, CW, CH);
    }

    if (isSBS) {
      drawPanel(0, "matter");
      drawPanel(CW+2, "dm");
      drawPanel((CW+2)*2, "curve");

      // Particles on matter panel
      if (showP) {
        for (const p of s.particles) {
          const sz = Math.max(1.2,Math.min(2.8,p.mass*1.3));
          ctx.fillStyle = "rgba(255,245,200,0.8)";
          ctx.beginPath(); ctx.arc(p.x,p.y,sz,0,Math.PI*2); ctx.fill();
        }
      }

      // Labels
      ctx.font = "bold 11px monospace"; ctx.textAlign = "center";
      [[`MATTER (${s.particles.length})`, "#fbbf24", CW/2],
       ["DARK MATTER MEDIUM", "#c4b5fd", CW+2+CW/2],
       ["SPACETIME CURVATURE", "#fb923c", (CW+2)*2+CW/2]
      ].forEach(([t,c,x])=>{
        ctx.fillStyle="rgba(0,0,0,0.7)"; ctx.fillRect(x-78,6,156,20);
        ctx.fillStyle=c; ctx.fillText(t,x,20);
      });
      ctx.fillStyle="rgba(255,255,255,0.06)";
      ctx.fillRect(CW,0,2,CH); ctx.fillRect(CW*2+2,0,2,CH);
    } else {
      drawPanel(0, view==="darkmatter"?"dm":view==="curvature"?"curve":view==="matter"?"matter":"composite");
      if (showP) {
        for (const p of s.particles) {
          const spd = Math.sqrt(p.vx*p.vx+p.vy*p.vy);
          const heat = Math.min(1,spd/2.5);
          const sz = Math.max(1.3,Math.min(3.5,p.mass*1.6));
          ctx.fillStyle = `rgba(${190+heat*65},${170-(!heat)*50+heat*50},${90+(1-heat)*70-heat*70},0.85)`;
          ctx.beginPath(); ctx.arc(p.x,p.y,sz,0,Math.PI*2); ctx.fill();
          if(p.mass>1.2){
            ctx.fillStyle=`rgba(255,220,120,0.08)`;
            ctx.beginPath(); ctx.arc(p.x,p.y,sz*4,0,Math.PI*2); ctx.fill();
          }
        }
      }
    }

    ctx.fillStyle="rgba(0,0,0,0.6)"; ctx.fillRect(4,canvas.height-20,170,16);
    ctx.fillStyle="#6b7280"; ctx.font="9px monospace"; ctx.textAlign="left";
    ctx.fillText(`t=${step}  click to add matter`,8,canvas.height-8);
  }, [view, showP, step]);

  useEffect(()=>{
    if(!running){cancelAnimationFrame(animRef.current);return;}
    const loop=()=>{
      for(let i=0;i<2;i++)simulate();
      render();
      setStep(p=>p+2);
      animRef.current=requestAnimationFrame(loop);
    };
    animRef.current=requestAnimationFrame(loop);
    return()=>cancelAnimationFrame(animRef.current);
  },[running,simulate,render]);

  useEffect(()=>{render();},[view,showP,render]);

  const handleClick=(e)=>{
    const rect=canvasRef.current.getBoundingClientRect();
    const sx=canvasRef.current.width/rect.width;
    let cx=(e.clientX-rect.left)*sx;
    let cy=(e.clientY-rect.top)*(canvasRef.current.height/rect.height);
    if(view==="sidebyside")cx=cx%(CW+2);
    if(!stateRef.current)return;
    for(let i=0;i<25;i++){
      const a=Math.random()*Math.PI*2,r=Math.random()*12;
      stateRef.current.particles.push({
        x:cx+Math.cos(a)*r, y:cy+Math.sin(a)*r,
        vx:(Math.random()-.5)*.4, vy:(Math.random()-.5)*.4,
        mass:0.5+Math.random()*1.5,
      });
    }
    render();
  };

  return (
    <div style={{minHeight:"100vh",background:"#04040a",color:"#d1d5db",fontFamily:"'DM Sans',sans-serif",padding:16}}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet"/>
      <div style={{maxWidth:1780,margin:"0 auto"}}>
        <h1 style={{
          fontSize:22,fontWeight:700,marginBottom:4,
          background:"linear-gradient(90deg,#fbbf24,#c4b5fd,#06b6d4)",
          WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent",
        }}>Primordial Fluid Model</h1>
        <p style={{fontSize:11,color:"#6b7280",fontFamily:"'DM Mono',monospace",lineHeight:1.5,marginBottom:10}}>
          Matter clumps → curvature forms → <span style={{color:"#c4b5fd"}}>dark medium accumulates as halos + filaments</span> → amplifies gravity → cycle
        </p>

        <div style={{display:"flex",gap:6,flexWrap:"wrap",marginBottom:8}}>
          {[["bigbang","Big Bang"],["web","Cosmic Web"],["bullet","Bullet Cluster"],["soup","Random"]].map(([k,l])=>(
            <button key={k} onClick={()=>{setPreset(k);setRunning(false);}} style={{
              padding:"5px 12px",fontSize:11,borderRadius:5,cursor:"pointer",
              border:preset===k?"1px solid #fbbf24":"1px solid #1c1c2e",
              background:preset===k?"rgba(251,191,36,0.1)":"rgba(8,8,16,0.9)",
              color:preset===k?"#fbbf24":"#9ca3af",fontFamily:"'DM Mono',monospace",
            }}>{l}</button>
          ))}
          <span style={{width:8}}/>
          {[["composite","All Layers","#06b6d4"],["sidebyside","Side by Side","#e879f9"],["matter","Matter","#fbbf24"],["darkmatter","Dark Matter","#c4b5fd"],["curvature","Curvature","#fb923c"]].map(([k,l,c])=>(
            <button key={k} onClick={()=>setView(k)} style={{
              padding:"5px 12px",fontSize:11,borderRadius:5,cursor:"pointer",
              border:view===k?`1px solid ${c}`:"1px solid #1c1c2e",
              background:view===k?`${c}18`:"rgba(8,8,16,0.9)",
              color:view===k?c:"#6b7280",fontFamily:"'DM Mono',monospace",
            }}>{l}</button>
          ))}
        </div>

        <div style={{display:"flex",gap:14,flexWrap:"wrap"}}>
          <canvas ref={canvasRef} onClick={handleClick} style={{
            borderRadius:6,border:"1px solid #1a1a2e",cursor:"crosshair",
            maxWidth:"100%",height:"auto",
          }}/>

          <div style={{minWidth:210,maxWidth:260}}>
            <div style={{display:"flex",gap:6,marginBottom:10,flexWrap:"wrap"}}>
              <button onClick={()=>setRunning(!running)} style={{
                padding:"8px 18px",fontSize:12,borderRadius:6,border:"none",cursor:"pointer",
                fontWeight:600,background:running?"#dc2626":"#f59e0b",color:"#fff",
              }}>{running?"⏸ Pause":"▶ Run"}</button>
              <button onClick={()=>{simulate();render();setStep(s2=>s2+1);}} style={{
                padding:"8px 10px",fontSize:11,borderRadius:6,cursor:"pointer",
                border:"1px solid #2a2a3e",background:"rgba(8,8,16,0.9)",color:"#9ca3af",
                fontFamily:"'DM Mono',monospace",
              }}>Step</button>
              <button onClick={()=>{init(preset);setRunning(false);}} style={{
                padding:"8px 10px",fontSize:11,borderRadius:6,cursor:"pointer",
                border:"1px solid #2a2a3e",background:"rgba(8,8,16,0.9)",color:"#9ca3af",
                fontFamily:"'DM Mono',monospace",
              }}>Reset</button>
            </div>
            <label style={{display:"flex",alignItems:"center",gap:8,marginBottom:10,cursor:"pointer"}}>
              <input type="checkbox" checked={showP} onChange={e=>setShowP(e.target.checked)} style={{accentColor:"#fbbf24"}}/>
              <span style={{fontSize:11,color:"#9ca3af",fontFamily:"'DM Mono',monospace"}}>Show particles</span>
            </label>

            <div style={{fontSize:10,color:"#6b7280",fontFamily:"'DM Mono',monospace",marginBottom:6,textTransform:"uppercase",letterSpacing:1}}>Physics</div>
            {[
              ["gravity","Gravity",0.005,0.1,0.001],
              ["dmGrow","DM Growth",0.05,0.5,0.01],
              ["dmDecay","DM Decay",0.005,0.1,0.005],
              ["softening","Softening",5,30,1],
              ["damping","Damping",0.99,1.0,0.001],
            ].map(([k,l,mn,mx,st])=>(
              <div key={k} style={{display:"flex",alignItems:"center",gap:6,marginBottom:3}}>
                <span style={{width:80,fontSize:10,color:"#6b7280",fontFamily:"'DM Mono',monospace"}}>{l}</span>
                <input type="range" min={mn} max={mx} step={st} value={params[k]}
                  onChange={e=>setParams(p=>({...p,[k]:parseFloat(e.target.value)}))}
                  style={{flex:1,accentColor:"#c4b5fd"}}/>
                <span style={{width:36,fontSize:10,color:"#c4b5fd",fontFamily:"'DM Mono',monospace",textAlign:"right"}}>
                  {params[k]<1?params[k].toFixed(3):params[k].toFixed(1)}
                </span>
              </div>
            ))}

            <div style={{
              marginTop:14,padding:10,borderRadius:6,
              background:"rgba(6,6,14,0.9)",border:"1px solid #1a1a2e",
              fontSize:10,fontFamily:"'DM Mono',monospace",lineHeight:1.7,color:"#9ca3af",
            }}>
              <div style={{color:"#6b7280",marginBottom:4,textTransform:"uppercase",letterSpacing:1}}>What to watch</div>
              <div><span style={{color:"#fbbf24"}}>Left:</span> matter particles clumping</div>
              <div><span style={{color:"#c4b5fd"}}>Center:</span> DM halos forming around clusters</div>
              <div><span style={{color:"#c4b5fd"}}>       </span> dim trails = filament history</div>
              <div><span style={{color:"#fb923c"}}>Right:</span> curvature tracks total mass</div>
              <div style={{marginTop:6,color:"#4b5563"}}>
                The DM medium isn't particles — it's<br/>
                the primordial fluid responding to where<br/>
                matter warps spacetime geometry
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
