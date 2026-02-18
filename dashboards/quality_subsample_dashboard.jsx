import { useState, useMemo } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine, Legend, LineChart, Line, ScatterChart, Scatter, ComposedChart } from "recharts";

// === DATA FROM PIPELINE OUTPUT ===
const COMPARISON = {
  full: { n_gal: 98, n_pts: 2540, sigma: 0.1198, dense_sigma: 0.1122, field_sigma: 0.1223, delta: 0.0101, dense_gal: 29, field_gal: 69 },
  quality: { n_gal: 43, n_pts: 1063, sigma: 0.1109, dense_sigma: 0.1122, field_sigma: 0.1081, delta: -0.0041, dense_gal: 29, field_gal: 14 },
  tier1: { dense_sigma: 0.1233, field_sigma: 0.1081, delta: -0.0152, p: 0.239, dense_gal: 15, field_gal: 14 },
};

const METHOD_SCATTER = [
  { method: "Hubble flow", fD: 1, sigma: 0.1268, n: 1573, color: "#ef4444" },
  { method: "TRGB", fD: 2, sigma: 0.1202, n: 620, color: "#22c55e" },
  { method: "UMa cluster", fD: 4, sigma: 0.0823, n: 200, color: "#f59e0b" },
  { method: "Cepheid", fD: 3, sigma: 0.0777, n: 129, color: "#3b82f6" },
  { method: "SNe Ia", fD: 5, sigma: 0.0271, n: 18, color: "#a855f7" },
];

const BINNED_ENV = [
  { bin: "-12.0", dense: 0.0972, field: 0.0991, nd: 89, nf: 26, p: 0.627 },
  { bin: "-11.0", dense: 0.0894, field: 0.1158, nd: 362, nf: 212, p: 0.971 },
  { bin: "-10.0", dense: 0.1523, field: 0.0921, nd: 184, nf: 146, p: 0.092 },
];

const SKEW_PROFILE = [
  { c: -12.18, n: 27, std: 0.1088, sk: -0.44, qsk: 0.072, kurt: -0.6 },
  { c: -11.54, n: 283, std: 0.1075, sk: -1.20, qsk: 0.091, kurt: 5.7 },
  { c: -10.89, n: 344, std: 0.0976, sk: -2.11, qsk: -0.192, kurt: 8.3 },
  { c: -10.25, n: 308, std: 0.0932, sk: -4.07, qsk: -0.157, kurt: 27.3 },
  { c: -9.61, n: 84, std: 0.1856, sk: -4.45, qsk: 0.008, kurt: 21.8 },
];

const GALAXY_LIST = [
  // Dense environment - selected examples
  { name: "NGC2403", env: "dense", group: "M81", D: 3.16, method: "TRGB", sigma: 0.048, Vf: 131.2 },
  { name: "NGC0300", env: "dense", group: "Sculptor", D: 2.08, method: "TRGB", sigma: 0.065, Vf: 93.3 },
  { name: "NGC3741", env: "dense", group: "CVnI", D: 3.21, method: "TRGB", sigma: 0.037, Vf: 50.1 },
  { name: "NGC3917", env: "dense", group: "UMa", D: 18.0, method: "UMa", sigma: 0.044, Vf: 135.9 },
  { name: "NGC3949", env: "dense", group: "UMa", D: 18.0, method: "UMa", sigma: 0.036, Vf: 168.7 },
  { name: "NGC3972", env: "dense", group: "UMa", D: 18.0, method: "Ceph", sigma: 0.035, Vf: 132.7 },
  // Field environment - selected examples
  { name: "NGC6503", env: "field", group: "—", D: 6.26, method: "TRGB", sigma: 0.053, Vf: 116.3 },
  { name: "NGC2841", env: "field", group: "—", D: 14.1, method: "Ceph", sigma: 0.032, Vf: 284.8 },
  { name: "NGC3198", env: "field", group: "—", D: 13.8, method: "Ceph", sigma: 0.028, Vf: 150.1 },
  { name: "NGC7331", env: "field", group: "—", D: 14.7, method: "Ceph", sigma: 0.046, Vf: 239.0 },
  { name: "NGC4559", env: "field", group: "—", D: 9.0, method: "TRGB*", sigma: 0.078, Vf: 121.2 },
  { name: "DDO161", env: "field", group: "—", D: 7.5, method: "TRGB*", sigma: 0.101, Vf: 66.3 },
];

const DELTA_COMPARISON = [
  { label: "Full sample\n(all methods)", delta: 0.0101, p: 0.998, color: "#22c55e", note: "97 Hubble-flow gal included" },
  { label: "Quality sub\n(+UMa)", delta: -0.0041, p: 0.449, color: "#94a3b8", note: "43 gal, equalized dist" },
  { label: "Tier 1 only\n(TRGB/Ceph)", delta: -0.0152, p: 0.239, color: "#ef4444", note: "29 gal, strictest cut" },
];

const S = ({ children, c = "#e2e8f0" }) => <span style={{ color: c, fontWeight: 600 }}>{children}</span>;
const Tag = ({ children, bg = "#1e293b", fg = "#94a3b8" }) => (
  <span style={{ background: bg, color: fg, padding: "2px 8px", borderRadius: 4, fontSize: 11, fontWeight: 500 }}>{children}</span>
);

export default function QualitySubsampleDashboard() {
  const [view, setView] = useState("verdict");

  const compData = [
    { label: "Full sample", σ_dense: COMPARISON.full.dense_sigma, σ_field: COMPARISON.full.field_sigma },
    { label: "Quality sub", σ_dense: COMPARISON.quality.dense_sigma, σ_field: COMPARISON.quality.field_sigma },
    { label: "Tier 1 only", σ_dense: COMPARISON.tier1.dense_sigma, σ_field: COMPARISON.tier1.field_sigma },
  ];

  return (
    <div style={{ fontFamily: "'Newsreader', 'Iowan Old Style', 'Palatino Linotype', serif", background: "#08090d", color: "#d1d5db", minHeight: "100vh", padding: "24px 20px" }}>
      <div style={{ maxWidth: 960, margin: "0 auto" }}>

        {/* Header */}
        <div style={{ borderBottom: "2px solid #dc2626", paddingBottom: 16, marginBottom: 24 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
            <span style={{ background: "#dc2626", color: "#fff", fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 2, letterSpacing: 1.5, textTransform: "uppercase", fontFamily: "system-ui" }}>Null Result</span>
            <span style={{ color: "#6b7280", fontSize: 11, fontFamily: "system-ui" }}>Distance-Equalized Reanalysis</span>
          </div>
          <h1 style={{ fontSize: 28, fontWeight: 700, color: "#f9fafb", margin: "6px 0 0", lineHeight: 1.15, letterSpacing: "-0.5px" }}>
            Environmental RAR Signal Vanishes<br/>
            <span style={{ color: "#dc2626" }}>With Homogeneous Distances</span>
          </h1>
          <p style={{ color: "#9ca3af", fontSize: 14, margin: "8px 0 0", lineHeight: 1.5 }}>
            43 galaxies · TRGB / Cepheid / SNe distances only · ≲10% precision · Per-galaxy M/L
          </p>
        </div>

        {/* Key metrics */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 24 }}>
          {[
            { label: "Full-sample Δσ", val: "+0.010", sub: "P = 99.8%", color: "#22c55e" },
            { label: "Quality-sub Δσ", val: "−0.004", sub: "P = 44.9%", color: "#dc2626" },
            { label: "Signal remaining", val: "0%", sub: "Reversed sign", color: "#dc2626" },
            { label: "Hubble-flow σ", val: "0.127", sub: "vs 0.082 UMa", color: "#f59e0b" },
          ].map((m, i) => (
            <div key={i} style={{ background: "#111318", border: "1px solid #1f2937", borderRadius: 8, padding: "12px 14px", textAlign: "center" }}>
              <div style={{ fontSize: 10, color: "#6b7280", fontFamily: "system-ui", textTransform: "uppercase", letterSpacing: 1 }}>{m.label}</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: m.color, margin: "4px 0 2px", fontFamily: "'JetBrains Mono', monospace" }}>{m.val}</div>
              <div style={{ fontSize: 11, color: "#9ca3af", fontFamily: "system-ui" }}>{m.sub}</div>
            </div>
          ))}
        </div>

        {/* Tab navigation */}
        <div style={{ display: "flex", gap: 4, marginBottom: 20, borderBottom: "1px solid #1f2937", paddingBottom: 1 }}>
          {[
            { id: "verdict", label: "The Verdict" },
            { id: "methods", label: "Distance Methods" },
            { id: "env", label: "Environmental Test" },
            { id: "sample", label: "Galaxy Sample" },
            { id: "physics", label: "What Survives" },
          ].map(t => (
            <button key={t.id} onClick={() => setView(t.id)} style={{
              padding: "8px 16px", fontSize: 12, fontWeight: view === t.id ? 600 : 400,
              fontFamily: "system-ui",
              background: "transparent",
              borderBottom: view === t.id ? "2px solid #dc2626" : "2px solid transparent",
              border: "none", borderBottomWidth: 2, borderBottomStyle: "solid",
              borderBottomColor: view === t.id ? "#dc2626" : "transparent",
              color: view === t.id ? "#f9fafb" : "#6b7280", cursor: "pointer",
              marginBottom: -1
            }}>
              {t.label}
            </button>
          ))}
        </div>

        {/* === VERDICT === */}
        {view === "verdict" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ background: "#111318", borderRadius: 10, padding: 20, border: "1px solid #dc262633" }}>
              <div style={{ fontSize: 16, fontWeight: 600, color: "#f9fafb", marginBottom: 12, lineHeight: 1.3 }}>
                The 99.8% environmental signal was a distance-precision artifact
              </div>
              <div style={{ fontSize: 14, color: "#d1d5db", lineHeight: 1.7 }}>
                In the full SPARC sample, field galaxies showed 0.010 dex more RAR scatter than cluster/group galaxies at 99.8% bootstrap confidence.
                This looked like strong evidence for the fluid dark matter prediction that external DM pressure confines halo fluctuations.
                <br/><br/>
                But field galaxies predominantly used Hubble-flow distances (σ<sub>dist</sub> ≈ 20–30%), while cluster galaxies used group membership or TRGB
                (σ<sub>dist</sub> ≈ 5–10%). When we restrict to only galaxies with <S c="#f59e0b">TRGB, Cepheid, or SNe distances</S> — equalizing
                measurement precision across environments — <S c="#dc2626">the signal vanishes and the sign reverses</S>.
                Dense environments now show <em>slightly more</em> scatter, though the difference is not significant (Levene p = 0.89).
              </div>
            </div>

            {/* Δσ comparison chart */}
            <div style={{ background: "#111318", borderRadius: 10, padding: "16px 20px", border: "1px solid #1f2937" }}>
              <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 16 }}>Δσ (field − dense) Across Distance Cuts</div>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={DELTA_COMPARISON} margin={{ top: 10, right: 30, bottom: 5, left: 10 }} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis type="number" domain={[-0.025, 0.015]}
                    tick={{ fill: "#9ca3af", fontSize: 11, fontFamily: "system-ui" }} stroke="#374151" />
                  <YAxis type="category" dataKey="label" width={100}
                    tick={{ fill: "#d1d5db", fontSize: 11, fontFamily: "system-ui" }} stroke="#374151" />
                  <ReferenceLine x={0} stroke="#6b7280" strokeDasharray="4 4" />
                  <Tooltip content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const d = payload[0].payload;
                    return (
                      <div style={{ background: "#1f2937", border: "1px solid #374151", borderRadius: 6, padding: "8px 12px", fontSize: 12, fontFamily: "system-ui" }}>
                        <div style={{ fontWeight: 600, color: "#f9fafb" }}>{d.label.replace('\n', ' ')}</div>
                        <div>Δσ = {d.delta > 0 ? '+' : ''}{d.delta.toFixed(4)} dex</div>
                        <div>P(field {'>'} dense) = {(d.p * 100).toFixed(1)}%</div>
                        <div style={{ color: "#9ca3af", fontSize: 11 }}>{d.note}</div>
                      </div>
                    );
                  }} />
                  <Bar dataKey="delta" barSize={28} radius={[0, 4, 4, 0]}>
                    {DELTA_COMPARISON.map((d, i) => (
                      <Cell key={i} fill={d.color} fillOpacity={0.7} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div style={{ fontSize: 12, color: "#9ca3af", marginTop: 8, lineHeight: 1.5, fontFamily: "system-ui" }}>
                Green bar: original result with all distance methods. Gray/red: after restricting to high-precision distances.
                The positive Δσ (field {'>'} dense) requires Hubble-flow galaxies to be present.
              </div>
            </div>

            {/* What went wrong */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
              <div style={{ background: "#111318", borderRadius: 10, padding: 16, border: "1px solid #1f2937" }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: "#dc2626", marginBottom: 8, fontFamily: "system-ui" }}>The Problem</div>
                <div style={{ fontSize: 13, color: "#d1d5db", lineHeight: 1.6 }}>
                  97 of 175 SPARC galaxies use Hubble-flow distances with ≈25% uncertainty.
                  These are overwhelmingly field galaxies (by definition — they're identified via redshift).
                  Distance errors propagate into the RAR as σ<sub>RAR</sub> ∝ σ<sub>D</sub>, inflating apparent
                  scatter for field galaxies relative to cluster galaxies that use tighter distance methods.
                </div>
              </div>
              <div style={{ background: "#111318", borderRadius: 10, padding: 16, border: "1px solid #1f2937" }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: "#f59e0b", marginBottom: 8, fontFamily: "system-ui" }}>The Fix</div>
                <div style={{ fontSize: 13, color: "#d1d5db", lineHeight: 1.6 }}>
                  We restricted to 43 galaxies with TRGB, Cepheid, or SNe distances (all ≲10% precision),
                  plus UMa cluster members (10%). Five Hubble-flow galaxies were upgraded via NED-D cross-match.
                  This equalizes distance quality across both environments — and eliminates the signal.
                </div>
              </div>
            </div>
          </div>
        )}

        {/* === DISTANCE METHODS === */}
        {view === "methods" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ background: "#111318", borderRadius: 10, padding: "16px 20px", border: "1px solid #1f2937" }}>
              <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 14 }}>RAR Scatter by Distance Method</div>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={METHOD_SCATTER} margin={{ top: 10, right: 30, bottom: 20, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="method" tick={{ fill: "#9ca3af", fontSize: 11, fontFamily: "system-ui" }} stroke="#374151" />
                  <YAxis domain={[0, 0.14]} tick={{ fill: "#9ca3af", fontSize: 11 }} stroke="#374151"
                    label={{ value: "σ (dex)", angle: -90, position: "insideLeft", style: { fill: "#9ca3af", fontSize: 11 } }} />
                  <Tooltip content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const d = payload[0].payload;
                    return (
                      <div style={{ background: "#1f2937", border: "1px solid #374151", borderRadius: 6, padding: "8px 12px", fontSize: 12, fontFamily: "system-ui" }}>
                        <div style={{ fontWeight: 600 }}>{d.method} (fD={d.fD})</div>
                        <div>σ = {d.sigma.toFixed(4)} dex</div>
                        <div>N = {d.n} data points</div>
                      </div>
                    );
                  }} />
                  <Bar dataKey="sigma" barSize={48} radius={[4, 4, 0, 0]}>
                    {METHOD_SCATTER.map((d, i) => (
                      <Cell key={i} fill={d.color} fillOpacity={0.7} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div style={{ fontSize: 12, color: "#9ca3af", marginTop: 8, lineHeight: 1.5, fontFamily: "system-ui" }}>
                Hubble flow galaxies (red) show 54% more scatter than Cepheid galaxies and 42% more than UMa members.
                This distance-precision gradient maps directly onto the environment classification, creating the spurious signal.
              </div>
            </div>

            <div style={{ background: "#111318", borderRadius: 10, padding: 16, border: "1px solid #1f2937" }}>
              <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 10 }}>The Confound: Distance Method ↔ Environment</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 10, fontFamily: "system-ui" }}>
                <div style={{ background: "#0d1017", borderRadius: 8, padding: 14 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: "#3b82f6", marginBottom: 8 }}>Field Galaxies (full sample)</div>
                  <div style={{ fontSize: 12, color: "#d1d5db" }}>
                    69 galaxies total<br/>
                    55 use Hubble flow (80%) → σ<sub>dist</sub> ≈ 25%<br/>
                    14 use TRGB/Cepheid (20%) → σ<sub>dist</sub> ≈ 5%
                  </div>
                </div>
                <div style={{ background: "#0d1017", borderRadius: 8, padding: 14 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: "#dc2626", marginBottom: 8 }}>Dense Environment (full sample)</div>
                  <div style={{ fontSize: 12, color: "#d1d5db" }}>
                    29 galaxies total<br/>
                    0 use Hubble flow (0%) → none<br/>
                    15 TRGB, 14 UMa (100%) → σ<sub>dist</sub> ≈ 5–10%
                  </div>
                </div>
              </div>
              <div style={{ fontSize: 12, color: "#f59e0b", marginTop: 12, lineHeight: 1.5, fontFamily: "system-ui" }}>
                ⚠ Zero Hubble-flow galaxies are in the "dense" group. 80% of field galaxies use Hubble flow.
                This perfect correlation between distance method and environment classification makes the environmental
                signal impossible to disentangle from the distance-precision systematic without homogeneous distances.
              </div>
            </div>

            {/* NED-D upgrade results */}
            <div style={{ background: "#111318", borderRadius: 10, padding: 16, border: "1px solid #22c55e33" }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#22c55e", marginBottom: 8 }}>NED-D Cross-Match: 5 Upgrades Found</div>
              <div style={{ fontSize: 13, color: "#d1d5db", lineHeight: 1.6 }}>
                Queried all 175 SPARC galaxies against the NED Extragalactic Distance Database. Five Hubble-flow galaxies
                now have published TRGB measurements: DDO 161, NGC 4559, UGC 5721, UGC 7323, and UGC 9992.
                These were added to the quality subsample. The remaining 92 Hubble-flow galaxies lack any TRGB or Cepheid
                measurement in the literature as of February 2026.
              </div>
            </div>
          </div>
        )}

        {/* === ENVIRONMENTAL TEST === */}
        {view === "env" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ background: "#111318", borderRadius: 10, padding: "16px 20px", border: "1px solid #1f2937" }}>
              <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 14 }}>Environmental Scatter: Full vs Quality Subsample</div>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={compData} margin={{ top: 10, right: 30, bottom: 20, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="label" tick={{ fill: "#9ca3af", fontSize: 11, fontFamily: "system-ui" }} stroke="#374151" />
                  <YAxis domain={[0.06, 0.14]} tick={{ fill: "#9ca3af", fontSize: 11 }} stroke="#374151"
                    label={{ value: "σ (dex)", angle: -90, position: "insideLeft", style: { fill: "#9ca3af", fontSize: 11 } }} />
                  <Legend wrapperStyle={{ fontSize: 11, fontFamily: "system-ui" }} />
                  <Bar dataKey="σ_dense" name="Dense (cluster+group)" fill="#dc2626" fillOpacity={0.7} barSize={28} />
                  <Bar dataKey="σ_field" name="Field" fill="#3b82f6" fillOpacity={0.7} barSize={28} />
                </BarChart>
              </ResponsiveContainer>
              <div style={{ fontSize: 12, color: "#9ca3af", marginTop: 8, lineHeight: 1.5, fontFamily: "system-ui" }}>
                In the full sample, blue bar (field) exceeds red bar (dense). In the quality subsample,
                the bars converge — and with Tier 1 only, the dense bar actually exceeds field. The dense environment
                bar is <strong>identical</strong> across all three (0.112 → 0.112 → 0.123) because dense galaxies
                already had good distances. Only the field bar changes — it drops from 0.122 to 0.108 when
                Hubble-flow galaxies are removed.
              </div>
            </div>

            <div style={{ background: "#111318", borderRadius: 10, padding: "16px 20px", border: "1px solid #1f2937" }}>
              <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 14 }}>Binned Environmental Test (Quality Subsample)</div>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={BINNED_ENV} margin={{ top: 10, right: 30, bottom: 20, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="bin" tick={{ fill: "#9ca3af", fontSize: 11, fontFamily: "system-ui" }} stroke="#374151"
                    label={{ value: "log₁₀(g_bar)", position: "bottom", style: { fill: "#9ca3af", fontSize: 11 } }} />
                  <YAxis domain={[0, 0.18]} tick={{ fill: "#9ca3af", fontSize: 11 }} stroke="#374151" />
                  <Legend wrapperStyle={{ fontSize: 11, fontFamily: "system-ui" }} />
                  <Bar dataKey="dense" name="Dense" fill="#dc2626" fillOpacity={0.65} barSize={24} />
                  <Bar dataKey="field" name="Field" fill="#3b82f6" fillOpacity={0.65} barSize={24} />
                </BarChart>
              </ResponsiveContainer>
              <div style={{ fontSize: 12, color: "#9ca3af", marginTop: 8, lineHeight: 1.5, fontFamily: "system-ui" }}>
                One bin shows field {'>'} dense (−11.0, P=97.1%), another shows dense {'>'} field (−10.0, P=90.8%).
                These cancel out globally. The bin at −11.0 is interesting but could reflect sample composition
                rather than physics — only 14 field galaxies contribute to all bins.
              </div>
            </div>

            {/* Levene's test box */}
            <div style={{ background: "#0d1017", borderRadius: 10, padding: 16, border: "1px solid #374151" }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: "#9ca3af", marginBottom: 8, fontFamily: "system-ui" }}>Statistical Tests</div>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, fontFamily: "system-ui" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid #1f2937" }}>
                    <th style={{ padding: "6px 8px", textAlign: "left", color: "#6b7280" }}>Test</th>
                    <th style={{ padding: "6px 8px", textAlign: "right", color: "#6b7280" }}>Statistic</th>
                    <th style={{ padding: "6px 8px", textAlign: "right", color: "#6b7280" }}>P-value</th>
                    <th style={{ padding: "6px 8px", textAlign: "left", color: "#6b7280" }}>Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { test: "Bootstrap Δσ", stat: "−0.0041", p: "0.551", v: "Not significant", vc: "#ef4444" },
                    { test: "Levene's test", stat: "F = 0.018", p: "0.893", v: "No difference", vc: "#ef4444" },
                    { test: "Tier 1 bootstrap", stat: "−0.0152", p: "0.761", v: "Not significant", vc: "#ef4444" },
                    { test: "Bin −11.0 only", stat: "+0.0264", p: "0.029", v: "Marginal", vc: "#f59e0b" },
                  ].map((r, i) => (
                    <tr key={i} style={{ borderBottom: "1px solid #111318" }}>
                      <td style={{ padding: "6px 8px", color: "#d1d5db" }}>{r.test}</td>
                      <td style={{ padding: "6px 8px", textAlign: "right", color: "#9ca3af", fontFamily: "monospace" }}>{r.stat}</td>
                      <td style={{ padding: "6px 8px", textAlign: "right", color: "#9ca3af", fontFamily: "monospace" }}>{r.p}</td>
                      <td style={{ padding: "6px 8px", color: r.vc }}>{r.v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* === GALAXY SAMPLE === */}
        {view === "sample" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ background: "#111318", borderRadius: 10, padding: 16, border: "1px solid #1f2937" }}>
              <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>Subsample Construction Pipeline</div>
              <div style={{ display: "flex", gap: 12, marginBottom: 16, fontFamily: "system-ui" }}>
                {[
                  { step: "SPARC", n: 175, desc: "All galaxies" },
                  { step: "Quality dist", n: 55, desc: "fD=2,3,5 + NED" },
                  { step: "+ UMa", n: 83, desc: "+ fD=4 cluster" },
                  { step: "Quality cuts", n: 43, desc: "Q,Inc,N₀₁₀" },
                ].map((s, i) => (
                  <div key={i} style={{ flex: 1, textAlign: "center" }}>
                    <div style={{ background: "#0d1017", borderRadius: 8, padding: "10px 8px", border: "1px solid #1f2937" }}>
                      <div style={{ fontSize: 20, fontWeight: 700, color: i === 3 ? "#22c55e" : "#d1d5db", fontFamily: "monospace" }}>{s.n}</div>
                      <div style={{ fontSize: 11, color: "#6b7280" }}>{s.step}</div>
                    </div>
                    <div style={{ fontSize: 10, color: "#4b5563", marginTop: 4 }}>{s.desc}</div>
                    {i < 3 && <span style={{ position: "relative", top: -44, left: 50, color: "#374151", fontSize: 16 }}>→</span>}
                  </div>
                ))}
              </div>
              <div style={{ fontSize: 12, color: "#9ca3af", lineHeight: 1.5, fontFamily: "system-ui" }}>
                <strong>Cuts applied:</strong> Q ≤ 2 (removed 7), 30° ≤ Inc ≤ 85° (removed 11), N ≥ 10 data points (removed 22).
                The N ≥ 10 cut removes the most galaxies — many nearby dwarfs with TRGB distances have sparse rotation curves.
              </div>
            </div>

            <div style={{ background: "#111318", borderRadius: 10, padding: 16, border: "1px solid #1f2937" }}>
              <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 10 }}>Sample Galaxies (Selected)</div>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, fontFamily: "system-ui" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid #1f2937" }}>
                    <th style={{ padding: "6px", textAlign: "left", color: "#6b7280" }}>Galaxy</th>
                    <th style={{ padding: "6px", textAlign: "center", color: "#6b7280" }}>Env</th>
                    <th style={{ padding: "6px", textAlign: "left", color: "#6b7280" }}>Group</th>
                    <th style={{ padding: "6px", textAlign: "right", color: "#6b7280" }}>D (Mpc)</th>
                    <th style={{ padding: "6px", textAlign: "center", color: "#6b7280" }}>Method</th>
                    <th style={{ padding: "6px", textAlign: "right", color: "#6b7280" }}>V<sub>flat</sub></th>
                  </tr>
                </thead>
                <tbody>
                  {GALAXY_LIST.map((g, i) => (
                    <tr key={i} style={{ borderBottom: "1px solid #0d1017" }}>
                      <td style={{ padding: "5px 6px", color: "#e5e7eb", fontWeight: 500 }}>{g.name}</td>
                      <td style={{ padding: "5px 6px", textAlign: "center" }}>
                        <Tag bg={g.env === "dense" ? "#7f1d1d" : "#1e3a5f"} fg={g.env === "dense" ? "#fca5a5" : "#93c5fd"}>
                          {g.env}
                        </Tag>
                      </td>
                      <td style={{ padding: "5px 6px", color: "#9ca3af" }}>{g.group}</td>
                      <td style={{ padding: "5px 6px", textAlign: "right", color: "#d1d5db", fontFamily: "monospace" }}>{g.D.toFixed(1)}</td>
                      <td style={{ padding: "5px 6px", textAlign: "center" }}>
                        <Tag bg={g.method.includes("TRGB") ? "#14532d" : g.method.includes("Ceph") ? "#1e3a5f" : "#3b2f0f"}
                             fg={g.method.includes("TRGB") ? "#86efac" : g.method.includes("Ceph") ? "#93c5fd" : "#fcd34d"}>
                          {g.method}
                        </Tag>
                      </td>
                      <td style={{ padding: "5px 6px", textAlign: "right", color: "#d1d5db", fontFamily: "monospace" }}>{g.Vf.toFixed(0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div style={{ fontSize: 11, color: "#6b7280", marginTop: 8, fontFamily: "system-ui" }}>
                * = NED upgrade (Hubble flow → TRGB). Showing representative subset of 43 galaxies.
              </div>
            </div>
          </div>
        )}

        {/* === WHAT SURVIVES === */}
        {view === "physics" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ background: "#111318", borderRadius: 10, padding: "16px 20px", border: "1px solid #1f2937" }}>
              <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 14 }}>Scatter & Skewness Profile (Quality Subsample)</div>
              <ResponsiveContainer width="100%" height={240}>
                <ComposedChart data={SKEW_PROFILE} margin={{ top: 10, right: 30, bottom: 20, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="c" type="number" domain={[-12.5, -9]}
                    tick={{ fill: "#9ca3af", fontSize: 11 }} stroke="#374151"
                    label={{ value: "log₁₀(g_bar)", position: "bottom", style: { fill: "#9ca3af", fontSize: 11 } }} />
                  <YAxis yAxisId="left" domain={[0, 0.2]} tick={{ fill: "#9ca3af", fontSize: 11 }} stroke="#374151" />
                  <YAxis yAxisId="right" orientation="right" domain={[-0.3, 0.2]} tick={{ fill: "#9ca3af", fontSize: 11 }} stroke="#374151" />
                  <Line yAxisId="left" dataKey="std" stroke="#3b82f6" strokeWidth={2} dot={{ r: 5, fill: "#3b82f6" }} name="σ (dex)" />
                  <Bar yAxisId="right" dataKey="qsk" barSize={24} fillOpacity={0.6} name="Quantile skew" radius={[3, 3, 0, 0]}>
                    {SKEW_PROFILE.map((d, i) => (
                      <Cell key={i} fill={d.qsk > 0 ? "#22c55e" : "#ef4444"} />
                    ))}
                  </Bar>
                  <ReferenceLine yAxisId="right" y={0} stroke="#4b5563" strokeDasharray="4 4" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            <div style={{ background: "#111318", borderRadius: 10, padding: 16, border: "1px solid #22c55e33" }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#22c55e", marginBottom: 8 }}>What Still Works</div>
              <div style={{ fontSize: 13, color: "#d1d5db", lineHeight: 1.7 }}>
                <strong>Mass dependence is robust.</strong> Low-mass galaxies (V<sub>flat</sub> {'<'} 110 km/s) show
                0.140 dex scatter vs 0.068 dex for high-mass galaxies — a factor of 2× that cannot be explained by
                distance errors since both subsamples use the same distance methods. The σ<sub>fluid</sub> ≈ 136 km/s
                prediction remains physically reasonable. High kurtosis (up to 27) persists in the quality subsample,
                indicating genuine non-Gaussian tails from intermittent density fluctuations.
              </div>
            </div>

            <div style={{ background: "#111318", borderRadius: 10, padding: 16, border: "1px solid #dc262633" }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#dc2626", marginBottom: 8 }}>What Doesn't Work</div>
              <div style={{ fontSize: 13, color: "#d1d5db", lineHeight: 1.7 }}>
                <strong>Environmental test:</strong> Definitively killed by distance equalization. The 99.8% signal was
                entirely driven by the Hubble-flow distance systematic. Cannot claim this as evidence for the fluid model.
                <br/><br/>
                <strong>Skewness trend:</strong> With only 1,063 points in 5 bins, statistical power is too low to
                detect the predicted gradient. The quantile skewness oscillates between positive and negative with
                no clear trend. BIG-SPARC (~4,000 galaxies with updated distances) remains essential.
                <br/><br/>
                <strong>Moment skewness:</strong> Goes the wrong direction — becomes <em>more</em> negative at
                high g<sub>bar</sub> rather than less, driven by extreme outliers (kurtosis = 27 at log g<sub>bar</sub> ≈ −10.25).
              </div>
            </div>

            <div style={{ background: "#0d1017", borderRadius: 10, padding: 16, border: "1px solid #374151" }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#9ca3af", marginBottom: 10, fontFamily: "system-ui" }}>Revised Assessment</div>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, fontFamily: "system-ui" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid #1f2937" }}>
                    <th style={{ padding: "6px", textAlign: "left", color: "#6b7280" }}>Test</th>
                    <th style={{ padding: "6px", textAlign: "center", color: "#6b7280" }}>Original</th>
                    <th style={{ padding: "6px", textAlign: "center", color: "#6b7280" }}>After equalization</th>
                    <th style={{ padding: "6px", textAlign: "left", color: "#6b7280" }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { test: "Field σ > Cluster σ", orig: "99.8%", now: "44.9%", status: "Killed", sc: "#ef4444" },
                    { test: "Skewness ↑ with g_bar", orig: "81%", now: "~50%", status: "Inconclusive", sc: "#6b7280" },
                    { test: "Low mass σ > High mass σ", orig: "✓", now: "✓ (2×)", status: "Robust", sc: "#22c55e" },
                    { test: "σ_fluid ≈ 136 km/s", orig: "✓", now: "✓", status: "Robust", sc: "#22c55e" },
                    { test: "Excess kurtosis", orig: "κ ≤ 37", now: "κ ≤ 27", status: "Robust", sc: "#22c55e" },
                    { test: "High mass less skewed", orig: "✓", now: "✗ (reversed)", status: "Failed", sc: "#ef4444" },
                  ].map((r, i) => (
                    <tr key={i} style={{ borderBottom: "1px solid #111318" }}>
                      <td style={{ padding: "6px", color: "#d1d5db" }}>{r.test}</td>
                      <td style={{ padding: "6px", textAlign: "center", color: "#9ca3af" }}>{r.orig}</td>
                      <td style={{ padding: "6px", textAlign: "center", color: "#d1d5db" }}>{r.now}</td>
                      <td style={{ padding: "6px", color: r.sc, fontWeight: 500 }}>{r.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Footer */}
        <div style={{ marginTop: 24, paddingTop: 16, borderTop: "1px solid #1f2937", fontSize: 11, color: "#4b5563", fontFamily: "system-ui", lineHeight: 1.5 }}>
          SPARC database (Lelli+ 2016) · NED-D cross-match (175 galaxies queried, 5 upgrades) ·
          Per-galaxy M/L optimization · Bootstrap: 10,000 permutations · Quality cuts: Q≤2, 30°≤Inc≤85°, N≥10
        </div>
      </div>
    </div>
  );
}
