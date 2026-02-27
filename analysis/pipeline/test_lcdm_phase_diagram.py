#!/usr/bin/env python3
"""
LCDM Phase Diagram Null Test
============================

Tests whether EAGLE and/or TNG100-1 LCDM simulations reproduce the SPARC
phase-transition signature in RAR residual variance.

Outputs (saved under analysis/results/lcdm_phase_diagram/):
  - summary_lcdm_phase_diagram.json
  - plot_a_variance_vs_gbar.png
  - plot_b_window_sweep_stability.png
  - plot_c_permutation_null.png
"""

import hashlib
import json
import os
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "analysis", "results", "lcdm_phase_diagram")
RESULTS_DIR = os.environ.get("LCDM_PHASE_OUTPUT_DIR", DEFAULT_RESULTS_DIR)

G = 6.674e-11
M_SUN = 1.989e30
KPC_M = 3.086e19
G_DAGGER = 1.2e-10
LOG_G_DAGGER = np.log10(G_DAGGER)

BIN_WIDTH = 0.25
BIN_RANGE = (-13.0, -8.0)
MIN_POINTS_PER_BIN = 10
WINDOW_CUTOFFS = [-8.8, -9.0, -9.2, -9.4, -9.6]

N_RADII = 12  # >=8, adapted radii grid per galaxy
N_PERM = 500
N_CV_FOLDS = 5
ADAPT_RMIN_FACTOR = float(os.environ.get("LCDM_ADAPT_RMIN_FACTOR", "2.0"))
ADAPT_RMAX_FACTOR = float(os.environ.get("LCDM_ADAPT_RMAX_FACTOR", "5.0"))

if ADAPT_RMIN_FACTOR <= 0:
    ADAPT_RMIN_FACTOR = 2.0
if ADAPT_RMAX_FACTOR <= ADAPT_RMIN_FACTOR:
    ADAPT_RMAX_FACTOR = max(ADAPT_RMIN_FACTOR + 0.5, 5.0)

# Keep runtime manageable during local sanity checks
FAST_MODE = ("--fast" in sys.argv) or (os.environ.get("FAST", "0") == "1")
if FAST_MODE:
    N_PERM = 80


SPARC_REFERENCE = {
    "mu_peak": -9.922,
    "delta_gdagger": 0.0015,
    "daic_edge_vs_M1": -224.9,
    "w_peak": 0.378,
    "perm_p": 0.0,
    "cv_edge_wins": 0.55,
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def setup_plot_style():
    plt.style.use("dark_background")
    bg = "#0d1117"
    fg = "#c9d1d9"
    grid = "#30363d"
    plt.rcParams.update({
        "figure.facecolor": bg,
        "axes.facecolor": bg,
        "savefig.facecolor": bg,
        "savefig.edgecolor": bg,
        "axes.edgecolor": fg,
        "axes.labelcolor": fg,
        "axes.titlecolor": fg,
        "xtick.color": fg,
        "ytick.color": fg,
        "grid.color": grid,
        "text.color": fg,
        "legend.facecolor": "#161b22",
        "legend.edgecolor": grid,
    })


def stable_seed(value: str) -> int:
    h = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def rar_pred(log_gbar: np.ndarray, gdagger: float = G_DAGGER) -> np.ndarray:
    gbar = 10.0 ** log_gbar
    term = 1.0 - np.exp(-np.sqrt(np.maximum(gbar / gdagger, 1e-30)))
    return np.log10(np.maximum(gbar / np.maximum(term, 1e-30), 1e-30))


def concentration_dutton_maccio(m_halo: float, h_val: float) -> float:
    log_c = 0.905 - 0.101 * (np.log10(m_halo) - 12.0 + np.log10(h_val))
    return 10.0 ** log_c


def nfw_enclosed_mass(r_kpc: np.ndarray, m200: float, c: float, r200_kpc: float) -> np.ndarray:
    rs = r200_kpc / max(c, 1e-6)
    x = np.maximum(r_kpc / max(rs, 1e-6), 1e-8)
    norm = np.log(1.0 + c) - c / (1.0 + c)
    return m200 * (np.log(1.0 + x) - x / (1.0 + x)) / max(norm, 1e-12)


def exponential_enclosed_mass(r_kpc: np.ndarray, m_total: float, rd: float) -> np.ndarray:
    y = np.maximum(r_kpc / max(rd, 1e-6), 0.0)
    return m_total * (1.0 - (1.0 + y) * np.exp(-y))


def nll_general(resid: np.ndarray, log_sigma: np.ndarray) -> float:
    sigma = np.exp(np.clip(log_sigma, -20.0, 20.0))
    return float(np.sum(log_sigma + 0.5 * resid**2 / sigma**2) + 0.5 * len(resid) * np.log(2 * np.pi))


def gauss_bump(x: np.ndarray, mu: float, w: float) -> np.ndarray:
    ww = max(w, 1e-6)
    return np.exp(-0.5 * ((x - mu) / ww) ** 2)


def nll_m0(params: np.ndarray, r: np.ndarray, x: np.ndarray) -> float:
    mu_r, ls = params
    return nll_general(r - mu_r, np.full_like(x, ls, dtype=float))


def nll_m1(params: np.ndarray, r: np.ndarray, x: np.ndarray) -> float:
    mu_r, s0, s1 = params
    ls = s0 + s1 * x
    return nll_general(r - mu_r, ls)


def nll_m2(params: np.ndarray, r: np.ndarray, x: np.ndarray) -> float:
    mu_r, s0, s1, c, mu0, lw = params
    w = np.exp(lw)
    ls = s0 + s1 * x + c * gauss_bump(x, mu0, w)
    return nll_general(r - mu_r, ls)


def nll_m2b(params: np.ndarray, r: np.ndarray, x: np.ndarray) -> float:
    mu_r, s0, s1, ap, mup, lwp, ad, mud, lwd = params
    apv = np.exp(ap)
    adv = -np.exp(ad)
    wp = np.exp(lwp)
    wd = np.exp(lwd)
    ls = s0 + s1 * x + apv * gauss_bump(x, mup, wp) + adv * gauss_bump(x, mud, wd)
    return nll_general(r - mu_r, ls)


def nll_m2b_edge(params: np.ndarray, r: np.ndarray, x: np.ndarray) -> float:
    mu_r, s0, s1, ap, mup, lwp, ad, mud, lwd, log_e, xe, log_de = params
    apv = np.exp(ap)
    adv = -np.exp(ad)
    wp = np.exp(lwp)
    wd = np.exp(lwd)
    edge_amp = np.exp(log_e)
    de = np.exp(log_de)
    z = np.clip((x - xe) / max(de, 1e-6), -50.0, 50.0)
    edge = edge_amp / (1.0 + np.exp(-z))
    ls = s0 + s1 * x + apv * gauss_bump(x, mup, wp) + adv * gauss_bump(x, mud, wd) + edge
    return nll_general(r - mu_r, ls)


def eval_log_sigma_m2b_edge(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    _, s0, s1, ap, mup, lwp, ad, mud, lwd, log_e, xe, log_de = params
    apv = np.exp(ap)
    adv = -np.exp(ad)
    wp = np.exp(lwp)
    wd = np.exp(lwd)
    edge_amp = np.exp(log_e)
    de = np.exp(log_de)
    z = np.clip((x - xe) / max(de, 1e-6), -50.0, 50.0)
    edge = edge_amp / (1.0 + np.exp(-z))
    return s0 + s1 * x + apv * gauss_bump(x, mup, wp) + adv * gauss_bump(x, mud, wd) + edge


def eval_log_sigma_m1(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    _, s0, s1 = params
    return s0 + s1 * x


def fit_best(fn, p0_list, args, bounds=None, maxiter=3000):
    best = None
    for p0 in p0_list:
        try:
            res = minimize(
                fn,
                p0,
                args=args,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": maxiter},
            )
            if not np.isfinite(res.fun):
                continue
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue
    return best


def fit_phase_models(x: np.ndarray, r: np.ndarray, quick: bool = False):
    x = np.asarray(x, dtype=float)
    r = np.asarray(r, dtype=float)
    mask = np.isfinite(x) & np.isfinite(r)
    x = x[mask]
    r = r[mask]

    mr = float(np.mean(r))
    sr = float(np.std(r))
    lsr = float(np.log(max(sr, 1e-4)))

    # M0
    b0 = [(-1.0, 1.0), (-6.0, 2.0)]
    res0 = fit_best(nll_m0, [[mr, lsr]], args=(r, x), bounds=b0, maxiter=2000)
    if res0 is None:
        raise RuntimeError("M0 fit failed")
    aic0 = 2.0 * res0.fun + 2.0 * 2

    # M1
    b1 = [(-1.0, 1.0), (-6.0, 2.0), (-2.0, 2.0)]
    p1 = [[mr, lsr, 0.0], [mr, lsr, -0.05], [mr, lsr, 0.05]]
    res1 = fit_best(nll_m1, p1, args=(r, x), bounds=b1, maxiter=2500)
    if res1 is None:
        raise RuntimeError("M1 fit failed")
    aic1 = 2.0 * res1.fun + 2.0 * 3

    # M2 single bump
    b2 = [
        (-1.0, 1.0),
        (-6.0, 2.0),
        (-2.0, 2.0),
        (-6.0, 6.0),
        (-13.0, -8.0),
        (np.log(0.02), np.log(3.0)),
    ]
    p2 = []
    mu_grid = [LOG_G_DAGGER, -10.5, -10.0, -9.5] if quick else [LOG_G_DAGGER, -10.8, -10.5, -10.0, -9.7, -9.4]
    c_grid = [-0.8, -0.4, 0.4] if quick else [-1.0, -0.6, -0.3, 0.3, 0.6]
    w_grid = [0.2, 0.5] if quick else [0.15, 0.3, 0.6, 1.0]
    for mu0 in mu_grid:
        for c in c_grid:
            for w in w_grid:
                p2.append([mr, res1.x[1], res1.x[2], c, mu0, np.log(w)])
    res2 = fit_best(nll_m2, p2, args=(r, x), bounds=b2, maxiter=3000)
    if res2 is None:
        raise RuntimeError("M2 fit failed")
    aic2 = 2.0 * res2.fun + 2.0 * 6

    # M2b peak+dip
    b2b = [
        (-1.0, 1.0),
        (-6.0, 2.0),
        (-2.0, 2.0),
        (-6.0, 6.0),
        (-13.0, -8.0),
        (np.log(0.02), np.log(3.0)),
        (-6.0, 6.0),
        (-13.0, -8.0),
        (np.log(0.02), np.log(3.0)),
    ]
    p2b = []
    peak_mu_grid = [LOG_G_DAGGER, -10.2, -9.9] if quick else [LOG_G_DAGGER, -10.4, -10.1, -9.8, -9.5]
    dip_mu_grid = [-10.4, -10.1, -9.7] if quick else [-10.6, -10.3, -10.0, -9.7]
    amp_grid = [-1.0, -0.4, 0.2] if quick else [-1.2, -0.5, 0.2]
    wp_grid = [0.2, 0.4] if quick else [0.15, 0.35, 0.7]
    wd_grid = [0.1, 0.2] if quick else [0.08, 0.18, 0.35]
    for mup in peak_mu_grid:
        for mud in dip_mu_grid:
            for ap in amp_grid:
                for ad in amp_grid:
                    for wp in wp_grid:
                        for wd in wd_grid:
                            p2b.append([mr, res1.x[1], res1.x[2], ap, mup, np.log(wp), ad, mud, np.log(wd)])
    if not quick and len(p2b) > 180:
        rng = np.random.default_rng(123)
        keep = np.sort(rng.choice(len(p2b), size=180, replace=False))
        p2b = [p2b[i] for i in keep]
    res2b = fit_best(nll_m2b, p2b, args=(r, x), bounds=b2b, maxiter=3500 if quick else 4500)
    if res2b is None:
        raise RuntimeError("M2b fit failed")
    aic2b = 2.0 * res2b.fun + 2.0 * 9

    # M2b edge
    b_edge = [
        (-1.0, 1.0),
        (-6.0, 2.0),
        (-2.0, 2.0),
        (-6.0, 6.0),
        (-13.0, -8.0),
        (np.log(0.02), np.log(3.0)),
        (-6.0, 6.0),
        (-13.0, -8.0),
        (np.log(0.02), np.log(3.0)),
        (-6.0, 6.0),
        (-13.0, -8.0),
        (np.log(0.02), np.log(3.0)),
    ]
    p_edge = []
    edge_amp_grid = [-1.0, -0.3, 0.2] if quick else [-1.2, -0.8, -0.3, 0.2, 0.8]
    edge_xe_grid = [-9.4, -9.1, -8.8] if quick else [-9.6, -9.3, -9.1, -8.9, -8.7]
    edge_de_grid = [0.08, 0.15, 0.30] if quick else [0.05, 0.10, 0.20, 0.40]
    # Seed from M2b
    for eamp in edge_amp_grid:
        for xe in edge_xe_grid:
            for de in edge_de_grid:
                p_edge.append(
                    [
                        res2b.x[0],
                        res2b.x[1],
                        res2b.x[2],
                        res2b.x[3],
                        res2b.x[4],
                        res2b.x[5],
                        res2b.x[6],
                        res2b.x[7],
                        res2b.x[8],
                        eamp,
                        xe,
                        np.log(de),
                    ]
                )
    # Add a g† anchored seed
    p_edge.append(
        [
            mr,
            res1.x[1],
            res1.x[2],
            -0.3,
            LOG_G_DAGGER,
            np.log(0.35),
            -0.6,
            -10.2,
            np.log(0.12),
            -0.2,
            -9.0,
            np.log(0.10),
        ]
    )
    res_edge = fit_best(nll_m2b_edge, p_edge, args=(r, x), bounds=b_edge, maxiter=4000 if quick else 5500)
    if res_edge is None:
        raise RuntimeError("M2b_edge fit failed")
    aic_edge = 2.0 * res_edge.fun + 2.0 * 12

    return {
        "M0": {"k": 2, "nll": float(res0.fun), "aic": float(aic0), "params": [float(v) for v in res0.x]},
        "M1": {"k": 3, "nll": float(res1.fun), "aic": float(aic1), "params": [float(v) for v in res1.x]},
        "M2_single_bump": {"k": 6, "nll": float(res2.fun), "aic": float(aic2), "params": [float(v) for v in res2.x]},
        "M2b_peak_dip": {"k": 9, "nll": float(res2b.fun), "aic": float(aic2b), "params": [float(v) for v in res2b.x]},
        "M2b_edge_final": {"k": 12, "nll": float(res_edge.fun), "aic": float(aic_edge), "params": [float(v) for v in res_edge.x]},
    }


def fit_m1_and_edge_only(x: np.ndarray, r: np.ndarray, seed_edge_params=None):
    """
    Fast fitter for iterative loops (window sweep, CV, permutation):
    only M1 and M2b_edge_final are fitted.
    """
    x = np.asarray(x, dtype=float)
    r = np.asarray(r, dtype=float)
    mask = np.isfinite(x) & np.isfinite(r)
    x = x[mask]
    r = r[mask]

    mr = float(np.mean(r))
    sr = float(np.std(r))
    lsr = float(np.log(max(sr, 1e-4)))

    b1 = [(-1.0, 1.0), (-6.0, 2.0), (-2.0, 2.0)]
    p1 = [[mr, lsr, 0.0], [mr, lsr, -0.05], [mr, lsr, 0.05]]
    res1 = fit_best(nll_m1, p1, args=(r, x), bounds=b1, maxiter=1800)
    if res1 is None:
        raise RuntimeError("M1 fit failed")
    aic1 = 2.0 * res1.fun + 2.0 * 3

    b_edge = [
        (-1.0, 1.0),
        (-6.0, 2.0),
        (-2.0, 2.0),
        (-6.0, 6.0),
        (-13.0, -8.0),
        (np.log(0.02), np.log(3.0)),
        (-6.0, 6.0),
        (-13.0, -8.0),
        (np.log(0.02), np.log(3.0)),
        (-6.0, 6.0),
        (-13.0, -8.0),
        (np.log(0.02), np.log(3.0)),
    ]

    p_edge = [
        [
            mr,
            res1.x[1],
            res1.x[2],
            -0.3,
            LOG_G_DAGGER,
            np.log(0.35),
            -0.6,
            -10.2,
            np.log(0.12),
            -0.2,
            -9.0,
            np.log(0.10),
        ],
        [
            mr,
            res1.x[1],
            res1.x[2],
            0.2,
            -10.2,
            np.log(0.40),
            -0.4,
            -9.7,
            np.log(0.20),
            0.1,
            -8.9,
            np.log(0.20),
        ],
    ]
    if seed_edge_params is not None and len(seed_edge_params) == 12:
        seed = np.array(seed_edge_params, dtype=float).copy()
        seed[0] = mr
        p_edge.append(seed.tolist())

    res_edge = fit_best(nll_m2b_edge, p_edge, args=(r, x), bounds=b_edge, maxiter=2500)
    if res_edge is None:
        raise RuntimeError("Edge fit failed")
    aic_edge = 2.0 * res_edge.fun + 2.0 * 12

    return {
        "M1": {"k": 3, "nll": float(res1.fun), "aic": float(aic1), "params": [float(v) for v in res1.x]},
        "M2b_edge_final": {"k": 12, "nll": float(res_edge.fun), "aic": float(aic_edge), "params": [float(v) for v in res_edge.x]},
    }


def bin_sigma(x: np.ndarray, r: np.ndarray, bin_range=BIN_RANGE, width=BIN_WIDTH, min_pts=MIN_POINTS_PER_BIN):
    lo, hi = bin_range
    edges = np.arange(lo, hi + width, width)
    centers = []
    sigma = []
    sigma_err = []
    counts = []
    for i in range(len(edges) - 1):
        m = (x >= edges[i]) & (x < edges[i + 1])
        n = int(np.sum(m))
        if n < min_pts:
            continue
        s = float(np.std(r[m], ddof=1))
        e = float(s / np.sqrt(max(2 * (n - 1), 1)))
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        sigma.append(s)
        sigma_err.append(e)
        counts.append(n)
    return np.array(centers), np.array(sigma), np.array(sigma_err), np.array(counts), edges


def parse_marasco_table(path: str, sim_name: str):
    galaxies = []
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                galaxies.append({
                    "id": str(parts[0]),
                    "logMs": float(parts[1]),
                    "logMh": float(parts[2]),
                    "vflat": float(parts[3]),
                    "Reff": float(parts[4]),
                    "Rs": float(parts[5]),
                    "sim": sim_name,
                })
            except ValueError:
                continue
    return galaxies


def build_simulated_rar_points(galaxies, sim_name: str):
    """
    Build adapted-radii RAR points from catalog-level galaxy properties.
    Adapted radii: [ADAPT_RMIN_FACTOR, ADAPT_RMAX_FACTOR] x stellar half-mass radius.
    """
    if len(galaxies) == 0:
        return pd.DataFrame(columns=["sim", "galaxy", "R_kpc", "log_gbar", "log_gobs"])

    h_val = 0.6777 if sim_name.upper() == "EAGLE" else 0.6774
    rho_crit = 1.27e11  # Msun / Mpc^3 for h=1

    rows = []
    for g in galaxies:
        m_star = 10.0 ** g["logMs"]
        m_halo = 10.0 ** g["logMh"]
        reff = g["Reff"]
        if not np.isfinite(reff) or reff <= 0:
            continue

        c200 = concentration_dutton_maccio(m_halo, h_val)
        rho200 = 200.0 * rho_crit * h_val**2
        r200_mpc = (3.0 * m_halo / (4.0 * np.pi * rho200)) ** (1.0 / 3.0)
        r200_kpc = r200_mpc * 1000.0

        rd = reff / 1.678
        gas_fraction = 0.15
        m_gas = gas_fraction * m_star
        rg = 2.0 * rd
        fb = min((m_star + m_gas) / m_halo, 0.90)

        r_min = max(ADAPT_RMIN_FACTOR * reff, 1.0)
        r_max = min(ADAPT_RMAX_FACTOR * reff, 0.2 * r200_kpc)
        if r_max <= r_min:
            r_max = max(r_min * 1.4, r_min + 0.5)
        radii = np.linspace(r_min, r_max, N_RADII)

        m_star_enc = exponential_enclosed_mass(radii, m_star, rd)
        m_gas_enc = exponential_enclosed_mass(radii, m_gas, rg)
        m_bar_enc = m_star_enc + m_gas_enc
        m_dm_enc = nfw_enclosed_mass(radii, m_halo * (1.0 - fb), c200, r200_kpc)
        m_total_enc = m_bar_enc + m_dm_enc

        r_m = radii * KPC_M
        g_bar = G * m_bar_enc * M_SUN / np.maximum(r_m**2, 1e-30)
        g_obs = G * m_total_enc * M_SUN / np.maximum(r_m**2, 1e-30)

        # Match prior simulation-treatment: small observational-like noise
        seed = stable_seed(f"{sim_name}_{g['id']}")
        rng = np.random.default_rng(seed)
        log_noise = 0.087
        lgb = np.log10(np.maximum(g_bar, 1e-15)) + rng.normal(0.0, 0.5 * log_noise, len(radii))
        lgo = np.log10(np.maximum(g_obs, 1e-15)) + rng.normal(0.0, log_noise, len(radii))

        valid = (
            np.isfinite(lgb)
            & np.isfinite(lgo)
            & (lgb >= BIN_RANGE[0])
            & (lgb <= BIN_RANGE[1])
            & (lgo >= BIN_RANGE[0])
            & (lgo <= BIN_RANGE[1])
        )
        if np.sum(valid) < 8:
            continue

        gid = f"{sim_name}_{g['id']}"
        for rk, xg, xo in zip(radii[valid], lgb[valid], lgo[valid]):
            rows.append({
                "sim": sim_name,
                "galaxy": gid,
                "R_kpc": float(rk),
                "log_gbar": float(xg),
                "log_gobs": float(xo),
            })

    return pd.DataFrame(rows)


def try_load_existing_sim_file(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext not in {".csv", ".tsv", ".txt"}:
        return None
    try:
        if ext == ".tsv":
            df = pd.read_csv(path, sep="\t")
        else:
            # Let pandas infer separator for txt/csv if possible
            if ext == ".txt":
                df = pd.read_csv(path, sep=None, engine="python")
            else:
                df = pd.read_csv(path)
    except Exception:
        return None

    cols = {c.lower(): c for c in df.columns}
    if "log_gbar" not in cols or "log_gobs" not in cols:
        return None

    out = pd.DataFrame({
        "log_gbar": pd.to_numeric(df[cols["log_gbar"]], errors="coerce"),
        "log_gobs": pd.to_numeric(df[cols["log_gobs"]], errors="coerce"),
    })
    if "r_kpc" in cols:
        out["R_kpc"] = pd.to_numeric(df[cols["r_kpc"]], errors="coerce")
    else:
        out["R_kpc"] = np.nan
    if "galaxy" in cols:
        out["galaxy"] = df[cols["galaxy"]].astype(str)
    else:
        out["galaxy"] = [f"g{i}" for i in range(len(out))]

    out = out.dropna(subset=["log_gbar", "log_gobs"]).copy()
    return out


def load_simulation_points():
    """
    Option A: search local existing simulation RAR point files first.
    Fallback: synthesize adapted-radii points from Marasco+2020 catalogs.
    """
    search_roots = [
        os.path.join(PROJECT_ROOT, "analysis", "results"),
        os.path.join(PROJECT_ROOT, "data"),
    ]
    key_tokens = ("eagle", "tng", "lcdm", "simulation")

    by_sim = {"EAGLE": None, "TNG": None}
    source_notes = []

    # Option A: local simulation point files
    for root in search_roots:
        for cur_root, _, files in os.walk(root):
            for fn in files:
                low = fn.lower()
                if not any(t in low for t in key_tokens):
                    continue
                path = os.path.join(cur_root, fn)
                maybe = try_load_existing_sim_file(path)
                if maybe is None or len(maybe) == 0:
                    continue
                if "eagle" in low and by_sim["EAGLE"] is None:
                    tmp = maybe.copy()
                    tmp["sim"] = "EAGLE"
                    by_sim["EAGLE"] = tmp
                    source_notes.append(f"EAGLE from file: {path}")
                elif "tng" in low and by_sim["TNG"] is None:
                    tmp = maybe.copy()
                    tmp["sim"] = "TNG"
                    by_sim["TNG"] = tmp
                    source_notes.append(f"TNG from file: {path}")
                if by_sim["EAGLE"] is not None and by_sim["TNG"] is not None:
                    break
            if by_sim["EAGLE"] is not None and by_sim["TNG"] is not None:
                break
        if by_sim["EAGLE"] is not None and by_sim["TNG"] is not None:
            break

    # Fallback to Marasco table synthesis where needed
    eagle_path = os.path.join(DATA_DIR, "eagle_rar", "tablea1e.dat")
    tng_path = os.path.join(DATA_DIR, "eagle_rar", "tablea1t.dat")
    if by_sim["EAGLE"] is None and os.path.exists(eagle_path):
        eagle_gals = parse_marasco_table(eagle_path, "EAGLE")
        by_sim["EAGLE"] = build_simulated_rar_points(eagle_gals, "EAGLE")
        source_notes.append(
            f"EAGLE synthesized from Marasco tablea1e.dat with adapted radii {ADAPT_RMIN_FACTOR:g}-{ADAPT_RMAX_FACTOR:g}*Reff"
        )
    if by_sim["TNG"] is None and os.path.exists(tng_path):
        tng_gals = parse_marasco_table(tng_path, "TNG")
        by_sim["TNG"] = build_simulated_rar_points(tng_gals, "TNG")
        source_notes.append(
            f"TNG synthesized from Marasco tablea1t.dat with adapted radii {ADAPT_RMIN_FACTOR:g}-{ADAPT_RMAX_FACTOR:g}*Reff"
        )

    return by_sim, source_notes


def load_sparc_points():
    table2_path = os.path.join(DATA_DIR, "sparc", "SPARC_table2_rotmods.dat")
    mrt_path = os.path.join(DATA_DIR, "sparc", "SPARC_Lelli2016c.mrt")
    if not (os.path.exists(table2_path) and os.path.exists(mrt_path)):
        return pd.DataFrame(columns=["sim", "galaxy", "R_kpc", "log_gbar", "log_gobs"])

    rc_data = {}
    with open(table2_path, "r") as f:
        for line in f:
            if len(line.strip()) < 50:
                continue
            try:
                name = line[0:11].strip()
                if not name:
                    continue
                r = float(line[19:25].strip())
                vobs = float(line[26:32].strip())
                vgas = float(line[39:45].strip())
                vdisk = float(line[46:52].strip())
                vbul = float(line[53:59].strip())
            except (ValueError, IndexError):
                continue
            if name not in rc_data:
                rc_data[name] = {"R": [], "Vobs": [], "Vgas": [], "Vdisk": [], "Vbul": []}
            rc_data[name]["R"].append(r)
            rc_data[name]["Vobs"].append(vobs)
            rc_data[name]["Vgas"].append(vgas)
            rc_data[name]["Vdisk"].append(vdisk)
            rc_data[name]["Vbul"].append(vbul)

    for name in rc_data:
        for k in rc_data[name]:
            rc_data[name][k] = np.array(rc_data[name][k], dtype=float)

    props = {}
    with open(mrt_path, "r") as f:
        lines = f.readlines()
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("---") and i > 50:
            data_start = i + 1
            break
    for line in lines[data_start:]:
        if not line.strip() or line.startswith("#"):
            continue
        try:
            name = line[0:11].strip()
            parts = line[11:].split()
            if len(parts) < 17:
                continue
            props[name] = {"Inc": float(parts[4]), "Q": int(parts[16])}
        except (ValueError, IndexError):
            continue

    rows = []
    for name, d in rc_data.items():
        if name not in props:
            continue
        inc = props[name]["Inc"]
        q = props[name]["Q"]
        if q > 2 or inc < 30 or inc > 85:
            continue
        r = d["R"]
        vobs = d["Vobs"]
        vgas = d["Vgas"]
        vdisk = d["Vdisk"]
        vbul = d["Vbul"]
        vbar_sq = 0.5 * vdisk**2 + vgas * np.abs(vgas) + 0.7 * vbul * np.abs(vbul)
        gbar = np.where(r > 0, np.abs(vbar_sq) * 1e6 / (r * KPC_M), 1e-15)
        gobs = np.where(r > 0, (vobs * 1e3) ** 2 / (r * KPC_M), 1e-15)
        valid = (
            np.isfinite(gbar)
            & np.isfinite(gobs)
            & (gbar > 1e-15)
            & (gobs > 1e-15)
            & (vobs > 5)
        )
        if np.sum(valid) < 5:
            continue
        lg = np.log10(gbar[valid])
        lo = np.log10(gobs[valid])
        rr = r[valid]
        in_range = (lg >= BIN_RANGE[0]) & (lg <= BIN_RANGE[1]) & (lo >= BIN_RANGE[0]) & (lo <= BIN_RANGE[1])
        if np.sum(in_range) < 5:
            continue
        for rk, xg, xo in zip(rr[in_range], lg[in_range], lo[in_range]):
            rows.append({
                "sim": "SPARC",
                "galaxy": str(name),
                "R_kpc": float(rk),
                "log_gbar": float(xg),
                "log_gobs": float(xo),
            })
    return pd.DataFrame(rows)


@dataclass
class DatasetResult:
    name: str
    n_galaxies: int
    n_points: int
    centers: np.ndarray
    sigma: np.ndarray
    sigma_err: np.ndarray
    counts: np.ndarray
    models: dict
    mu_peak: float
    delta_gdagger: float
    daic_edge_vs_m1: float
    daic_edge_vs_m2b: float
    w_peak: float
    ap: float
    window_sweep: list
    window_sweep_stable: bool
    window_sweep_shift: float
    window_sweep_informative: bool
    perm_daic: np.ndarray
    perm_p: float
    cv_edge_wins: float
    cv_n_valid: int
    verdict: str


def classify_lcdm(mu_peak: float, daic_edge_vs_m1: float, perm_p: float) -> str:
    if np.isfinite(mu_peak) and np.isfinite(daic_edge_vs_m1) and np.isfinite(perm_p):
        if abs(mu_peak - LOG_G_DAGGER) < 0.1 and daic_edge_vs_m1 < -50 and perm_p < 0.05:
            return "LCDM_REPRODUCES"
        if daic_edge_vs_m1 < -50 and abs(mu_peak - LOG_G_DAGGER) >= 0.1:
            return "LCDM_SHIFTED"
        if daic_edge_vs_m1 >= -20:
            return "LCDM_NO_PEAK"
        return "LCDM_WEAK"
    return "PENDING"


def window_sweep_fit(df: pd.DataFrame, edge_seed_params=None):
    out = []
    mu_vals = []
    for cut in WINDOW_CUTOFFS:
        sub = df[df["log_gbar"] < cut]
        if len(sub) < 120:
            out.append({
                "cutoff": float(cut),
                "n_points": int(len(sub)),
                "mu_peak": None,
                "daic_edge_vs_M1": None,
            })
            continue
        try:
            models = fit_m1_and_edge_only(
                sub["log_gbar"].to_numpy(),
                sub["log_res"].to_numpy(),
                seed_edge_params=edge_seed_params,
            )
            p = np.array(models["M2b_edge_final"]["params"], dtype=float)
            mu_peak = float(p[4])
            daic = float(models["M2b_edge_final"]["aic"] - models["M1"]["aic"])
            mu_vals.append(mu_peak)
            out.append({
                "cutoff": float(cut),
                "n_points": int(len(sub)),
                "mu_peak": mu_peak,
                "daic_edge_vs_M1": daic,
            })
        except Exception:
            out.append({
                "cutoff": float(cut),
                "n_points": int(len(sub)),
                "mu_peak": None,
                "daic_edge_vs_M1": None,
            })

    valid_rows = [row for row in out if row["mu_peak"] is not None]
    npts = [row["n_points"] for row in valid_rows]
    informative = len(set(npts)) >= 2
    if informative and len(mu_vals) >= 2:
        shift = float(np.max(mu_vals) - np.min(mu_vals))
        stable = bool(shift <= 0.3)
    else:
        shift = np.nan
        stable = False
    return out, stable, shift, informative


def permutation_test(df: pd.DataFrame, observed_daic: float, seed: int, edge_seed_params=None):
    gal_groups = {}
    for g, gdf in df.groupby("galaxy"):
        gsort = gdf.sort_values("R_kpc")
        gal_groups[g] = {
            "x": gsort["log_gbar"].to_numpy(),
            "r": gsort["log_res"].to_numpy(),
            "idx": gsort.index.to_numpy(),
        }
    gals = list(gal_groups.keys())
    if len(gals) < 10:
        return np.array([]), np.nan

    # Destination arrays keep x fixed
    x_full = df["log_gbar"].to_numpy()
    order_full = df.index.to_numpy()
    idx_pos = {idx: i for i, idx in enumerate(order_full)}
    dest_indices = {g: np.array([idx_pos[i] for i in gal_groups[g]["idx"]], dtype=int) for g in gals}

    base_resid = df["log_res"].to_numpy()
    rng = np.random.default_rng(seed)
    daics = []

    for _ in range(N_PERM):
        perm = rng.permutation(len(gals))
        pr = np.empty_like(base_resid)
        for d_i, g_dest in enumerate(gals):
            g_src = gals[perm[d_i]]
            src_seq = gal_groups[g_src]["r"]
            dpos = dest_indices[g_dest]
            nd = len(dpos)
            if len(src_seq) == nd:
                vals = src_seq
            elif len(src_seq) == 1:
                vals = np.repeat(src_seq[0], nd)
            else:
                src_t = np.linspace(0.0, 1.0, len(src_seq))
                dst_t = np.linspace(0.0, 1.0, nd)
                vals = np.interp(dst_t, src_t, src_seq)
            pr[dpos] = vals

        try:
            models = fit_m1_and_edge_only(x_full, pr, seed_edge_params=edge_seed_params)
            daic = float(models["M2b_edge_final"]["aic"] - models["M1"]["aic"])
            if np.isfinite(daic):
                daics.append(daic)
        except Exception:
            continue

    daics = np.array(daics, dtype=float)
    if len(daics) == 0:
        return daics, np.nan
    p_val = float(np.mean(daics < observed_daic))
    return daics, p_val


def cross_validate_edge_vs_m1(df: pd.DataFrame, seed: int, edge_seed_params=None):
    gals = np.array(sorted(df["galaxy"].unique()))
    if len(gals) < max(20, N_CV_FOLDS):
        return np.nan, 0
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(gals)
    folds = np.array_split(shuffled, N_CV_FOLDS)

    wins = 0
    valid_folds = 0
    for fold in folds:
        test_gals = set(fold.tolist())
        train = df[~df["galaxy"].isin(test_gals)]
        test = df[df["galaxy"].isin(test_gals)]
        if len(train) < 200 or len(test) < 80:
            continue
        try:
            models = fit_m1_and_edge_only(
                train["log_gbar"].to_numpy(),
                train["log_res"].to_numpy(),
                seed_edge_params=edge_seed_params,
            )
        except Exception:
            continue

        p1 = np.array(models["M1"]["params"], dtype=float)
        pe = np.array(models["M2b_edge_final"]["params"], dtype=float)

        x_te = test["log_gbar"].to_numpy()
        r_te = test["log_res"].to_numpy()

        nll1 = nll_general(r_te - p1[0], eval_log_sigma_m1(p1, x_te))
        nlle = nll_general(r_te - pe[0], eval_log_sigma_m2b_edge(pe, x_te))

        valid_folds += 1
        if nlle < nll1:
            wins += 1

    if valid_folds == 0:
        return np.nan, 0
    return float(wins / valid_folds), int(valid_folds)


def analyze_dataset(name: str, df: pd.DataFrame, seed: int):
    if df is None or len(df) == 0:
        return None

    data = df.copy()
    data = data[["galaxy", "R_kpc", "log_gbar", "log_gobs"]].dropna()
    data = data[
        (data["log_gbar"] >= BIN_RANGE[0])
        & (data["log_gbar"] <= BIN_RANGE[1])
        & (data["log_gobs"] >= BIN_RANGE[0])
        & (data["log_gobs"] <= BIN_RANGE[1])
    ].copy()
    if len(data) < 200:
        return None

    data["log_res"] = data["log_gobs"].to_numpy() - rar_pred(data["log_gbar"].to_numpy())
    data = data[np.isfinite(data["log_res"])].copy()
    if len(data) < 200:
        return None

    data = data.sort_values(["galaxy", "R_kpc"]).reset_index(drop=True)

    models = fit_phase_models(data["log_gbar"].to_numpy(), data["log_res"].to_numpy(), quick=False)
    edge_params = np.array(models["M2b_edge_final"]["params"], dtype=float)

    mu_peak = float(edge_params[4])
    w_peak = float(np.exp(edge_params[5]))
    ap = float(np.exp(edge_params[3]))
    daic_edge_vs_m1 = float(models["M2b_edge_final"]["aic"] - models["M1"]["aic"])
    daic_edge_vs_m2b = float(models["M2b_edge_final"]["aic"] - models["M2b_peak_dip"]["aic"])
    delta_gdagger = float(mu_peak - LOG_G_DAGGER)

    centers, sigma, sigma_err, counts, _ = bin_sigma(
        data["log_gbar"].to_numpy(),
        data["log_res"].to_numpy(),
    )

    sweep, sweep_stable, sweep_shift, sweep_informative = window_sweep_fit(
        data, edge_seed_params=edge_params
    )
    perm_daic, perm_p = permutation_test(data, daic_edge_vs_m1, seed=seed + 1000, edge_seed_params=edge_params)
    cv_wins, cv_valid = cross_validate_edge_vs_m1(data, seed=seed + 2000, edge_seed_params=edge_params)
    if name.upper() == "SPARC":
        verdict = "SPARC_REFERENCE"
    else:
        verdict = classify_lcdm(mu_peak, daic_edge_vs_m1, perm_p)

    return DatasetResult(
        name=name,
        n_galaxies=int(data["galaxy"].nunique()),
        n_points=int(len(data)),
        centers=centers,
        sigma=sigma,
        sigma_err=sigma_err,
        counts=counts,
        models=models,
        mu_peak=mu_peak,
        delta_gdagger=delta_gdagger,
        daic_edge_vs_m1=daic_edge_vs_m1,
        daic_edge_vs_m2b=daic_edge_vs_m2b,
        w_peak=w_peak,
        ap=ap,
        window_sweep=sweep,
        window_sweep_stable=sweep_stable,
        window_sweep_shift=sweep_shift,
        window_sweep_informative=sweep_informative,
        perm_daic=perm_daic,
        perm_p=perm_p,
        cv_edge_wins=cv_wins,
        cv_n_valid=cv_valid,
        verdict=verdict,
    )


def dataset_json_block(ds: DatasetResult):
    if ds is None:
        return {
            "n_galaxies": 0,
            "n_points": 0,
            "mu_peak": None,
            "delta_gdagger": None,
            "daic_edge_vs_M1": None,
            "daic_edge_vs_M2b": None,
            "w_peak": None,
            "Ap": None,
            "perm_p": None,
            "cv_edge_wins": None,
            "window_sweep_stable": False,
            "window_sweep_informative": False,
            "verdict": "PENDING",
        }
    return {
        "n_galaxies": ds.n_galaxies,
        "n_points": ds.n_points,
        "mu_peak": ds.mu_peak,
        "delta_gdagger": ds.delta_gdagger,
        "daic_edge_vs_M1": ds.daic_edge_vs_m1,
        "daic_edge_vs_M2b": ds.daic_edge_vs_m2b,
        "w_peak": ds.w_peak,
        "Ap": ds.ap,
        "perm_p": ds.perm_p,
        "cv_edge_wins": ds.cv_edge_wins,
        "window_sweep_stable": ds.window_sweep_stable,
        "window_sweep_informative": ds.window_sweep_informative,
        "window_sweep_shift_dex": ds.window_sweep_shift,
        "window_sweep": ds.window_sweep,
        "verdict": ds.verdict,
    }


def plot_a_variance(datasets: dict, outpath: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    order = ["SPARC", "EAGLE", "TNG"]
    colors = {"SPARC": "#f59e0b", "EAGLE": "#22c55e", "TNG": "#38bdf8"}

    for ax, name in zip(axes, order):
        ds = datasets.get(name)
        ax.axvline(LOG_G_DAGGER, color="#f87171", linestyle="--", linewidth=1.2, label="g†")
        if ds is None:
            ax.text(0.5, 0.5, f"{name}\nNo data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name)
            ax.set_xlim(BIN_RANGE)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.2)
            continue

        c = colors[name]
        if len(ds.centers) > 0:
            ax.errorbar(ds.centers, ds.sigma, yerr=ds.sigma_err, fmt="o", color=c, ecolor=c, ms=4, lw=1.0, label="Binned σ")
        x_model = np.linspace(BIN_RANGE[0], BIN_RANGE[1], 500)
        p_edge = np.array(ds.models["M2b_edge_final"]["params"], dtype=float)
        sigma_model = np.exp(eval_log_sigma_m2b_edge(p_edge, x_model))
        ax.plot(x_model, sigma_model, color="#e5e7eb", lw=1.5, label="Edge fit")
        ax.set_title(
            f"{name}\nμp={ds.mu_peak:.3f}, ΔAIC={ds.daic_edge_vs_m1:.1f}"
        )
        ax.set_xlim(BIN_RANGE)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
        ax.set_xlabel("log g_bar")
        if name == "SPARC":
            ax.set_ylabel("σ(residual)")
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Plot A: Variance vs log g_bar (SPARC vs EAGLE vs TNG)", fontsize=13)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_b_window_sweep(datasets: dict, outpath: str):
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    colors = {"SPARC": "#f59e0b", "EAGLE": "#22c55e", "TNG": "#38bdf8"}
    for name in ["SPARC", "EAGLE", "TNG"]:
        ds = datasets.get(name)
        if ds is None:
            continue
        xs = []
        ys = []
        for row in ds.window_sweep:
            if row["mu_peak"] is None:
                continue
            xs.append(row["cutoff"])
            ys.append(row["mu_peak"])
        if len(xs) == 0:
            continue
        ax.plot(xs, ys, marker="o", lw=1.8, color=colors[name], label=f"{name}")
    ax.axhline(LOG_G_DAGGER, color="#f87171", linestyle="--", lw=1.2, label="g†")
    ax.set_xlabel("Upper cutoff on log g_bar")
    ax.set_ylabel("Fitted mu_peak")
    ax.set_title("Plot B: Window Sweep Stability")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_c_permutation(datasets: dict, outpath: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    colors = {"SPARC": "#f59e0b", "EAGLE": "#22c55e", "TNG": "#38bdf8"}
    for ax, name in zip(axes, ["SPARC", "EAGLE", "TNG"]):
        ds = datasets.get(name)
        if ds is None or ds.perm_daic is None or len(ds.perm_daic) == 0:
            ax.text(0.5, 0.5, f"{name}\nNo permutation data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name)
            ax.grid(alpha=0.2)
            continue
        c = colors[name]
        ax.hist(ds.perm_daic, bins=32, color=c, alpha=0.85)
        ax.axvline(ds.daic_edge_vs_m1, color="#f87171", linestyle="--", lw=1.8, label=f"Observed {ds.daic_edge_vs_m1:.1f}")
        ax.set_title(f"{name}\np={ds.perm_p:.4f} (n={len(ds.perm_daic)})")
        ax.set_xlabel("ΔAIC (edge - M1)")
        if name == "SPARC":
            ax.set_ylabel("Count")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Plot C: Permutation Null Distribution", fontsize=13)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def interpret_discriminating(eagle: DatasetResult, tng: DatasetResult):
    available = [d for d in [eagle, tng] if d is not None]
    if len(available) == 0:
        return True, "No LCDM simulation dataset was available locally; discriminant remains unresolved."

    reproduces_all = all(d.verdict == "LCDM_REPRODUCES" for d in available)
    if reproduces_all:
        near_cv = all(np.isfinite(d.cv_edge_wins) and abs(d.cv_edge_wins - SPARC_REFERENCE["cv_edge_wins"]) <= 0.15 for d in available)
        if near_cv:
            return False, "LCDM reproduces peak location/significance and shows similar holdout behavior; result is not discriminating."
        return False, "LCDM reproduces key peak/significance criteria; phase signature is not a clean discriminator."

    shifted = [d.name for d in available if d.verdict == "LCDM_SHIFTED"]
    weak_or_none = [d.name for d in available if d.verdict in {"LCDM_NO_PEAK", "LCDM_WEAK"}]

    if len(shifted) > 0 and len(weak_or_none) == 0:
        return True, f"LCDM shows a peak but shifted away from g† ({', '.join(shifted)}), preserving discriminating power."
    if len(weak_or_none) > 0:
        return True, f"LCDM lacks a strong correctly-located peak in {', '.join(weak_or_none)}; SPARC phase structure remains discriminating."
    return True, "At least one LCDM dataset fails reproduction criteria; SPARC phase structure remains discriminating."


def print_dataset_summary(ds: DatasetResult):
    if ds is None:
        return
    print(f"  {ds.name}: {ds.n_galaxies} galaxies, {ds.n_points} points")
    print(f"    mu_peak={ds.mu_peak:.3f}, delta_gdagger={ds.delta_gdagger:+.3f}")
    print(f"    daic(edge-M1)={ds.daic_edge_vs_m1:+.1f}, daic(edge-M2b)={ds.daic_edge_vs_m2b:+.1f}")
    print(f"    w_peak={ds.w_peak:.3f}, Ap={ds.ap:.3f}, perm_p={ds.perm_p:.4f}")
    if np.isfinite(ds.cv_edge_wins):
        print(f"    cv_edge_wins={ds.cv_edge_wins:.2f} (n={ds.cv_n_valid})")
    else:
        print("    cv_edge_wins=NA")
    print(
        f"    window_stable={ds.window_sweep_stable} "
        f"(informative={ds.window_sweep_informative}, shift={ds.window_sweep_shift:.3f} dex)"
    )
    print(f"    verdict={ds.verdict}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    setup_plot_style()

    print("=" * 72)
    print("LCDM PHASE DIAGRAM NULL TEST")
    print("=" * 72)
    print(f"  g† = {G_DAGGER:.2e} m/s^2, log g† = {LOG_G_DAGGER:.3f}")
    print(f"  FAST mode: {'ON' if FAST_MODE else 'OFF'} | permutations per dataset: {N_PERM}")
    print(f"  Adapted radii factors: {ADAPT_RMIN_FACTOR:g} to {ADAPT_RMAX_FACTOR:g} x Reff")
    print(f"  Output directory: {RESULTS_DIR}")

    print("\n[1] Loading SPARC and simulation datasets...")
    sparc_df = load_sparc_points()
    sim_by_name, source_notes = load_simulation_points()

    print(f"  SPARC points: {len(sparc_df)}")
    for note in source_notes:
        print(f"  - {note}")

    print("\n[2] Running phase-diagram fits...")
    sparc_res = analyze_dataset("SPARC", sparc_df, seed=101)
    eagle_res = analyze_dataset("EAGLE", sim_by_name.get("EAGLE"), seed=202) if sim_by_name.get("EAGLE") is not None else None
    tng_res = analyze_dataset("TNG", sim_by_name.get("TNG"), seed=303) if sim_by_name.get("TNG") is not None else None

    if sparc_res is not None:
        print_dataset_summary(sparc_res)
    if eagle_res is not None:
        print_dataset_summary(eagle_res)
    else:
        print("  EAGLE: pending (no usable local dataset)")
    if tng_res is not None:
        print_dataset_summary(tng_res)
    else:
        print("  TNG: pending (no usable local dataset)")

    print("\n[3] Generating plots...")
    datasets = {"SPARC": sparc_res, "EAGLE": eagle_res, "TNG": tng_res}
    plot_a_path = os.path.join(RESULTS_DIR, "plot_a_variance_vs_gbar.png")
    plot_b_path = os.path.join(RESULTS_DIR, "plot_b_window_sweep_stability.png")
    plot_c_path = os.path.join(RESULTS_DIR, "plot_c_permutation_null.png")
    plot_a_variance(datasets, plot_a_path)
    plot_b_window_sweep(datasets, plot_b_path)
    plot_c_permutation(datasets, plot_c_path)

    print("\n[4] Writing summary JSON...")
    discriminating, interpretation = interpret_discriminating(eagle_res, tng_res)
    summary = {
        "sparc_reference": SPARC_REFERENCE,
        "sparc_recomputed": dataset_json_block(sparc_res),
        "eagle": dataset_json_block(eagle_res),
        "tng": dataset_json_block(tng_res),
        "discriminating": bool(discriminating),
        "interpretation": interpretation,
        "method": {
            "bin_width_dex": BIN_WIDTH,
            "bin_range": [BIN_RANGE[0], BIN_RANGE[1]],
            "min_points_per_bin": MIN_POINTS_PER_BIN,
            "window_cutoffs": WINDOW_CUTOFFS,
            "n_perm": N_PERM,
            "cv_folds": N_CV_FOLDS,
            "simulation_adapted_radii": f"{ADAPT_RMIN_FACTOR:g}x to {ADAPT_RMAX_FACTOR:g}x stellar half-mass radius (Reff)",
            "adapted_rmin_factor": ADAPT_RMIN_FACTOR,
            "adapted_rmax_factor": ADAPT_RMAX_FACTOR,
            "fast_mode": FAST_MODE,
        },
        "paths": {
            "output_dir": RESULTS_DIR,
            "plot_a": plot_a_path,
            "plot_b": plot_b_path,
            "plot_c": plot_c_path,
        },
        "data_source_notes": source_notes,
    }
    out_json = os.path.join(RESULTS_DIR, "summary_lcdm_phase_diagram.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Wrote: {out_json}")
    print(f"  Wrote: {plot_a_path}")
    print(f"  Wrote: {plot_b_path}")
    print(f"  Wrote: {plot_c_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
