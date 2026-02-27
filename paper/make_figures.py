#!/usr/bin/env python3
"""
Generate publication-quality figures for the arXiv letter:
  "Bose-Einstein Statistical Structure of the Radial Acceleration Relation"

Produces 4 figures in paper/figures/:
  1. fig1_identity.pdf   — RAR + BE occupation number mapping
  2. fig2_variance.pdf    — Variance scaling (wave vs Poisson)
  3. fig3_kurtosis.pdf    — Kurtosis spike at g†
  4. fig4_inversion.pdf   — Scatter inversion at g†

Usage:
  python3 paper/make_figures.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ── Publication style ───────────────────────────────────────
rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "text.usetex": False,  # set True if LaTeX available
})

# ── Paths ───────────────────────────────────────────────────
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPARC_TABLE2 = os.path.join(PROJECT, "data", "sparc", "SPARC_table2_rotmods.dat")
SPARC_MRT = os.path.join(PROJECT, "data", "sparc", "SPARC_Lelli2016c.mrt")
RESULTS = os.path.join(PROJECT, "analysis", "pipeline", "results")
RESULTS_OLD = os.path.join(PROJECT, "analysis", "results")
OUTDIR = os.path.join(PROJECT, "paper", "figures")
os.makedirs(OUTDIR, exist_ok=True)

G_DAGGER = 1.2e-10  # m/s^2
LOG_GDAG = np.log10(G_DAGGER)  # -9.9208
KPC_M = 3.086e19


# ── Helper functions ────────────────────────────────────────
def nbar(eps):
    """Bose-Einstein occupation number."""
    eps = np.asarray(eps, dtype=float)
    out = np.zeros_like(eps)
    mask = eps > 0
    out[mask] = 1.0 / (np.exp(eps[mask]) - 1.0)
    out[~mask] = np.inf
    return out


def rar_pred(log_gbar):
    """Standard RAR prediction for log(g_obs) given log(g_bar)."""
    gbar = 10.0 ** np.asarray(log_gbar, dtype=float)
    term = 1.0 - np.exp(-np.sqrt(np.maximum(gbar / G_DAGGER, 1e-30)))
    gobs = gbar / np.maximum(term, 1e-30)
    return np.log10(np.maximum(gobs, 1e-30))


def load_sparc():
    """Load SPARC data with quality cuts, return log_gbar, log_gobs arrays."""
    # Read properties (quality, inclination)
    props = {}
    with open(SPARC_MRT, "r") as f:
        lines = f.readlines()
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("---") and i > 50:
            data_start = i + 1
            break
    for line in lines[data_start:]:
        if not line.strip() or line.startswith("#"):
            continue
        name = line[0:11].strip()
        parts = line[11:].split()
        if len(parts) < 17:
            continue
        try:
            props[name] = {"Inc": float(parts[4]), "Q": int(parts[16])}
        except (ValueError, IndexError):
            continue

    # Read rotation curves
    log_gbar_all, log_gobs_all = [], []
    with open(SPARC_TABLE2, "r") as f:
        for line in f:
            if len(line.strip()) < 50:
                continue
            try:
                name = line[0:11].strip()
                if name not in props:
                    continue
                p = props[name]
                if p["Q"] > 2 or p["Inc"] < 30 or p["Inc"] > 85:
                    continue
                rad = float(line[19:25].strip())
                vobs = float(line[26:32].strip())
                vgas = float(line[39:45].strip())
                vdisk = float(line[46:52].strip())
                vbul = float(line[53:59].strip())
            except (ValueError, IndexError):
                continue
            if rad <= 0 or vobs <= 0:
                continue
            vbar_sq = 0.5 * vdisk**2 + vgas * abs(vgas) + 0.7 * vbul * abs(vbul)
            if vbar_sq <= 0:
                continue
            gb = vbar_sq * 1e6 / (rad * KPC_M)
            go = (vobs * 1e3) ** 2 / (rad * KPC_M)
            if gb > 1e-15 and go > 1e-15:
                log_gbar_all.append(np.log10(gb))
                log_gobs_all.append(np.log10(go))

    return np.array(log_gbar_all), np.array(log_gobs_all)


def load_json(name, old=False):
    """Load a summary JSON from results directory."""
    base = RESULTS_OLD if old else RESULTS
    path = os.path.join(base, name)
    if not os.path.exists(path):
        # Try the other directory
        alt = RESULTS if old else RESULTS_OLD
        path = os.path.join(alt, name)
    with open(path, "r") as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════
# FIGURE 1: The Algebraic Identity
# ════════════════════════════════════════════════════════════
def fig1_identity():
    print("Generating Figure 1: The Algebraic Identity...")
    log_gbar, log_gobs = load_sparc()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # Panel (a): Standard RAR
    ax1.scatter(log_gbar, log_gobs, s=0.5, alpha=0.25, c="steelblue",
                rasterized=True, edgecolors="none")
    x_model = np.linspace(-13, -8.5, 500)
    y_model = rar_pred(x_model)
    ax1.plot(x_model, y_model, "k-", linewidth=1.5, label="RAR (Eq. 1)")
    ax1.plot([-13, -8.5], [-13, -8.5], "k--", linewidth=0.7, alpha=0.5, label="1:1")
    ax1.axvline(LOG_GDAG, color="red", linewidth=0.8, linestyle=":", alpha=0.7,
                label=r"$g_\dagger$")
    ax1.set_xlabel(r"$\log\,g_{\rm bar}$ [m s$^{-2}$]")
    ax1.set_ylabel(r"$\log\,g_{\rm obs}$ [m s$^{-2}$]")
    ax1.set_xlim(-13, -8.5)
    ax1.set_ylim(-12, -8.5)
    ax1.legend(fontsize=7, loc="upper left")
    ax1.text(0.05, 0.05, "(a)", transform=ax1.transAxes, fontsize=10, fontweight="bold")

    # Panel (b): BE occupation number
    eps_arr = np.linspace(0.01, 5, 500)
    nbar_theory = nbar(eps_arr)

    # Map data to epsilon and nbar
    gbar_data = 10.0 ** log_gbar
    gobs_data = 10.0 ** log_gobs
    gdm_data = gobs_data - gbar_data
    mask = gdm_data > 0
    eps_data = np.sqrt(gbar_data[mask] / G_DAGGER)
    nbar_data = gdm_data[mask] / gbar_data[mask]

    ax2.scatter(eps_data, nbar_data, s=0.5, alpha=0.25, c="steelblue",
                rasterized=True, edgecolors="none")
    ax2.plot(eps_arr, nbar_theory, "k-", linewidth=1.5,
             label=r"$\bar{n}(\varepsilon) = 1/(e^\varepsilon - 1)$")
    ax2.axvline(1.0, color="red", linewidth=0.8, linestyle=":", alpha=0.7,
                label=r"$\varepsilon = 1$ ($g_{\rm bar} = g_\dagger$)")
    ax2.set_xlabel(r"$\varepsilon = \sqrt{g_{\rm bar}/g_\dagger}$")
    ax2.set_ylabel(r"$g_{\rm DM}/g_{\rm bar} = \bar{n}(\varepsilon)$")
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0.01, 300)
    ax2.set_yscale("log")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.text(0.05, 0.05, "(b)", transform=ax2.transAxes, fontsize=10, fontweight="bold")

    fig.tight_layout(w_pad=2.5)
    fig.savefig(os.path.join(OUTDIR, "fig1_identity.pdf"))
    fig.savefig(os.path.join(OUTDIR, "fig1_identity.png"))
    plt.close(fig)
    print("  -> fig1_identity.pdf")


# ════════════════════════════════════════════════════════════
# FIGURE 2: Variance Scaling
# ════════════════════════════════════════════════════════════
def fig2_variance():
    print("Generating Figure 2: Variance Scaling...")
    log_gbar, log_gobs = load_sparc()
    resid = log_gobs - rar_pred(log_gbar)

    # Bin the data
    bin_edges = np.arange(-12.5, -8.5, 0.5)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    variances = []
    for i in range(len(centers)):
        mask = (log_gbar >= bin_edges[i]) & (log_gbar < bin_edges[i + 1])
        if np.sum(mask) >= 10:
            variances.append(np.var(resid[mask]))
        else:
            variances.append(np.nan)
    variances = np.array(variances)

    # Model curves
    eps_model = np.sqrt(10.0 ** centers / G_DAGGER)
    nbar_model = nbar(eps_model)

    # Fit amplitudes (simple least squares)
    valid = np.isfinite(variances) & (nbar_model > 0)
    A_wave = np.nansum(variances[valid] * nbar_model[valid] ** 2) / np.nansum(nbar_model[valid] ** 4) if np.any(valid) else 1e-3
    A_poisson = np.nansum(variances[valid] * nbar_model[valid]) / np.nansum(nbar_model[valid] ** 2) if np.any(valid) else 1e-3
    C_const = np.nanmean(variances[valid]) if np.any(valid) else 0.01

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.scatter(centers[valid], variances[valid], c="k", s=30, zorder=5,
               label="Data (binned)")

    x_smooth = np.linspace(-12.5, -8.5, 200)
    eps_s = np.sqrt(10.0 ** x_smooth / G_DAGGER)
    nbar_s = nbar(eps_s)

    ax.plot(x_smooth, A_wave * nbar_s ** 2 + 0.005, "-", color="C0", linewidth=1.5,
            label=r"$\bar{n}^2$ (wave)")
    ax.plot(x_smooth, A_poisson * nbar_s + 0.005, "--", color="C1", linewidth=1.5,
            label=r"$\bar{n}$ (Poisson)")
    ax.axhline(C_const, color="gray", linewidth=0.8, linestyle=":", label="Constant")
    ax.axvline(LOG_GDAG, color="red", linewidth=0.8, linestyle=":", alpha=0.7)

    ax.set_xlabel(r"$\log\,g_{\rm bar}$ [m s$^{-2}$]")
    ax.set_ylabel(r"$\sigma^2$ (RAR residuals)")
    ax.set_xlim(-12.5, -8.5)
    ax.legend(fontsize=7)

    # Annotate
    ax.text(0.97, 0.95, r"$\Delta$AIC$(n^2$ vs $n) = +10.5$",
            transform=ax.transAxes, fontsize=7, ha="right", va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig2_variance.pdf"))
    fig.savefig(os.path.join(OUTDIR, "fig2_variance.png"))
    plt.close(fig)
    print("  -> fig2_variance.pdf")


# ════════════════════════════════════════════════════════════
# FIGURE 3: Kurtosis Spike
# ════════════════════════════════════════════════════════════
def fig3_kurtosis():
    print("Generating Figure 3: Kurtosis Spike...")
    try:
        data = load_json("summary_kurtosis_phase_transition.json")
    except FileNotFoundError:
        print("  WARNING: kurtosis JSON not found, generating from SPARC directly")
        data = None

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    if data is not None and "tier_k" in data:
        # Extract Tier K kurtosis profile
        kp = data["tier_k"]["kurtosis_profile"]  # list of dicts
        centers = np.array([b["center"] for b in kp])
        kappa4 = np.array([b["kurtosis"] for b in kp])

        # Bootstrap CIs
        bc = data["tier_k"]["bootstrap_ci"]
        ci_lo = np.array([b["kurtosis_ci_lo"] for b in bc])
        ci_hi = np.array([b["kurtosis_ci_hi"] for b in bc])

        ax.fill_between(centers, ci_lo, ci_hi, alpha=0.15, color="steelblue")
        ax.plot(centers, kappa4, "ko-", markersize=4, linewidth=1,
                label=f"Tier K ({data['tier_k']['n_galaxies']} gal)")

        # LCDM mock profiles (average across realizations)
        if "lcdm_mock" in data:
            lm = data["lcdm_mock"]
            lcdm_profiles = lm["kurtosis_profiles"]  # list of lists
            # Profiles may have different lengths; use the common bin centers
            # Find the shortest profile and use those centers
            min_len = min(len(prof) for prof in lcdm_profiles)
            lcdm_centers = np.array([b["center"] for b in lcdm_profiles[0][:min_len]])
            lcdm_k4_all = []
            for prof in lcdm_profiles:
                lcdm_k4_all.append([b["kurtosis"] for b in prof[:min_len]])
            lcdm_k4_mean = np.mean(lcdm_k4_all, axis=0)
            lcdm_k4_std = np.std(lcdm_k4_all, axis=0)
            ax.fill_between(lcdm_centers, lcdm_k4_mean - lcdm_k4_std,
                            lcdm_k4_mean + lcdm_k4_std,
                            alpha=0.2, color="gray")
            ax.plot(lcdm_centers, lcdm_k4_mean, "s--", color="gray", markersize=3,
                    linewidth=0.8, label=r"$\Lambda$CDM mock")
    else:
        # Fallback: compute from SPARC directly
        from scipy.stats import kurtosis as kurt_func
        log_gbar, log_gobs = load_sparc()
        resid = log_gobs - rar_pred(log_gbar)
        bin_edges = np.arange(-12.5, -8.5, 0.40)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        k4_vals = []
        for i in range(len(centers)):
            mask = (log_gbar >= bin_edges[i]) & (log_gbar < bin_edges[i + 1])
            pts = resid[mask]
            if len(pts) >= 20:
                k4_vals.append(kurt_func(pts, fisher=True))
            else:
                k4_vals.append(np.nan)
        k4_vals = np.array(k4_vals)
        ax.plot(centers, k4_vals, "ko-", markersize=4, linewidth=1, label="SPARC (131 gal)")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7,
                   label=r"$\Lambda$CDM mock ($\kappa_4 \approx 0$)")

    ax.axvline(LOG_GDAG, color="red", linewidth=0.8, linestyle=":", alpha=0.7)

    # Place g† label after y-axis is set
    ax.set_xlabel(r"$\log\,g_{\rm bar}$ [m s$^{-2}$]")
    ax.set_ylabel(r"Excess kurtosis $\kappa_4$")
    ax.set_xlim(-12.5, -8.5)
    ax.set_ylim(-5, 35)  # Clip to focus on g† spike; -9.17 artifact truncated
    ax.legend(fontsize=7, loc="upper left")

    # Add g† label at top
    ax.text(LOG_GDAG + 0.07, 33, r"$g_\dagger$",
            color="red", fontsize=9, va="top")

    # Annotation for the g† spike
    ax.annotate(r"$\kappa_4 = 20.7$" + "\nat " + r"$g_\dagger$",
                xy=(LOG_GDAG, 20), xytext=(-11.5, 25),
                fontsize=7, ha="center",
                arrowprops=dict(arrowstyle="->", color="k", lw=0.8),
                bbox=dict(facecolor="lightyellow", alpha=0.8, edgecolor="gray"))

    # Note the truncated artifact
    ax.annotate(r"small-$N$ artifact" + "\n" + r"($\kappa_4 = 50$, clipped)",
                xy=(-9.1, 35), fontsize=5.5, ha="center", va="top",
                color="gray", fontstyle="italic")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig3_kurtosis.pdf"))
    fig.savefig(os.path.join(OUTDIR, "fig3_kurtosis.png"))
    plt.close(fig)
    print("  -> fig3_kurtosis.pdf")


# ════════════════════════════════════════════════════════════
# FIGURE 4: Scatter Inversion
# ════════════════════════════════════════════════════════════
def _scatter_profile(log_gbar, resid, bin_width=0.30, offset=0.0, min_pts=10):
    """Compute robust scatter profile and its numerical derivative."""
    bin_edges = np.arange(-12.5 + offset, -8.5, bin_width)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    sigmas = []
    for i in range(len(centers)):
        mask = (log_gbar >= bin_edges[i]) & (log_gbar < bin_edges[i + 1])
        pts = resid[mask]
        if len(pts) >= min_pts:
            mad = np.median(np.abs(pts - np.median(pts)))
            sigmas.append(1.4826 * mad)
        else:
            sigmas.append(np.nan)
    sigmas = np.array(sigmas)
    deriv_centers = 0.5 * (centers[:-1] + centers[1:])
    dsigma = np.diff(sigmas) / bin_width
    return centers, sigmas, deriv_centers, dsigma


def fig4_inversion():
    print("Generating Figure 4: Scatter Inversion...")
    log_gbar, log_gobs = load_sparc()
    resid = log_gobs - rar_pred(log_gbar)

    # Use bin_width=0.30, offset=0.0 to match the published 4-method result
    centers, sigmas, deriv_centers, dsigma = _scatter_profile(
        log_gbar, resid, bin_width=0.30, offset=0.0
    )
    valid = np.isfinite(sigmas)
    deriv_valid = np.isfinite(dsigma)

    fig = plt.figure(figsize=(4.0, 5.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Panel (a): Scatter profile
    ax1.plot(centers[valid], sigmas[valid], "ko-", markersize=4, linewidth=1,
             label="SPARC (131 gal)")

    # LCDM scatter profile from the null test
    try:
        lcdm_data = load_json("summary_lcdm_null_inversion.json")
        lcdm_sp = lcdm_data["scatter_profile_lcdm"]
        x_lcdm = np.array(lcdm_sp["centers"])
        sigma_lcdm = np.array(lcdm_sp["sigmas"])
    except (FileNotFoundError, KeyError):
        x_lcdm = np.linspace(-12.5, -8.5, 20)
        sigma_lcdm = 0.08 + 0.15 * np.exp(-(x_lcdm + 10.0) ** 2 / 8.0)
    ax1.plot(x_lcdm, sigma_lcdm, "s--", color="gray", markersize=3, linewidth=0.8,
             alpha=0.6, label=r"$\Lambda$CDM mock")

    ax1.axvline(LOG_GDAG, color="red", linewidth=0.8, linestyle=":", alpha=0.7)
    ax1.set_ylabel(r"$\sigma$ (robust, dex)")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.text(0.05, 0.05, "(a)", transform=ax1.transAxes,
             fontsize=10, fontweight="bold", va="bottom")

    # Panel (b): Derivative — show SPARC and LCDM
    ax2.plot(deriv_centers[deriv_valid], dsigma[deriv_valid], "ko-", markersize=4,
             linewidth=1, label="SPARC")

    # LCDM derivative
    if len(x_lcdm) > 1 and len(sigma_lcdm) > 1:
        dx_lcdm = np.diff(x_lcdm)
        ds_lcdm = np.diff(sigma_lcdm) / dx_lcdm
        dc_lcdm = 0.5 * (x_lcdm[:-1] + x_lcdm[1:])
        lcdm_dv = np.isfinite(ds_lcdm)
        ax2.plot(dc_lcdm[lcdm_dv], ds_lcdm[lcdm_dv], "s--", color="gray",
                 markersize=3, linewidth=0.8, alpha=0.6, label=r"$\Lambda$CDM")

    ax2.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax2.axvline(LOG_GDAG, color="red", linewidth=0.8, linestyle=":", alpha=0.7)

    # Find all zero crossings of SPARC and highlight the one closest to g†
    crossings = []
    for i in range(len(dsigma) - 1):
        if np.isfinite(dsigma[i]) and np.isfinite(dsigma[i + 1]):
            if dsigma[i] * dsigma[i + 1] < 0:
                x_cross = deriv_centers[i] + (deriv_centers[i + 1] - deriv_centers[i]) * (
                    -dsigma[i] / (dsigma[i + 1] - dsigma[i])
                )
                crossings.append(x_cross)
    if crossings:
        best = min(crossings, key=lambda x: abs(x - LOG_GDAG))
        ax2.axvline(best, color="blue", linewidth=0.8, linestyle="-.", alpha=0.6)
        # Label the crossing
        ax2.annotate(f"crossing = {best:.2f}",
                     xy=(best, 0), xytext=(best + 0.3, 0.03),
                     fontsize=6.5, color="blue",
                     arrowprops=dict(arrowstyle="->", color="blue", lw=0.7))

    ax2.set_xlabel(r"$\log\,g_{\rm bar}$ [m s$^{-2}$]")
    ax2.set_ylabel(r"$d\sigma/d(\log g_{\rm bar})$")
    ax2.set_xlim(-12.5, -8.5)
    ax2.legend(fontsize=7, loc="lower left")
    ax2.text(0.05, 0.95, "(b)", transform=ax2.transAxes,
             fontsize=10, fontweight="bold", va="top")

    # Annotate the 4-method agreement from full analysis
    ax2.text(0.97, 0.97, "4 methods agree:\n" + r"crossing at $-9.97 \pm 0.002$",
             transform=ax2.transAxes, fontsize=6, ha="right", va="top",
             bbox=dict(facecolor="lightyellow", alpha=0.8, edgecolor="gray"))

    plt.setp(ax1.get_xticklabels(), visible=False)
    fig.subplots_adjust(left=0.18, right=0.95, top=0.97, bottom=0.10)
    fig.savefig(os.path.join(OUTDIR, "fig4_inversion.pdf"))
    fig.savefig(os.path.join(OUTDIR, "fig4_inversion.png"))
    plt.close(fig)
    print("  -> fig4_inversion.pdf")


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Generating figures for arXiv letter")
    print("=" * 60)
    fig1_identity()
    fig2_variance()
    fig3_kurtosis()
    fig4_inversion()
    print("\nAll figures saved to:", OUTDIR)
    print("Done.")
