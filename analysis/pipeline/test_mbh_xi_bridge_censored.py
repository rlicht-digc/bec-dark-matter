#!/usr/bin/env python3
"""
Censored MBH bridge fit (detections + upper limits).

Reads the direct-bridge per-galaxy table and runs:
  1) Detection-only OLS (for reproducibility baseline)
  2) Tobit-style censored Gaussian MLE on y = log10(MBH)

Upper-limit rows are handled with CDF terms at the reported limit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import median_abs_deviation, norm


def _float(x: Any) -> float:
    return float(x) if np.isfinite(x) else float("nan")


def sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, np.ndarray):
        return [sanitize_json(v) for v in obj.tolist()]
    return obj


def ols_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    A = np.column_stack([np.ones_like(x), x])
    beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ beta
    resid = y - yhat
    rms = float(np.sqrt(np.mean(resid ** 2)))
    mad = float(median_abs_deviation(resid, scale="normal"))
    sigma = float(np.std(resid, ddof=2)) if len(resid) > 2 else float(np.std(resid))
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "sigma": float(max(sigma, 1e-8)),
        "rms": rms,
        "mad": mad,
    }


def tobit_fit(x: np.ndarray, y: np.ndarray, is_upper_limit: np.ndarray) -> Dict[str, Any]:
    det = ~is_upper_limit
    if det.sum() < 2:
        raise ValueError("Need >=2 detections for stable initialization.")

    init = ols_fit(x[det], y[det])
    p0 = np.array([init["intercept"], init["slope"], np.log(init["sigma"])], dtype=float)

    def nll(params: np.ndarray) -> float:
        a, b, log_sigma = params
        sigma = np.exp(log_sigma)
        mu = a + b * x
        z = (y - mu) / sigma
        ll_det = (-0.5 * np.log(2.0 * np.pi) - np.log(sigma) - 0.5 * z[det] ** 2).sum()
        ll_ul = log_ndtr(z[is_upper_limit]).sum()
        val = -(ll_det + ll_ul)
        if not np.isfinite(val):
            return 1e100
        return float(val)

    res = minimize(
        nll,
        p0,
        method="L-BFGS-B",
        bounds=[(None, None), (None, None), (np.log(1e-6), np.log(10.0))],
    )
    if not res.success:
        res_nm = minimize(nll, p0, method="Nelder-Mead")
        if res_nm.success:
            res = res_nm

    a, b, log_sigma = res.x
    sigma = float(np.exp(log_sigma))
    mu = a + b * x
    z = (y - mu) / sigma
    ll_det = (-0.5 * np.log(2.0 * np.pi) - np.log(sigma) - 0.5 * z[det] ** 2).sum()
    ll_ul = log_ndtr(z[is_upper_limit]).sum()
    loglik = float(ll_det + ll_ul)
    k = 3
    n = len(x)
    aic = float(2 * k - 2 * loglik)
    bic = float(np.log(max(n, 1)) * k - 2 * loglik)

    out: Dict[str, Any] = {
        "success": bool(res.success),
        "message": str(res.message),
        "n_iter": int(getattr(res, "nit", -1)),
        "intercept": float(a),
        "slope": float(b),
        "sigma": sigma,
        "loglik": loglik,
        "aic": aic,
        "bic": bic,
    }

    try:
        if hasattr(res, "hess_inv") and hasattr(res.hess_inv, "todense"):
            cov = np.asarray(res.hess_inv.todense(), dtype=float)
            if cov.shape == (3, 3) and np.all(np.isfinite(cov)):
                se = np.sqrt(np.maximum(np.diag(cov), 0.0))
                out["se_intercept"] = float(se[0])
                out["se_slope"] = float(se[1])
                out["se_log_sigma"] = float(se[2])
    except Exception:
        pass

    return out


def _column_map(x_choice: str, y_choice: str) -> Tuple[str, str, str]:
    x_lookup = {
        "xi": ("log_xi", "log10 xi [kpc]"),
        "mdyn": ("log_Mdyn", "log10 Mdyn [Msun]"),
    }
    y_lookup = {
        "mbh": ("log10_MBH_Msun", "log10 MBH [Msun]"),
    }
    if x_choice not in x_lookup:
        raise ValueError(f"Unsupported x_choice={x_choice}; use one of {sorted(x_lookup)}")
    if y_choice not in y_lookup:
        raise ValueError(f"Unsupported y_choice={y_choice}; use one of {sorted(y_lookup)}")
    x_col, x_label = x_lookup[x_choice]
    y_col, y_label = y_lookup[y_choice]
    return x_col, y_col, f"{y_label} vs {x_label}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Censored MBH bridge fit")
    parser.add_argument("--in_dir", type=str, required=True, help="Input direct-bridge folder")
    parser.add_argument("--out_dir", type=str, required=True, help="Output folder")
    parser.add_argument("--x_choice", type=str, default="xi", choices=["xi", "mdyn"])
    parser.add_argument("--y_choice", type=str, default="mbh", choices=["mbh"])
    parser.add_argument("--method", type=str, default="tobit", choices=["tobit"])
    args = parser.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    input_csv = in_dir / "direct_plus_upper_matches.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing required input table: {input_csv}")

    df = pd.read_csv(input_csv)
    if "is_upper_limit" not in df.columns:
        if "UPPERLIMIT" in df.columns:
            df["is_upper_limit"] = df["UPPERLIMIT"].astype(int) == 1
        else:
            raise ValueError("Input CSV needs `is_upper_limit` or `UPPERLIMIT` column.")

    x_col, y_col, title = _column_map(args.x_choice, args.y_choice)
    req = [x_col, y_col, "is_upper_limit", "galaxy"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    df = df[np.isfinite(df[x_col]) & np.isfinite(df[y_col])].copy()
    df["is_upper_limit"] = df["is_upper_limit"].astype(bool)
    det = ~df["is_upper_limit"].to_numpy(dtype=bool)

    if det.sum() < 3:
        raise RuntimeError("Need at least 3 detections for OLS baseline.")

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    is_upper = df["is_upper_limit"].to_numpy(dtype=bool)

    ols = ols_fit(x[det], y[det])
    tobit = tobit_fit(x, y, is_upper)

    x_grid = np.linspace(float(np.min(x)) - 0.05, float(np.max(x)) + 0.05, 200)
    y_ols = ols["intercept"] + ols["slope"] * x_grid
    y_tobit = tobit["intercept"] + tobit["slope"] * x_grid

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=160)
    d_det = df[~df["is_upper_limit"]]
    d_ul = df[df["is_upper_limit"]]
    ax.scatter(d_det[x_col], d_det[y_col], s=40, c="#1f77b4", edgecolors="black", linewidths=0.4, label=f"Detections (N={len(d_det)})")
    if len(d_ul):
        ax.scatter(d_ul[x_col], d_ul[y_col], s=60, marker="v", c="#d62728", edgecolors="black", linewidths=0.4, label=f"Upper limits (N={len(d_ul)})")
    ax.plot(x_grid, y_ols, color="#2ca02c", lw=1.8, label=f"OLS det-only slope={ols['slope']:.3f}")
    ax.plot(x_grid, y_tobit, color="#ff7f0e", lw=1.8, ls="--", label=f"Tobit censored slope={tobit['slope']:.3f}")
    ax.set_xlabel("log10 xi [kpc]" if args.x_choice == "xi" else "log10 Mdyn [Msun]")
    ax.set_ylabel("log10 MBH [Msun]")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    fig_path = out_dir / "fig_mbh_censored_fit.png"
    fig.savefig(fig_path, facecolor="white")
    plt.close(fig)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    n_total = int(len(df))
    n_detections = int((~df["is_upper_limit"]).sum())
    n_upper_limits = int(df["is_upper_limit"].sum())
    ols_flat = {
        "slope": ols.get("slope"),
        "intercept": ols.get("intercept"),
        "sigma": ols.get("sigma"),
        "rms": ols.get("rms"),
        "r2": None,
        "n": n_detections,
    }
    tobit_flat = {
        "slope": tobit.get("slope"),
        "intercept": tobit.get("intercept"),
        "sigma": tobit.get("sigma"),
        "rms": None,
        "nll": (-tobit["loglik"] if tobit.get("loglik") is not None else None),
        "converged": tobit.get("success"),
        "n": n_total,
    }
    summary = {
        "test": "mbh_xi_bridge_censored",
        "timestamp": ts,
        "in_dir": str(in_dir),
        "input_csv": str(input_csv),
        "out_dir": str(out_dir),
        "x_choice": args.x_choice,
        "y_choice": args.y_choice,
        "method": args.method,
        # Flat compatibility keys for older readers.
        "n_total": n_total,
        "n_detections": n_detections,
        "n_upper_limits": n_upper_limits,
        "ols": ols_flat,
        "tobit": tobit_flat,
        "counts": {
            "n_total": n_total,
            "n_detections": n_detections,
            "n_upper_limits": n_upper_limits,
        },
        "models": {
            "ols_detection_only": ols,
            "tobit_censored": tobit,
        },
        "artifacts": {
            "figure": str(fig_path),
        },
    }
    summary_path = out_dir / "summary_mbh_censored_fit.json"
    summary_path.write_text(json.dumps(sanitize_json(summary), indent=2))

    report_lines = [
        "# MBH Bridge Censored Fit Report",
        "",
        f"- Input folder: `{in_dir}`",
        f"- Input table: `{input_csv.name}`",
        f"- Method: `{args.method}`",
        f"- Counts: total={len(df)}, detections={(~df['is_upper_limit']).sum()}, upper_limits={df['is_upper_limit'].sum()}",
        "",
        "## OLS (detections only)",
        f"- intercept={ols['intercept']:.6f}",
        f"- slope={ols['slope']:.6f}",
        f"- sigma={ols['sigma']:.6f}",
        f"- rms={ols['rms']:.6f}",
        f"- mad={ols['mad']:.6f}",
        "",
        "## Tobit censored MLE",
        f"- success={tobit['success']}",
        f"- message={tobit['message']}",
        f"- intercept={_float(tobit['intercept']):.6f}",
        f"- slope={_float(tobit['slope']):.6f}",
        f"- sigma={_float(tobit['sigma']):.6f}",
        f"- loglik={_float(tobit['loglik']):.6f}",
        f"- aic={_float(tobit['aic']):.6f}",
        f"- bic={_float(tobit['bic']):.6f}",
        "",
        f"Figure: `{fig_path.name}`",
        f"Summary: `{summary_path.name}`",
    ]
    (out_dir / "report_mbh_censored_fit.md").write_text("\n".join(report_lines) + "\n")

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote figure:  {fig_path}")
    print(f"OLS slope (detections): {ols['slope']:.6f}")
    print(f"Tobit slope (censored): {tobit['slope']:.6f}")


if __name__ == "__main__":
    main()
