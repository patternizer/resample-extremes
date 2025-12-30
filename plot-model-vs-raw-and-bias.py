#!/usr/bin/env python3

# plot-model-vs-raw-and-bias.py
#
# michael.taylor@cefas.gov.uk
# 30-Dec-2025

#!/usr/bin/env python3
"""
plot-model-vs-raw-and-bias.py

Improved plotting:
- robust scaling for outliers (quantile clipping)
- optional log axes
- obs-vs-fitted scatter/hexbin
- residual diagnostics (hist + ACF + QQ + residual vs fitted)
- HF thinning comparison (sampling-intensity bias check)

NEW (Dec-2025 patch):
- Auto-detect actual triggered HF period from conc.csv sampling intensity:
    --auto-hf (default True)
  This is intended for "triggered then sustained ~daily HF for ~5 years" synthetic data.

Inputs:
  daily_truth.csv: Date, Concentration   (true daily concentration)
  flow.csv:        Date, Flow
  conc.csv:        Date, Concentration   (observed samples, irregular)
  monthly-loads.csv from estimator
  diag.csv from estimator (optional; avoids refitting)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--daily-truth-csv", required=True)
    p.add_argument("--conc-csv", required=True)
    p.add_argument("--flow-csv", required=True)
    p.add_argument("--monthly-csv", required=True)
    p.add_argument("--diag-csv", default=None)

    # HF window: can still be provided, but auto detection can override
    p.add_argument("--hf-start", default="2000-01-01")
    p.add_argument("--hf-end", default="2005-12-31")

    # NEW: auto HF detection from conc sampling intensity
    p.add_argument("--auto-hf", action="store_true", default=True,
                   help="Auto-detect triggered HF period from sampling intensity in conc.csv (default: on).")
    p.add_argument("--no-auto-hf", action="store_false", dest="auto_hf",
                   help="Disable auto HF detection; use --hf-start/--hf-end as provided.")

    p.add_argument("--auto-hf-window-days", type=int, default=30,
                   help="Window (days) for rolling sampling intensity used in HF detection (default: 30).")
    p.add_argument("--auto-hf-threshold", type=float, default=10.0,
                   help="Threshold for rolling count to classify HF-active (default: 10 obs per window).")
    p.add_argument("--auto-hf-gap-days", type=int, default=14,
                   help="Allow gaps up to this many days inside HF segment (default: 14).")

    # Presentation/scaling
    p.add_argument("--conc-y", choices=["linear", "log"], default="log",
                   help="Y-scale for concentration/FWMC plots. log recommended.")
    p.add_argument("--clip-q", type=float, default=0.995,
                   help="Upper quantile for ylim clipping. Set <=0 or >=1 to disable.")
    p.add_argument("--trim-q", type=float, default=0.99,
                   help="Trim quantile for optional trimmed daily plot. Set <=0 or >=1 to disable.")
    p.add_argument("--scatter-scale", choices=["linear", "loglog"], default="loglog",
                   help="Scale for obs-vs-fitted plot.")
    p.add_argument("--use-hexbin", action="store_true",
                   help="Use hexbin instead of scatter for obs-vs-fitted.")

    p.add_argument("--hf-thin-interval-days", type=int, default=7)
    p.add_argument("--log-load-axis", action="store_true")

    # Optional sampling strip display (kept for compatibility; if your script uses it)
    p.add_argument("--sampling-strip", action="store_true", help="Add a thin strip showing sample days (if implemented).")

    p.add_argument("--outdir", default="plots_out")

    return p.parse_args()


# ----------------------------
# Styling helpers
# ----------------------------

def set_style():
    plt.rcParams.update({
        "figure.figsize": (11.5, 6.2),
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "savefig.dpi": 220,
    })


def _read_csv_with_date_index(path: str, date_col_guess: str = "Date") -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col_guess not in df.columns:
        # fallback: first column that looks like date
        for c in df.columns:
            if str(c).lower() in ("date", "datetime", "time", "timestamp", "index"):
                date_col_guess = c
                break
        else:
            date_col_guess = df.columns[0]
    df[date_col_guess] = pd.to_datetime(df[date_col_guess], errors="coerce")
    df = df.dropna(subset=[date_col_guess]).sort_values(date_col_guess).set_index(date_col_guess)
    df.index = df.index.floor("D")
    return df


def ensure_daily_unique(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index, errors="coerce").floor("D")
    s = s.dropna()
    s = s.groupby(level=0).mean()
    return s.sort_index()


def apply_date_formatting(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))


def maybe_set_log_y(ax, yscale: str):
    if yscale == "log":
        ax.set_yscale("log")


def clip_ylim(ax, y: np.ndarray, clip_q: float):
    if clip_q is None or clip_q <= 0 or clip_q >= 1:
        return
    y = np.asarray(y)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return
    hi = np.quantile(y, clip_q)
    lo = np.quantile(y, 0.01)
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        ax.set_ylim(bottom=lo, top=hi)


# ----------------------------
# HF detection
# ----------------------------

def detect_hf_period_from_conc_sampling(
    conc_obs: pd.Series,
    window_days: int = 30,
    threshold: float = 10.0,
    gap_days: int = 14,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """
    Detect a sustained HF period based on rolling observation counts.
    Intended for synthetic data where HF sampling is near-daily for years.

    Method:
    - build a daily indicator series (1 on observation days)
    - rolling sum over window_days
    - classify day as HF-active if rolling_sum >= threshold
    - find longest contiguous HF-active segment, allowing small gaps (<= gap_days)
    """
    if conc_obs.empty:
        return None, None

    idx = pd.DatetimeIndex(conc_obs.index).sort_values().floor("D").unique()
    d0, d1 = idx.min(), idx.max()
    daily = pd.Series(0.0, index=pd.date_range(d0, d1, freq="D"))
    daily.loc[idx] = 1.0

    w = max(7, int(window_days))
    roll = daily.rolling(w, min_periods=max(3, w // 4)).sum()
    hf_flag = roll >= float(threshold)

    # Convert hf_flag True days to segments allowing gaps <= gap_days
    true_days = hf_flag[hf_flag].index
    if len(true_days) == 0:
        return None, None

    gap = pd.Timedelta(days=max(0, int(gap_days)))

    segments = []
    start = true_days[0]
    prev = true_days[0]
    for t in true_days[1:]:
        if (t - prev) <= gap + pd.Timedelta(days=1):
            prev = t
        else:
            segments.append((start, prev))
            start = t
            prev = t
    segments.append((start, prev))

    # Choose longest segment
    best = max(segments, key=lambda ab: (ab[1] - ab[0]).days)
    return pd.Timestamp(best[0]), pd.Timestamp(best[1])


def shade_hf(ax, hf_start, hf_end, label="HF"):
    if hf_start is None or hf_end is None:
        return
    ax.axvspan(hf_start, hf_end, alpha=0.12)
    ax.text(hf_start, 0.98, label, transform=ax.get_xaxis_transform(), va="top", ha="left")


def add_period_label(idx: pd.DatetimeIndex, hf_start: pd.Timestamp | None, hf_end: pd.Timestamp | None) -> pd.Series:
    period = pd.Series(index=idx, dtype="object")
    if hf_start is None or hf_end is None:
        period.loc[:] = "all"
        return period
    period.loc[idx < hf_start] = "pre"
    period.loc[(idx >= hf_start) & (idx <= hf_end)] = "hf"
    period.loc[idx > hf_end] = "post"
    return period


# ----------------------------
# Monthly computations
# ----------------------------

def compute_true_monthly(truth_conc: pd.Series, flow_daily: pd.Series) -> pd.DataFrame:
    c = ensure_daily_unique(truth_conc).rename("C_true")
    q = ensure_daily_unique(flow_daily).rename("Q")
    df = pd.concat([c, q], axis=1).dropna()
    df = df[df["Q"] > 0].copy()

    df["load_day"] = df["C_true"] * df["Q"]
    m_load = df["load_day"].groupby(pd.Grouper(freq="ME")).sum(min_count=1)
    m_q = df["Q"].groupby(pd.Grouper(freq="ME")).sum(min_count=1)
    m_fwmc = (df["C_true"] * df["Q"]).groupby(pd.Grouper(freq="ME")).sum(min_count=1) / m_q

    out = pd.DataFrame({"TrueMonthlyLoad": m_load, "TrueFWMC": m_fwmc})
    out.index.name = "MonthEnd"
    return out


def compute_naive_monthly_from_samples(conc_obs: pd.Series, flow_daily: pd.Series, prefix: str) -> pd.DataFrame:
    c_m = conc_obs.groupby(pd.Grouper(freq="ME")).mean()
    q_m = flow_daily.groupby(pd.Grouper(freq="ME")).sum(min_count=1)
    load_m = c_m * q_m
    n_m = conc_obs.groupby(pd.Grouper(freq="ME")).size()
    return pd.DataFrame({f"{prefix}Cmean": c_m, f"{prefix}MonthlyLoad": load_m, f"{prefix}N_Obs": n_m})


def hf_thin_to_interval(conc_obs: pd.Series, hf_start: pd.Timestamp | None, hf_end: pd.Timestamp | None, interval_days: int) -> pd.Series:
    interval_days = max(1, int(interval_days))
    s = conc_obs.copy().sort_index()

    if hf_start is None or hf_end is None:
        # Nothing to thin
        return s

    in_hf = (s.index >= hf_start) & (s.index <= hf_end)
    s_hf = s.loc[in_hf].sort_index()
    s_other = s.loc[~in_hf].sort_index()
    if s_hf.empty:
        return s

    kept_idx = []
    last = None
    for t in s_hf.index:
        if last is None or (t - last).days >= interval_days:
            kept_idx.append(t)
            last = t
    return pd.concat([s_other, s_hf.loc[kept_idx]]).sort_index()


def read_monthly_model(csv_path: str) -> pd.DataFrame:
    m = pd.read_csv(csv_path)
    if "MonthEnd" in m.columns:
        m["MonthEnd"] = pd.to_datetime(m["MonthEnd"], errors="coerce")
        m = m.dropna(subset=["MonthEnd"]).set_index("MonthEnd").sort_index()
    else:
        c0 = m.columns[0]
        m[c0] = pd.to_datetime(m[c0], errors="coerce")
        m = m.dropna(subset=[c0]).set_index(c0).sort_index()
    return m


def read_diag(diag_csv: str) -> pd.DataFrame:
    d = pd.read_csv(diag_csv)
    if "Date" not in d.columns:
        raise KeyError(f"diag.csv must contain 'Date'. Found: {d.columns.tolist()}")
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce").dt.floor("D")
    d = d.dropna(subset=["Date"]).sort_values("Date")
    return d


# ----------------------------
# Main
# ----------------------------

def main():
    args = parse_args()
    set_style()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # read flow
    flow_df = _read_csv_with_date_index(args.flow_csv, "Date")
    if "Flow" not in flow_df.columns:
        raise KeyError(f"flow.csv must contain 'Flow'. Found: {flow_df.columns.tolist()}")
    flow = ensure_daily_unique(flow_df["Flow"].astype(float)).rename("Flow")

    # read truth concentration
    truth_df = _read_csv_with_date_index(args.daily_truth_csv, "Date")
    if "Concentration_true" in truth_df.columns:
        truth_c = ensure_daily_unique(truth_df["Concentration_true"].astype(float))
    elif "Concentration" in truth_df.columns:
        truth_c = ensure_daily_unique(truth_df["Concentration"].astype(float))
    else:
        raise KeyError(f"daily_truth.csv must contain 'Concentration_true' or 'Concentration'. Found: {truth_df.columns.tolist()}")

    # observed concentration
    conc_df = _read_csv_with_date_index(args.conc_csv, "Date")
    if "Concentration" not in conc_df.columns:
        raise KeyError(f"conc.csv must contain 'Concentration'. Found: {conc_df.columns.tolist()}")
    conc_obs = ensure_daily_unique(conc_df["Concentration"].astype(float))

    # HF window: either auto-detected, or from CLI
    hf_start_cli = pd.Timestamp(args.hf_start) if args.hf_start else None
    hf_end_cli = pd.Timestamp(args.hf_end) if args.hf_end else None

    if args.auto_hf:
        hf_start, hf_end = detect_hf_period_from_conc_sampling(
            conc_obs,
            window_days=args.auto_hf_window_days,
            threshold=args.auto_hf_threshold,
            gap_days=args.auto_hf_gap_days,
        )
        if hf_start is None or hf_end is None:
            hf_start, hf_end = hf_start_cli, hf_end_cli
            hf_label = "HF (CLI)"
            print("Auto HF detection: no HF segment detected; falling back to --hf-start/--hf-end.")
        else:
            hf_label = "HF (auto)"
            print(f"Auto HF detection: {hf_start.date()} .. {hf_end.date()}  (duration ~{(hf_end-hf_start).days+1} days)")
    else:
        hf_start, hf_end = hf_start_cli, hf_end_cli
        hf_label = "HF (CLI)"

    # monthly outputs
    model_m = read_monthly_model(args.monthly_csv)

    true_m = compute_true_monthly(truth_c, flow)
    naive_m = compute_naive_monthly_from_samples(conc_obs, flow, prefix="Naive_")
    conc_thin = hf_thin_to_interval(conc_obs, hf_start, hf_end, args.hf_thin_interval_days)
    thin_m = compute_naive_monthly_from_samples(conc_thin, flow, prefix="Thin_")

    keep_model_cols = [c for c in ["MonthlyLoad", "FWMC", "MonthlyFlowSum",
                                  "MonthlyLoad_CI_Lo", "MonthlyLoad_CI_Hi"] if c in model_m.columns]

    m = true_m.join(naive_m, how="outer").join(thin_m, how="outer").join(model_m[keep_model_cols], how="outer")
    m["period"] = add_period_label(m.index, hf_start, hf_end)

    # --- Plot 01: daily truth vs obs ---
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(truth_c.index, truth_c.values, linewidth=1.0, label="True daily concentration")
    ax.scatter(conc_obs.index, conc_obs.values, s=8, alpha=0.5, label="Observed samples")
    shade_hf(ax, hf_start, hf_end, label=hf_label)
    maybe_set_log_y(ax, args.conc_y)
    clip_ylim(ax, truth_c.values, args.clip_q)
    apply_date_formatting(ax)
    ax.set_title("Daily concentration: truth vs observations")
    ax.set_xlabel("Date")
    ax.set_ylabel("Concentration")
    ax.legend(loc="upper right")
    fig.savefig(outdir / "01_daily_truth_vs_samples.png")
    plt.close(fig)

    # --- Plot 01b: zoom around HF ---
    fig, ax = plt.subplots(constrained_layout=True)
    if hf_start is not None and hf_end is not None:
        pad = pd.Timedelta(days=365)
        t0, t1 = hf_start - pad, hf_end + pad
    else:
        # fallback: mid-range zoom
        t0, t1 = truth_c.index.min(), truth_c.index.min() + pd.Timedelta(days=365 * 7)

    z_truth = truth_c.loc[(truth_c.index >= t0) & (truth_c.index <= t1)]
    z_obs = conc_obs.loc[(conc_obs.index >= t0) & (conc_obs.index <= t1)]
    ax.plot(z_truth.index, z_truth.values, linewidth=1.0, label="True daily concentration")
    ax.scatter(z_obs.index, z_obs.values, s=10, alpha=0.6, label="Observed samples")
    shade_hf(ax, hf_start, hf_end, label=hf_label)
    maybe_set_log_y(ax, args.conc_y)
    clip_ylim(ax, z_truth.values, args.clip_q)
    ax.set_title("Zoom around HF period: truth vs observations")
    ax.set_xlabel("Date")
    ax.set_ylabel("Concentration")
    ax.legend(loc="upper right")
    fig.savefig(outdir / "01b_zoom_hf_truth_vs_samples.png")
    plt.close(fig)

    # --- Plot 02: monthly FWMC ---
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(m.index, m["TrueFWMC"], linewidth=1.3, label="True monthly FWMC")
    ax.plot(m.index, m["Naive_Cmean"], linewidth=1.0, label="Naive monthly mean(C_samples)")
    if "FWMC" in m.columns:
        ax.plot(m.index, m["FWMC"], linewidth=1.1, label="Model monthly FWMC")
    shade_hf(ax, hf_start, hf_end, label=hf_label)
    maybe_set_log_y(ax, args.conc_y)
    clip_ylim(ax, m["TrueFWMC"].values, args.clip_q)
    apply_date_formatting(ax)
    ax.set_title("Monthly FWMC: truth vs naive vs model")
    ax.set_xlabel("Month")
    ax.set_ylabel("FWMC")
    ax.legend(loc="upper left")
    fig.savefig(outdir / "02_monthly_fwmc_comparison.png")
    plt.close(fig)

    # --- Plot 02b: sampling intensity ---
    fig, ax = plt.subplots(constrained_layout=True)
    ax.bar(m.index, m["Naive_N_Obs"].fillna(0).values, width=25, alpha=0.75, label="Samples per month (raw)")
    shade_hf(ax, hf_start, hf_end, label=hf_label)
    apply_date_formatting(ax)
    ax.set_title("Sampling intensity: observations per month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    fig.savefig(outdir / "02b_sampling_intensity_counts.png")
    plt.close(fig)

    # --- Plot 03: monthly load ---
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(m.index, m["TrueMonthlyLoad"], linewidth=1.3, label="True monthly load")
    ax.plot(m.index, m["Naive_MonthlyLoad"], linewidth=1.0, label="Naive load (mean(C_samples)*sum(Q))")
    if "MonthlyLoad" in m.columns:
        ax.plot(m.index, m["MonthlyLoad"], linewidth=1.1, label="Model monthly load")
    shade_hf(ax, hf_start, hf_end, label=hf_label)
    if args.log_load_axis:
        ax.set_yscale("log")
    apply_date_formatting(ax)
    ax.set_title("Monthly load: truth vs naive vs model")
    ax.set_xlabel("Month")
    ax.set_ylabel("Load")
    ax.legend(loc="upper left")
    fig.savefig(outdir / "03_monthly_load_comparison.png")
    plt.close(fig)

    # --- Plot 04: error boxplots (pre/hf/post) ---
    eps = 1e-12
    m["rel_err_load_naive"] = (m["Naive_MonthlyLoad"] - m["TrueMonthlyLoad"]) / (np.abs(m["TrueMonthlyLoad"]) + eps)
    if "MonthlyLoad" in m.columns:
        m["rel_err_load_model"] = (m["MonthlyLoad"] - m["TrueMonthlyLoad"]) / (np.abs(m["TrueMonthlyLoad"]) + eps)

        fig, ax = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
        periods = ["pre", "hf", "post"]
        data, labels = [], []
        for p_lab in periods:
            data.append(m.loc[m["period"] == p_lab, "rel_err_load_naive"].dropna().values); labels.append(f"Naive ({p_lab})")
            data.append(m.loc[m["period"] == p_lab, "rel_err_load_model"].dropna().values); labels.append(f"Model ({p_lab})")
        ax.boxplot(data, tick_labels=labels, showfliers=False)  # tick_labels avoids Matplotlib 3.9 deprecation
        ax.axhline(0.0, linewidth=1)
        ax.set_title("Monthly load relative error by period")
        ax.set_xlabel("Estimator / period")
        ax.set_ylabel("Relative error: (estimate - true) / |true|")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.savefig(outdir / "04_relative_error_load_boxplots.png")
        plt.close(fig)

    # --- Plot 05: HF thinning comparison ---
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(m.index, m["TrueFWMC"], linewidth=1.3, label="True monthly FWMC")
    ax.plot(m.index, m["Naive_Cmean"], linewidth=1.0, label="Naive mean(C_samples) (all samples)")
    ax.plot(m.index, m["Thin_Cmean"], linewidth=1.0,
            label=f"Naive mean(C_samples) (HF thinned ~{args.hf_thin_interval_days}d)")
    shade_hf(ax, hf_start, hf_end, label=hf_label)
    maybe_set_log_y(ax, args.conc_y)
    clip_ylim(ax, m["TrueFWMC"].values, args.clip_q)
    apply_date_formatting(ax)
    ax.set_title("HF thinning test: sampling-intensity bias in naive monthly mean")
    ax.set_xlabel("Month")
    ax.set_ylabel("Concentration (FWMC scale)")
    ax.legend(loc="upper left")
    fig.savefig(outdir / "05_hf_thinning_comparison.png")
    plt.close(fig)

    # --- Diagnostics from diag.csv ---
    if args.diag_csv:
        d = read_diag(args.diag_csv)

        # Use auto HF for period labels if available
        if "period" not in d.columns or (args.auto_hf and hf_start is not None and hf_end is not None):
            d["period"] = add_period_label(pd.DatetimeIndex(d["Date"]), hf_start, hf_end).values

        # Obs vs fitted
        fig, ax = plt.subplots(figsize=(8.6, 6.4), constrained_layout=True)
        x = d["C_obs"].to_numpy(float)
        y = d["C_fitted"].to_numpy(float)
        per = d["period"].astype(str).values

        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        x, y, per = x[mask], y[mask], per[mask]
        hf_mask = (per == "hf")

        if args.scatter_scale == "loglog":
            ax.set_xscale("log")
            ax.set_yscale("log")

        if args.use_hexbin:
            ax.hexbin(x, y, gridsize=60, mincnt=1)
        else:
            ax.scatter(x[~hf_mask], y[~hf_mask], s=18, alpha=0.45, label="Non-HF")
            ax.scatter(x[hf_mask], y[hf_mask], s=18, alpha=0.55, label="HF")

        lo = float(np.nanmin([x.min(), y.min()]))
        hi = float(np.nanmax([x.max(), y.max()]))
        ax.plot([lo, hi], [lo, hi], linewidth=1.0, label="1:1")

        ax.set_title("Observed vs fitted concentration (diag.csv)")
        ax.set_xlabel("Observed concentration")
        ax.set_ylabel("Fitted concentration")
        ax.legend(loc="upper left")
        fig.savefig(outdir / "06_obs_vs_fitted_scatter.png")
        plt.close(fig)

        # Residual diagnostics
        resid = d["log_resid"].to_numpy(float)
        fitc = d["C_fitted"].to_numpy(float)
        t = pd.to_datetime(d["Date"])

        ok = np.isfinite(resid) & np.isfinite(fitc) & (fitc > 0)
        resid = resid[ok]
        fitc = fitc[ok]
        t = t[ok]

        rstd = resid / (np.std(resid) + 1e-12)

        fig = plt.figure(figsize=(16.5, 9.5), constrained_layout=True)
        gs = fig.add_gridspec(2, 3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(np.log(fitc), resid, s=18, alpha=0.45)
        ax1.axhline(0.0, linewidth=1)
        ax1.set_title("Residual vs log(fitted)")
        ax1.set_xlabel("log(Fitted concentration)")
        ax1.set_ylabel("Log residual")

        ax2 = fig.add_subplot(gs[0, 1])
        sm.qqplot(rstd, line="45", ax=ax2)
        ax2.set_title("Q–Q plot (standardised log residuals)")

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(t, resid, linewidth=0.9)
        ax3.axhline(0.0, linewidth=1)
        shade_hf(ax3, hf_start, hf_end, label=hf_label)
        ax3.set_title("Residuals over time")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Log residual")

        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(resid[np.isfinite(resid)], bins=60)
        ax4.set_title("Residual histogram (log residuals)")
        ax4.set_xlabel("Log residual")
        ax4.set_ylabel("Count")

        ax5 = fig.add_subplot(gs[1, 1:])
        plot_acf(resid, ax=ax5, lags=60, alpha=0.05)
        ax5.set_title("Residual ACF (log residuals)")

        fig.savefig(outdir / "07_residual_diagnostics.png")
        plt.close(fig)

    print(f"Wrote plots to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
