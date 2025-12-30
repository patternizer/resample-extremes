#!/usr/bin/env python3

# monthly-load-estimator.py
#
# michael.taylor@cefas.gov.uk
# 30-Dec-2025

"""
Estimate monthly loads from irregular concentration observations and daily flow.

Key features:
- Fits log(C) ~ smooth(time) + smooth(logQ) + seasonal harmonics
- Optional persistence/hysteresis enhancement:
    --use-persistence
    --n-lags N          (adds logQ_lag1..logQ_lagN)
    --roll-days K       (adds logQ_rollmean_K)
    --use-rising-limb   (adds rising_limb indicator based on daily dQ)
- Optional observation weighting to avoid HF windows dominating the fit:
    --obs-weighting {none,by_month_equal} (default by_month_equal)
- Monthly load = sum_over_days( C_hat_day * Q_day )
- Block bootstrap on residuals (bootstrapped daily residual sequences)
- Optional diagnostic CSV on observation days: obs vs fitted + residuals + period

Expected inputs:
- conc.csv with date column and concentration column (irregular sampling)
- flow.csv with daily (or subdaily) date column and flow column
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix


# -----------------------------
# Utilities
# -----------------------------

def _as_daily_series(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index, errors="coerce").floor("D")
    s = s.dropna()
    s = s.groupby(level=0).mean()
    return s.sort_index()


def _month_end_index(idx: pd.DatetimeIndex) -> pd.PeriodIndex:
    return idx.to_period("M")


def _block_bootstrap_out_indices(n_resid: int, n_out: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap indices."""
    block_len = max(1, int(block_len))
    out = np.empty(n_out, dtype=int)
    pos = 0
    while pos < n_out:
        start = int(rng.integers(0, n_resid))
        block = (start + np.arange(block_len)) % n_resid
        take = min(block_len, n_out - pos)
        out[pos:pos + take] = block[:take]
        pos += take
    return out


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.maximum(x, eps))


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class ModelConfig:
    df_time: int = 12
    df_logq: int = 6
    seasonal_k: int = 2
    cap_quantile: float = 0.999


@dataclass
class PersistenceConfig:
    use_persistence: bool = False
    n_lags: int = 3
    roll_days: int = 7
    use_rising_limb: bool = False


@dataclass
class FitResult:
    model: sm.regression.linear_model.RegressionResultsWrapper
    resid_sd: float

    obs_days: pd.DatetimeIndex
    obs_conc: pd.Series
    obs_flow: pd.Series
    fitted_conc: pd.Series
    log_resid: pd.Series
    weights: pd.Series


# -----------------------------
# Feature engineering
# -----------------------------

def _build_persistence_features(
    df: pd.DataFrame,
    pcfg: PersistenceConfig,
) -> pd.DataFrame:
    """
    df must have columns: Q (positive), and datetime index at daily resolution.
    Returns a DataFrame of extra covariates indexed like df.
    """
    feats = {}

    q = df["Q"].to_numpy(dtype=float)
    logq = _safe_log(q)

    if pcfg.n_lags and pcfg.n_lags > 0:
        for k in range(1, int(pcfg.n_lags) + 1):
            feats[f"logQ_lag{k}"] = pd.Series(logq, index=df.index).shift(k)

    if pcfg.roll_days and pcfg.roll_days > 1:
        w = int(pcfg.roll_days)
        feats[f"logQ_rollmean_{w}"] = pd.Series(logq, index=df.index).rolling(w, min_periods=max(2, w // 2)).mean()

    if pcfg.use_rising_limb:
        dq = pd.Series(q, index=df.index).diff()
        feats["rising_limb"] = (dq > 0).astype(float)

    if not feats:
        return pd.DataFrame(index=df.index)

    return pd.DataFrame(feats, index=df.index)


# -----------------------------
# Model fitting
# -----------------------------

def fit_concentration_model(
    conc_obs: pd.Series,
    q_daily: pd.Series,
    cfg: ModelConfig,
    obs_weighting: str = "by_month_equal",
    pcfg: Optional[PersistenceConfig] = None,
) -> FitResult:
    """
    Fit log(C) regression on observation days.
    """
    if pcfg is None:
        pcfg = PersistenceConfig(use_persistence=False)

    conc_obs = _as_daily_series(conc_obs)
    q_daily = _as_daily_series(q_daily)

    # Align on observation days
    df = pd.DataFrame({
        "C": conc_obs,
        "Q": q_daily.reindex(conc_obs.index),
    }).dropna()

    # Guard against nonpositive flows (log undefined)
    df = df[df["Q"] > 0].copy()
    if df.empty:
        raise ValueError("No overlapping conc/flow data after alignment and Q>0 filtering.")

    # Time covariate: days since start (float)
    t = (df.index - df.index.min()).days.astype(float).to_numpy()
    logq = _safe_log(df["Q"].to_numpy(dtype=float))

    # Spline bases
    B_time = dmatrix(
        f"bs(t, df={cfg.df_time}, degree=3, include_intercept=False)",
        {"t": t},
        return_type="dataframe",
    )
    B_q = dmatrix(
        f"bs(x, df={cfg.df_logq}, degree=3, include_intercept=False)",
        {"x": logq},
        return_type="dataframe",
    )

    # Ensure index alignment
    B_time.index = df.index
    B_q.index = df.index

    # Seasonal harmonics
    doy = df.index.dayofyear.to_numpy(dtype=float)
    season = {}
    for k in range(1, cfg.seasonal_k + 1):
        season[f"sin{k}"] = np.sin(2.0 * np.pi * k * doy / 365.25)
        season[f"cos{k}"] = np.cos(2.0 * np.pi * k * doy / 365.25)
    B_season = pd.DataFrame(season, index=df.index)

    X_parts = [B_time, B_q, B_season]

    # Optional persistence/hysteresis features
    if pcfg.use_persistence:
        extra = _build_persistence_features(df, pcfg)
        X_parts.append(extra)

    X = pd.concat(X_parts, axis=1)
    X = sm.add_constant(X, has_constant="add")

    y = _safe_log(df["C"].to_numpy(dtype=float))

    # weights
    ow = (obs_weighting or "none").strip().lower()
    if ow not in {"none", "by_month_equal"}:
        raise ValueError("--obs-weighting must be one of: none, by_month_equal")

    if ow == "by_month_equal":
        m = _month_end_index(df.index)
        counts = pd.Series(1.0, index=df.index).groupby(m).sum()
        w = 1.0 / counts.reindex(m).to_numpy(dtype=float)
    else:
        w = np.ones(len(df), dtype=float)

    # Drop any rows with missing/inf (lags/rolling introduce NaNs)
    X_np = X.to_numpy(dtype=float)
    good = np.isfinite(X_np).all(axis=1) & np.isfinite(y) & np.isfinite(w) & (w > 0)

    if not np.all(good):
        X = X.loc[good]
        df = df.loc[good]
        y = y[good]
        w = w[good]

    if len(df) < 20:
        raise ValueError(f"Too few usable observations after feature construction: n={len(df)}")

    model = sm.WLS(y, X, weights=w).fit()

    resid = np.asarray(model.resid, dtype=float)
    resid_sd = float(np.sqrt(np.mean(resid ** 2)))
    fitted = np.exp(np.asarray(model.fittedvalues, dtype=float))

    obs_days = pd.DatetimeIndex(df.index)
    obs_conc_s = pd.Series(df["C"].to_numpy(dtype=float), index=obs_days, name="C_obs")
    obs_flow_s = pd.Series(df["Q"].to_numpy(dtype=float), index=obs_days, name="Q_obs")
    fitted_s = pd.Series(fitted, index=obs_days, name="C_fitted")
    resid_s = pd.Series(resid, index=obs_days, name="log_resid")
    w_s = pd.Series(np.asarray(w, dtype=float), index=obs_days, name="weight")

    return FitResult(
        model=model,
        resid_sd=resid_sd,
        obs_days=obs_days,
        obs_conc=obs_conc_s,
        obs_flow=obs_flow_s,
        fitted_conc=fitted_s,
        log_resid=resid_s,
        weights=w_s,
    )


def predict_daily_logC(
    fit: FitResult,
    q_daily: pd.Series,
    cfg: ModelConfig,
    pcfg: Optional[PersistenceConfig] = None,
) -> pd.Series:
    """
    Predict daily log(C) for the full daily flow index.
    """
    if pcfg is None:
        pcfg = PersistenceConfig(use_persistence=False)

    q_daily = _as_daily_series(q_daily)
    q_daily = q_daily[q_daily > 0].copy()
    if q_daily.empty:
        raise ValueError("Daily flow series is empty after filtering Q>0.")

    full_index = pd.DatetimeIndex(q_daily.index)
    t_full = (full_index - fit.obs_days.min()).days.astype(float).to_numpy()
    logq_full = _safe_log(q_daily.to_numpy(dtype=float))

    B_time = dmatrix(
        f"bs(t, df={cfg.df_time}, degree=3, include_intercept=False)",
        {"t": t_full},
        return_type="dataframe",
    )
    B_q = dmatrix(
        f"bs(x, df={cfg.df_logq}, degree=3, include_intercept=False)",
        {"x": logq_full},
        return_type="dataframe",
    )
    B_time.index = full_index
    B_q.index = full_index

    doy = full_index.dayofyear.to_numpy(dtype=float)
    season = {}
    for k in range(1, cfg.seasonal_k + 1):
        season[f"sin{k}"] = np.sin(2.0 * np.pi * k * doy / 365.25)
        season[f"cos{k}"] = np.cos(2.0 * np.pi * k * doy / 365.25)
    B_season = pd.DataFrame(season, index=full_index)

    X_parts = [B_time, B_q, B_season]

    if pcfg.use_persistence:
        # Need a DataFrame with Q for feature builder
        df_full = pd.DataFrame({"Q": q_daily.to_numpy(dtype=float)}, index=full_index)
        extra = _build_persistence_features(df_full, pcfg)
        X_parts.append(extra)

    X_full = pd.concat(X_parts, axis=1)
    X_full = sm.add_constant(X_full, has_constant="add")

    X_full = X_full.replace([np.inf, -np.inf], np.nan).dropna()
    logC_hat = pd.Series(fit.model.predict(X_full), index=X_full.index, name="logC_hat")
    return logC_hat


# -----------------------------
# Monthly loads + CI
# -----------------------------

def monthly_load_with_ci(
    conc_obs: pd.Series,
    q_daily: pd.Series,
    cfg: ModelConfig,
    pcfg: PersistenceConfig,
    n_boot: int,
    seed: int,
    block_len_resid: int,
    obs_weighting: str,
) -> Tuple[pd.DataFrame, FitResult]:

    rng = np.random.default_rng(seed)
    fit = fit_concentration_model(conc_obs, q_daily, cfg, obs_weighting=obs_weighting, pcfg=pcfg)

    q_daily = _as_daily_series(q_daily)
    q_daily = q_daily[q_daily > 0].copy()

    logC_hat = predict_daily_logC(fit, q_daily, cfg, pcfg=pcfg)
    q_aligned = q_daily.reindex(logC_hat.index)

    C_hat = np.exp(logC_hat.to_numpy(dtype=float))
    cap = float(np.quantile(C_hat, cfg.cap_quantile))
    C_hat = np.minimum(C_hat, cap)

    daily_load = pd.Series(C_hat * q_aligned.to_numpy(dtype=float), index=logC_hat.index, name="daily_load")

    m_load = daily_load.groupby(pd.Grouper(freq="ME")).sum(min_count=1)
    m_qsum = q_aligned.groupby(pd.Grouper(freq="ME")).sum(min_count=1)
    m_cq = (pd.Series(C_hat, index=logC_hat.index) * q_aligned).groupby(pd.Grouper(freq="ME")).sum(min_count=1)
    m_fwmc = m_cq / m_qsum

    out = pd.DataFrame({
        "MonthEnd": m_load.index,
        "MonthlyLoad": m_load.values,
        "MonthlyFlowSum": m_qsum.values,
        "FWMC": m_fwmc.values,
    }).set_index("MonthEnd")

    resid = fit.log_resid.to_numpy(dtype=float)
    if len(resid) < 5 or n_boot <= 0:
        out["MonthlyLoad_CI_Lo"] = np.nan
        out["MonthlyLoad_CI_Hi"] = np.nan
        return out.reset_index(), fit

    n_days = len(logC_hat)
    n_res = len(resid)

    boot_monthly = np.zeros((n_boot, out.shape[0]), dtype=float)
    for b in range(n_boot):
        boot_idx = _block_bootstrap_out_indices(n_res, n_days, block_len_resid, rng)
        eps_daily = resid[boot_idx]
        logC_b = logC_hat.to_numpy(dtype=float) + eps_daily
        C_b = np.exp(logC_b)
        C_b = np.minimum(C_b, cap)

        daily_load_b = pd.Series(C_b * q_aligned.to_numpy(dtype=float), index=logC_hat.index)
        m_b = daily_load_b.groupby(pd.Grouper(freq="ME")).sum(min_count=1).reindex(out.index)
        boot_monthly[b, :] = m_b.to_numpy(dtype=float)

    out["MonthlyLoad_CI_Lo"] = np.nanpercentile(boot_monthly, 2.5, axis=0)
    out["MonthlyLoad_CI_Hi"] = np.nanpercentile(boot_monthly, 97.5, axis=0)

    return out.reset_index(), fit


# -----------------------------
# Diagnostics CSV
# -----------------------------

def diagnostics_dataframe(fit: FitResult, hf_start: Optional[str], hf_end: Optional[str]) -> pd.DataFrame:
    df = pd.DataFrame({
        "Date": fit.obs_days,
        "C_obs": fit.obs_conc.values,
        "C_fitted": fit.fitted_conc.values,
        "log_resid": fit.log_resid.values,
        "Q_obs": fit.obs_flow.values,
        "weight": fit.weights.values,
    })
    df["Date"] = pd.to_datetime(df["Date"]).dt.floor("D")

    if hf_start and hf_end:
        hs, he = pd.Timestamp(hf_start), pd.Timestamp(hf_end)
        df["period"] = np.where(df["Date"] < hs, "pre", np.where(df["Date"] > he, "post", "hf"))
    else:
        df["period"] = "all"

    return df.sort_values("Date")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monthly load estimator with HF-robust observation weighting.")
    p.add_argument("--conc-csv", required=True)
    p.add_argument("--flow-csv", required=True)
    p.add_argument("--conc-time-col", default="Date")
    p.add_argument("--conc-value-col", default="Concentration")
    p.add_argument("--flow-time-col", default="Date")
    p.add_argument("--flow-value-col", default="Flow")

    p.add_argument("--out-csv", required=True)
    p.add_argument("--diag-csv", default=None)
    p.add_argument("--hf-start", default=None)
    p.add_argument("--hf-end", default=None)

    p.add_argument("--obs-weighting", choices=["none", "by_month_equal"], default="by_month_equal")

    p.add_argument("--df-time", type=int, default=12)
    p.add_argument("--df-logq", type=int, default=6)
    p.add_argument("--seasonal-k", type=int, default=2)
    p.add_argument("--cap-quantile", type=float, default=0.999)

    # Persistence/hysteresis enhancement
    p.add_argument("--use-persistence", action="store_true", help="Enable antecedent/hysteresis covariates.")
    p.add_argument("--n-lags", type=int, default=3, help="Number of daily logQ lags to include (default: 3).")
    p.add_argument("--roll-days", type=int, default=7, help="Rolling window (days) for logQ mean (default: 7).")
    p.add_argument("--use-rising-limb", action="store_true", help="Add rising-limb indicator based on dQ>0.")

    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--block-len-resid", type=int, default=7)

    return p.parse_args()


def main():
    args = parse_args()

    conc_df = pd.read_csv(args.conc_csv)
    flow_df = pd.read_csv(args.flow_csv)

    if args.conc_time_col not in conc_df.columns:
        raise KeyError(f"Concentration CSV missing time column '{args.conc_time_col}'. Found {conc_df.columns.tolist()}")
    if args.conc_value_col not in conc_df.columns:
        raise KeyError(f"Concentration CSV missing value column '{args.conc_value_col}'. Found {conc_df.columns.tolist()}")

    if args.flow_time_col not in flow_df.columns:
        raise KeyError(f"Flow CSV missing time column '{args.flow_time_col}'. Found {flow_df.columns.tolist()}")
    if args.flow_value_col not in flow_df.columns:
        raise KeyError(f"Flow CSV missing value column '{args.flow_value_col}'. Found {flow_df.columns.tolist()}")

    conc_df[args.conc_time_col] = pd.to_datetime(conc_df[args.conc_time_col], errors="coerce")
    flow_df[args.flow_time_col] = pd.to_datetime(flow_df[args.flow_time_col], errors="coerce")

    conc = conc_df.dropna(subset=[args.conc_time_col]).set_index(args.conc_time_col)[args.conc_value_col].astype(float)
    flow = flow_df.dropna(subset=[args.flow_time_col]).set_index(args.flow_time_col)[args.flow_value_col].astype(float)

    conc = _as_daily_series(conc)
    flow = _as_daily_series(flow)

    cfg = ModelConfig(
        df_time=args.df_time,
        df_logq=args.df_logq,
        seasonal_k=args.seasonal_k,
        cap_quantile=args.cap_quantile,
    )

    pcfg = PersistenceConfig(
        use_persistence=bool(args.use_persistence),
        n_lags=int(args.n_lags),
        roll_days=int(args.roll_days),
        use_rising_limb=bool(args.use_rising_limb),
    )

    monthly, fit = monthly_load_with_ci(
        conc_obs=conc,
        q_daily=flow,
        cfg=cfg,
        pcfg=pcfg,
        n_boot=args.n_boot,
        seed=args.seed,
        block_len_resid=args.block_len_resid,
        obs_weighting=args.obs_weighting,
    )

    monthly.to_csv(args.out_csv, index=False, float_format="%.6g")
    print(f"Wrote: {args.out_csv}")

    if args.diag_csv:
        ddf = diagnostics_dataframe(fit, args.hf_start, args.hf_end)
        ddf.to_csv(args.diag_csv, index=False, float_format="%.6g")
        print(f"Wrote diagnostics: {args.diag_csv}")


if __name__ == "__main__":
    main()

