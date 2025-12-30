#!/usr/bin/env python3

# generate-synthetic-river-inputs.py
#
# michael.taylor@cefas.gov.uk
# 30-Dec-2025

"""
Generate synthetic daily flow, true concentration, and observed irregular concentration
with storm-driven dynamics and a triggered high-frequency (HF) sampling programme.

Design goals (for load estimation stress-testing):
- Short, intense storms that can dominate monthly load.
- Daily flow always available.
- Concentration observed mostly weekly, with a HF window that is triggered.
- Concentration exhibits persistence (antecedent / memory) and hysteresis (rising vs falling limb).
- Weekly sampling has a meaningful probability of missing storm peaks.
- HF sampling is responsive to elevated runoff / concentration (informative sampling).

Outputs (default):
- flow.csv        columns: Date, Flow
- daily-truth.csv columns: Date, Concentration
- conc.csv        columns: Date, Concentration   (irregular observations)

Author: michael.taylor@cefas.gov.uk
Date:   30-Dec-2025
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------
# SYNTHETIC CONFIG PRESETS
# -----------------------
# Choose a preset by name. You can override any parameter via CLI flags.
DEFAULT_CONFIG_NAME = "BASELINE"

CONFIGS: Dict[str, Dict[str, float]] = {
    # Conservative baseline (close to your earlier working generator)
    "BASELINE": dict(
        # Storm occurrence/intensity
        p_start_wet=0.030,
        p_start_dry=0.010,
        mean_duration_days=2.0,
        ln_mean=3.6,
        ln_sigma=0.75,
        # Hydrograph reservoir recession
        hydro_alpha=0.85,
        # Concentration memory and event strength
        mem_alpha=0.65,
        storm_term_beta=0.70,
        # Hysteresis parameters
        hyster_rise_boost=0.55,
        hyster_fall_suppress=0.30,
        # Flow->conc effect strength (logQ effect)
        flow_effect_beta=0.12,
        # Noise levels (lognormal)
        conc_ln_noise_sd=0.12,
        obs_ln_noise_sd=0.25,
        # HF triggering and sampling
        trigger_flow_quantile=0.90,
        trigger_conc_quantile=0.95,
        hf_hold_days=60,
        hf_daily_prob=0.80,
        hf_flow_weight_k=2.0,
        # HF operation mode: if True, once triggered HF runs continuously for hf_hold_days
        hf_continuous=1.0,
    ),

    # Designed to produce a clear HF vs non-HF contrast while remaining plausible.
    # Key: multi-year HF programme once triggered, flashier events, shorter chemistry memory.
    "HF_PROGRAMME_5Y_PLAUSIBLE": dict(
        p_start_wet=0.040,
        p_start_dry=0.012,
        mean_duration_days=1.5,
        ln_mean=3.75,
        ln_sigma=0.95,
        hydro_alpha=0.78,
        mem_alpha=0.30,
        storm_term_beta=1.10,
        hyster_rise_boost=0.75,
        hyster_fall_suppress=0.40,
        flow_effect_beta=0.12,
        conc_ln_noise_sd=0.10,
        obs_ln_noise_sd=0.22,
        trigger_flow_quantile=0.88,
        trigger_conc_quantile=0.95,
        hf_hold_days=365 * 5,
        hf_daily_prob=0.85,
        hf_flow_weight_k=4.0,
        hf_continuous=1.0,
    ),

    # Stronger stress-test (still hydrologically interpretable, but more extreme tails)
    "HF_CONTRAST_STRONG": dict(
        p_start_wet=0.045,
        p_start_dry=0.010,
        mean_duration_days=1.2,
        ln_mean=3.90,
        ln_sigma=1.10,
        hydro_alpha=0.72,
        mem_alpha=0.25,
        storm_term_beta=1.35,
        hyster_rise_boost=0.85,
        hyster_fall_suppress=0.45,
        flow_effect_beta=0.12,
        conc_ln_noise_sd=0.09,
        obs_ln_noise_sd=0.22,
        trigger_flow_quantile=0.85,
        trigger_conc_quantile=0.95,
        hf_hold_days=365 * 5,
        hf_daily_prob=0.90,
        hf_flow_weight_k=6.0,
        hf_continuous=1.0,
    ),
}


# -----------------------
# Utilities
# -----------------------

def _ensure_dir(path: str) -> None:
    if path and path != ".":
        os.makedirs(path, exist_ok=True)


def _dates(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")


def _seasonal_signal(dates: pd.DatetimeIndex, base: float, amp: float, phase: float = 0.0) -> np.ndarray:
    doy = dates.dayofyear.to_numpy(dtype=float)
    return base + amp * np.sin(2.0 * np.pi * (doy / 365.25) + phase)


def _clustered_storm_intensity(
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    p_start_wet: float,
    p_start_dry: float,
    wet_months: Tuple[int, ...],
    mean_duration_days: float,
    ln_mean: float,
    ln_sigma: float,
) -> np.ndarray:
    """
    Clustered storm starts with geometric durations and heavy-tailed lognormal intensities.

    Returns:
      inten: daily storm input intensity (nonnegative), length len(dates)
    """
    n = len(dates)
    months = dates.month.to_numpy()
    wet = np.isin(months, np.array(wet_months, dtype=int))
    p_start = np.where(wet, float(p_start_wet), float(p_start_dry))

    mean_duration_days = max(1.0, float(mean_duration_days))
    p_end = 1.0 / mean_duration_days  # geometric end prob

    inten = np.zeros(n, dtype=float)
    t = 0
    while t < n:
        if rng.random() < p_start[t]:
            dur = 1
            while (t + dur) < n and rng.random() > p_end:
                dur += 1
            mag = rng.lognormal(mean=float(ln_mean), sigma=float(ln_sigma))
            inten[t:t + dur] += mag
            t += dur
        else:
            t += 1
    return inten


def _linear_reservoir(input_series: np.ndarray, alpha: float) -> np.ndarray:
    """
    Simple AR(1) / linear reservoir response: x[t] = alpha*x[t-1] + input[t]
    """
    n = len(input_series)
    x = np.zeros(n, dtype=float)
    a = float(alpha)
    a = min(0.995, max(0.0, a))
    for t in range(1, n):
        x[t] = a * x[t - 1] + input_series[t]
    return x


def _weekly_days(
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    weekly_interval_days: int,
    weekly_phase_random: bool = True,
) -> pd.DatetimeIndex:
    weekly_interval_days = max(1, int(weekly_interval_days))
    if weekly_phase_random and weekly_interval_days > 1:
        phase = int(rng.integers(0, weekly_interval_days))
    else:
        phase = 0
    return dates[phase::weekly_interval_days]


def _make_sampling_days_triggered_hf(
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    flow: np.ndarray,
    conc_true: np.ndarray,
    weekly_interval_days: int,
    hf_start: str,
    hf_end: str,
    trigger_flow_quantile: float,
    trigger_conc_quantile: float,
    hf_hold_days: int,
    hf_daily_prob: float,
    hf_flow_weight_k: float = 1.0,
    weekly_phase_random: bool = True,
    hf_continuous: bool = True,
) -> Tuple[pd.DatetimeIndex, np.ndarray, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[pd.Timestamp]]]:
    """
    Weekly sampling throughout entire record.

    Within [hf_start, hf_end], search for a trigger day.
    - Trigger if conc >= q_conc (always)
    - Or probabilistically if flow exceeds q_flow using hf_flow_weight_k

    If triggered:
      - If hf_continuous=True: HF runs continuously for hf_hold_days
      - Else: HF runs for hf_hold_days as well (same in this implementation; flag retained for future variants)

    During HF-active days, sample each day with probability hf_daily_prob.
    Max 1 sample/day is guaranteed by construction (set union on dates).

    Returns:
      obs_days: unique sample dates
      hf_on: boolean array of HF-active days (len(dates))
      hf_period: (trigger_date, active_start, active_end) or (None,None,None)
    """
    weekly_interval_days = max(1, int(weekly_interval_days))
    hf_hold_days = max(1, int(hf_hold_days))
    hf_daily_prob = float(np.clip(hf_daily_prob, 0.0, 1.0))
    hf_flow_weight_k = float(max(0.0, hf_flow_weight_k))

    weekly_days = _weekly_days(rng, dates, weekly_interval_days, weekly_phase_random)

    hs = pd.Timestamp(hf_start)
    he = pd.Timestamp(hf_end)
    in_search = (dates >= hs) & (dates <= he)
    idx_search = np.where(in_search)[0]
    n = len(dates)

    hf_on = np.zeros(n, dtype=bool)
    if idx_search.size == 0:
        obs = pd.DatetimeIndex(sorted(set(weekly_days)))
        return obs, hf_on, (None, None, None)

    trigger_flow_quantile = float(np.clip(trigger_flow_quantile, 0.0, 1.0))
    trigger_conc_quantile = float(np.clip(trigger_conc_quantile, 0.0, 1.0))

    flow_search = flow[idx_search]
    conc_search = conc_true[idx_search]

    q_flow = np.quantile(flow_search[np.isfinite(flow_search)], trigger_flow_quantile)
    q_conc = np.quantile(conc_search[np.isfinite(conc_search)], trigger_conc_quantile)

    trigger_idx = None
    for t in idx_search:
        flow_hi = (flow[t] >= q_flow)
        conc_hi = (conc_true[t] >= q_conc)
        if not (flow_hi or conc_hi):
            continue

        if conc_hi:
            trigger_idx = int(t)
            break

        # flow-based probabilistic start
        if hf_flow_weight_k > 0 and np.isfinite(flow[t]) and flow[t] > 0:
            ratio = max(0.0, (flow[t] / max(q_flow, 1e-12)) - 1.0)
            p_start = 1.0 - np.exp(-hf_flow_weight_k * ratio)
            if rng.random() < p_start:
                trigger_idx = int(t)
                break
        else:
            trigger_idx = int(t)
            break

    if trigger_idx is None:
        obs = pd.DatetimeIndex(sorted(set(weekly_days)))
        return obs, hf_on, (None, None, None)

    hf_active_start = dates[trigger_idx]
    hf_active_end = hf_active_start + pd.Timedelta(days=hf_hold_days - 1)
    if hf_active_end > dates[-1]:
        hf_active_end = dates[-1]

    hf_on[(dates >= hf_active_start) & (dates <= hf_active_end)] = True

    obs_extra = []
    for t in np.where(hf_on)[0]:
        if rng.random() < hf_daily_prob:
            obs_extra.append(dates[int(t)])

    obs = pd.DatetimeIndex(sorted(set(weekly_days).union(set(obs_extra))))
    return obs, hf_on, (dates[trigger_idx], hf_active_start, hf_active_end)


def _merge_cfg(preset: Dict[str, float], args: argparse.Namespace) -> Dict[str, float]:
    """
    Start from preset dict, then override with any CLI values that were explicitly provided.
    """
    cfg = dict(preset)

    # Mapping: cfg_key -> args_attr (if not None, override)
    overrides = {
        # storms
        "p_start_wet": "p_start_wet",
        "p_start_dry": "p_start_dry",
        "mean_duration_days": "mean_duration_days",
        "ln_mean": "ln_mean",
        "ln_sigma": "ln_sigma",
        # hydro
        "hydro_alpha": "hydro_alpha",
        # chemistry
        "mem_alpha": "mem_alpha",
        "storm_term_beta": "storm_term_beta",
        "hyster_rise_boost": "hyster_rise_boost",
        "hyster_fall_suppress": "hyster_fall_suppress",
        "flow_effect_beta": "flow_effect_beta",
        # noise
        "conc_ln_noise_sd": "conc_ln_noise_sd",
        "obs_ln_noise_sd": "obs_ln_noise_sd",
        # HF
        "trigger_flow_quantile": "trigger_flow_quantile",
        "trigger_conc_quantile": "trigger_conc_quantile",
        "hf_hold_days": "hf_hold_days",
        "hf_daily_prob": "hf_daily_prob",
        "hf_flow_weight_k": "hf_flow_weight_k",
        "hf_continuous": "hf_continuous",
    }

    for k, a in overrides.items():
        v = getattr(args, a, None)
        if v is not None:
            cfg[k] = float(v) if k not in {"hf_hold_days"} else int(v)

    # basic guardrails
    cfg["p_start_wet"] = float(np.clip(cfg["p_start_wet"], 0.0, 1.0))
    cfg["p_start_dry"] = float(np.clip(cfg["p_start_dry"], 0.0, 1.0))
    cfg["mean_duration_days"] = max(1.0, float(cfg["mean_duration_days"]))
    cfg["hydro_alpha"] = float(np.clip(cfg["hydro_alpha"], 0.0, 0.995))
    cfg["mem_alpha"] = float(np.clip(cfg["mem_alpha"], 0.0, 0.995))
    cfg["trigger_flow_quantile"] = float(np.clip(cfg["trigger_flow_quantile"], 0.0, 1.0))
    cfg["trigger_conc_quantile"] = float(np.clip(cfg["trigger_conc_quantile"], 0.0, 1.0))
    cfg["hf_daily_prob"] = float(np.clip(cfg["hf_daily_prob"], 0.0, 1.0))
    cfg["hf_flow_weight_k"] = max(0.0, float(cfg["hf_flow_weight_k"]))
    cfg["hf_hold_days"] = max(1, int(cfg["hf_hold_days"]))
    cfg["hf_continuous"] = 1.0 if float(cfg.get("hf_continuous", 1.0)) >= 0.5 else 0.0

    return cfg


# -----------------------
# Main generator
# -----------------------

def generate(
    outdir: str,
    seed: int,
    start: str,
    end: str,
    hf_start: str,
    hf_end: str,
    weekly_interval_days: int,
    cfg: Dict[str, float],
) -> None:
    rng = np.random.default_rng(int(seed))
    dates = _dates(start, end)
    n = len(dates)

    # -----------------------
    # FLOW (daily)
    # -----------------------
    flow_base = _seasonal_signal(dates, base=20.0, amp=10.0, phase=0.2)

    storm_input = _clustered_storm_intensity(
        rng=rng,
        dates=dates,
        p_start_wet=cfg["p_start_wet"],
        p_start_dry=cfg["p_start_dry"],
        wet_months=(10, 11, 12, 1, 2, 3),
        mean_duration_days=cfg["mean_duration_days"],
        ln_mean=cfg["ln_mean"],
        ln_sigma=cfg["ln_sigma"],
    )

    storm_hydro = _linear_reservoir(storm_input, alpha=cfg["hydro_alpha"])
    flow_noise = np.exp(rng.normal(0.0, 0.06, n))

    flow = (flow_base + storm_hydro) * flow_noise
    flow = np.maximum(flow, 0.1)  # strictly positive

    # -----------------------
    # TRUE CONCENTRATION (daily)
    # -----------------------
    c_base = _seasonal_signal(dates, base=5.0, amp=2.0, phase=-0.4)
    c_base = np.maximum(c_base, 0.05)

    dq = np.diff(flow, prepend=flow[0])
    rising = (dq > 0).astype(float)

    # Normalised storm input for "event memory"
    if np.any(storm_input > 0):
        denom = np.percentile(storm_input[storm_input > 0], 90)
        denom = max(float(denom), 1e-12)
    else:
        denom = 1.0
    s_norm = np.clip(storm_input / denom, 0.0, 10.0)

    mem = np.zeros(n, dtype=float)
    mem_alpha = float(cfg["mem_alpha"])
    for t in range(1, n):
        mem[t] = mem_alpha * mem[t - 1] + s_norm[t]

    hyster = 1.0 + float(cfg["hyster_rise_boost"]) * rising - float(cfg["hyster_fall_suppress"]) * (1.0 - rising)
    hyster = np.clip(hyster, 0.4, 2.5)

    # Flow effect in log space
    logq = np.log(flow)
    logq_centered = logq - np.median(logq)
    flow_effect = float(cfg["flow_effect_beta"]) * logq_centered

    storm_term = float(cfg["storm_term_beta"]) * mem * hyster

    conc_true = c_base * np.exp(flow_effect + storm_term)
    conc_true *= np.exp(rng.normal(0.0, float(cfg["conc_ln_noise_sd"]), n))
    conc_true = np.maximum(conc_true, 1e-6)

    # -----------------------
    # OBSERVED CONCENTRATION (irregular) with TRIGGERED HF
    # -----------------------
    obs_days, hf_on, hf_period = _make_sampling_days_triggered_hf(
        rng=rng,
        dates=dates,
        flow=flow,
        conc_true=conc_true,
        weekly_interval_days=weekly_interval_days,
        hf_start=hf_start,
        hf_end=hf_end,
        trigger_flow_quantile=float(cfg["trigger_flow_quantile"]),
        trigger_conc_quantile=float(cfg["trigger_conc_quantile"]),
        hf_hold_days=int(cfg["hf_hold_days"]),
        hf_daily_prob=float(cfg["hf_daily_prob"]),
        hf_flow_weight_k=float(cfg["hf_flow_weight_k"]),
        weekly_phase_random=True,
        hf_continuous=bool(cfg["hf_continuous"] >= 0.5),
    )

    obs_idx = dates.get_indexer(obs_days)
    c_obs = conc_true[obs_idx].copy()
    c_obs *= np.exp(rng.normal(0.0, float(cfg["obs_ln_noise_sd"]), len(c_obs)))
    c_obs = np.maximum(c_obs, 1e-6)

    # -----------------------
    # OUTPUT
    # -----------------------
    _ensure_dir(outdir)
    flow_path = os.path.join(outdir, "flow.csv")
    truth_path = os.path.join(outdir, "daily-truth.csv")
    conc_path = os.path.join(outdir, "conc.csv")

    pd.DataFrame({"Date": dates, "Flow": flow}).to_csv(flow_path, index=False)
    pd.DataFrame({"Date": dates, "Concentration": conc_true}).to_csv(truth_path, index=False)
    pd.DataFrame({"Date": obs_days, "Concentration": c_obs}).to_csv(conc_path, index=False)

    # -----------------------
    # Diagnostics
    # -----------------------
    hs = pd.Timestamp(hf_start)
    he = pd.Timestamp(hf_end)
    in_window = (obs_days >= hs) & (obs_days <= he)
    eligible_days = int(((dates >= hs) & (dates <= he)).sum())

    trig, a0, a1 = hf_period

    print(f"Wrote: {flow_path}")
    print(f"Wrote: {truth_path}")
    print(f"Wrote: {conc_path}")
    print(f"Obs count: {len(obs_days)} | weekly interval: {weekly_interval_days}d")
    print(f"Obs in HF window: {int(in_window.sum())} | outside window: {int((~in_window).sum())}")
    print(f"HF search window: {hf_start} .. {hf_end} | eligible_days={eligible_days}")
    print(
        "HF trigger params: "
        f"flow_q={cfg['trigger_flow_quantile']:.3f}, conc_q={cfg['trigger_conc_quantile']:.3f}, "
        f"flow_weight_k={cfg['hf_flow_weight_k']:.3f}"
    )
    print(
        "HF sampling: "
        f"hold_days={int(cfg['hf_hold_days'])}, daily_prob={cfg['hf_daily_prob']:.3f}, "
        f"continuous={'yes' if cfg['hf_continuous'] >= 0.5 else 'no'}"
    )
    print(f"HF-on days (mode active): {int(hf_on.sum())} out of {eligible_days} eligible days")

    if trig is None:
        print("HF trigger: NONE (no HF period activated).")
    else:
        print(
            f"HF trigger date: {trig.date()} | HF active: {a0.date()} .. {a1.date()} "
            f"| duration_days={int(hf_on.sum())}"
        )


# -----------------------
# CLI
# -----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic river flow + concentration with storm-driven bias and triggered HF sampling."
    )

    p.add_argument("--outdir", default=".", help="Output directory (default: current directory).")
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    p.add_argument("--start", default="1980-01-01", help="Start date (YYYY-MM-DD).")
    p.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD).")

    # HF search window during which triggering is allowed
    p.add_argument("--hf-start", default="2000-01-01", help="HF programme window start (YYYY-MM-DD).")
    p.add_argument("--hf-end", default="2005-12-31", help="HF programme window end (YYYY-MM-DD).")

    # Baseline sampling
    p.add_argument("--weekly-interval-days", type=int, default=7, help="Baseline sampling interval in days (default: 7).")

    # Preset selection
    p.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        choices=sorted(CONFIGS.keys()),
        help=f"Config preset name (default: {DEFAULT_CONFIG_NAME}).",
    )

    # Optional overrides (all default None => use preset)
    # Storms
    p.add_argument("--p-start-wet", type=float, default=None)
    p.add_argument("--p-start-dry", type=float, default=None)
    p.add_argument("--mean-duration-days", type=float, default=None)
    p.add_argument("--ln-mean", type=float, default=None)
    p.add_argument("--ln-sigma", type=float, default=None)

    # Hydro
    p.add_argument("--hydro-alpha", type=float, default=None)

    # Chemistry
    p.add_argument("--mem-alpha", type=float, default=None)
    p.add_argument("--storm-term-beta", type=float, default=None)
    p.add_argument("--hyster-rise-boost", type=float, default=None)
    p.add_argument("--hyster-fall-suppress", type=float, default=None)
    p.add_argument("--flow-effect-beta", type=float, default=None)

    # Noise
    p.add_argument("--conc-ln-noise-sd", type=float, default=None)
    p.add_argument("--obs-ln-noise-sd", type=float, default=None)

    # HF triggering and sampling
    p.add_argument("--trigger-flow-quantile", type=float, default=None)
    p.add_argument("--trigger-conc-quantile", type=float, default=None)
    p.add_argument("--hf-hold-days", type=int, default=None)
    p.add_argument("--hf-daily-prob", type=float, default=None)
    p.add_argument("--hf-flow-weight-k", type=float, default=None)
    p.add_argument(
        "--hf-continuous",
        type=int,
        default=None,
        help="1 => once triggered, HF runs continuously for hf_hold_days; 0 => (reserved) non-continuous mode.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.config_name not in CONFIGS:
        raise KeyError(f"Unknown config '{args.config_name}'. Options: {sorted(CONFIGS.keys())}")

    preset = CONFIGS[args.config_name]
    cfg = _merge_cfg(preset, args)

    generate(
        outdir=args.outdir,
        seed=args.seed,
        start=args.start,
        end=args.end,
        hf_start=args.hf_start,
        hf_end=args.hf_end,
        weekly_interval_days=args.weekly_interval_days,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()

