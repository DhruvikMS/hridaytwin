"""Fused feature matrix: static EHR + causal stream windows + twin state.

Everything is computed on (n_patients, n_days) arrays so the same code scores the
real cohort and the what-if simulator's synthetic futures.  Every window ends at
day d inclusive; nothing after d is ever read (tests/test_leakage.py enforces it).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .twin import B_NAMES, G_NAMES

EHR_FEATURES = ["age", "male", "bmi", "waist_cm", "tobacco", "family_hx_cvd", "t2d", "htn",
                "dm_years", "hba1c", "ldl", "egfr", "log_uacr", "sbp_clinic", "prior_cvd",
                "prior_hypo", "med_metformin", "med_sulfonylurea", "med_dpp4i", "med_sglt2i",
                "med_insulin", "med_raas", "med_ccb", "med_statin", "observes_fasting",
                "has_cgm"]


# --------------------------------------------------------------------------- windows
def _roll(x: np.ndarray, w: int, how: str) -> np.ndarray:
    """Trailing window stat over the last w days (inclusive), NaN-aware."""
    n, D = x.shape
    out = np.full((n, D), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if how in ("mean", "count", "sum"):
            v = np.nan_to_num(x)
            c = (~np.isnan(x)).astype(float)
            cs = np.concatenate([np.zeros((n, 1)), np.cumsum(v, 1)], 1)
            cc = np.concatenate([np.zeros((n, 1)), np.cumsum(c, 1)], 1)
            lo = np.maximum(0, np.arange(D) - w + 1)
            s = cs[:, np.arange(D) + 1] - cs[:, lo]
            k = cc[:, np.arange(D) + 1] - cc[:, lo]
            if how == "count":
                return k
            if how == "sum":
                return s
            return np.where(k > 0, s / np.maximum(k, 1), np.nan)
        from numpy.lib.stride_tricks import sliding_window_view
        pad = np.concatenate([np.full((n, w - 1), np.nan), x], 1)
        win = sliding_window_view(pad, w, axis=1)
        fn = {"max": np.nanmax, "min": np.nanmin, "std": np.nanstd}[how]
        return fn(win, axis=2)


def _count_where(x: np.ndarray, cond, w: int) -> np.ndarray:
    flag = np.where(np.isnan(x), np.nan, cond(x).astype(float))
    return _roll(flag, w, "sum")


# --------------------------------------------------------------------------- build
def build_daily_features(A: dict, T: dict, ehr: pd.DataFrame, cal: pd.DataFrame,
                         event_counts: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Return {feature_name: (n, D) array} for every day."""
    n, D = A["glucose"].shape
    F: dict[str, np.ndarray] = {}
    g, gmin, sbp = A["glucose"], A["glucose_min"], A["sbp"]

    F["glucose_m7"] = _roll(g, 7, "mean")
    F["glucose_m28"] = _roll(g, 28, "mean")
    F["glucose_sd14"] = _roll(g, 14, "std")
    F["glucose_max7"] = _roll(g, 7, "max")
    prev7 = np.concatenate([np.full((n, 7), np.nan), F["glucose_m7"][:, :-7]], 1)
    F["glucose_trend7"] = F["glucose_m7"] - prev7
    F["glucose_n_hi7"] = _count_where(g, lambda v: v > 250, 7)
    F["gmin_min7"] = _roll(gmin, 7, "min")
    F["gmin_m28"] = _roll(gmin, 28, "mean")
    F["gmin_n_low28"] = _count_where(gmin, lambda v: v < 70, 28)
    F["gmin_n_low7"] = _count_where(gmin, lambda v: v < 70, 7)

    F["sbp_m7"] = _roll(sbp, 7, "mean")
    F["sbp_m28"] = _roll(sbp, 28, "mean")
    F["sbp_max7"] = _roll(sbp, 7, "max")
    prevb = np.concatenate([np.full((n, 7), np.nan), F["sbp_m7"][:, :-7]], 1)
    F["sbp_trend7"] = F["sbp_m7"] - prevb
    F["dbp_m7"] = _roll(A["dbp"], 7, "mean")
    F["sbp_n_hi14"] = _count_where(sbp, lambda v: v >= 160, 14)

    F["steps_m7"] = _roll(A["steps"], 7, "mean")
    F["steps_vs_base"] = F["steps_m7"] / np.maximum(T["reg"]["steps_base"], 300)
    F["rhr_dev"] = _roll(A["rhr"], 3, "mean") - _roll(A["rhr"], 28, "mean")
    F["hrv_dev"] = _roll(A["hrv"], 3, "mean") - _roll(A["hrv"], 28, "mean")
    F["sleep_m7"] = _roll(A["sleep_h"], 7, "mean")
    F["adh_m7"] = _roll(A["med_taken"], 7, "mean")
    F["adh_m28"] = _roll(A["med_taken"], 28, "mean")
    F["adh_m3"] = _roll(A["med_taken"], 3, "mean")
    F["salt_m14"] = _roll(A["salt_score"], 14, "mean")
    F["carb_m14"] = _roll(A["carb_score"], 14, "mean")
    F["wear_frac7"] = _roll(A["steps"], 7, "count") / 7.0
    F["bp_n7"] = _roll(sbp, 7, "count")
    F["glucose_n7"] = _roll(g, 7, "count")

    # ---- twin state ----------------------------------------------------------
    tg, tb = T["glucose"], T["sbp"]
    for j, name in enumerate(G_NAMES[1:], start=1):
        F[f"twin_{name}"] = tg["theta"][:, :, j]
    for j, name in enumerate(B_NAMES[1:], start=1):
        F[f"twin_{name}"] = tb["theta"][:, :, j]
    F["twin_g_drug_sd"] = tg["theta_sd"][:, :, 1]
    F["twin_b_drug_sd"] = tb["theta_sd"][:, :, 1]
    F["twin_g_expected"] = tg["yhat"]
    F["twin_b_expected"] = tb["yhat"]
    F["twin_g_surprise7"] = _roll(tg["z"], 7, "mean")
    F["twin_b_surprise7"] = _roll(tb["z"], 7, "mean")
    F["twin_g_abs_surprise7"] = _roll(np.abs(tg["z"]), 7, "mean")
    # "adherence-adjusted" glucose: what glucose would be if every dose were taken
    F["twin_g_full_adherence"] = tg["yhat"] + tg["theta"][:, :, 1] * (1 - T["reg"]["adh"])

    # ---- context known in advance -------------------------------------------
    fast = cal["fast_day"].to_numpy()[:D].astype(float)
    feast = cal["feast"].to_numpy()[:D].astype(float)
    fast_ahead = np.array([fast[d + 1:d + 15].sum() for d in range(D)])
    feast_ahead = np.array([feast[d + 1:d + 15].sum() for d in range(D)])
    obs_fast = ehr["observes_fasting"].to_numpy()[:, None].astype(float)
    F["fast_days_next14"] = obs_fast * fast_ahead[None, :]
    F["feast_days_next14"] = np.broadcast_to(feast_ahead[None, :], (n, D)).copy()
    F["monsoon"] = np.broadcast_to(cal["monsoon"].to_numpy()[:D][None, :], (n, D)).astype(float)

    if event_counts is None:
        event_counts = np.zeros((n, D))
    F["events_past"] = event_counts
    return F


def stack_rows(F: dict, ehr: pd.DataFrame, days, patient_idx=None) -> pd.DataFrame:
    """Pick prediction days and attach static EHR columns -> model-ready frame."""
    n = len(ehr)
    patient_idx = np.arange(n) if patient_idx is None else np.asarray(patient_idx)
    days = np.asarray(days)
    ii = np.repeat(patient_idx, len(days))
    dd = np.tile(days, len(patient_idx))
    out = {"patient_id": ehr["patient_id"].to_numpy()[ii], "day": dd}
    e = ehr.assign(log_uacr=np.log1p(ehr["uacr"]))
    for c in EHR_FEATURES:
        out[c] = e[c].to_numpy()[ii]
    for k, v in F.items():
        out[k] = v[ii, dd]
    return pd.DataFrame(out)


def event_count_array(events: pd.DataFrame, patient_ids, D: int) -> np.ndarray:
    """Cumulative count of past adverse-event encounters known by day d (inclusive)."""
    pos = {p: i for i, p in enumerate(patient_ids)}
    arr = np.zeros((len(pos), D))
    for p, d in zip(events.patient_id, events.day):
        if p in pos and d < D:
            arr[pos[p], d] += 1
    return np.cumsum(arr, 1)


def labels(events: pd.DataFrame, rows: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Any event (and each type) in (day, day+horizon]."""
    from .synth import EVENT_TYPES
    out = pd.DataFrame(index=rows.index)
    ev = events.groupby("patient_id")
    groups = {p: (g.day.to_numpy(), g.event_type.to_numpy()) for p, g in ev}
    y_any = np.zeros(len(rows), int)
    y_type = {t: np.zeros(len(rows), int) for t in EVENT_TYPES}
    first = np.full(len(rows), np.nan)
    for k, (p, d) in enumerate(zip(rows.patient_id.to_numpy(), rows.day.to_numpy())):
        if p not in groups:
            continue
        days, types = groups[p]
        m = (days > d) & (days <= d + horizon)
        if m.any():
            y_any[k] = 1
            first[k] = days[m].min() - d
            for t in set(types[m]):
                y_type[t][k] = 1
    out["y"] = y_any
    out["days_to_event"] = first
    for t in EVENT_TYPES:
        out[f"y_{t}"] = y_type[t]
    return out
