"""The digital twin core: a per-patient Bayesian state-space model.

Each patient gets two small linear-Gaussian models whose *coefficients are the
patient's personal physiology*:

    glucose_t = th0 + th1*adherence_t + th2*activity_t + th3*carbs_t + th4*illness_t + e
    sbp_t     = ph0 + ph1*adherence_t + ph2*activity_t + ph3*salt_t  + ph4*illness_t + e

* The PRIOR on every coefficient comes from the EHR (HbA1c -> eAG, clinic BP,
  the medication list x published average drug effects).  This is the EHR half
  of the twin.
* The POSTERIOR is updated every day by a Kalman filter from the wearable /
  home-device stream (adherence from a smart pill-box, steps and resting HR from
  the wearable, glucose from CGM/SMBG, home BP, diet log).  This is the
  real-time half.
* Coefficients follow a slow random walk, so the twin tracks drift (disease
  progression, de-conditioning) instead of freezing on old data.

What the twin produces each day (all strictly causal - uses data <= day d):
  - its expectation of today's glucose / BP before seeing the reading (yhat)
  - the standardised surprise when the reading arrives (z = innovation / sd)
  - the personal coefficients and their uncertainty (e.g. "this patient's
    medicines lower glucose by 41 +/- 9 mg/dL when taken")
These feed the risk model and drive the what-if simulator.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .synth import GLUCOSE_DRUG_EFFECT, SBP_DRUG_EFFECT, drug_load

STREAM_COLS = ["steps", "rhr", "hrv", "sleep_h", "glucose", "glucose_min", "sbp", "dbp",
               "med_taken", "salt_score", "carb_score", "weight_kg", "in_hospital"]

G_NAMES = ["g_base", "g_drug", "g_steps", "g_carb", "g_ill"]
B_NAMES = ["b_base", "b_drug", "b_steps", "b_salt", "b_ill"]


# --------------------------------------------------------------------------- helpers
def to_arrays(daily: pd.DataFrame, patient_ids, days: int) -> dict[str, np.ndarray]:
    """Long patient-day frame -> dict of (n_patients, n_days) arrays (NaN = missing)."""
    idx = pd.MultiIndex.from_product([list(patient_ids), range(days)], names=["patient_id", "day"])
    wide = daily.set_index(["patient_id", "day"]).reindex(idx)
    n = len(patient_ids)
    return {c: wide[c].to_numpy(float).reshape(n, days) for c in STREAM_COLS if c in wide}


def ema_skipna(x: np.ndarray, alpha: float, init) -> np.ndarray:
    """Exponential moving average along axis 1; missing days keep the last value."""
    n, D = x.shape
    out = np.empty_like(x)
    e = np.broadcast_to(np.asarray(init, float), (n,)).copy()
    for d in range(D):
        v = x[:, d]
        ok = ~np.isnan(v)
        e[ok] = (1 - alpha) * e[ok] + alpha * v[ok]
        out[:, d] = e
    return out


def expanding_mean_skipna(x: np.ndarray, fallback) -> np.ndarray:
    s = np.nancumsum(x, axis=1)
    c = np.cumsum(~np.isnan(x), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / c
    fb = np.broadcast_to(np.asarray(fallback, float)[:, None], m.shape)
    return np.where(c > 0, m, fb)


def regressors(A: dict, ehr: pd.DataFrame) -> dict[str, np.ndarray]:
    """Causal regressors shared by the twin and the feature builder."""
    n = len(ehr)
    adh = ema_skipna(A["med_taken"], 0.25, 0.85)
    steps_ema = ema_skipna(A["steps"], 0.2, np.nan)
    first_steps = np.array([row[~np.isnan(row)][0] if (~np.isnan(row)).any() else 5000.0
                            for row in A["steps"]])
    steps_ema = np.where(np.isnan(steps_ema), first_steps[:, None], steps_ema)
    # personal activity baseline: mean of the first 28 observed days, then frozen
    base_window = A["steps"][:, :28]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        steps_base = np.nanmean(base_window, axis=1)
    steps_base = np.where(np.isnan(steps_base), first_steps, steps_base)
    exp_steps = expanding_mean_skipna(A["steps"], first_steps)
    D = A["steps"].shape[1]
    steps_base_t = np.where(np.arange(D)[None, :] < 28, exp_steps, steps_base[:, None])

    rhr_ema = ema_skipna(A["rhr"], 0.5, np.nan)
    rhr_base = expanding_mean_skipna(A["rhr"], np.full(n, 75.0))
    rhr_ema = np.where(np.isnan(rhr_ema), rhr_base, rhr_ema)
    carb = ema_skipna(A["carb_score"], 0.4, 0.0)
    salt = ema_skipna(A["salt_score"], 0.2, 10.0)
    return {
        "adh": adh,
        "activity": (steps_ema - steps_base_t) / 1000.0,
        "steps_base": steps_base_t,
        "carb": carb,
        "salt": salt - 10.0,
        "illness": np.clip((rhr_ema - rhr_base) / 10.0, -1.5, 3.0),
    }


def lag(x: np.ndarray, alpha: float) -> np.ndarray:
    """First-order lag (complete series, no NaN) along the day axis."""
    out = np.empty_like(x)
    e = x[:, 0].copy()
    for d in range(x.shape[1]):
        e = (1 - alpha) * e + alpha * x[:, d]
        out[:, d] = e
    return out


def priors(ehr: pd.DataFrame):
    """EHR-derived prior means / sds for both twin models."""
    Lg = drug_load(ehr, GLUCOSE_DRUG_EFFECT).astype(float)
    Lb = drug_load(ehr, SBP_DRUG_EFFECT).astype(float)
    eag = 28.7 * ehr.hba1c.to_numpy() - 46.7
    n = len(ehr)
    mg = np.column_stack([eag + 0.85 * Lg, -Lg, np.full(n, -4.0), np.full(n, 7.0), np.full(n, 45.0)])
    sg = np.column_stack([np.full(n, 30.0), np.maximum(8.0, 0.6 * Lg), np.full(n, 2.5),
                          np.full(n, 3.5), np.full(n, 25.0)])
    mb = np.column_stack([ehr.sbp_clinic.to_numpy() + 0.85 * Lb, -Lb, np.full(n, -0.7),
                          np.full(n, 1.1), np.full(n, -4.0)])
    sb = np.column_stack([np.full(n, 15.0), np.maximum(5.0, 0.6 * Lb), np.full(n, 0.6),
                          np.full(n, 0.8), np.full(n, 6.0)])
    return (mg, sg), (mb, sb)


def kalman_regression(y, X, m0, s0, q, r):
    """Vectorised Kalman filter over patients for y_t = X_t . theta_t + e.

    y: (n, D) with NaN for missing; X: (n, D, k); m0/s0: (n, k); q: (k,) daily
    random-walk sd; r: (n,) observation sd.  Returns dict of (n, D[, k]) arrays.
    """
    n, D, k = X.shape
    th = m0.copy()
    P = np.zeros((n, k, k))
    P[:, range(k), range(k)] = s0 ** 2
    Q = np.diag(q ** 2)
    R = r ** 2
    yhat = np.full((n, D), np.nan)
    ysd = np.full((n, D), np.nan)
    z = np.full((n, D), np.nan)
    TH = np.zeros((n, D, k))
    SD = np.zeros((n, D, k))
    for d in range(D):
        P = P + Q
        x = X[:, d, :]
        yh = np.einsum("nk,nk->n", x, th)
        Px = np.einsum("nij,nj->ni", P, x)
        S = np.einsum("ni,ni->n", x, Px) + R
        yhat[:, d] = yh
        ysd[:, d] = np.sqrt(S)
        obs = ~np.isnan(y[:, d])
        if obs.any():
            innov = y[obs, d] - yh[obs]
            z[obs, d] = innov / np.sqrt(S[obs])
            K = Px[obs] / S[obs, None]
            th[obs] = th[obs] + K * innov[:, None]
            P[obs] = P[obs] - np.einsum("ni,nj->nij", K, Px[obs])
        TH[:, d, :] = th
        SD[:, d, :] = np.sqrt(np.clip(np.einsum("nii->ni", P), 0, None))
    return {"yhat": yhat, "ysd": ysd, "z": z, "theta": TH, "theta_sd": SD}


def run_twin(A: dict, ehr: pd.DataFrame, reg: dict | None = None) -> dict:
    reg = reg or regressors(A, ehr)
    n, D = A["glucose"].shape
    ones = np.ones((n, D))
    # physiology responds with a lag (glucose ~0.3/day, BP ~0.35/day), so the
    # behavioural drivers are passed through the same first-order lag
    lg = {k: lag(reg[k], 0.3) for k in ("adh", "activity", "carb", "illness")}
    lb = {k: lag(reg[k], 0.35) for k in ("adh", "activity", "salt", "illness")}
    Xg = np.stack([ones, lg["adh"], lg["activity"], lg["carb"], lg["illness"]], axis=2)
    Xb = np.stack([ones, lb["adh"], lb["activity"], lb["salt"], lb["illness"]], axis=2)
    (mg, sg), (mb, sb) = priors(ehr)
    cgm = ehr.has_cgm.to_numpy().astype(bool)
    rg = np.where(cgm, 14.0, 19.0)
    rb = np.full(n, 9.5)
    qg = np.array([0.6, 0.15, 0.03, 0.03, 0.15])
    qb = np.array([0.3, 0.08, 0.01, 0.01, 0.05])
    g = kalman_regression(A["glucose"], Xg, mg, sg, qg, rg)
    b = kalman_regression(A["sbp"], Xb, mb, sb, qb, rb)
    return {"glucose": g, "sbp": b, "reg": reg, "Xg": Xg, "Xb": Xb}
