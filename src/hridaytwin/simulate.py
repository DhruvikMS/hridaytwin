"""What-if engine: simulate a patient's next 90 days under candidate care plans.

The twin is the *generative* half of the system: it knows this patient's personal
coefficients (drug response, activity response, salt sensitivity, residual noise,
glucose excursions on fasting days) learned from their own stream.  For each
candidate plan we roll the twin forward with Monte-Carlo noise, produce synthetic
wearable / home-device streams with the patient's own missingness pattern, and
hand those streams to the *same* risk engine used in production.  So a plan is
judged exactly the way the patient will be judged once it is in place.

Design choices that keep this honest
  * Common random numbers: every plan sees the same noise draws, so differences
    between plans are not Monte-Carlo luck.
  * New drugs use published average effects scaled by a *shrunk* version of this
    patient's learned response (0.5 + 0.5 x personal multiplier) - responders to
    one agent tend to respond, but we do not assume full transfer.
  * Safety guardrails (eGFR, age, current regimen) remove plans a clinician would
    not consider, and attach cautions to the rest. Output is decision support:
    every plan is labelled "for clinician review".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

from . import config
from .features import build_daily_features, stack_rows
from .synth import GLUCOSE_DRUG_EFFECT, SBP_DRUG_EFFECT, drug_load, india_calendar
from .twin import STREAM_COLS, regressors, run_twin, lag

# literature average effects at full adherence (mean glucose mg/dL, SBP mmHg)
NEW_DRUG = {
    "add_sglt2i": {"g": -17, "b": -4},
    "add_dpp4i": {"g": -17, "b": 0},
    "add_metformin": {"g": -30, "b": 0},
    "add_ccb": {"g": 0, "b": -9},
    "start_raas": {"g": 0, "b": -9},
    "uptitrate_raas": {"g": 0, "b": -5},
}
PHARM_LABEL = {
    "none": "Continue current medicines",
    "add_sglt2i": "Add SGLT2 inhibitor",
    "add_dpp4i": "Add DPP-4 inhibitor",
    "add_metformin": "Start metformin",
    "switch_su_to_dpp4i": "Switch sulfonylurea to DPP-4 inhibitor",
    "fasting_dose_plan": "Fasting-day dose plan (SU/insulin)",
    "add_ccb": "Add amlodipine (CCB)",
    "start_raas": "Start ACE-i / ARB",
    "uptitrate_raas": "Up-titrate ACE-i / ARB",
}


@dataclass
class Plan:
    pharm: str = "none"
    extra_steps: int = 0
    salt_cut_g: float = 0.0
    adherence_support: bool = False
    cautions: list = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.pharm}|{self.extra_steps}|{self.salt_cut_g:g}|{int(self.adherence_support)}"

    def label(self) -> str:
        parts = [PHARM_LABEL[self.pharm]] if self.pharm != "none" else []
        if self.extra_steps:
            parts.append(f"+{self.extra_steps:,} steps/day")
        if self.salt_cut_g:
            parts.append(f"-{self.salt_cut_g:g} g salt/day")
        if self.adherence_support:
            parts.append("adherence support (pill-box + SMS)")
        return " + ".join(parts) if parts else "Continue current care"

    def burden(self) -> float:
        """Rough patient burden, used only to break near-ties between plans."""
        return (0.004 * (self.pharm != "none") + 0.003 * (self.extra_steps > 0)
                + 0.002 * (self.salt_cut_g > 0) + 0.001 * self.adherence_support)


# --------------------------------------------------------------------------- guardrails
def allowed_pharm(e: pd.Series, recent_sbp: float, recent_glu: float) -> dict[str, list[str]]:
    """Return {pharm_option: [cautions]} for options a clinician could consider."""
    out: dict[str, list[str]] = {"none": []}
    egfr, age = e.egfr, e.age
    if e.t2d:
        if not e.med_sglt2i and egfr >= 20:
            c = ["Check volume status; counsel on genital hygiene"]
            if age >= 75:
                c.append("Age >= 75: watch for dehydration")
            out["add_sglt2i"] = c
        if not e.med_dpp4i and not e.med_insulin:
            c = ["Renal dose adjustment needed (eGFR < 45)"] if egfr < 45 else []
            out["add_dpp4i"] = c
        if not e.med_metformin and egfr >= 30:
            c = ["Max 1 g/day while eGFR 30-45"] if egfr < 45 else []
            out["add_metformin"] = c
        if e.med_sulfonylurea and not e.med_dpp4i:
            out["switch_su_to_dpp4i"] = ["Expect small rise in glucose; recheck HbA1c in 3 months"]
        if (e.med_sulfonylurea or e.med_insulin) and e.observes_fasting:
            out["fasting_dose_plan"] = ["Per IDF-DAR fasting guidance; educate on hypo symptoms"]
    if e.med_sulfonylurea or e.med_insulin:
        for k in ("add_sglt2i", "add_dpp4i", "add_metformin"):
            if k in out:
                out[k].append("On SU/insulin: consider lowering that dose to avoid hypoglycaemia")
    if recent_sbp >= 130:
        if not e.med_ccb:
            out["add_ccb"] = ["Watch for ankle oedema"]
        if not e.med_raas:
            out["start_raas"] = ["Check creatinine and potassium in 1-2 weeks"]
        else:
            out["uptitrate_raas"] = ["Check creatinine and potassium in 1-2 weeks"]
    return out


def candidate_plans(e: pd.Series, recent_sbp: float, recent_glu: float) -> list[Plan]:
    plans = []
    for pharm, cautions in allowed_pharm(e, recent_sbp, recent_glu).items():
        for steps, salt, adh in product([0, 2000], [0.0, 3.0], [False, True]):
            c = list(cautions)
            if steps and e.prior_cvd:
                c.append("Prior CVD: graded activity plan, clear with physician")
            plans.append(Plan(pharm, steps, salt, adh, c))
    return plans


# --------------------------------------------------------------------------- simulation
def _personal_stats(A: dict, T: dict, i: int, D: int) -> dict:
    sl = slice(max(0, D - 28), D)
    def m(col, default):
        v = A[col][i, sl]
        v = v[~np.isnan(v)]
        return float(v.mean()) if len(v) else default
    def rate(col):
        return float((~np.isnan(A[col][i, max(0, D - 60):D])).mean())
    steps = A["steps"][i, sl]
    steps = steps[~np.isnan(steps)]
    resid_g = (A["glucose"][i, :D] - T["glucose"]["yhat"][i, :D])
    resid_b = (A["sbp"][i, :D] - T["sbp"]["yhat"][i, :D])
    g, gm = A["glucose"][i, :D], A["glucose_min"][i, :D]
    exc = g - gm
    return {
        "steps_mean": float(steps.mean()) if len(steps) else 4500.0,
        "steps_logsd": float(np.std(np.log(np.maximum(steps, 200)))) if len(steps) > 3 else 0.3,
        "adh": m("med_taken", 0.85), "salt": m("salt_score", 10.0), "carb": m("carb_score", 0.0),
        "rhr": m("rhr", 75.0), "hrv": m("hrv", 30.0), "sleep": m("sleep_h", 6.5),
        "sd_g": float(np.nanstd(resid_g[-90:])) if np.isfinite(resid_g).sum() > 5 else 20.0,
        "sd_b": float(np.nanstd(resid_b[-90:])) if np.isfinite(resid_b).sum() > 5 else 10.0,
        "excursion": float(np.nanmean(exc)) if np.isfinite(exc).sum() > 5 else 50.0,
        "rates": {c: rate(c) for c in ["steps", "glucose", "glucose_min", "sbp", "med_taken",
                                       "salt_score", "weight_kg"]},
    }


def simulate_patient(i: int, A: dict, ehr: pd.DataFrame, events_past: np.ndarray,
                     bundle: dict, plans: list[Plan], n_draws: int = 16, seed: int = 0,
                     horizon: int | None = None) -> dict:
    """Roll patient i forward under every plan. Returns per-plan summaries + trajectories."""
    H = horizon or config.WHATIF_DAYS
    D = A["glucose"].shape[1]
    e = ehr.iloc[i]
    ehr1 = ehr.iloc[[i]].reset_index(drop=True)
    A1 = {k: v[i:i + 1] for k, v in A.items()}
    T1 = run_twin(A1, ehr1)
    st = _personal_stats(A1, T1, 0, D)
    thg = T1["glucose"]["theta"][0, D - 1]
    thb = T1["sbp"]["theta"][0, D - 1]
    Lg = drug_load(ehr1, GLUCOSE_DRUG_EFFECT)[0]
    Lb = drug_load(ehr1, SBP_DRUG_EFFECT)[0]
    r_g_pers = float(np.clip(-thg[1] / Lg, 0.3, 2.5)) if Lg > 0 else 1.0
    r_b_pers = float(np.clip(-thb[1] / Lb, 0.3, 2.5)) if Lb > 0 else 1.0
    cal = india_calendar(D + H)
    fast = cal["fast_day"].to_numpy()[D:].astype(bool) & bool(e.observes_fasting)
    feast = cal["feast"].to_numpy()[D:]
    su_share = (GLUCOSE_DRUG_EFFECT["med_sulfonylurea"] / Lg) if (Lg > 0 and e.med_sulfonylurea) else 0
    su_or_ins = bool(e.med_sulfonylurea or e.med_insulin)

    P, M = len(plans), n_draws
    N = P * M
    rng = np.random.default_rng(seed)
    # ---- common random numbers (identical across plans) ----------------------
    z_steps = rng.normal(0, 1, (M, H))
    u_adh = rng.random((M, H))
    z_g = rng.normal(0, 1, (M, H))
    z_b = rng.normal(0, 1, (M, H))
    z_gm = rng.normal(0, 1, (M, H))
    z_salt = rng.normal(0, 2, (M, H))
    z_carb = rng.normal(0, 1, (M, H))
    ill_on = rng.random((M, H)) < 1 / 140
    ill_len = rng.integers(3, 10, (M, H))
    obs_u = {c: rng.random((M, H)) for c in st["rates"]}

    illness = np.zeros((M, H))
    for mth in range(M):
        for d in np.where(ill_on[mth])[0]:
            illness[mth, d:d + ill_len[mth, d]] = rng.uniform(0.4, 1.0)

    def tile(x):  # (M,H) -> (N,H) with plans as outer index
        return np.tile(x, (P, 1))

    plan_steps = np.repeat([p.extra_steps for p in plans], M)[:, None]
    plan_salt = np.repeat([p.salt_cut_g for p in plans], M)[:, None]
    plan_adh = np.repeat([0.95 if p.adherence_support else st["adh"] for p in plans], M)[:, None]
    ramp = np.minimum(1, np.arange(1, H + 1) / 21)[None, :]   # behaviour change over 3 weeks
    ILL = tile(illness)

    steps = np.maximum(300, (st["steps_mean"] + plan_steps * ramp) * np.exp(st["steps_logsd"] * tile(z_steps) * 0.8)) * (1 - 0.6 * ILL)
    p_take = np.minimum(0.99, plan_adh * (1 - 0.25 * ILL) + (1 - ramp) * (st["adh"] - plan_adh))
    took = (tile(u_adh) < p_take).astype(float)
    salt = st["salt"] - plan_salt * ramp + tile(z_salt) + 3 * feast[None, :]
    carb = st["carb"] + tile(z_carb) * 0.8 + 1.8 * feast[None, :] - 2.2 * fast[None, :]
    rhr = st["rhr"] + 11 * ILL + tile(rng.normal(0, 2, (M, H)))
    hrv = st["hrv"] - 9 * ILL + tile(rng.normal(0, 3, (M, H)))
    sleep = st["sleep"] - 1.2 * ILL + tile(rng.normal(0, 0.6, (M, H)))

    # ---- full-information regressors through the twin's own transforms -----
    def full(col, fut):
        hist = np.repeat(A1[col], N, 0)
        return np.concatenate([hist, fut], 1)
    Afull = {"steps": full("steps", steps), "rhr": full("rhr", rhr),
             "med_taken": full("med_taken", took), "carb_score": full("carb_score", carb),
             "salt_score": full("salt_score", salt)}
    ehrN = pd.concat([ehr1] * N, ignore_index=True)
    reg = regressors({**{c: np.repeat(A1[c], N, 0) for c in A1}, **Afull}, ehrN)
    lgk = {k: lag(reg[k], 0.3)[:, D:] for k in ("adh", "activity", "carb", "illness")}
    lbk = {k: lag(reg[k], 0.35)[:, D:] for k in ("adh", "activity", "salt", "illness")}

    # ---- drug-change deltas (ramp in with the same physiological lag) -------
    dg = np.zeros((N, 1)); db = np.zeros((N, 1))
    hypo_dip_scale = np.ones((N, 1)); exc_shift = np.zeros((N, 1))
    shrink_g = 0.5 + 0.5 * r_g_pers
    shrink_b = 0.5 + 0.5 * r_b_pers
    for k, p in enumerate(plans):
        s = slice(k * M, (k + 1) * M)
        if p.pharm in NEW_DRUG:
            dg[s] = NEW_DRUG[p.pharm]["g"] * shrink_g
            db[s] = NEW_DRUG[p.pharm]["b"] * shrink_b
        elif p.pharm == "switch_su_to_dpp4i":
            dg[s] = -thg[1] * su_share + NEW_DRUG["add_dpp4i"]["g"] * shrink_g
            hypo_dip_scale[s] = 0.0 if not e.med_insulin else 0.5
            exc_shift[s] = -8
        elif p.pharm == "fasting_dose_plan":
            hypo_dip_scale[s] = 0.45
    drug_ramp = 1 - (0.7 ** np.arange(1, H + 1))[None, :]
    adh_eff = lgk["adh"]

    # AR(1) residuals with the patient's own residual sd
    def ar1(z, sd, phi=0.6):
        out = np.zeros_like(z)
        acc = np.zeros(z.shape[0])
        for d in range(z.shape[1]):
            acc = phi * acc + np.sqrt(1 - phi ** 2) * z[:, d]
            out[:, d] = acc
        return out * sd
    eg = ar1(tile(z_g), min(st["sd_g"], 35))
    eb = ar1(tile(z_b), min(st["sd_b"], 14))

    G = (thg[0] + thg[1] * adh_eff + thg[2] * lgk["activity"] + thg[3] * lgk["carb"]
         + thg[4] * lgk["illness"] + dg * adh_eff * drug_ramp + eg)
    if not e.t2d:
        G = np.minimum(G, 125)
    G = np.clip(G, 70, 480)
    S = (thb[0] + thb[1] * lbk["adh"] + thb[2] * lbk["activity"] + thb[3] * lbk["salt"]
         + thb[4] * lbk["illness"] + db * lbk["adh"] * drug_ramp + eb)
    surge = steps > 1.45 * np.maximum(500, st["steps_mean"])
    dip = su_or_ins * took * (45 * fast[None, :] + 20 * surge + 18 * ILL) * hypo_dip_scale
    Gmin = np.clip(G - (st["excursion"] + exc_shift) - dip + 9 * tile(z_gm), 30, None)
    if not su_or_ins:
        Gmin = np.maximum(Gmin, 66)

    # ---- observation layer with the patient's own missingness ----------------
    def seen(col):
        return tile(obs_u[col]) < st["rates"][col]
    worn = seen("steps")
    nan = np.nan
    fut = {
        "steps": np.where(worn, steps, nan), "rhr": np.where(worn, rhr, nan),
        "hrv": np.where(worn, hrv, nan), "sleep_h": np.where(worn, sleep, nan),
        "glucose": np.where(seen("glucose"), G + tile(rng.normal(0, 8, (M, H))), nan),
        "glucose_min": np.where(seen("glucose_min"), Gmin, nan),
        "sbp": np.where(seen("sbp"), S + tile(rng.normal(0, 7, (M, H))), nan),
        "dbp": np.where(seen("sbp"), 0.52 * S + 12 + tile(rng.normal(0, 4, (M, H))), nan),
        "med_taken": np.where(seen("med_taken"), took, nan),
        "salt_score": np.where(seen("salt_score"), salt, nan),
        "carb_score": np.where(seen("salt_score"), carb, nan),
        "weight_kg": np.full(G.shape, nan), "in_hospital": np.zeros(G.shape),
    }
    AN = {c: np.concatenate([np.repeat(A1[c], N, 0), fut[c]], 1) for c in STREAM_COLS}
    TN = run_twin(AN, ehrN)
    evN = np.concatenate([np.repeat(events_past[i:i + 1], N, 0),
                          np.repeat(events_past[i:i + 1, -1:], N, 0).repeat(H, 1)], 1)
    FN = build_daily_features(AN, TN, ehrN, cal, evN)
    score_days = np.arange(D + 13, D + H - config.HORIZON_DAYS + 1, 7)
    rows = stack_rows(FN, ehrN, score_days)
    X = rows[bundle["features"]]
    risk = bundle["model"].predict_proba(X)[:, 1].reshape(N, len(score_days))
    heads = {t: m.predict_proba(X)[:, 1].reshape(N, len(score_days)) for t, m in bundle["heads"].items()}

    results = []
    for k, p in enumerate(plans):
        s = slice(k * M, (k + 1) * M)
        g90 = G[s].mean()
        results.append({
            "key": p.key, "label": p.label(), "pharm": p.pharm, "extra_steps": p.extra_steps,
            "salt_cut_g": p.salt_cut_g, "adherence_support": p.adherence_support,
            "cautions": p.cautions, "burden": p.burden(),
            "mean_risk_14d": float(risk[s].mean()),
            "risk_by_week": risk[s].mean(0).round(4).tolist(),
            "type_risk": {t: float(h[s].mean()) for t, h in heads.items()},
            "projected_hba1c": round(float((g90 + 46.7) / 28.7), 2),
            "projected_sbp": round(float(S[s][:, -30:].mean()), 1),
            "hypo_days_per_90": round(float((Gmin[s] < 54).sum(1).mean()), 2) if su_or_ins else 0.0,
            "glucose_p50": np.percentile(G[s], 50, 0).round(1).tolist(),
            "glucose_p10": np.percentile(G[s], 10, 0).round(1).tolist(),
            "glucose_p90": np.percentile(G[s], 90, 0).round(1).tolist(),
            "sbp_p50": np.percentile(S[s], 50, 0).round(1).tolist(),
            "sbp_p10": np.percentile(S[s], 10, 0).round(1).tolist(),
            "sbp_p90": np.percentile(S[s], 90, 0).round(1).tolist(),
        })
    return {"plans": results, "score_days": (score_days - D + 1).tolist(),
            "personal": {"glucose_drug_response": round(r_g_pers, 2),
                         "bp_drug_response": round(r_b_pers, 2)}}


def recommend(results: list[dict], top: int = 3) -> list[dict]:
    """Rank plans by simulated 14-day risk + small burden penalty; baseline first."""
    base = next(r for r in results if r["key"] == "none|0|0|0")
    ranked = sorted(results, key=lambda r: r["mean_risk_14d"] + r["burden"])
    out = []
    for r in ranked:
        if r["key"] == base["key"]:
            continue
        rel = 1 - r["mean_risk_14d"] / max(base["mean_risk_14d"], 1e-9)
        out.append({**{k: r[k] for k in ("key", "label", "mean_risk_14d", "projected_hba1c",
                                         "projected_sbp", "hypo_days_per_90", "cautions")},
                    "relative_risk_reduction": round(float(rel), 3)})
        if len(out) == top:
            break
    return out
