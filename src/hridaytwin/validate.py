"""Two validations that only a synthetic cohort makes possible.

1. Twin calibration vs hidden truth.  The generator knows each patient's real
   drug-response multiplier; the EHR prior does not (it assumes the population
   average, so its correlation with the truth is 0 by construction).  We measure
   how well the twin recovers the multiplier from the stream after 4, 8, 13 and
   26 weeks.  Patients whose treatment changed after an admission are excluded,
   because their truth changed mid-stream.

2. Seed robustness.  The whole pipeline (new cohort, new twin, new models) is
   re-run under several generator seeds; we report mean +/- sd of the headline
   comparison so the result is not one lucky draw.

Writes model/twin_validation.json and model/seed_robustness.json.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from . import config
from .model import PRODUCTION, build_dataset, feature_sets, oof_scores, timing_auroc
from .synth import GLUCOSE_DRUG_EFFECT, SBP_DRUG_EFFECT, drug_load, simulate
from .twin import run_twin, to_arrays

CHECKPOINTS = [27, 55, 90, 179]   # clipped to the cohort length at run time


def twin_validation(seed: int | None = None) -> dict:
    ehr, daily, events, latent, truth, cal = simulate(seed=seed)
    A = to_arrays(daily, ehr.patient_id, config.DAYS)
    T = run_twin(A, ehr)
    stable = ~ehr.patient_id.isin(events.patient_id).to_numpy()
    out = {}
    for name, table, col, key, min_load in [
            ("glucose", GLUCOSE_DRUG_EFFECT, "r_g", "glucose", 45),
            ("blood_pressure", SBP_DRUG_EFFECT, "r_b", "sbp", 9)]:
        L = drug_load(ehr, table).astype(float)
        m = stable & (L >= min_load)
        rows = []
        for d in sorted({min(c, config.DAYS - 1) for c in CHECKPOINTS}):
            est = -T[key]["theta"][:, d, 1] / np.where(L > 0, L, 1)
            tr = truth[col].to_numpy()
            rows.append({"week": round((d + 1) / 7, 1),
                         "corr_with_truth": round(float(np.corrcoef(est[m], tr[m])[0, 1]), 3),
                         "mae_multiplier": round(float(np.abs(est[m] - tr[m]).mean()), 3),
                         "mean_posterior_sd_units": round(float(T[key]["theta_sd"][m, d, 1].mean()), 2)})
        prior_mae = float(np.abs(1 - truth[col].to_numpy()[m]).mean())
        out[name] = {"n_patients": int(m.sum()), "prior_corr_with_truth": 0.0,
                     "prior_mae_multiplier": round(prior_mae, 3), "checkpoints": rows}
    # calibration of the twin's predictive distribution
    zg = T["glucose"]["z"]
    zb = T["sbp"]["z"]
    out["predictive_calibration"] = {
        "glucose_z_mean": round(float(np.nanmean(zg)), 3), "glucose_z_sd": round(float(np.nanstd(zg)), 3),
        "glucose_within_2sd": round(float(np.mean(np.abs(zg[np.isfinite(zg)]) < 2)), 3),
        "sbp_z_mean": round(float(np.nanmean(zb)), 3), "sbp_z_sd": round(float(np.nanstd(zb)), 3),
        "sbp_within_2sd": round(float(np.mean(np.abs(zb[np.isfinite(zb)]) < 2)), 3),
        "note": "A calibrated twin has z mean ~0, sd ~1 and ~95% of readings within 2 sd."}
    return out


def seed_sweep(seeds=(2026, 11, 37)) -> dict:
    res = []
    for s in seeds:
        ehr, daily, events, latent, truth, cal = simulate(seed=s)
        df, _ = build_dataset(ehr, daily, events, cal)
        sets = feature_sets([c for c in df.columns if not c.startswith("y") and c != "days_to_event"])
        folds = list(GroupKFold(config.N_FOLDS).split(df, df.y, df.patient_id))
        row = {"seed": s, "n_rows": len(df), "positive_rate": round(float(df.y.mean()), 4)}
        oof = {}
        for k in ["ehr_only", PRODUCTION, "fused_plus_twin_state"]:
            oof[k] = oof_scores(df, sets[k], folds)
            row[f"{k}_auroc"] = round(roc_auc_score(df.y, oof[k]), 4)
        row["fused_minus_ehr"] = round(row[f"{PRODUCTION}_auroc"] - row["ehr_only_auroc"], 4)
        row["timing_fused"] = timing_auroc(df, oof[PRODUCTION])["mean_within_patient_auroc"]
        res.append(row)
        print(row, flush=True)
    keys = [k for k in res[0] if k.endswith("auroc") or k in ("fused_minus_ehr", "timing_fused")]
    summary = {k: {"mean": round(float(np.mean([r[k] for r in res])), 4),
                   "sd": round(float(np.std([r[k] for r in res], ddof=1)), 4)} for k in keys}
    summary["fused_beats_ehr_in"] = f"{sum(r['fused_minus_ehr'] > 0 for r in res)}/{len(res)} seeds"
    return {"per_seed": res, "summary": summary}


def main(skip_seeds: bool = False):
    config.MODEL_DIR.mkdir(exist_ok=True)
    tv = twin_validation()
    (config.MODEL_DIR / "twin_validation.json").write_text(json.dumps(tv, indent=2))
    print(json.dumps(tv["predictive_calibration"], indent=1))
    if skip_seeds:
        return
    ss = seed_sweep()
    (config.MODEL_DIR / "seed_robustness.json").write_text(json.dumps(ss, indent=2))
    print(json.dumps(ss["summary"], indent=1))


if __name__ == "__main__":
    main()
