"""Adverse-event risk model: training, honest evaluation and ablations.

Task: at a weekly check-in on day d, predict whether the patient has an adverse
cardiometabolic event (severe hypoglycaemia, hyperglycaemic crisis, hypertensive
emergency or acute CV event) in the next 14 days.

Evaluation protocol (all reported in model/metrics.json -> docs/RESULTS.md):
  * 5-fold GroupKFold by patient -> a patient is never in train and test.
  * Temporal + patient hold-out: train on early weeks of some patients, test on
    later weeks of *other* patients (the deployment situation).
  * Ablations with identical folds: EHR-only, stream-only, fused without twin
    features, fused (full), and a clinical threshold-rule baseline.
  * Paired deltas with patient-level cluster bootstrap p-values.
  * Label-permutation control (must be ~0.50).
  * Calibration (Brier, ECE) and an operating point set by an alert budget.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from . import config
from .features import (EHR_FEATURES, build_daily_features, event_count_array, labels,
                       stack_rows)
from .synth import EVENT_TYPES
from .twin import run_twin, to_arrays

CAL_FEATURES = ["fast_days_next14", "feast_days_next14", "monsoon"]


def feature_sets(cols) -> dict[str, list[str]]:
    twin = [c for c in cols if c.startswith("twin_")]
    ehr = EHR_FEATURES + ["events_past"]
    skip = set(ehr) | set(twin) | set(CAL_FEATURES) | {"patient_id", "day"}
    stream = [c for c in cols if c not in skip]
    return {
        "ehr_only": ehr,
        "stream_only": stream + CAL_FEATURES,
        "fused": ehr + stream + CAL_FEATURES,                       # production model
        "fused_plus_twin_state": ehr + stream + CAL_FEATURES + twin,  # ablation
    }


PRODUCTION = "fused"


def pred_days() -> np.ndarray:
    return np.arange(config.WARMUP_DAYS - 1, config.DAYS - config.HORIZON_DAYS, config.PRED_EVERY_DAYS)


def build_dataset(ehr, daily, events, cal):
    D = config.DAYS
    A = to_arrays(daily, ehr.patient_id, D)
    T = run_twin(A, ehr)
    evc = event_count_array(events, ehr.patient_id, D)
    F = build_daily_features(A, T, ehr, cal, evc)
    rows = stack_rows(F, ehr, pred_days())
    hosp = A["in_hospital"][np.repeat(np.arange(len(ehr)), len(pred_days())),
                            np.tile(pred_days(), len(ehr))]
    rows = rows[hosp == 0].reset_index(drop=True)
    y = labels(events, rows, config.HORIZON_DAYS)
    return pd.concat([rows, y], axis=1), (A, T, F, evc)


def rule_score(X: pd.DataFrame) -> np.ndarray:
    """What a busy clinic would flag from the same readings without a model."""
    s = np.zeros(len(X))
    s += (X["sbp_max7"].fillna(0) >= 170)
    s += (X["glucose_max7"].fillna(0) >= 300)
    s += (X["gmin_n_low7"].fillna(0) >= 1)
    s += (X["adh_m7"].fillna(1) < 0.5)
    s += 0.5 * X["prior_cvd"] + 0.25 * (X["age"] >= 65)
    return s


def gbm():
    return HistGradientBoostingClassifier(random_state=config.SEED, **config.GBM_PARAMS)


def ece(y, p, bins=10) -> float:
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def oof_scores(df, feats, folds, target="y", model_fn=gbm):
    p = np.zeros(len(df))
    for tr, te in folds:
        m = model_fn()
        m.fit(df.iloc[tr][feats], df.iloc[tr][target])
        p[te] = m.predict_proba(df.iloc[te][feats])[:, 1]
    return p


def cluster_bootstrap_delta(y, p_a, p_b, groups, n_boot=1000, seed=0):
    """Patient-level bootstrap of AUROC(a) - AUROC(b); two-sided p-value."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    deltas = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[k] for k in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        deltas.append(roc_auc_score(yy, p_a[idx]) - roc_auc_score(yy, p_b[idx]))
    deltas = np.array(deltas)
    point = roc_auc_score(y, p_a) - roc_auc_score(y, p_b)
    pval = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {"delta": round(float(point), 4), "ci95": [round(float(np.quantile(deltas, .025)), 4),
            round(float(np.quantile(deltas, .975)), 4)], "p_value": max(float(pval), 1 / len(deltas))}


def operating_point(df, p, budget):
    thr = float(np.quantile(p, 1 - budget))
    flag = p >= thr
    y = df["y"].to_numpy()
    tp = (flag & (y == 1)).sum()
    lead = df.loc[flag & (y == 1), "days_to_event"]
    return {"alert_budget_per_patient_week": budget, "threshold": round(thr, 4),
            "sensitivity": round(float(tp / max(1, y.sum())), 3),
            "ppv": round(float(tp / max(1, flag.sum())), 3),
            "base_rate": round(float(y.mean()), 4),
            "lift_over_base_rate": round(float(tp / max(1, flag.sum()) / y.mean()), 2),
            "median_lead_time_days": float(lead.median()) if len(lead) else None}


def evaluate(df: pd.DataFrame, quick: bool = False) -> dict:
    sets = feature_sets([c for c in df.columns if not c.startswith("y") and c != "days_to_event"])
    groups = df["patient_id"].to_numpy()
    folds = list(GroupKFold(n_splits=config.N_FOLDS).split(df, df["y"], groups))
    y = df["y"].to_numpy()
    res = {"n_rows": int(len(df)), "n_patients": int(df.patient_id.nunique()),
           "positive_rate": round(float(y.mean()), 4), "variants": {}}
    oof = {}
    for name, feats in sets.items():
        p = oof_scores(df, feats, folds)
        oof[name] = p
        res["variants"][name] = {"n_features": len(feats),
                                 "auroc": round(roc_auc_score(y, p), 4),
                                 "auprc": round(average_precision_score(y, p), 4),
                                 "brier": round(brier_score_loss(y, p), 5),
                                 "ece": round(ece(y, p), 4)}
    r = rule_score(df)
    oof["clinical_rules"] = r
    res["variants"]["clinical_rules"] = {"auroc": round(roc_auc_score(y, r), 4),
                                         "auprc": round(average_precision_score(y, r), 4)}
    lr = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               LogisticRegression(max_iter=2000, C=0.3))
    p_lr = oof_scores(df, sets[PRODUCTION], folds, model_fn=lr)
    oof["fused_logistic"] = p_lr
    res["variants"]["fused_logistic"] = {"auroc": round(roc_auc_score(y, p_lr), 4),
                                         "auprc": round(average_precision_score(y, p_lr), 4)}

    nb = 300 if quick else 1000
    full = oof[PRODUCTION]
    res["paired_deltas"] = {
        f"fused_vs_{k}": cluster_bootstrap_delta(y, full, oof[k], groups, nb)
        for k in ["ehr_only", "stream_only", "fused_plus_twin_state", "fused_logistic",
                  "clinical_rules"]}
    res["timing_pre_first_event"] = {k: timing_auroc(df, oof[k]) for k in
                                     ["ehr_only", "stream_only", PRODUCTION,
                                      "fused_plus_twin_state"]}

    # label-permutation control: shuffle labels within the training folds only
    rng = np.random.default_rng(1)
    dperm = df.copy()
    dperm["y_perm"] = rng.permutation(y)
    pp = oof_scores(dperm, sets[PRODUCTION], folds, target="y_perm")
    res["controls"] = {"label_permutation_auroc": round(roc_auc_score(dperm["y_perm"], pp), 4)}

    # temporal + patient hold-out
    split = config.TEMPORAL_SPLIT_DAY
    tp = np.zeros(len(df))
    mask_te = np.zeros(len(df), bool)
    for tr, te in folds:
        tr = tr[df.iloc[tr]["day"].to_numpy() < split - config.HORIZON_DAYS]
        te = te[df.iloc[te]["day"].to_numpy() >= split]
        m = gbm().fit(df.iloc[tr][sets[PRODUCTION]], df.iloc[tr]["y"])
        tp[te] = m.predict_proba(df.iloc[te][sets[PRODUCTION]])[:, 1]
        mask_te[te] = True
    m2 = np.zeros(len(df))
    for tr, te in folds:
        tr = tr[df.iloc[tr]["day"].to_numpy() < split - config.HORIZON_DAYS]
        te = te[df.iloc[te]["day"].to_numpy() >= split]
        mm = gbm().fit(df.iloc[tr][sets["ehr_only"]], df.iloc[tr]["y"])
        m2[te] = mm.predict_proba(df.iloc[te][sets["ehr_only"]])[:, 1]
    res["temporal_holdout"] = {
        "train_days": f"< {split - config.HORIZON_DAYS} (labels fully observed before split)",
        "test_days": f">= {split}", "n_test": int(mask_te.sum()),
        "fused_auroc": round(roc_auc_score(y[mask_te], tp[mask_te]), 4),
        "ehr_only_auroc": round(roc_auc_score(y[mask_te], m2[mask_te]), 4)}

    # per-event-type heads (fused features)
    res["event_types"] = {}
    for t in EVENT_TYPES:
        yt = df[f"y_{t}"].to_numpy()
        pt = oof_scores(df, sets[PRODUCTION], folds, target=f"y_{t}")
        pe = oof_scores(df, sets["ehr_only"], folds, target=f"y_{t}")
        oof[f"type_{t}"] = pt
        res["event_types"][t] = {"positives": int(yt.sum()),
                                 "fused_auroc": round(roc_auc_score(yt, pt), 4),
                                 "ehr_only_auroc": round(roc_auc_score(yt, pe), 4),
                                 "delta_vs_ehr": cluster_bootstrap_delta(yt, pt, pe, groups, nb // 2)}

    res["operating_point"] = operating_point(df, full, config.ALERT_BUDGET)
    res["operating_point_rules"] = operating_point(df, r + 1e-6 * full, config.ALERT_BUDGET)
    # reliability curve for the report
    q = np.quantile(full, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(q, full, side="right") - 1, 0, 9)
    res["reliability"] = [{"mean_pred": round(float(full[idx == b].mean()), 4),
                           "observed": round(float(y[idx == b].mean()), 4)} for b in range(10)]
    return res, oof, sets


def timing_auroc(df: pd.DataFrame, p: np.ndarray) -> dict:
    """Mean within-patient AUROC over weeks *before the patient's first event*.

    Static EHR features are constant there, so EHR-only scores exactly 0.50; anything
    above 0.50 is the stream telling us *when* a given patient is in danger.
    """
    d = df.assign(p=p)
    d = d[d["events_past"] == 0]
    aucs = [roc_auc_score(s.y, s.p) for _, s in d.groupby("patient_id")
            if 0 < s.y.sum() < len(s)]
    return {"mean_within_patient_auroc": round(float(np.mean(aucs)), 4),
            "n_patients": len(aucs)}


def fit_final(df: pd.DataFrame, sets: dict):
    feats = sets[PRODUCTION]
    main = gbm().fit(df[feats], df["y"])
    heads = {t: gbm().fit(df[feats], df[f"y_{t}"]) for t in EVENT_TYPES}
    return {"features": feats, "model": main, "heads": heads, "feature_sets": sets}


def main():
    from .synth import simulate
    t0 = time.time()
    ehr, daily, events, latent, truth, cal = simulate()
    df, _ = build_dataset(ehr, daily, events, cal)
    res, oof, sets = evaluate(df)
    bundle = fit_final(df, sets)
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, config.MODEL_DIR / "risk_model.joblib")
    out = df[["patient_id", "day", "y", "days_to_event"]].copy()
    for k, v in oof.items():
        out[f"p_{k}"] = np.round(v, 5)
    out.to_csv(config.MODEL_DIR / "oof_predictions.csv", index=False)
    res["runtime_seconds"] = round(time.time() - t0, 1)
    res["config"] = {"n_patients": config.N_PATIENTS, "days": config.DAYS, "seed": config.SEED,
                     "horizon_days": config.HORIZON_DAYS, "folds": config.N_FOLDS}
    (config.MODEL_DIR / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "reliability"}, indent=1))


if __name__ == "__main__":
    main()
