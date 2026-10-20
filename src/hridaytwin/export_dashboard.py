"""Build the clinician console: dashboard/data.json -> dashboard/index.html.

Demo patients are HELD OUT: the risk model used here is re-fitted on every other
patient, so the console never shows a patient the model was trained on.  Daily
risk is therefore an honest out-of-sample replay of what the console would have
shown each morning.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from . import config
from .features import build_daily_features, event_count_array, stack_rows
from .model import PRODUCTION, build_dataset, feature_sets, gbm
from .simulate import candidate_plans, recommend, simulate_patient
from .synth import EVENT_TYPES, india_calendar, simulate
from .twin import B_NAMES, G_NAMES, run_twin, to_arrays

FRIENDLY = {
    "age": "Age", "hba1c": "HbA1c (EHR)", "egfr": "Kidney function (eGFR)", "prior_cvd": "Prior heart/stroke event",
    "tobacco": "Tobacco use", "ldl": "LDL cholesterol", "sbp_clinic": "Clinic BP", "dm_years": "Years with diabetes",
    "log_uacr": "Urine albumin", "events_past": "Recent adverse events", "med_sulfonylurea": "On sulfonylurea",
    "med_insulin": "On insulin", "glucose_m7": "Glucose, 7-day avg", "glucose_max7": "Glucose peak (7d)",
    "glucose_m28": "Glucose, 28-day avg", "gmin_min7": "Lowest glucose (7d)", "gmin_n_low28": "Low-glucose days (28d)",
    "gmin_n_low7": "Low-glucose days (7d)", "sbp_m7": "Home BP, 7-day avg", "sbp_max7": "Home BP peak (7d)",
    "sbp_m28": "Home BP, 28-day avg", "sbp_n_hi14": "Days BP >= 160 (14d)", "steps_m7": "Steps, 7-day avg",
    "steps_vs_base": "Activity vs personal baseline", "rhr_dev": "Resting HR vs baseline",
    "hrv_dev": "HRV vs baseline", "sleep_m7": "Sleep, 7-day avg", "adh_m7": "Doses taken (7d)",
    "adh_m28": "Doses taken (28d)", "adh_m3": "Doses taken (3d)", "salt_m14": "Salt intake (diet log)",
    "fast_days_next14": "Fasting days ahead", "feast_days_next14": "Festival days ahead",
    "glucose_sd14": "Glucose variability", "glucose_trend7": "Glucose trend", "sbp_trend7": "BP trend",
    "dbp_m7": "Diastolic BP (7d)", "glucose_n_hi7": "Days glucose > 250", "monsoon": "Monsoon season",
    "bmi": "BMI", "htn": "Hypertension", "t2d": "Type 2 diabetes", "male": "Sex (male)",
}


def pick_demo(ehr, df_rows, events, A):
    """Pick 8 diverse, clinically recognisable archetypes deterministically."""
    risk = df_rows.groupby("patient_id")["y"].sum()
    had = set(events.patient_id)
    late_event = set(events[events.day > 120].patient_id)
    picks, used = [], set()

    def take(mask, tag, prefer_event=None):
        cands = ehr[mask & ~ehr.patient_id.isin(used)]
        if prefer_event is True:
            c2 = cands[cands.patient_id.isin(late_event)]
            cands = c2 if len(c2) else cands
        elif prefer_event is False:
            c2 = cands[~cands.patient_id.isin(had)]
            cands = c2 if len(c2) else cands
        if len(cands):
            pid = cands.patient_id.iloc[0]
            used.add(pid)
            picks.append((pid, tag))

    E = ehr
    take((E.med_sulfonylurea == 1) & (E.observes_fasting == 1) & (E.has_cgm == 1) & (E.age >= 58), "Fasting on a sulfonylurea", True)
    take((E.htn == 1) & (E.sbp_clinic >= 160), "Uncontrolled hypertension", True)
    take((E.t2d == 1) & (E.hba1c >= 10), "Poorly controlled diabetes", True)
    take((E.prior_cvd == 1) & (E.age >= 65), "Prior heart attack / stroke", True)
    take((E.egfr < 45) & (E.t2d == 1), "Diabetic kidney disease", None)
    take((E.t2d == 1) & (E.htn == 1) & (E.hba1c < 7.2) & (E.sbp_clinic < 135), "Well controlled", False)
    take((E.age <= 40) & (E.t2d == 1), "Young-onset diabetes", None)
    take((E.htn == 1) & (E.t2d == 0) & (E.tobacco == 1), "Hypertension + tobacco", None)
    return picks[: config.N_DEMO_PATIENTS]


def r1(x, n=1):
    return [None if (v is None or not np.isfinite(v)) else round(float(v), n) for v in x]


def main():
    ehr, daily, events, latent, truth, cal = simulate()
    df, (A, T, F, evc) = build_dataset(ehr, daily, events, cal)
    metrics = json.loads((config.MODEL_DIR / "metrics.json").read_text())
    thr = metrics["operating_point"]["threshold"]
    demo = pick_demo(ehr, df, events, A)
    demo_ids = [p for p, _ in demo]
    sets = feature_sets([c for c in df.columns if not c.startswith("y") and c != "days_to_event"])
    feats = sets[PRODUCTION]

    # hold-out model: trained on everyone except the demo patients
    tr = df[~df.patient_id.isin(demo_ids)]
    hold = {"features": feats, "model": gbm().fit(tr[feats], tr["y"]),
            "heads": {t: gbm().fit(tr[feats], tr[f"y_{t}"]) for t in EVENT_TYPES}}

    import shap
    explainer = shap.TreeExplainer(hold["model"])

    D = config.DAYS
    out_patients = []
    for pid, tag in demo:
        i = int(np.where(ehr.patient_id == pid)[0][0])
        e = ehr.iloc[i]
        days = np.arange(config.WARMUP_DAYS - 1, D)
        rows = stack_rows(F, ehr, days, patient_idx=[i])
        X = rows[feats]
        risk = hold["model"].predict_proba(X)[:, 1]
        heads = {t: hold["heads"][t].predict_proba(X)[:, 1] for t in EVENT_TYPES}
        inhosp = A["in_hospital"][i, days] > 0
        risk = np.where(inhosp, np.nan, risk)

        # SHAP for the latest day and for every weekly check-in (drivers timeline)
        sv = explainer.shap_values(X)
        def drivers(row_idx, k=6):
            vals = sv[row_idx]
            order = np.argsort(-np.abs(vals))[:k]
            return [{"feature": FRIENDLY.get(feats[j], feats[j].replace("_", " ")),
                     "value": None if pd.isna(X.iloc[row_idx, j]) else round(float(X.iloc[row_idx, j]), 2),
                     "impact": round(float(vals[j]), 3)} for j in order]
        weekly_idx = list(range(0, len(days), 7)) + [len(days) - 1]
        drv = {int(days[k]): drivers(k) for k in weekly_idx}

        # explainable alerts
        alerts = []
        zg, zb = T["glucose"]["z"][i], T["sbp"]["z"][i]
        prev_flag = False
        for k, d in enumerate(days):
            flag = bool(np.isfinite(risk[k]) and risk[k] >= thr)
            if flag and not prev_flag:
                top = max(EVENT_TYPES, key=lambda t: heads[t][k])
                alerts.append({"day": int(d), "kind": "risk", "level": "high",
                               "text": f"14-day risk {risk[k]*100:.1f}% (threshold {thr*100:.1f}%) - most likely: {top.replace('_', ' ')}"})
            prev_flag = flag
            gm = A["glucose_min"][i, d]
            if np.isfinite(gm) and gm < 54:
                alerts.append({"day": int(d), "kind": "hypo", "level": "high", "text": f"CGM low {gm:.0f} mg/dL (level-2 hypoglycaemia)"})
            if np.isfinite(zg[d]) and zg[d] > 2.5:
                alerts.append({"day": int(d), "kind": "twin", "level": "med", "text": f"Glucose {A['glucose'][i, d]:.0f} far above twin expectation ({T['glucose']['yhat'][i, d]:.0f})"})
            if np.isfinite(zb[d]) and zb[d] > 2.5:
                alerts.append({"day": int(d), "kind": "twin", "level": "med", "text": f"Home BP {A['sbp'][i, d]:.0f} far above twin expectation ({T['sbp']['yhat'][i, d]:.0f})"})
            if d >= 3:
                mt = A["med_taken"][i, d - 2:d + 1]
                if np.all(mt == 0):
                    alerts.append({"day": int(d), "kind": "adherence", "level": "med", "text": "3 days of missed doses (pill-box)"})
            rd = F["rhr_dev"][i, d]
            if np.isfinite(rd) and rd > 8 and (d == days[0] or not (F["rhr_dev"][i, d - 1] > 8)):
                alerts.append({"day": int(d), "kind": "illness", "level": "med", "text": f"Resting HR +{rd:.0f} bpm vs baseline - possible infection"})

        ev_i = events[events.patient_id == pid]
        recent_sbp = float(np.nanmean(A["sbp"][i, -28:])) if np.isfinite(A["sbp"][i, -28:]).any() else float(e.sbp_clinic)
        recent_glu = float(np.nanmean(A["glucose"][i, -28:])) if np.isfinite(A["glucose"][i, -28:]).any() else 28.7 * e.hba1c - 46.7
        plans = candidate_plans(e, recent_sbp, recent_glu)
        sim = simulate_patient(i, A, ehr, evc, hold, plans, n_draws=16, seed=7)
        step = 3
        for p in sim["plans"]:
            for k in ("glucose_p50", "glucose_p10", "glucose_p90", "sbp_p50", "sbp_p10", "sbp_p90"):
                p[k] = p[k][::step]
            p["type_risk"] = {t: round(v, 4) for t, v in p["type_risk"].items()}
            p["mean_risk_14d"] = round(p["mean_risk_14d"], 4)

        tg, tb = T["glucose"], T["sbp"]
        out_patients.append({
            "id": pid, "archetype": tag,
            "ehr": {k: (v.item() if hasattr(v, "item") else v) for k, v in e.items()},
            "series": {
                "steps": r1(A["steps"][i], 0), "rhr": r1(A["rhr"][i]), "hrv": r1(A["hrv"][i]),
                "sleep": r1(A["sleep_h"][i], 2), "glucose": r1(A["glucose"][i], 0),
                "glucose_min": r1(A["glucose_min"][i], 0), "sbp": r1(A["sbp"][i], 0),
                "dbp": r1(A["dbp"][i], 0), "med_taken": r1(A["med_taken"][i], 0),
                "salt": r1(A["salt_score"][i]), "in_hospital": r1(A["in_hospital"][i], 0),
                "twin_g": r1(tg["yhat"][i], 0), "twin_g_sd": r1(tg["ysd"][i], 0),
                "twin_b": r1(tb["yhat"][i], 0), "twin_b_sd": r1(tb["ysd"][i], 0),
                "g_drug": r1(-tg["theta"][i, :, 1], 1), "g_drug_sd": r1(tg["theta_sd"][i, :, 1], 1),
                "b_drug": r1(-tb["theta"][i, :, 1], 1), "b_drug_sd": r1(tb["theta_sd"][i, :, 1], 1),
                "b_salt": r1(tb["theta"][i, :, 3], 2),
            },
            "risk": {"days": days.tolist(), "p": r1(risk, 4),
                     "types": {t: r1(heads[t], 4) for t in EVENT_TYPES}},
            "drivers": drv, "alerts": alerts,
            "events": [{"day": int(d), "type": t} for d, t in zip(ev_i.day, ev_i.event_type)],
            "whatif": sim, "recommended": recommend(sim["plans"]),
            "truth": {"glucose_drug_response": round(float(truth.r_g.iloc[i]), 2),
                      "bp_drug_response": round(float(truth.r_b.iloc[i]), 2)},
        })
        print(f"{pid} {tag}: events={len(ev_i)} alerts={len(alerts)} plans={len(plans)}")

    calx = india_calendar(D + config.WHATIF_DAYS)
    payload = {
        "generated_note": "Synthetic patients. Not for clinical use.",
        "start_date": str(calx.date.iloc[0].date()), "days": D,
        "threshold": thr, "horizon": config.HORIZON_DAYS,
        "calendar": {"monsoon": calx.monsoon.tolist(), "fast": calx.fast_day.tolist(),
                     "feast": calx.feast.tolist()},
        "metrics": {k: metrics[k] for k in ("variants", "operating_point", "event_types",
                                            "timing_pre_first_event", "temporal_holdout",
                                            "positive_rate", "n_patients")},
        "patients": out_patients,
    }
    config.DASH_DIR.mkdir(exist_ok=True)
    js = json.dumps(payload, separators=(",", ":"), allow_nan=False, default=lambda o: None)
    (config.DASH_DIR / "data.json").write_text(js)
    tpl = (config.DASH_DIR / "template.html").read_text()
    (config.DASH_DIR / "index.html").write_text(tpl.replace("/*__DATA__*/null", js))
    print(f"dashboard/index.html written ({len(js)/1e6:.2f} MB of data)")


if __name__ == "__main__":
    main()
