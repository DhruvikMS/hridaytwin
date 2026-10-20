"""Synthetic cohort generator: static EHR + 180 days of daily wearable / home-device data.

Why synthetic: real paired EHR + wearable data for Indian cardiometabolic patients
is not openly available, and the Digital Personal Data Protection Act 2023 rightly
restricts it. The generator is *mechanistic*, not a lookup table: each patient has
hidden physiological parameters (drug response, salt sensitivity, activity response,
adherence propensity) and their latent glucose and blood pressure evolve day by day
under medication, activity, diet, illness, monsoon season and festival fasting.
Adverse events are drawn stochastically from the *latent* state, while the model
only ever sees *noisy, partially missing observations* of it - so labels cannot be
read back from any single observed column.

Population priors are set to published Indian figures where they exist
(see docs/CLINICAL_RATIONALE.md for sources).

Outputs (data/):
    ehr.csv           one row per patient (FHIR-mappable fields)
    daily.csv         one row per patient-day (observed streams only, with NaNs)
    events.csv        adverse events (patient_id, day, type)
    latent.csv        hidden ground truth (for tests / audit only - never a feature)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

START_DATE = pd.Timestamp("2026-04-04")

EVENT_TYPES = ["severe_hypoglycaemia", "hyperglycaemic_crisis",
               "hypertensive_emergency", "acute_cv_event"]


# --------------------------------------------------------------------------- calendar
def india_calendar(days: int) -> pd.DataFrame:
    """Day-level context that is known in advance (illustrative 2026 calendar)."""
    dates = pd.date_range(START_DATE, periods=days, freq="D")
    cal = pd.DataFrame({"day": np.arange(days), "date": dates})
    cal["dow"] = dates.dayofweek
    # South-west monsoon over most of India: ~8 Jun - ~20 Sep
    cal["monsoon"] = ((dates >= "2026-06-08") & (dates <= "2026-09-20")).astype(int)
    # Fasting days (approximate): Ekadashi roughly every 15 days + Shravan Mondays
    ekadashi = (np.arange(days) % 15 == 7)
    shravan_monday = (dates >= "2026-07-29") & (dates <= "2026-08-27") & (cal["dow"] == 0)
    navratri = (dates >= "2026-10-11") & (dates <= "2026-10-19")   # used by the simulator
    cal["fast_day"] = (ekadashi | shravan_monday | navratri).astype(int)
    # Festival feasting windows (sweets, salty snacks): Eid al-Adha, Ganesh Chaturthi, Diwali
    feast = ((dates >= "2026-05-26") & (dates <= "2026-05-29")) | \
            ((dates >= "2026-09-14") & (dates <= "2026-09-24")) | \
            ((dates >= "2026-11-06") & (dates <= "2026-11-11"))
    cal["feast"] = feast.astype(int)
    return cal


# --------------------------------------------------------------------------- EHR
def make_ehr(n: int, rng: np.random.Generator) -> pd.DataFrame:
    age = np.clip(rng.normal(54, 10.5, n), 30, 84).round()
    male = rng.random(n) < 0.55
    region = rng.choice(["North", "South", "East", "West", "Northeast"], n,
                        p=[0.27, 0.27, 0.19, 0.21, 0.06])
    bmi = np.clip(rng.normal(26.6, 4.0, n), 18, 42).round(1)
    waist = np.clip(bmi * 3.4 + np.where(male, 4, -2) + rng.normal(0, 5, n), 65, 140).round()
    tobacco = rng.random(n) < np.where(male, 0.26, 0.06)
    family_hx = rng.random(n) < 0.35

    t2d = rng.random(n) < 0.72
    htn = rng.random(n) < 0.66
    neither = ~(t2d | htn)
    coin = rng.random(n) < 0.5
    t2d = t2d | (neither & coin)
    htn = htn | (neither & ~coin)

    dm_years = np.where(t2d, np.clip(rng.gamma(2.0, 3.6, n), 0.3, 30), 0).round(1)
    egfr = np.clip(rng.normal(92 - 0.65 * (age - 50) - 9 * (dm_years > 10), 16, n), 14, 125).round()
    uacr = np.round(np.exp(rng.normal(2.6 + 0.9 * t2d + 0.4 * htn, 1.0, n)), 1)  # mg/g
    ldl = np.clip(rng.normal(116, 32, n), 45, 250).round()
    hba1c = np.where(t2d, np.clip(rng.normal(8.1, 1.4, n), 6.0, 13.5),
                     np.clip(rng.normal(5.8, 0.35, n), 4.8, 6.4)).round(1)
    sbp_clinic = np.where(htn, rng.normal(146, 14, n), rng.normal(126, 10, n)).round()
    prior_cvd = rng.random(n) < (0.05 + 0.06 * (age > 60) + 0.04 * (t2d & htn) + 0.03 * tobacco)

    metformin = t2d & (egfr >= 30) & (rng.random(n) < 0.88)
    sulfonylurea = t2d & (rng.random(n) < 0.46)
    dpp4i = t2d & (rng.random(n) < 0.30)
    sglt2i = t2d & (egfr >= 25) & (rng.random(n) < 0.14)
    insulin = t2d & ((hba1c > 9.5) | (dm_years > 12)) & (rng.random(n) < 0.45)
    raas = (htn | (t2d & (uacr > 30))) & (rng.random(n) < 0.72)
    ccb = htn & (rng.random(n) < 0.40)
    statin = (t2d | prior_cvd | (ldl > 160)) & (rng.random(n) < 0.52)
    prior_hypo = (sulfonylurea | insulin) & (rng.random(n) < 0.16)
    observes_fasting = rng.random(n) < 0.38
    has_cgm = t2d & (rng.random(n) < 0.28)

    ehr = pd.DataFrame({
        "patient_id": [f"HT{idx:05d}" for idx in range(n)],
        "age": age.astype(int), "male": male.astype(int), "region": region,
        "bmi": bmi, "waist_cm": waist, "tobacco": tobacco.astype(int),
        "family_hx_cvd": family_hx.astype(int),
        "t2d": t2d.astype(int), "htn": htn.astype(int), "dm_years": dm_years,
        "hba1c": hba1c, "ldl": ldl, "egfr": egfr, "uacr": uacr,
        "sbp_clinic": sbp_clinic, "prior_cvd": prior_cvd.astype(int),
        "prior_hypo": prior_hypo.astype(int),
        "med_metformin": metformin.astype(int), "med_sulfonylurea": sulfonylurea.astype(int),
        "med_dpp4i": dpp4i.astype(int), "med_sglt2i": sglt2i.astype(int),
        "med_insulin": insulin.astype(int), "med_raas": raas.astype(int),
        "med_ccb": ccb.astype(int), "med_statin": statin.astype(int),
        "observes_fasting": observes_fasting.astype(int), "has_cgm": has_cgm.astype(int),
    })
    return ehr


# Literature-anchored average effects (mean glucose mg/dL via ADAG eAG = 28.7*A1c - 46.7)
GLUCOSE_DRUG_EFFECT = {"med_metformin": 30, "med_sulfonylurea": 30, "med_dpp4i": 17,
                       "med_sglt2i": 17, "med_insulin": 45}
SBP_DRUG_EFFECT = {"med_raas": 9, "med_ccb": 9, "med_sglt2i": 4}


def drug_load(ehr: pd.DataFrame, table: dict) -> np.ndarray:
    return sum(ehr[k].to_numpy() * v for k, v in table.items())


# --------------------------------------------------------------------------- simulate
def simulate(n: int | None = None, days: int | None = None, seed: int | None = None):
    n = n or config.N_PATIENTS
    days = days or config.DAYS
    seed = config.SEED if seed is None else seed
    rng = np.random.default_rng(seed)

    ehr = make_ehr(n, rng)
    cal = india_calendar(days)
    E = ehr  # alias

    # ---- hidden physiology (the "true" twin parameters) ----------------------
    p_adh = rng.beta(6.0, 1.6, n)                       # adherence propensity
    r_g = np.exp(rng.normal(0, 0.35, n))                # glycaemic drug-response multiplier
    r_b = np.exp(rng.normal(0, 0.35, n))                # BP drug-response multiplier
    k_steps_g = np.clip(rng.normal(4.0, 1.6, n), 0.5, 9)   # mg/dL per 1000 steps (7d)
    k_steps_b = np.clip(rng.normal(0.7, 0.3, n), 0.1, 1.6) # mmHg per 1000 steps
    k_carb = np.clip(rng.normal(7.0, 2.5, n), 1, 14)    # mg/dL per carb-score unit
    salt_sens = np.exp(rng.normal(np.log(1.1), 0.5, n)) # mmHg per g salt (He et al. ~1.2)
    salt_mean = np.clip(rng.normal(10.0, 2.2, n), 5, 16)  # g/day (India ~10 g)
    steps_mean = np.clip(rng.lognormal(np.log(5200), 0.45, n), 1200, 16000)
    sleep_mean = np.clip(rng.normal(6.6, 0.7, n), 4.5, 8.5)
    rhr_base = np.clip(rng.normal(76, 7, n) - (steps_mean - 5000) / 1500, 55, 100)
    hrv_base = np.clip(rng.normal(32, 9, n) - 0.15 * (E.age.to_numpy() - 50), 10, 70)
    wear_p = rng.beta(9, 1.6, n)
    t2d = E.t2d.to_numpy().astype(bool)
    htn = E.htn.to_numpy().astype(bool)
    su_or_ins = (E.med_sulfonylurea.to_numpy() | E.med_insulin.to_numpy()).astype(bool)
    egfr = E.egfr.to_numpy()
    age = E.age.to_numpy()

    # untreated set points, back-calculated from clinic values measured on treatment
    eag = 28.7 * E.hba1c.to_numpy() - 46.7
    g_load = drug_load(E, GLUCOSE_DRUG_EFFECT)
    G_set = eag + r_g * g_load * p_adh
    b_load = drug_load(E, SBP_DRUG_EFFECT)
    S_set = E.sbp_clinic.to_numpy() + r_b * b_load * p_adh

    # baseline cardiovascular hazard (per day), loosely WHO-South-Asia-chart shaped
    lin = (0.065 * (age - 55) + 0.45 * E.male + 0.55 * E.tobacco + 0.85 * E.prior_cvd
           + 0.006 * (E.ldl - 120) + 0.35 * t2d + 0.012 * np.maximum(0, 75 - egfr)
           + 0.25 * (E.uacr > 30) - 0.25 * E.med_statin).to_numpy()
    h0_cv = 2.4e-4 * np.exp(lin)

    # ---- state arrays --------------------------------------------------------
    G = eag.copy() + rng.normal(0, 8, n)
    S = E.sbp_clinic.to_numpy().astype(float) + rng.normal(0, 4, n)
    adh7 = p_adh.copy()
    steps7 = steps_mean.copy()
    salt_ema = salt_mean.copy()
    carb_ema = np.zeros(n)
    missed_yesterday = np.zeros(n, bool)
    refill_gap = np.zeros(n, int)
    ill_left = np.zeros(n, int)
    ill_sev = np.zeros(n)
    hosp_left = np.zeros(n, int)
    weight = E.bmi.to_numpy() * 1.62 ** 2

    med_g = g_load.astype(float)        # current glycaemic drug load (changes after events)
    med_b = b_load.astype(float)
    su_active = su_or_ins.copy()

    rows, latent_rows, events = [], [], []
    ar_steps = np.zeros(n)

    for d in range(days):
        c = cal.iloc[d]
        # ---- context -----------------------------------------------------------
        weekday_f = 0.88 if c.dow == 6 else 1.0
        monsoon_f = 0.83 if c.monsoon else 1.0
        fasting = (c.fast_day == 1) & E.observes_fasting.to_numpy().astype(bool)

        # illness onset (viral fevers / dengue season doubles in monsoon)
        onset = (ill_left == 0) & (rng.random(n) < (1 / 140) * (2.2 if c.monsoon else 1.0))
        ill_left = np.where(onset, rng.integers(3, 11, n), ill_left)
        ill_sev = np.where(onset, rng.uniform(0.4, 1.0, n), ill_sev)
        ill = (ill_left > 0) * ill_sev
        ill_left = np.maximum(0, ill_left - 1)

        # ---- adherence (Markov lapses + pharmacy refill gaps) -------------------
        new_gap = (refill_gap == 0) & (rng.random(n) < 1 / 110)
        refill_gap = np.where(new_gap, rng.integers(3, 9, n), refill_gap)
        p_take = np.where(missed_yesterday, p_adh * 0.55, np.minimum(0.995, p_adh + 0.08))
        p_take = p_take * (1 - 0.25 * ill)
        took = (rng.random(n) < p_take) & (refill_gap == 0)
        refill_gap = np.maximum(0, refill_gap - 1)
        missed_yesterday = ~took
        adh7 = 0.75 * adh7 + 0.25 * took

        # ---- behaviour ---------------------------------------------------------
        ar_steps = 0.6 * ar_steps + rng.normal(0, 0.22, n)
        steps = steps_mean * weekday_f * monsoon_f * (1 - 0.6 * ill) * np.exp(ar_steps)
        steps7 = 0.8 * steps7 + 0.2 * steps
        salt = salt_mean + 3.0 * c.feast + rng.normal(0, 1.6, n)
        salt_ema = 0.8 * salt_ema + 0.2 * salt
        carb = rng.normal(0, 1, n) + 1.8 * c.feast - 2.2 * fasting
        carb_ema = 0.6 * carb_ema + 0.4 * carb
        sleep = sleep_mean - 1.2 * ill + rng.normal(0, 0.7, n)

        # ---- physiology ----------------------------------------------------------
        progression = 0.03 * d * t2d          # slow beta-cell decline (mg/dL per day)
        G_target = (G_set + progression - r_g * med_g * adh7
                    - k_steps_g * (steps7 - steps_mean) / 1000
                    + k_carb * carb_ema + 55 * ill)
        G_target = np.where(t2d, G_target, np.minimum(G_target, 118))
        G = G + 0.3 * (G_target - G) + rng.normal(0, 5, n)
        G = np.clip(G, 70, 480)

        exercise_surge = steps > 1.45 * steps7
        hypo_push = su_active * took * (62 * fasting + 26 * exercise_surge
                                        + 16 * (egfr < 45) + 10 * (age >= 70) + 18 * ill * (carb < -0.5))
        spread = 22 + 0.18 * G
        Gmin = G - spread - hypo_push + rng.normal(0, 9, n)
        Gmin = np.where(su_active, Gmin, np.maximum(Gmin, 68 + rng.normal(0, 4, n)))
        Gmin = np.clip(Gmin, 28, None)

        S_target = (S_set - r_b * med_b * adh7 + salt_sens * (salt_ema - 10)
                    - k_steps_b * (steps7 - steps_mean) / 1000
                    + 2.0 * np.maximum(0, 6.5 - sleep) - 6 * ill)
        S = S + 0.35 * (S_target - S) + rng.normal(0, 3.5, n)
        Dbp = 0.52 * S + 12 + rng.normal(0, 3, n)

        rhr = rhr_base + 11 * ill + 1.5 * np.maximum(0, 6.5 - sleep) \
            - (steps7 - steps_mean) / 2500 + rng.normal(0, 2, n)
        hrv = hrv_base - 9 * ill - 0.4 * (rhr - rhr_base) + rng.normal(0, 3.5, n)
        weight = weight + rng.normal(0, 0.08, n) + 0.004 * (carb_ema)

        # ---- adverse events from the latent state --------------------------------
        in_hosp = hosp_left > 0
        p_hypo = np.where(Gmin < 56, 0.10 + 0.25 * (Gmin < 44) + 0.08 * (age >= 70), 0.0)
        p_hyper = 0.05 / (1 + np.exp(-(G - 330) / 22)) * (1 + 2.5 * ill)
        p_htn = 0.06 / (1 + np.exp(-(S - 184) / 6))
        h_cv = h0_cv * np.exp(0.030 * (S - 135) + 0.006 * (G - 150)
                              + 1.8 * ill + 0.03 * (rhr - rhr_base))
        p_cv = 1 - np.exp(-h_cv)
        draws = rng.random((4, n))
        hit = np.vstack([draws[0] < p_hypo, draws[1] < p_hyper,
                         draws[2] < p_htn, draws[3] < p_cv]) & ~in_hosp
        any_hit = hit.any(0)
        for k, et in enumerate(EVENT_TYPES):
            for i in np.where(hit[k])[0]:
                events.append((E.patient_id.iat[i], d, et))
        # care response: admission, then treatment intensification
        hosp_left = np.where(any_hit, rng.integers(2, 6, n), np.maximum(0, hosp_left - 1))
        # hypoglycaemia -> de-intensify SU/insulin; hyperglycaemia -> add insulin
        su_active = su_active & ~hit[0]
        med_g = np.where(hit[0], med_g * 0.7, med_g)
        med_g = np.where(hit[1], med_g + 40, med_g)
        med_b = np.where(hit[2] | hit[3], med_b + 9, med_b)
        h0_cv = np.where(hit[3], h0_cv * 0.85, h0_cv)   # statin/antiplatelet post-event
        G = np.where(any_hit, 0.5 * G + 0.5 * np.clip(G_target, 90, 220), G)
        S = np.where(any_hit, np.minimum(S, 145), S)

        # ---- observation layer (noise + missingness) -----------------------------
        hosp_now = hosp_left > 0
        worn = (rng.random(n) < wear_p * (0.9 if c.monsoon else 1.0)) & ~hosp_now
        cgm = E.has_cgm.to_numpy().astype(bool)
        g_obs_p = np.where(cgm, 0.93, np.where(t2d, 0.55, 0.08))
        g_seen = (rng.random(n) < g_obs_p) & ~hosp_now
        bp_seen = (rng.random(n) < np.where(htn, 0.55, 0.22)) & ~hosp_now
        adh_seen = (rng.random(n) < 0.86) & ~hosp_now
        diet_seen = (rng.random(n) < 0.38) & ~hosp_now

        nan = np.nan
        rows.append(pd.DataFrame({
            "patient_id": E.patient_id.to_numpy(), "day": d,
            "steps": np.where(worn, np.round(steps), nan),
            "rhr": np.where(worn, np.round(rhr, 1), nan),
            "hrv": np.where(worn, np.round(np.clip(hrv, 5, 120), 1), nan),
            "sleep_h": np.where(worn, np.round(np.clip(sleep, 2, 11), 2), nan),
            "glucose": np.where(g_seen, np.round(G + rng.normal(0, np.where(cgm, 7, 14), n)), nan),
            "glucose_min": np.where(g_seen & cgm, np.round(Gmin + rng.normal(0, 5, n)), nan),
            "sbp": np.where(bp_seen, np.round(S + rng.normal(0, 7, n)), nan),
            "dbp": np.where(bp_seen, np.round(Dbp + rng.normal(0, 5, n)), nan),
            "med_taken": np.where(adh_seen, took.astype(float), nan),
            "salt_score": np.where(diet_seen, np.round(salt + rng.normal(0, 2, n), 1), nan),
            "carb_score": np.where(diet_seen, np.round(carb + rng.normal(0, 0.6, n), 2), nan),
            "weight_kg": np.where(worn & (rng.random(n) < 0.3), np.round(weight, 1), nan),
            "in_hospital": hosp_now.astype(int),
        }))
        latent_rows.append(pd.DataFrame({
            "patient_id": E.patient_id.to_numpy(), "day": d, "G": G.round(1),
            "Gmin": Gmin.round(1), "S": S.round(1), "ill": ill.round(2),
            "adh7": adh7.round(3), "steps7": steps7.round(0)}))

    daily = pd.concat(rows, ignore_index=True).sort_values(["patient_id", "day"])
    latent = pd.concat(latent_rows, ignore_index=True).sort_values(["patient_id", "day"])
    ev = pd.DataFrame(events, columns=["patient_id", "day", "event_type"])
    truth = pd.DataFrame({"patient_id": E.patient_id, "p_adh": p_adh, "r_g": r_g, "r_b": r_b,
                          "k_steps_g": k_steps_g, "salt_sens": salt_sens,
                          "steps_mean": steps_mean})
    return ehr, daily.reset_index(drop=True), ev, latent.reset_index(drop=True), truth, cal


def main():
    config.DATA_DIR.mkdir(exist_ok=True)
    ehr, daily, ev, latent, truth, cal = simulate()
    ehr.to_csv(config.DATA_DIR / "ehr.csv", index=False)
    daily.to_csv(config.DATA_DIR / "daily.csv", index=False)
    ev.to_csv(config.DATA_DIR / "events.csv", index=False)
    latent.to_csv(config.DATA_DIR / "latent.csv", index=False)
    truth.to_csv(config.DATA_DIR / "truth_params.csv", index=False)
    cal.to_csv(config.DATA_DIR / "calendar.csv", index=False)
    print(f"patients={len(ehr)} patient-days={len(daily)} events={len(ev)}")
    print(ev.event_type.value_counts().to_string())


if __name__ == "__main__":
    main()
