"""Correctness, leakage, determinism and safety tests."""
import numpy as np
import pandas as pd
import pytest

from hridaytwin import features, fhir, simulate, synth, twin


# ---------------------------------------------------------------- generator
def test_generator_is_deterministic():
    a = synth.simulate(n=40, days=30, seed=7)
    b = synth.simulate(n=40, days=30, seed=7)
    pd.testing.assert_frame_equal(a[1], b[1])
    pd.testing.assert_frame_equal(a[2], b[2])


def test_generator_shapes_and_ranges(small_cohort):
    ehr, daily, events, latent, truth, cal = small_cohort
    assert len(daily) == len(ehr) * 90
    assert (ehr.t2d | ehr.htn).all(), "every patient has at least one condition"
    assert daily.glucose.dropna().between(20, 600).all()
    assert daily.sbp.dropna().between(60, 260).all()
    assert set(events.event_type) <= set(synth.EVENT_TYPES)


def test_observations_are_not_the_latent_state(small_cohort):
    """Labels come from the hidden state; inputs are noisy, gappy views of it."""
    ehr, daily, events, latent, truth, cal = small_cohort
    m = daily.glucose.notna()
    assert not np.allclose(daily.glucose[m], latent.G[m])
    assert daily.glucose.isna().mean() > 0.2


# ---------------------------------------------------------------- leakage
def _features(ehr, daily, events, cal, D):
    A = twin.to_arrays(daily, ehr.patient_id, D)
    T = twin.run_twin(A, ehr)
    evc = features.event_count_array(events, ehr.patient_id, D)
    return features.build_daily_features(A, T, ehr, cal, evc)


def test_features_do_not_look_into_the_future(small_cohort):
    """Scramble every reading after day d: features at day d must not change."""
    ehr, daily, events, latent, truth, cal = small_cohort
    D, d = 90, 50
    F1 = _features(ehr, daily, events, cal, D)
    rng = np.random.default_rng(0)
    scrambled = daily.copy()
    late = scrambled.day > d
    for c in ["steps", "rhr", "glucose", "glucose_min", "sbp", "dbp", "med_taken", "salt_score", "carb_score"]:
        scrambled.loc[late, c] = rng.permutation(scrambled.loc[late, c].to_numpy())
    ev2 = events[events.day <= d]
    F2 = _features(ehr, scrambled, ev2, cal, D)
    for k in F1:
        if k in ("fast_days_next14", "feast_days_next14"):   # calendar is known in advance by design
            continue
        np.testing.assert_allclose(F1[k][:, d], F2[k][:, d], equal_nan=True, err_msg=k)


def test_labels_window(small_cohort):
    ehr, daily, events, latent, truth, cal = small_cohort
    pid, day = events.patient_id.iloc[0], int(events.day.iloc[0])
    rows = pd.DataFrame({"patient_id": [pid] * 3, "day": [day - 14, day - 1, day]})
    y = features.labels(events, rows, 14)
    assert y.y.tolist()[:2] == [1, 1]
    later = events[(events.patient_id == pid) & (events.day > day) & (events.day <= day + 14)]
    assert y.y.iloc[2] == int(len(later) > 0)


# ---------------------------------------------------------------- twin
def test_twin_is_calibrated(small_cohort):
    ehr, daily, events, latent, truth, cal = small_cohort
    A = twin.to_arrays(daily, ehr.patient_id, 90)
    T = twin.run_twin(A, ehr)
    z = T["glucose"]["z"]
    z = z[np.isfinite(z)]
    assert abs(z.mean()) < 0.25
    assert 0.75 < z.std() < 1.3
    assert 0.88 < np.mean(np.abs(z) < 2) < 0.99


def test_twin_uncertainty_shrinks_with_data(small_cohort):
    ehr, daily, events, latent, truth, cal = small_cohort
    A = twin.to_arrays(daily, ehr.patient_id, 90)
    T = twin.run_twin(A, ehr)
    sd = T["glucose"]["theta_sd"][:, :, 1]
    assert np.mean(sd[:, -1]) < np.mean(sd[:, 0])


def test_twin_learns_drug_response():
    ehr, daily, events, latent, truth, cal = synth.simulate(n=500, days=150, seed=5)
    A = twin.to_arrays(daily, ehr.patient_id, 150)
    T = twin.run_twin(A, ehr)
    L = synth.drug_load(ehr, synth.GLUCOSE_DRUG_EFFECT)
    m = (L >= 45) & ~ehr.patient_id.isin(events.patient_id).to_numpy()
    est = -T["glucose"]["theta"][:, -1, 1] / np.where(L > 0, L, 1)
    assert np.corrcoef(est[m], truth.r_g.to_numpy()[m])[0, 1] > 0.3


# ---------------------------------------------------------------- guardrails
def _patient(**kw):
    base = dict(age=60, egfr=80, t2d=1, htn=1, prior_cvd=0, observes_fasting=0,
                med_metformin=1, med_sulfonylurea=0, med_dpp4i=0, med_sglt2i=0, med_insulin=0,
                med_raas=0, med_ccb=0, med_statin=0)
    base.update(kw)
    return pd.Series(base)


def test_no_sglt2i_below_egfr_20():
    assert "add_sglt2i" not in simulate.allowed_pharm(_patient(egfr=15), 150, 200)
    assert "add_sglt2i" in simulate.allowed_pharm(_patient(egfr=40), 150, 200)


def test_no_metformin_below_egfr_30():
    assert "add_metformin" not in simulate.allowed_pharm(_patient(egfr=25, med_metformin=0), 150, 200)


def test_no_duplicate_raas_blockade():
    opts = simulate.allowed_pharm(_patient(med_raas=1), 150, 200)
    assert "start_raas" not in opts and "uptitrate_raas" in opts


def test_no_bp_drugs_when_bp_controlled():
    opts = simulate.allowed_pharm(_patient(), 122, 200)
    assert not {"add_ccb", "start_raas", "uptitrate_raas"} & set(opts)


def test_sulfonylurea_caution_on_glucose_add_on():
    opts = simulate.allowed_pharm(_patient(med_sulfonylurea=1), 150, 200)
    assert any("hypoglycaemia" in c for c in opts["add_sglt2i"])
    assert "switch_su_to_dpp4i" in opts


def test_fasting_plan_only_for_fasting_su_users():
    assert "fasting_dose_plan" in simulate.allowed_pharm(_patient(med_sulfonylurea=1, observes_fasting=1), 150, 200)
    assert "fasting_dose_plan" not in simulate.allowed_pharm(_patient(observes_fasting=1), 150, 200)


# ---------------------------------------------------------------- FHIR
def test_fhir_round_trip(small_cohort):
    ehr, daily, *_ = small_cohort
    for i in range(10):
        e = ehr.iloc[i]
        row = fhir.bundle_to_ehr_row(fhir.patient_bundle(e, daily))
        for k in ["age", "male", "t2d", "htn", "hba1c", "egfr", "ldl", "med_sulfonylurea", "med_raas"]:
            assert float(row[k]) == float(e[k]), k


def test_fhir_codes_are_standard(small_cohort):
    ehr, daily, *_ = small_cohort
    b = fhir.patient_bundle(ehr.iloc[0], daily)
    systems = {c["system"] for ent in b["entry"] for c in ent["resource"].get("code", {}).get("coding", [])}
    assert systems <= {"http://loinc.org", "http://snomed.info/sct"}
