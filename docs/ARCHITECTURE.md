# Architecture

## 1. Components and data flow

| Stage | Module | What it does | Production counterpart |
|---|---|---|---|
| Sources | `synth.py` | Generates EHR rows, a daily device stream and adverse events | ABDM Health Information Exchange (EHR, FHIR R4), device-vendor APIs (wearable, CGM, BP cuff, smart pill-box) |
| Ingestion | `twin.to_arrays`, `fhir.py` | Patient-day grid (patients x days), NaN = not measured; FHIR in/out | FHIR server + streaming ingestion (e.g. Kafka) into a time-series store |
| Twin | `twin.py` | EHR prior → daily Kalman update of personal physiology | Stateful per-patient service; state persisted after each update |
| Features | `features.py` | Causal 3/7/14/28-day windows, calendar context, event history | Feature store with point-in-time correctness |
| Risk engine | `model.py` | Gradient boosting, 14-day risk + 4 event-type heads | Model-serving endpoint, versioned, monitored for drift |
| What-if engine | `simulate.py` | Candidate plans → guardrails → twin roll-outs → risk engine → ranking | Batch job each evening per high-risk patient, or on demand |
| Explanations and alerts | `export_dashboard.py` | SHAP drivers, alert rules | Notification service with alert-fatigue controls |
| Console | `dashboard/` | Clinician UI | Care-team web app inside the hospital's EHR (SMART-on-FHIR launch) |
| Write-back | `fhir.RiskAssessment` | Risk returned to the record | FHIR `RiskAssessment` posted to the EHR |

## 2. The twin, precisely

For each patient *i* and day *t*:

```
glucose_t = θ0 + θ1·adh̃_t + θ2·act̃_t + θ3·carb̃_t + θ4·ill̃_t + ε_t
sbp_t     = φ0 + φ1·adh̃_t + φ2·act̃_t + φ3·salt̃_t + φ4·ill̃_t + ε_t
θ_t = θ_{t-1} + η_t   (slow random walk: tracks disease progression and drift)
```

- `adh` = exponential moving average of pill-box doses taken. `act` = steps EMA minus the patient's own 28-day baseline (thousands). `carb` and `salt` come from the diet log. `ill` = 3-day resting HR above the 28-day baseline (/10 bpm), a proxy for infection.
- The tilde means each driver passes through a first-order lag matched to physiology (glucose about 0.3/day, BP about 0.35/day). Without it the twin under-estimates drug effects; adding it raised the correlation with truth from 0.81 to 0.85 in development.
- **Prior (the EHR half).** θ0 = eAG(HbA1c) + 0.85·L, θ1 = −L, where L is the sum of published average effects of the patient's current medicines. φ is built the same way from clinic BP. Prior standard deviations are wide (for example 0.6·L on the drug coefficient), so the stream can move them.
- **Posterior (the real-time half).** A vectorised Kalman filter updates all patients at once, once per day. Missing readings are skipped, not imputed: the uncertainty grows, and the console shows it.
- **Outputs per day:** the expectation before the reading arrives (`yhat`, `ysd`), a standardised surprise `z` once it arrives, and the coefficients with their posterior sd.

Why a linear-Gaussian twin and not a neural ODE? It is identifiable from sparse, noisy home data. Each coefficient has a clinical meaning ("medicines lower this patient's glucose by 41 ± 9 mg/dL when taken"). It updates in microseconds. And it gives honest uncertainty. The same interface would take a richer mechanistic model (for example a Bergman minimal model on dense CGM) where the data support it.

## 3. Why the twin is not inside the risk model

We tested it: adding the twin's coefficients and surprise scores to the fused risk model *lowered* AUROC by 0.017 (patient-level bootstrap p < 0.001; mixed across seeds). The raw stream windows already carry that information, and the extra 16 correlated features made the boosted trees overfit the rare event classes. So the system keeps two roles apart:

- **Risk engine (discriminative):** EHR + stream windows → probability.
- **Twin (generative):** personal parameters → simulated futures, which the risk engine then scores.

## 4. What-if engine

1. **Candidate plans.** One medicine change at a time (clinical convention), crossed with +2,000 steps/day, −3 g salt/day and adherence support. That is 16 to 56 plans per patient after guardrails.
2. **Guardrails** (`allowed_pharm`): eGFR cut-offs (SGLT2i ≥ 20 per KDIGO 2022, metformin ≥ 30), no duplicate drug classes, ACE-i/ARB up-titration instead of adding a second RAAS blocker, BP drugs only if recent home SBP ≥ 130. Cautions are attached: creatinine and potassium check after starting RAAS, volume status for SGLT2i in older patients, lower sulfonylurea/insulin doses when adding another glucose-lowering drug, IDF-DAR fasting advice.
3. **Roll-out.** 90 days x 16 Monte Carlo draws per plan. Behaviour change ramps in over 3 weeks; drug effects ramp with physiological lag; illness episodes occur at the background rate; the patient's own residual noise and observation gaps apply. The future calendar includes Navratri and Diwali. **Common random numbers**: every plan sees identical noise, so differences between plans are not Monte Carlo luck.
4. **Scoring.** Simulated streams → the same feature builder → the same risk model and event heads.
5. **Ranking.** Mean simulated 14-day risk plus a small burden penalty per change, top three shown with projected HbA1c (ADAG), home SBP, hypoglycaemia days and cautions.

## 5. Point-in-time correctness (no leakage)

- Every window ends at day *d* inclusive; labels look at (*d*, *d*+14]. `tests/test_leakage.py` changes all data after day *d* and checks that the features at *d* do not move.
- Rows where the patient is in hospital at *d* are excluded.
- Patients never span train and test folds; the temporal hold-out also requires that a training label window closes before the test period opens.
- The console's model is re-fitted without the eight demo patients.

## 6. Scaling

The twin update is O(patients x k²) per day with k = 5, so about 1 ms per 1,000 patients on one CPU core. The full pipeline (1,200 patients x 180 days: twin, 15 cross-validated models, bootstrap) runs in about 6 minutes on a laptop. A district-hospital deployment of 50,000 patients would need one modest VM for the twin and nightly batch scoring.
