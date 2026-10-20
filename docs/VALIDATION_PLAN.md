# From synthetic PoC to real-world evidence

The PoC shows the machinery works and is honest about itself. Clinical claims need real patients. This is the plan.

## Phase 1: retrospective validation (3 to 4 months)
- **Data:** a partner hospital or diabetes chain with remote-monitoring patients who have both an EHR (HbA1c, eGFR, medicines, admissions) and synced device data (glucometer or CGM, BP cuff, wearable). Indian diabetes clinic networks already run such programmes; data are accessed through ABDM consent artefacts or under an IEC-approved retrospective protocol.
- **Mapping:** the FHIR layer (`src/hridaytwin/fhir.py`) and data dictionary define the target schema. Only feature *inputs* change; the code does not.
- **Questions, in order:**
  1. Is the twin calibrated on real data? (z-scores, coverage of the 2-sd band)
  2. Does the twin's estimated drug response predict the HbA1c change after a later dose change? (the real-world stand-in for the hidden truth we used here)
  3. Does fused beat EHR-only on admissions and ED visits, with patient-level bootstrap CIs, by event type?
  4. Subgroup performance and missing-data bias (see `MODEL_CARD.md`).
- **Pre-registration:** analysis plan, metrics and the alert budget are fixed before unblinding, to avoid the model-selection optimism we disclose in RESULTS.

## Phase 2: prospective silent mode (6 months)
The twin runs every night on live patients and clinicians do not see it. We measure discrimination, calibration drift and how far ahead alerts come. It also gives the alert-fatigue baseline: how many alerts per clinician per day at each threshold.

## Phase 3: clinician-in-the-loop pilot
- Cluster-randomised by clinic or care-coordinator.
- **Primary outcome:** adverse-event admissions per 100 patient-years.
- **Secondary:** severe hypoglycaemia, HbA1c and BP at 6 months, clinician time per patient, alert acceptance rate, patient-reported burden.
- **Safety:** every planner recommendation is logged with the clinician's decision; independent review of any harm.

## Regulatory path
Software that informs treatment decisions is likely to be a medical device under India's Medical Devices Rules 2017 (CDSCO), with risk class to be confirmed. Phase 2 and 3 data form the clinical evidence. Model updates follow a pre-defined change-control plan.
