"""FHIR R4 interoperability layer (ABDM-ready).

India's Ayushman Bharat Digital Mission exchanges health records as HL7 FHIR R4
documents. This module maps the twin's inputs and outputs onto standard FHIR
resources so the PoC can plug into an ABDM Health Information User/Provider flow:

  EHR input  -> Patient, Condition (SNOMED CT), Observation (LOINC), MedicationStatement
  Device feed-> Observation (LOINC) with device-reported values
  Twin output-> RiskAssessment (14-day adverse-event probability + per-type breakdown)

The reverse direction (FHIR -> twin) is `bundle_to_ehr_row`, used by the tests to
prove the mapping round-trips.  Codes are the standard published ones; ABDM/NRCeS
profile URLs would be added in a production build.
"""
from __future__ import annotations

import json
from datetime import timedelta

import pandas as pd

from . import config
from .synth import START_DATE

LOINC = {
    "hba1c": ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", "%"),
    "ldl": ("13457-7", "Cholesterol in LDL [Mass/volume] in Serum or Plasma by calculation", "mg/dL"),
    "egfr": ("62238-1", "Glomerular filtration rate/1.73 sq M.predicted (CKD-EPI)", "mL/min/{1.73_m2}"),
    "uacr": ("9318-7", "Albumin/Creatinine [Mass Ratio] in Urine", "mg/g"),
    "bmi": ("39156-5", "Body mass index (BMI) [Ratio]", "kg/m2"),
    "waist_cm": ("8280-0", "Waist Circumference at umbilicus by Tape measure", "cm"),
    "sbp_clinic": ("8480-6", "Systolic blood pressure", "mm[Hg]"),
}
DEVICE_LOINC = {
    "glucose": ("2339-0", "Glucose [Mass/volume] in Blood", "mg/dL"),
    "sbp": ("8480-6", "Systolic blood pressure", "mm[Hg]"),
    "dbp": ("8462-4", "Diastolic blood pressure", "mm[Hg]"),
    "rhr": ("40443-4", "Heart rate --resting", "/min"),
    "steps": ("55423-8", "Number of steps in unspecified time Pedometer", "{steps}"),
    "sleep_h": ("93832-4", "Sleep duration", "h"),
    "weight_kg": ("29463-7", "Body weight", "kg"),
}
SNOMED = {"t2d": ("44054006", "Diabetes mellitus type 2"),
          "htn": ("38341003", "Hypertensive disorder"),
          "prior_cvd": ("49601007", "Disorder of cardiovascular system")}
MEDS = {"med_metformin": "Metformin", "med_sulfonylurea": "Sulfonylurea (e.g. gliclazide)",
        "med_dpp4i": "DPP-4 inhibitor (e.g. sitagliptin)", "med_sglt2i": "SGLT2 inhibitor (e.g. dapagliflozin)",
        "med_insulin": "Insulin", "med_raas": "ACE inhibitor / ARB (e.g. telmisartan)",
        "med_ccb": "Calcium channel blocker (amlodipine)", "med_statin": "Statin (e.g. atorvastatin)"}


def _date(day: int) -> str:
    return (START_DATE + timedelta(days=int(day))).strftime("%Y-%m-%d")


def _obs(pid, code, value, day, category, oid):
    c, disp, unit = code
    return {"resource": {
        "resourceType": "Observation", "id": oid, "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                  "code": category}]}],
        "code": {"coding": [{"system": "http://loinc.org", "code": c, "display": disp}]},
        "subject": {"reference": f"Patient/{pid}"}, "effectiveDateTime": _date(day),
        "valueQuantity": {"value": float(value), "unit": unit, "system": "http://unitsofmeasure.org", "code": unit}}}


def patient_bundle(e: pd.Series, daily: pd.DataFrame | None = None,
                   risk: dict | None = None, last_n_days: int = 7) -> dict:
    pid = e.patient_id
    entries = [{"resource": {
        "resourceType": "Patient", "id": pid,
        "gender": "male" if e.male else "female",
        "extension": [{"url": "urn:hridaytwin:age-years", "valueInteger": int(e.age)},
                      {"url": "urn:hridaytwin:region", "valueString": str(e.region)}]}}]
    for k, (code, disp) in SNOMED.items():
        if e[k]:
            entries.append({"resource": {
                "resourceType": "Condition", "id": f"{pid}-{k}",
                "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
                "code": {"coding": [{"system": "http://snomed.info/sct", "code": code, "display": disp}]},
                "subject": {"reference": f"Patient/{pid}"}}})
    for k, code in LOINC.items():
        entries.append(_obs(pid, code, e[k], 0, "laboratory" if k in ("hba1c", "ldl", "egfr", "uacr") else "vital-signs", f"{pid}-{k}"))
    for k, name in MEDS.items():
        if e[k]:
            entries.append({"resource": {
                "resourceType": "MedicationStatement", "id": f"{pid}-{k}", "status": "active",
                "medicationCodeableConcept": {"text": name}, "subject": {"reference": f"Patient/{pid}"}}})
    if daily is not None:
        d = daily[daily.patient_id == pid].tail(last_n_days)
        for _, r in d.iterrows():
            for k, code in DEVICE_LOINC.items():
                if pd.notna(r[k]):
                    entries.append(_obs(pid, code, r[k], r.day, "vital-signs", f"{pid}-{k}-{int(r.day)}"))
    if risk is not None:
        entries.append({"resource": {
            "resourceType": "RiskAssessment", "id": f"{pid}-risk", "status": "preliminary",
            "subject": {"reference": f"Patient/{pid}"}, "occurrenceDateTime": _date(risk["day"]),
            "method": {"text": "HridayTwin fused EHR + wearable model (research PoC, synthetic training data)"},
            "prediction": [{"outcome": {"text": t.replace("_", " ")}, "probabilityDecimal": round(float(p), 4),
                            "whenPeriod": {"start": _date(risk["day"]), "end": _date(risk["day"] + config.HORIZON_DAYS)}}
                           for t, p in risk["probabilities"].items()],
            "note": [{"text": "Decision support only. Requires clinician review."}]}})
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


def bundle_to_ehr_row(bundle: dict) -> dict:
    """Minimal FHIR -> twin input mapping (inverse of patient_bundle for static fields)."""
    row = {k: 0 for k in list(SNOMED) + list(MEDS)}
    inv_loinc = {v[0]: k for k, v in LOINC.items()}
    inv_snomed = {v[0]: k for k, v in SNOMED.items()}
    inv_med = {v: k for k, v in MEDS.items()}
    for ent in bundle["entry"]:
        r = ent["resource"]
        t = r["resourceType"]
        if t == "Patient":
            row["patient_id"] = r["id"]
            row["male"] = int(r["gender"] == "male")
            for x in r.get("extension", []):
                if x["url"].endswith("age-years"):
                    row["age"] = x["valueInteger"]
        elif t == "Condition":
            row[inv_snomed[r["code"]["coding"][0]["code"]]] = 1
        elif t == "MedicationStatement":
            row[inv_med[r["medicationCodeableConcept"]["text"]]] = 1
        elif t == "Observation":
            key = r["id"].split("-", 1)[1] if "-" in r["id"] else ""
            if key in LOINC:
                row[key] = r["valueQuantity"]["value"]
    return row


def main():
    """Export the first console patient, with the risk the console shows on the last day."""
    ehr = pd.read_csv(config.DATA_DIR / "ehr.csv")
    daily = pd.read_csv(config.DATA_DIR / "daily.csv")
    dash = json.loads((config.DASH_DIR / "data.json").read_text())
    p = dash["patients"][0]
    e = ehr[ehr.patient_id == p["id"]].iloc[0]
    probs = {"any_adverse_cardiometabolic_event": p["risk"]["p"][-1]}
    probs.update({t: v[-1] for t, v in p["risk"]["types"].items()})
    b = patient_bundle(e, daily, risk={"day": config.DAYS - 1, "probabilities": probs})
    out = config.DOCS_DIR / "sample_fhir_bundle.json"
    out.write_text(json.dumps(b, indent=1))
    print(f"wrote {out} with {len(b['entry'])} resources")


if __name__ == "__main__":
    main()
