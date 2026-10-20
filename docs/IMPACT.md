# Impact and adoption

## The gap it fills
An Indian adult with diabetes and hypertension typically sees a doctor every one to three months for a few minutes. Between visits, the glucometer readings, BP cuff readings, missed doses and fasting days are invisible, and that is when emergencies build up. Remote-monitoring programmes collect this data but leave a care-coordinator with a spreadsheet of readings and no way to know *who* to call first or *what* to change.

HridayTwin turns that stream into three things a coordinator can act on in a morning: a ranked call list with reasons, a personal picture of how each patient responds, and a tested shortlist of plan changes for the doctor.

## Who uses it
| User | Value |
|---|---|
| Care-coordinator or diabetes educator | Ranked daily worklist; reasons in plain language; fasting-season and festival warnings |
| Physician | Personal drug-response estimates; plan options with projected HbA1c, BP and hypoglycaemia risk; less trial and error |
| Patient | Fewer emergencies; plans that respect fasting and festivals; feedback that their adherence and salt changes matter |
| Hospital, insurer or employer health programme | Fewer avoidable admissions; evidence of care quality |

## Why it fits India specifically
- **Cultural calendar built in:** Ekadashi, Shravan and Navratri fasts and festival feasting are model inputs, and the planner offers IDF-DAR-style fasting-day dose plans.
- **Sulfonylurea-heavy prescribing:** the hypoglycaemia head (AUROC 0.94 on synthetic data) targets the most common preventable harm in Indian diabetes care.
- **Monsoon:** seasonal infection waves and lower activity are modelled explicitly.
- **ABDM-native:** FHIR R4 in and out, so it can plug into the national digital health stack instead of another silo.
- **Low-cost devices:** works on daily summaries from a glucometer, a BP cuff and any step counter; CGM is optional.

## How we would measure success in a pilot
- Avoidable admissions per 100 patient-years (primary)
- Severe hypoglycaemia episodes, especially in fasting periods
- Share of patients reaching HbA1c < 7% and home SBP < 130 at 6 months
- Coordinator minutes per patient per month
- Alert acceptance rate (signal of trust and of alert fatigue)

## Adoption path
1. **Pilot with a remote-monitoring diabetes programme** (hospital chain, digital health company or corporate wellness programme) on its existing data: retrospective validation first.
2. **Embed in the existing care-coordinator workflow** as a SMART-on-FHIR app or a worklist plug-in; no new app for patients.
3. **Expand** to related chronic conditions where the same twin pattern applies (CKD, heart failure, gestational diabetes) by swapping the state equations and event heads.
