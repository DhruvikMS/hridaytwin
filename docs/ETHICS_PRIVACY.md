# Ethics, privacy and safety

## Privacy by design (DPDP Act 2023 and ABDM)
| Principle | How HridayTwin handles it |
|---|---|
| Consent and purpose limitation | Data enters only through ABDM consent artefacts that name the purpose (care management) and duration; the twin stores derived state, not raw history beyond what the windows need (28 days) |
| Data minimisation | Daily summaries only (mean and minimum glucose, steps, resting HR, home BP, dose taken yes/no). No GPS, no raw accelerometer, no free-text diet photos |
| Right to withdraw and erase | Twin state is per patient and can be deleted along with the source data; nothing is pooled into the model without consent for secondary use |
| Security | De-identified IDs in the analytics layer; re-identification only inside the care-team application |
| Training on patient data | Retraining needs a separate, explicit consent purpose or an IEC-approved de-identified dataset |
| This PoC | Uses no real personal data at all |

## Clinical safety
- **Decision support, not decision making.** Every plan is labelled "for clinician review"; no message goes to a patient automatically.
- **Guardrails before ranking.** Contraindicated plans (by eGFR, regimen or BP level) are never shown. Cautions travel with each plan.
- **Calibrated, budgeted alerts.** The threshold is set by the number of alerts a care team can handle, and every alert carries its reason.
- **Uncertainty is visible.** The twin's expectation band widens when data are missing; low-data patients do not look falsely reassuring.
- **Known failure modes are documented** (`MODEL_CARD.md`) and surfaced in the console's evidence tab.

## Equity
Device access is uneven across income, gender and rural/urban India. Mitigations: support glucometer-only and feature-phone (IVR/SMS dose confirmation) patients; treat missing data as a signal; report performance by subgroup; never deprioritise a patient because they send fewer readings.
