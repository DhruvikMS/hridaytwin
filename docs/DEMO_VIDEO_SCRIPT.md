# HridayTwin Demo Video Script (15–20 minutes)

**Team:** Winmasters — Manjit Kalita, Dhruvik Sorathiya, Bishwarup Roy  
**College:** Goa Institute of Management  
**Target duration:** 16–18 minutes

> Recording note: show the actual dashboard and narrate what is visible. Do not claim clinical validation or real-patient deployment. The prototype uses synthetic patients only.

## Before recording
- Open the public dashboard: https://dhruvikms.github.io/hridaytwin/dashboard/
- Test screen recording and microphone.
- Close unrelated tabs and notifications.
- Use only synthetic data and avoid showing private account information.

## 0:00–1:15 — Introduction and team
Introduce Team Winmasters from Goa Institute of Management. Present HridayTwin as a proof of concept for a personal cardiometabolic digital twin for Indian adults with diabetes and/or hypertension. Explain that the goal is to explore longitudinal health signals for care-team review between clinic visits.

## 1:15–3:00 — Problem statement and healthcare use case
Explain the gap between intermittent clinic visits and daily changes in medication adherence, glucose, blood pressure, activity, meals and illness. State clearly that this is a research demonstration using synthetic data, not a clinical product.

## 3:00–5:00 — Architecture and inputs
Show the architecture diagram. Explain EHR context, wearable/pill-box/CGM or glucometer/home BP signals, and calendar context. Walk through synthetic data generation, feature fusion, personal twin, risk engine, what-if engine and clinician console.

## 5:00–7:30 — Personal twin and risk model
Explain that Bayesian linear state-space models estimate glucose and systolic BP and update using a Kalman filter. The gradient-boosting risk engine estimates 14-day risk and event-type risks. Explain that model explanations are intended to make risk drivers inspectable. Do not describe the risk estimates as clinically validated.

## 7:30–12:30 — Live dashboard demonstration
1. Open the dashboard and point out the synthetic-data warning.
2. Select a patient archetype and explain the displayed context.
3. Start the 180-day replay and show how risk, alerts and the twin expectation band change.
4. Compare observed glucose/BP with the twin's expectation.
5. Inspect the alert feed and risk drivers.
6. Use the what-if planner to compare a candidate plan and read the safety guardrails aloud.

## 12:30–14:30 — Results and evaluation
Report the documented synthetic-cohort results: 1,200 patients × 180 days; fused AUROC 0.726 versus 0.708 EHR-only; threshold-rule baseline AUROC 0.615; severe-hypoglycaemia AUROC 0.943 versus 0.907 EHR-only; label-permutation AUROC 0.515. Explicitly state that these results use synthetic data and do not establish clinical effectiveness.

## 14:30–16:15 — Limitations and safety
Mention that twin-state features reduced AUROC, individual step sensitivity was not learned, acute cardiovascular events were not improved by the stream, and timing the dangerous week remains modest. Confirm no real patient data are used and recommendations are for clinician review only.

## 16:15–17:15 — Roadmap and closing
Describe planned stages: retrospective validation under ethics approval, a prospective silent-mode pilot, and a clinician-in-the-loop pilot with safety and alert-fatigue monitoring. Thank the judges and display the repository and dashboard links.

## Final recording checklist
- [ ] Video duration is 15–20 minutes.
- [ ] Audio is clear and dashboard text is readable.
- [ ] No personal/private information is visible.
- [ ] Upload to YouTube as **Unlisted**.
- [ ] Test the link in an incognito/private window.
- [ ] Add the video URL to README.md.
