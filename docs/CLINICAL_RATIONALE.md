# Clinical rationale and sources

Every number the generator, twin prior or simulator assumes is listed here with its source. Where we simplified, we say so.

## Population (generator priors)

| Assumption | Value used | Source |
|---|---|---|
| Diabetes and hypertension burden in India | 101 M adults with diabetes, 315 M with hypertension; the cohort is a cardiometabolic clinic population (each patient has at least one) | Anjana RM et al., ICMR-INDIAB-17, *Lancet Diabetes Endocrinol* 2023;11:474-489 |
| Salt intake | Mean about 10 g/day (sd 2.2) | Johnson C et al., *J Am Heart Assoc* 2017;6:e004547 (about 9 to 11 g/day across urban and rural India) |
| Asian BMI distribution | Mean 26.6 (cut-off for obesity in Indians: ≥ 25) | Misra A et al., *JAPI* 2009 consensus on Asian Indian obesity cut-offs |
| Sulfonylurea use | 46% of people with diabetes in the cohort | Sulfonylureas remain among the most prescribed second-line agents in India (Indian prescribing audits); value is illustrative |
| Fasting | 38% observe regular religious fasts (Ekadashi, Shravan Mondays, Navratri) | Illustrative; fasting with diabetes is common enough in India that guidelines exist (IDF-DAR) |
| Monsoon | Steps fall 17%, infection incidence x2.2 from June to September | Illustrative; seasonal vector-borne and viral fever peaks are well documented in India |

## Physiology and treatment effects

| Effect | Value | Source |
|---|---|---|
| HbA1c ↔ mean glucose | eAG (mg/dL) = 28.7 x HbA1c − 46.7 | Nathan DM et al. (ADAG), *Diabetes Care* 2008;31:1473-1478 |
| Oral agents, HbA1c reduction | Metformin and sulfonylurea about 1.0%, DPP-4i and SGLT2i about 0.6%, insulin about 1.5% (as mean glucose: 30/30/17/17/45 mg/dL) | Sherifali D et al., *Diabetes Care* 2010;33:1859-1864 |
| BP drugs, standard dose | About 9 mmHg SBP per class (RAAS, CCB) | Law MR, Morris JK, Wald NJ, *BMJ* 2009;338:b1665 |
| SGLT2i on BP | About 4 mmHg SBP | Zinman B et al. (EMPA-REG OUTCOME), *NEJM* 2015;373:2117-2128 and BP meta-analyses |
| Salt and BP | About 1.1 mmHg per g/day salt (log-normal individual sensitivity) | He FJ, Li J, MacGregor GA, *BMJ* 2013;346:f1325 |
| Exercise and glycaemia | About 4 mg/dL per +1,000 steps/day (heterogeneous) | Umpierre D et al., *JAMA* 2011;305:1790-1799 (structured exercise about −0.67% HbA1c) |
| BP lowering and CV events | Hazard x exp(0.03 per mmHg) ≈ 26% per 10 mmHg | Ettehad D et al., *Lancet* 2016;387:957-967 (about 20% per 10 mmHg) |
| Acute infection as a CV trigger | Daily CV hazard x exp(1.8) ≈ 6x during illness | Kwong JC et al., *NEJM* 2018;378:345-353 (incidence ratio 6.05 for MI within 7 days of influenza) |
| Hypoglycaemia definitions | Level 2 < 54 mg/dL (CGM alert threshold) | American Diabetes Association, *Standards of Care in Diabetes 2024*, section 6 |
| Hypertensive crisis | SBP ≥ 180 mmHg | ACC/AHA 2017 hypertension guideline |

## Guardrails used by the planner

| Rule | Source |
|---|---|
| Start SGLT2i only if eGFR ≥ 20 | KDIGO 2022 Diabetes in CKD guideline, *Kidney Int* 2022;102(5S):S1-S127 |
| Metformin contraindicated below eGFR 30; maximum 1 g/day at eGFR 30-45 | KDIGO 2022; FDA 2016 labelling |
| DPP-4i renal dose adjustment (except linagliptin) below eGFR 45 | Product labelling |
| Do not combine ACE-i with ARB; check creatinine and potassium 1-2 weeks after starting or up-titrating | ONTARGET, *NEJM* 2008; KDIGO |
| Lower sulfonylurea/insulin doses when adding another glucose-lowering agent or on fasting days | ADA Standards 2024; Hassanein M et al., IDF-DAR practical guidelines, *Diabetes Res Clin Pract* 2022;185:109185 |

## Simplifications we know about
- Composite event hazards are stylised (logistic or exponential in latent glucose and BP), tuned to give about a 4.5% event rate per 14-day window in a high-risk clinic cohort. They are not fitted to Indian registry data.
- After an admission, the generator intensifies or de-intensifies treatment by simple rules (for example a hypoglycaemia event stops the sulfonylurea).
- Real CGM data has richer structure (meal spikes, dawn phenomenon). The twin works on daily summaries (mean, minimum) because that is what most Indian patients' devices and plans would sync.
