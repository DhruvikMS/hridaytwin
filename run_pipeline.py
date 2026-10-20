"""One command reproduces everything: data -> twin -> models -> validation -> console -> report.

    python run_pipeline.py            # full run (~10 min on a laptop CPU)
    python run_pipeline.py --quick    # CI-sized run (small cohort, no seed sweep)
"""
import os
import sys
import time

QUICK = "--quick" in sys.argv
if QUICK:
    os.environ.setdefault("HT_N_PATIENTS", "300")
    os.environ.setdefault("HT_DAYS", "120")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from hridaytwin import export_dashboard, fhir, model, report, synth, validate  # noqa: E402

steps = [
    ("1/6 Generate synthetic cohort (EHR + daily device stream)", synth.main),
    ("2/6 Run twin, train and evaluate risk models", model.main),
    ("3/6 Validate twin against hidden truth" + ("" if QUICK else " + seed sweep"), lambda: validate.main(skip_seeds=QUICK)),
    ("4/6 Build clinician console (dashboard/index.html)", export_dashboard.main),
    ("5/6 Export sample FHIR R4 bundle", fhir.main),
    ("6/6 Write docs/RESULTS.md and figures", report.main),
]
t0 = time.time()
for name, fn in steps:
    print(f"\n=== {name}", flush=True)
    fn()
print(f"\nDone in {time.time() - t0:.0f} s. Open dashboard/index.html in a browser.")
