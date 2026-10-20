"""Central configuration. Every script reads from here so runs are reproducible.

Environment overrides (used by CI and the seed-robustness sweep):
    HT_N_PATIENTS, HT_DAYS, HT_SEED
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
DASH_DIR = ROOT / "dashboard"
DOCS_DIR = ROOT / "docs"

N_PATIENTS = int(os.environ.get("HT_N_PATIENTS", 1200))
DAYS = int(os.environ.get("HT_DAYS", 180))
SEED = int(os.environ.get("HT_SEED", 2026))

# Prediction task
HORIZON_DAYS = 14          # predict an adverse event in the next 14 days
PRED_EVERY_DAYS = 7        # one scored prediction per patient per week
WARMUP_DAYS = 28           # twin needs 4 weeks of stream before first prediction
N_FOLDS = 5

# Temporal hold-out: train on predictions made before this day, test after
TEMPORAL_SPLIT_DAY = int(DAYS * 0.62)

# Alert budget used to set the operating threshold (alerts per patient-week)
ALERT_BUDGET = 0.10

# Gradient-boosting hyper-parameters (shared by every model variant)
# (chosen among two settings on the same grouped CV - see docs/RESULTS.md caveats)
GBM_PARAMS = dict(
    max_iter=250,
    learning_rate=0.04,
    max_leaf_nodes=12,
    min_samples_leaf=120,
    l2_regularization=3.0,
    early_stopping=False,
)

# Number of demo patients exported to the clinician dashboard
N_DEMO_PATIENTS = 8

# Simulation horizon for what-if scenarios
WHATIF_DAYS = 90
