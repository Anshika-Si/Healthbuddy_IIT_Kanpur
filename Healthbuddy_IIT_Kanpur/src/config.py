"""Central configuration for the GlucoTwin pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
DOCS = ROOT / "docs"

SEED = 42
N_PATIENTS = 120          # synthetic cohort size
N_DAYS = 14               # days of wearable data per patient
STEP_MIN = 5              # sensor resolution (minutes)
CALIBRATION_DAYS = 7      # days 1-7: digital-twin calibration ("onboarding week")
                          # days 8-14: prediction + evaluation

# Prediction task
HORIZON_MIN = 120         # predict 2 hours ahead
LOOKBACK_MIN = 120        # history window used for features
SAMPLE_EVERY_MIN = 15     # create a prediction point every 15 minutes
SPIKE_THRESHOLD = 180     # mg/dL (ADA post-prandial target is <180)
TEST_FRACTION = 0.25      # held-out *patients* (no patient appears in both splits)
ALERT_THRESHOLD = 0.5     # default probability for raising an alert on the dashboard

for p in (DATA_RAW, DATA_PROCESSED, OUTPUTS, FIGURES, DOCS):
    p.mkdir(parents=True, exist_ok=True)
