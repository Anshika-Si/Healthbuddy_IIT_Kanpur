"""
Run the full GlucoTwin pipeline end to end (about 3-5 minutes on a laptop CPU).

    python run_pipeline.py            # everything
    python run_pipeline.py --skip-cv  # (reserved) faster iteration
"""
import time

from src.data import generate_ehr, generate_wearables
from src.features import fusion
from src.models import train
from src.twin import validate_twin
from src.dashboard import export_data

STEPS = [
    ("1/6  Generate synthetic EHR cohort", generate_ehr.main),
    ("2/6  Generate synthetic wearable + CGM streams", generate_wearables.main),
    ("3/6  Calibrate digital twins + build fused features", fusion.main),
    ("4/6  Train, ablate, explain, evaluate", train.main),
    ("5/6  Validate twin + what-if simulation", validate_twin.main),
    ("6/6  Build clinician dashboard", export_data.main),
]

if __name__ == "__main__":
    t0 = time.time()
    for name, fn in STEPS:
        print(f"\n=== {name} ===")
        fn()
    print(f"\nDone in {time.time() - t0:.0f} s. Open dashboard/GlucoTwin_Dashboard.html in a browser.")
