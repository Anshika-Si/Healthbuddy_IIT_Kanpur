"""
Validate the digital twin and illustrate what-if simulation.

1. Parameter recovery: does the twin, calibrated only on observable data, recover each
   patient's HIDDEN insulin sensitivity used by the simulator? (Only possible with
   synthetic data - one of the reasons the sandbox rules favour synthetic cohorts.)
2. What-if: simulate the same dinner under different behaviours.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DATA_RAW, OUTPUTS, FIGURES
from src.twin.glucotwin import TwinParams, simulate

INK, CORAL, TEAL, GREY, SAND = "#12343B", "#E4572E", "#0F7C7A", "#8FA3A6", "#E9C46A"


def main():
    ehr = pd.read_csv(DATA_RAW / "ehr_patients.csv").set_index("patient_id")
    twins = json.load(open(OUTPUTS / "twin_params.json"))
    t = pd.DataFrame(twins.values()).set_index("patient_id")
    si = ehr.loc[t.index, "_latent_insulin_sensitivity"]
    r_cs = float(np.corrcoef(t.cs, si)[0, 1])
    r_gb = float(np.corrcoef(t.gb, ehr.loc[t.index, "fasting_glucose_mgdl"])[0, 1])

    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.scatter(si, t.cs, s=28, color=TEAL, alpha=0.8, edgecolor="white", lw=0.5)
    ax.set_xlabel("Hidden true insulin sensitivity (simulator only)")
    ax.set_ylabel("Twin-estimated carb sensitivity (mg/dL per g)")
    ax.set_title(f"Twin recovers hidden physiology from 7 days of data\nPearson r = {r_cs:.2f}",
                 color=INK, fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.2); fig.tight_layout(); fig.savefig(FIGURES / "twin_recovery.png", dpi=180); plt.close(fig)

    # What-if for the patient with the most sleep-sensitive twin
    pid = t.beta_sleep.idxmax()
    p = TwinParams(**twins[pid])
    scen = {
        "Usual: 85 g rice dinner, slept 7 h": dict(sleep_h=7, walk_min=0, carbs=85),
        "After a 4.5 h night": dict(sleep_h=4.5, walk_min=0, carbs=85),
        "15-min walk after dinner": dict(sleep_h=7, walk_min=15, carbs=85),
        "Smaller portion (55 g) + 15-min walk": dict(sleep_h=7, walk_min=15, carbs=55),
    }
    colors = [GREY, CORAL, TEAL, SAND]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    out = {}
    for (name, sc), c in zip(scen.items(), colors):
        traj = simulate(p, p.gb, [], 180, sleep_h=sc["sleep_h"], extra_meal=(10, sc["carbs"]),
                        walk_min=sc["walk_min"])
        x = np.arange(len(traj)) * 5
        ax.plot(x, traj, color=c, lw=2.4 if "night" in name else 1.8, label=f"{name} (peak {traj.max():.0f})")
        out[name] = float(traj.max())
    ax.axhline(180, ls="--", color="#bbb", lw=1); ax.text(2, 183, "180 mg/dL target", fontsize=8, color="#888")
    ax.set_xlabel("Minutes from now (dinner at +10 min)"); ax.set_ylabel("Simulated glucose (mg/dL)")
    ax.set_title(f"What-if simulation on the digital twin of {pid}", color=INK, fontsize=12)
    ax.legend(fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.2); fig.tight_layout(); fig.savefig(FIGURES / "whatif.png", dpi=180); plt.close(fig)

    res = {"r_carb_sensitivity_vs_true_si": r_cs, "r_basal_vs_fasting_glucose": r_gb,
           "whatif_patient": pid, "whatif_peaks": out}
    json.dump(res, open(OUTPUTS / "twin_validation.json", "w"), indent=2)
    print("Twin validation:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})


if __name__ == "__main__":
    main()
