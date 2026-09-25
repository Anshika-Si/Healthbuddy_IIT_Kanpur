"""
Multimodal fusion: turn the static EHR stream + dynamic wearable stream + digital-twin
state into one feature vector per (patient, time) prediction point.

Label: will CGM glucose exceed SPIKE_THRESHOLD (180 mg/dL) at any time in the next
HORIZON_MIN (120) minutes? Only points where glucose is currently below the threshold
are used, so the task is genuine *onset* prediction, not "is it already high".

Feature groups (the prefix is used for ablation studies and explainability):
  cgm_*    continuous glucose monitor history
  act_*    steps / activity
  hr_*     heart rate and HRV
  slp_*    last night's sleep architecture
  meal_*   app meal log
  time_*   circadian time
  ehr_*    static clinical record (demographics, labs, meds, genetics)
  twin_*   calibrated digital-twin parameters and simulated trajectory
"""
import json
import numpy as np
import pandas as pd

from src.config import (DATA_RAW, DATA_PROCESSED, OUTPUTS, STEP_MIN, HORIZON_MIN, LOOKBACK_MIN,
                        SAMPLE_EVERY_MIN, SPIKE_THRESHOLD, CALIBRATION_DAYS)
from src.twin.glucotwin import calibrate, simulate, expected_next_meal, nightly_sleep

EHR_COLS = ["age", "bmi", "waist_cm", "years_since_diagnosis", "family_history_t2d",
            "tcf7l2_risk_alleles", "hypertension", "dyslipidemia", "hba1c_pct",
            "fasting_glucose_mgdl", "ldl_mgdl", "triglycerides_mgdl", "egfr", "systolic_bp",
            "med_metformin", "med_sulfonylurea", "med_insulin", "med_dpp4"]


def _sleep_arch(df):
    d = df[["day", "minute_of_day", "sleep_stage"]].copy()
    d["night_of"] = np.where(d.minute_of_day >= 18 * 60, d.day + 1, d.day)
    d = d[((d.minute_of_day >= 18 * 60) | (d.minute_of_day < 12 * 60)) & (d.sleep_stage > 0)]
    g = d.groupby("night_of").sleep_stage
    return pd.DataFrame({"deep_frac": g.apply(lambda s: (s == 2).mean()),
                         "rem_frac": g.apply(lambda s: (s == 3).mean())})


def build_patient_features(df: pd.DataFrame, ehr_row: pd.Series, twin, include_high=False, only_day=None):
    df = df.sort_values("timestamp").reset_index(drop=True)
    g = df.glucose_cgm.to_numpy(float)
    steps = df.steps.to_numpy(float)
    hr = df.heart_rate.to_numpy(float)
    hrv = df.hrv_rmssd.to_numpy(float)
    carbs = df.meal_carbs_logged.to_numpy(float)
    tod = df.minute_of_day.to_numpy()
    day = df.day.to_numpy()
    sleep_h = nightly_sleep(df)
    arch = _sleep_arch(df)
    night_hrv = df[df.sleep_stage > 0].groupby("day").hrv_rmssd.mean()
    hrv_baseline = df[(df.day < CALIBRATION_DAYS) & (df.sleep_stage > 0)].hrv_rmssd.mean()
    rhr_baseline = df[(df.day < CALIBRATION_DAYS) & (df.steps < 10)].heart_rate.median()

    lb, hz = LOOKBACK_MIN // STEP_MIN, HORIZON_MIN // STEP_MIN
    stride = SAMPLE_EVERY_MIN // STEP_MIN
    cum_steps = np.cumsum(steps)

    rows = []
    for i in range(max(lb, CALIBRATION_DAYS * 288), len(g) - hz, stride):
        if g[i] >= SPIKE_THRESHOLD and not include_high:
            continue
        if only_day is not None and day[i] != only_day:
            continue
        hist = g[i - lb:i + 1]
        fut = g[i + 1:i + hz + 1]
        d = int(day[i])
        # logged meals in last 4 h
        meal_idx = [j for j in range(max(0, i - 48), i + 1) if carbs[j] > 0]
        recent = [((i - j) * STEP_MIN, carbs[j]) for j in meal_idx]
        mins_since_meal = recent[-1][0] if recent else 300.0
        sh = float(sleep_h.get(d, 7.0))

        twin_traj = simulate(twin, g[i], recent, HORIZON_MIN, sleep_h=sh)
        p_meal, exp_carbs, exp_off = expected_next_meal(twin, int(tod[i]), HORIZON_MIN, mins_since_meal)
        twin_traj_habit = simulate(twin, g[i], recent, HORIZON_MIN, sleep_h=sh,
                                   extra_meal=(exp_off, exp_carbs * p_meal))
        cob = sum(c * max(0.0, 1 - ago / (2.5 * twin.tpk)) for ago, c in recent)  # carbs on board

        crossing = np.flatnonzero(fut > SPIKE_THRESHOLD)
        steps_today = cum_steps[i] - (cum_steps[d * 288 - 1] if d > 0 else 0)
        r = {
            "patient_id": df.patient_id.iloc[0], "timestamp": df.timestamp.iloc[i], "day": d,
            "minute_of_day": int(tod[i]),
            # CGM
            "cgm_now": g[i], "cgm_mean_30": hist[-7:].mean(), "cgm_min_120": hist.min(),
            "cgm_max_120": hist.max(), "cgm_std_120": hist.std(),
            "cgm_slope_15": (g[i] - g[i - 3]) / 15, "cgm_slope_30": (g[i] - g[i - 6]) / 30,
            "cgm_slope_60": (g[i] - g[i - 12]) / 60,
            "cgm_accel": ((g[i] - g[i - 3]) - (g[i - 3] - g[i - 6])) / 15,
            "cgm_tar_120": (hist > SPIKE_THRESHOLD).mean(),
            # meal log
            "meal_mins_since": mins_since_meal, "meal_carbs_60": carbs[i - 12:i + 1].sum(),
            "meal_carbs_180": carbs[i - 36:i + 1].sum(), "meal_cob": cob,
            # activity
            "act_steps_30": steps[i - 6:i + 1].sum(), "act_steps_60": steps[i - 12:i + 1].sum(),
            "act_steps_120": steps[i - lb:i + 1].sum(), "act_steps_today": steps_today,
            # heart
            "hr_mean_30": hr[i - 6:i + 1].mean(), "hr_vs_rest": hr[i - 6:i + 1].mean() - rhr_baseline,
            "hr_hrv_60": hrv[i - 12:i + 1].mean(),
            "hr_hrv_night_z": (night_hrv.get(d, hrv_baseline) - hrv_baseline) / 5.0,
            # sleep
            "slp_hours": sh, "slp_deep_frac": float(arch.deep_frac.get(d, 0.18)),
            "slp_rem_frac": float(arch.rem_frac.get(d, 0.2)),
            # time
            "time_sin": np.sin(2 * np.pi * tod[i] / 1440), "time_cos": np.cos(2 * np.pi * tod[i] / 1440),
            # twin
            "twin_gb": twin.gb, "twin_k": twin.k, "twin_cs": twin.cs, "twin_tpk": twin.tpk,
            "twin_beta_sleep": twin.beta_sleep, "twin_walk_effect": twin.walk_effect,
            "twin_peak_2h": twin_traj.max(), "twin_end_2h": twin_traj[-1],
            "twin_peak_habit_2h": twin_traj_habit.max(), "twin_p_meal_2h": p_meal,
            # labels
            "y_spike": int(len(crossing) > 0),
            "twin_traj": twin_traj_habit.round(1).tolist() if only_day is not None else None,
            "y_peak_2h": fut.max(),
            "y_minutes_to_spike": float((crossing[0] + 1) * STEP_MIN) if len(crossing) else np.nan,
        }
        for c in EHR_COLS:
            r[f"ehr_{c}"] = float(ehr_row[c])
        r["ehr_sex_male"] = float(ehr_row["sex"] == "M")
        r["ehr_diet_rice"] = float(ehr_row["diet_pattern"] == "rice_dominant")
        rows.append(r)
    return rows


def main():
    ehr = pd.read_csv(DATA_RAW / "ehr_patients.csv").set_index("patient_id")
    wear = pd.read_csv(DATA_RAW / "wearables_5min.csv.gz", parse_dates=["timestamp"])
    wear = wear.drop(columns=[c for c in wear.columns if c.startswith("_")])  # hide ground truth
    rows, twins = [], {}
    for pid, df in wear.groupby("patient_id"):
        twin = calibrate(df)
        twins[pid] = twin.to_dict()
        rows.extend(build_patient_features(df, ehr.loc[pid], twin))
    feats = pd.DataFrame(rows).drop(columns=["twin_traj"])
    feats.to_csv(DATA_PROCESSED / "features.csv.gz", index=False)
    with open(OUTPUTS / "twin_params.json", "w") as f:
        json.dump(twins, f, indent=1)
    print(f"Features: {feats.shape[0]:,} prediction points x {feats.shape[1]} columns; "
          f"spike prevalence = {feats.y_spike.mean():.1%}")
    return feats


if __name__ == "__main__":
    main()
