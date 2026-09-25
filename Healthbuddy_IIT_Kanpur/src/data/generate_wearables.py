"""
Synthetic wearable / IoT time-series generator (the *dynamic / real-time* stream).

Simulates 5-minute resolution data that mimics what a CGM (e.g. FreeStyle Libre),
a smartwatch (Apple Health / Google Fit) and a food-logging app would export:

    glucose_cgm   continuous glucose monitor reading (mg/dL, with sensor noise)
    heart_rate    beats per minute
    hrv_rmssd     heart-rate variability, RMSSD (ms)
    steps         steps in the 5-minute bin
    sleep_stage   0 = awake, 1 = light, 2 = deep, 3 = REM
    meal_carbs_logged   carbs (g) logged in the app at that time (~70% of meals get logged)

The glucose trace is produced by a hidden physiological "world model" (a simplified
glucose-insulin minimal model with meal absorption, exercise uptake, dawn phenomenon,
stress and sleep-loss effects on insulin sensitivity). IMPORTANT: this world model is
separate from the digital twin in `src/twin/`. The twin never sees the hidden parameters;
it must infer them from data, exactly as it would for a real patient.

Physiological effects encoded (directionally consistent with published literature):
  * one night of short sleep reduces next-day insulin sensitivity (~20-25% at 4-5 h)
  * post-meal walking blunts the glucose peak
  * psychological stress raises glucose and lowers HRV
  * early-morning dawn phenomenon
  * Indian meal patterns: large carbohydrate lunch/dinner, late dinners, rice vs wheat
"""
import numpy as np
import pandas as pd

from src.config import DATA_RAW, N_DAYS, STEP_MIN, SEED

STEPS_PER_DAY = 24 * 60 // STEP_MIN
START = pd.Timestamp("2026-09-01 00:00")

MEAL_PLAN = {  # name: (mean hour, sd hour, carbs by diet {rice, wheat, mixed})
    "breakfast": (8.25, 0.6, {"rice_dominant": 55, "wheat_dominant": 50, "mixed": 45}),
    "lunch": (13.4, 0.6, {"rice_dominant": 95, "wheat_dominant": 75, "mixed": 75}),
    "snack": (17.2, 0.5, {"rice_dominant": 25, "wheat_dominant": 30, "mixed": 25}),
    "dinner": (21.0, 0.7, {"rice_dominant": 85, "wheat_dominant": 80, "mixed": 70}),
}
GI = {"rice_dominant": 1.15, "wheat_dominant": 1.0, "mixed": 0.95}


def _gamma_kernel(shape=2.6, scale_min=24.0, length_min=300):
    t = np.arange(0, length_min, STEP_MIN, dtype=float) + STEP_MIN / 2
    k = t ** (shape - 1) * np.exp(-t / scale_min)
    return k / k.sum()


def simulate_patient(p: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    n = N_DAYS * STEPS_PER_DAY
    tod_min = (np.arange(n) * STEP_MIN) % (24 * 60)
    day = np.arange(n) // STEPS_PER_DAY

    si_true = p["_latent_insulin_sensitivity"]
    walk_habit = rng.uniform(0.05, 0.6)            # prob. of a post-meal walk
    morning_walker = rng.random() < 0.4
    stress_prone = rng.uniform(0.1, 0.7)
    rhr = 62 + 10 * (1.2 - si_true) + (p["bmi"] - 25) * 0.5 + rng.normal(0, 4)
    hrv_base = np.clip(55 - 0.45 * (p["age"] - 30) + 12 * (si_true - 0.8) + rng.normal(0, 5), 12, 70)
    gb_base = p["fasting_glucose_mgdl"] - (8 if p["med_sulfonylurea"] or p["med_insulin"] else 0)

    steps = np.zeros(n)
    sleep_stage = np.zeros(n, dtype=int)
    carbs_true = np.zeros(n)
    carbs_logged = np.zeros(n)
    si_day = np.zeros(n)
    stress = np.zeros(n)
    gi_series = np.zeros(n)

    prev_sleep_h = 7.0
    for d in range(N_DAYS):
        base = d * STEPS_PER_DAY
        # ---- Sleep (night ending on the morning of day d) ----
        sleep_h = float(np.clip(rng.normal(6.8, 0.9), 4.8, 8.8))
        if rng.random() < 0.18:                     # occasional very short night
            sleep_h = float(rng.uniform(3.5, 5.0))
        wake_min = int(np.clip(rng.normal(6.6, 0.5), 5.0, 8.5) * 60)
        sleep_start = wake_min - int(sleep_h * 60)
        deep_frac = np.clip(rng.normal(0.19, 0.04) - 0.03 * (sleep_h < 5.5), 0.05, 0.3)
        for m in range(sleep_start, wake_min, STEP_MIN):
            idx = base + (m // STEP_MIN) if m >= 0 else base + (m // STEP_MIN)  # m<0 -> prev day
            if idx < 0:
                continue
            phase = ((m - sleep_start) % 90) / 90.0     # 90-min sleep cycles
            if phase < deep_frac * 1.8 and (m - sleep_start) < sleep_h * 60 * 0.6:
                sleep_stage[idx] = 2
            elif phase > 0.75:
                sleep_stage[idx] = 3
            else:
                sleep_stage[idx] = 1
        day_slice = slice(base, base + STEPS_PER_DAY)

        # ---- Day-level physiology ----
        stress_d = float(np.clip(rng.beta(2, 5) + (0.3 if rng.random() < stress_prone * 0.5 else 0), 0, 1))
        sleep_penalty = 0.25 * np.clip((7.0 - sleep_h) / 3.0, 0, 1)
        si_d = si_true * (1 - sleep_penalty) * (1 - 0.12 * stress_d) * (1.08 if p["med_metformin"] else 1.0)
        si_day[day_slice] = si_d
        stress[day_slice] = stress_d
        prev_sleep_h = sleep_h

        # ---- Background activity ----
        awake_day = (tod_min[day_slice] >= wake_min) & (tod_min[day_slice] < 22.8 * 60)
        steps[day_slice] = np.where(awake_day, rng.poisson(rng.uniform(25, 70), STEPS_PER_DAY), 0)
        if morning_walker and rng.random() < 0.7:
            s = base + (wake_min + 20) // STEP_MIN
            steps[s:s + int(rng.uniform(4, 8))] += rng.normal(480, 60)

        # ---- Meals ----
        for name, (mu_h, sd_h, carb_map) in MEAL_PLAN.items():
            if name == "snack" and rng.random() < 0.4:
                continue
            t = int(np.clip(rng.normal(mu_h, sd_h), 6.5, 23.2) * 60)
            carbs = max(10.0, rng.normal(carb_map[p["diet_pattern"]], 18))
            if rng.random() < 0.08:                  # festive / heavy meal
                carbs *= 1.6
            idx = base + t // STEP_MIN
            carbs_true[idx] += carbs
            gi_series[idx] = GI[p["diet_pattern"]] * rng.uniform(0.85, 1.15)
            if rng.random() < 0.7:                   # logged in app (with estimation error)
                carbs_logged[idx] += round(carbs * rng.uniform(0.75, 1.25))
            if name in ("lunch", "dinner") and rng.random() < walk_habit:
                w0 = idx + int(rng.uniform(2, 5))
                steps[w0:w0 + int(rng.uniform(3, 7))] += rng.normal(500, 60)

    steps = np.clip(steps, 0, None).round()

    # ---- Glucose: minimal-model style Euler integration ----
    kernel = _gamma_kernel()
    rise_per_g = 1.35 / np.power(np.clip(si_day, 0.2, None), 0.9)          # mg/dL per gram
    impulse = carbs_true * np.where(gi_series > 0, gi_series, 1.0) * rise_per_g
    ra = np.convolve(impulse, kernel)[:n]                               # mg/dL appearing per step
    dawn = 14 * np.exp(-0.5 * ((tod_min - 6.5 * 60) / 70) ** 2)
    gb = gb_base + dawn + 12 * stress
    recent_steps = pd.Series(steps).rolling(6, min_periods=1).sum().to_numpy()

    g = np.zeros(n)
    g[0] = gb[0]
    ar = 0.0
    for i in range(1, n):
        k_clear = 0.011 * si_day[i] * (1 + recent_steps[i] / 3000)          # exercise boosts uptake
        ex_uptake = 3.2 * steps[i] / 500 * (g[i - 1] / 130)
        ar = 0.9 * ar + rng.normal(0, 1.1)
        g[i] = g[i - 1] + ra[i] - k_clear * STEP_MIN * (g[i - 1] - gb[i]) - ex_uptake + ar * 0.35
        g[i] = max(g[i], 45)
    cgm = np.clip(g + rng.normal(0, 4.0, n), 40, 400).round(0)

    # ---- Heart rate & HRV ----
    asleep = sleep_stage > 0
    hr = rhr + 0.035 * steps + 8 * stress - 7 * asleep + rng.normal(0, 3, n)
    hrv = hrv_base * (1 - 0.35 * stress) * np.where(asleep, 1.25, 0.9) * (1 - 0.00025 * steps)
    hrv = np.clip(hrv + rng.normal(0, 4, n), 5, 120)

    ts = START + pd.to_timedelta(np.arange(n) * STEP_MIN, unit="min")
    return pd.DataFrame({
        "patient_id": p["patient_id"], "timestamp": ts, "day": day, "minute_of_day": tod_min,
        "glucose_cgm": cgm, "heart_rate": hr.round(0), "hrv_rmssd": hrv.round(1),
        "steps": steps.astype(int), "sleep_stage": sleep_stage,
        "meal_carbs_logged": carbs_logged,
        # hidden ground truth (for analysis only, dropped before modelling)
        "_carbs_true": carbs_true.round(0), "_si_day": si_day.round(3),
    })


def main():
    ehr = pd.read_csv(DATA_RAW / "ehr_patients.csv")
    rng = np.random.default_rng(SEED + 1)
    frames = [simulate_patient(p, rng) for _, p in ehr.iterrows()]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(DATA_RAW / "wearables_5min.csv.gz", index=False)
    df[df.patient_id == ehr.patient_id.iloc[0]].drop(columns=["_carbs_true", "_si_day"]).to_csv(
        DATA_RAW / "wearables_sample_P0000.csv", index=False)
    print(f"Wearables: {len(df):,} rows ({df.patient_id.nunique()} patients x {N_DAYS} days @ {STEP_MIN} min)")
    return df


if __name__ == "__main__":
    main()
