"""
GlucoTwin: a personalised, physiology-informed virtual patient.

The twin is a *hybrid* model:
  1. A compact mechanistic core (basal glucose, clearance rate, carbohydrate
     sensitivity, time-to-peak, sleep and walking modifiers) whose parameters are
     CALIBRATED per patient from their own data during an onboarding week.
  2. A machine-learning risk layer (src/models/train.py) that uses the twin's
     simulated trajectory as a physics-informed feature alongside the raw fused
     EHR + wearable features.

Because every twin parameter has a physiological meaning, a clinician can read
the twin ("this patient's glucose rises 2.8 mg/dL per gram of carbohydrate, 30%
more after a short night") and ask it what-if questions ("what if she walks
15 minutes after dinner?").

The twin never sees the simulator's hidden parameters - it learns from the
observable CGM, meal log, steps and sleep streams, as it would with a real patient.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from src.config import STEP_MIN, CALIBRATION_DAYS

# Population priors (used to shrink noisy individual estimates - empirical Bayes style)
PRIOR = {"k": 0.006, "cs": 1.25, "tpk": 55.0, "beta_sleep": 0.06, "walk_effect": 0.25}


@dataclass
class TwinParams:
    patient_id: str
    gb: float            # basal (fasting) glucose, mg/dL
    k: float             # clearance rate toward basal, 1/min
    cs: float            # carbohydrate sensitivity: peak rise per gram of carbs, mg/dL/g
    tpk: float           # minutes from meal to glucose peak
    beta_sleep: float    # fractional increase in meal response per hour of sleep below 7 h
    walk_effect: float   # fractional reduction in meal peak with a 30-min post-meal walk
    meal_hist: list      # probability of a meal starting in each hour of day (24 values)
    meal_carbs_hist: list  # mean logged carbs by hour of day (24 values)
    n_meals_used: int

    def to_dict(self):
        return asdict(self)


def nightly_sleep(df: pd.DataFrame) -> pd.Series:
    """Hours asleep in the night ending on each day's morning (18:00 prev day -> 12:00)."""
    d = df[["day", "minute_of_day", "sleep_stage"]].copy()
    d["night_of"] = np.where(d.minute_of_day >= 18 * 60, d.day + 1, d.day)
    d = d[(d.minute_of_day >= 18 * 60) | (d.minute_of_day < 12 * 60)]
    return d.groupby("night_of").sleep_stage.apply(lambda s: (s > 0).sum() * STEP_MIN / 60.0)


def _shrink(est, prior, n, strength=4):
    if est is None or not np.isfinite(est):
        return prior
    w = n / (n + strength)
    return w * est + (1 - w) * prior


def calibrate(df: pd.DataFrame) -> TwinParams:
    """Fit twin parameters for one patient from their calibration (onboarding) days."""
    pid = df.patient_id.iloc[0]
    cal = df[df.day < CALIBRATION_DAYS].reset_index(drop=True)
    g = cal.glucose_cgm.to_numpy(float)
    carbs = cal.meal_carbs_logged.to_numpy(float)
    steps = cal.steps.to_numpy(float)
    tod = cal.minute_of_day.to_numpy()
    sleep = nightly_sleep(df)

    # Basal glucose: median overnight value 02:00-04:30
    gb = float(np.median(g[(tod >= 120) & (tod <= 270)]))

    # Meal responses from logged meals
    rises, tpks, sleep_h, walked = [], [], [], []
    win = 180 // STEP_MIN
    for i in np.flatnonzero(carbs > 0):
        if i + win >= len(g) or carbs[max(0, i - 36):i].sum() > 0:
            continue  # skip truncated windows and stacked meals
        seg = g[i:i + win]
        j = int(np.argmax(seg))
        rises.append((seg[j] - g[i]) / carbs[i])
        tpks.append(j * STEP_MIN)
        sleep_h.append(sleep.get(cal.day.iloc[i], 7.0))
        walked.append(steps[i:i + 9].sum() > 1500)
    rises, tpks = np.array(rises), np.array(tpks)
    sleep_h, walked = np.array(sleep_h), np.array(walked)
    n = len(rises)

    cs_raw = float(np.median(rises[~walked])) if (~walked).sum() >= 2 else (float(np.median(rises)) if n else None)
    cs = _shrink(cs_raw, PRIOR["cs"], n)
    tpk = _shrink(float(np.median(tpks)) if n else None, PRIOR["tpk"], n)

    # Sleep sensitivity: slope of relative meal response vs sleep deficit
    beta = None
    if n >= 5 and np.ptp(sleep_h) > 1.0:
        rel = rises / max(np.median(rises), 1e-6) - 1
        deficit = np.clip(7 - sleep_h, 0, None)
        if deficit.std() > 0:
            beta = float(np.polyfit(deficit, rel, 1)[0])
    beta_sleep = float(np.clip(_shrink(beta, PRIOR["beta_sleep"], n, strength=12), 0, 0.3))

    # Walking effect
    we = None
    if walked.sum() >= 2 and (~walked).sum() >= 2:
        we = 1 - np.median(rises[walked]) / max(np.median(rises[~walked]), 1e-6)
    walk_effect = float(np.clip(_shrink(we, PRIOR["walk_effect"], int(walked.sum()), strength=6), 0, 0.6))

    # Clearance: regress 30-min change on excess glucose, >=150 min after any logged meal
    since_meal = np.full(len(g), 9999.0)
    last = -9999
    for i in range(len(g)):
        if carbs[i] > 0:
            last = i
        since_meal[i] = (i - last) * STEP_MIN
    lag = 30 // STEP_MIN
    idx = np.flatnonzero((since_meal >= 150) & (since_meal <= 360) & (g - gb > 15))
    idx = idx[idx + lag < len(g)]
    k = None
    if len(idx) > 20:
        x = g[idx] - gb
        y = -(g[idx + lag] - g[idx]) / 30.0
        k = float(np.clip(np.sum(x * y) / np.sum(x * x), 0.002, 0.04))
    k = _shrink(k, PRIOR["k"], len(idx) / 10)

    # Meal routine: hour-of-day meal probability from logged meals (Laplace smoothed)
    hrs = (tod[carbs > 0] // 60).astype(int)
    counts = np.bincount(hrs, minlength=24).astype(float)
    days = max(cal.day.nunique(), 1)
    meal_hist = ((counts + 0.05) / (days / 0.7)).clip(0, 1)  # /0.7 corrects for unlogged meals
    carb_sum = np.bincount(hrs, weights=carbs[carbs > 0], minlength=24)
    meal_carbs = np.where(counts > 0, carb_sum / np.maximum(counts, 1), 0)

    return TwinParams(pid, round(gb, 1), round(k, 5), round(cs, 3), round(tpk, 1),
                      round(beta_sleep, 3), round(walk_effect, 3),
                      [round(float(v), 3) for v in meal_hist],
                      [round(float(v), 1) for v in meal_carbs], n)


def meal_shape(tau, tpk):
    """Normalised meal response: 0 at tau=0, peak 1 at tau=tpk, gradual decay."""
    tau = np.maximum(tau, 0)
    return (tau / tpk) * np.exp(1 - tau / tpk)


def simulate(params: TwinParams, g_now: float, recent_meals: list, horizon_min=120,
             sleep_h=7.0, extra_meal=None, walk_min=0.0):
    """
    Forward-simulate glucose from now.

    recent_meals: list of (minutes_ago, carbs_g) for logged meals in the last ~4 h
    extra_meal:   optional (minutes_from_now, carbs_g) hypothetical/expected meal
    walk_min:     minutes of post-meal walking (what-if lever)
    Returns array of predicted glucose at t = 0, 5, ..., horizon_min.
    """
    t = np.arange(0, horizon_min + STEP_MIN, STEP_MIN, dtype=float)
    p = params
    sleep_mult = 1 + p.beta_sleep * max(0.0, 7.0 - sleep_h)
    walk_mult = 1 - p.walk_effect * min(walk_min, 30) / 30.0
    meals = list(recent_meals) + ([(-extra_meal[0], extra_meal[1])] if extra_meal else [])

    def meal_component(tt, mult_walk):
        out = np.zeros_like(tt)
        for ago, c in meals:
            m = walk_mult if (mult_walk and ago <= 30) else 1.0
            out += p.cs * c * sleep_mult * m * meal_shape(tt + ago, p.tpk)
        return out

    now_meal = meal_component(np.array([0.0]), False)[0]
    residual = g_now - p.gb - now_meal          # unexplained excess decays toward basal
    return p.gb + residual * np.exp(-p.k * t) + meal_component(t, True)


def expected_next_meal(params: TwinParams, minute_of_day: int, horizon_min=120, min_gap_since_meal=None):
    """Routine model: probability and size of a meal in the next horizon, from the twin's habit profile."""
    hours = [int(((minute_of_day + m) // 60) % 24) for m in range(0, horizon_min, 60)]
    probs = [params.meal_hist[h] for h in hours]
    p_any = 1 - np.prod([1 - min(q, 0.95) for q in probs])
    h_best = hours[int(np.argmax(probs))]
    carbs = params.meal_carbs_hist[h_best] or 50.0
    if min_gap_since_meal is not None and min_gap_since_meal < 120:
        p_any *= 0.3  # just ate: another meal soon is less likely
    offset = max(0, h_best * 60 - minute_of_day) % (24 * 60)
    return float(p_any), float(carbs), float(min(offset + 30, horizon_min - 30))
