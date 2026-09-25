"""
Export everything the clinician dashboard needs into one JSON file, then inject it into
the self-contained HTML template (no server, no internet needed to open the dashboard).
"""
import json
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from src.config import DATA_RAW, OUTPUTS, ROOT, SPIKE_THRESHOLD
from src.features.fusion import build_patient_features
from src.twin.glucotwin import TwinParams, nightly_sleep

warnings.filterwarnings("ignore")
N_SHOW = 6
READABLE = {
    "twin_peak_habit_2h": "Twin-simulated 2 h peak (incl. usual meal routine)",
    "twin_peak_2h": "Twin-simulated 2 h peak (logged meals only)",
    "twin_end_2h": "Twin-simulated glucose in 2 h",
    "twin_p_meal_2h": "Meal expected in next 2 h (routine)",
    "twin_gb": "Personal basal glucose", "twin_cs": "Personal carb sensitivity",
    "twin_k": "Personal glucose clearance rate", "twin_tpk": "Personal time-to-peak",
    "twin_beta_sleep": "Personal sleep sensitivity", "twin_walk_effect": "Personal walking benefit",
    "act_steps_30": "Steps in last 30 min", "act_steps_60": "Steps in last 60 min",
    "act_steps_120": "Steps in last 2 h", "act_steps_today": "Steps so far today",
    "time_sin": "Time of day", "time_cos": "Time of day",
    "slp_hours": "Last night's sleep", "slp_deep_frac": "Deep-sleep share", "slp_rem_frac": "REM share",
    "hr_hrv_60": "HRV (last hour)", "hr_hrv_night_z": "Overnight HRV vs baseline",
    "hr_mean_30": "Heart rate (30 min)", "hr_vs_rest": "Heart rate above resting",
    "meal_cob": "Carbs on board", "meal_mins_since": "Minutes since last logged meal",
    "meal_carbs_60": "Carbs logged in last hour", "meal_carbs_180": "Carbs logged in last 3 h",
    "cgm_now": "Current glucose", "cgm_slope_15": "Glucose trend (15 min)",
    "cgm_slope_30": "Glucose trend (30 min)", "cgm_slope_60": "Glucose trend (60 min)",
    "cgm_accel": "Glucose acceleration", "cgm_mean_30": "Mean glucose (30 min)",
    "cgm_max_120": "Highest glucose (2 h)", "cgm_min_120": "Lowest glucose (2 h)",
    "cgm_std_120": "Glucose variability (2 h)", "cgm_tar_120": "Time above range (2 h)",
    "ehr_hba1c_pct": "HbA1c", "ehr_fasting_glucose_mgdl": "Fasting glucose (lab)",
    "ehr_bmi": "BMI", "ehr_years_since_diagnosis": "Years since diagnosis",
    "ehr_tcf7l2_risk_alleles": "TCF7L2 risk alleles", "ehr_triglycerides_mgdl": "Triglycerides",
    "ehr_age": "Age", "ehr_waist_cm": "Waist circumference",
}


def fmt_val(c, v):
    if c.startswith("time_"):
        return None
    if "frac" in c or c in ("twin_p_meal_2h", "cgm_tar_120", "twin_walk_effect", "twin_beta_sleep"):
        return f"{v * 100:.0f}%"
    if c == "slp_hours":
        return f"{v:.1f} h"
    if "slope" in c or c == "cgm_accel":
        return f"{v:+.1f} mg/dL/min"
    if c.startswith("act_"):
        return f"{v:,.0f} steps"
    return f"{v:.1f}" if abs(v) < 20 else f"{v:,.0f}"


def main():
    meta = json.load(open(OUTPUTS / "test_patients.json"))
    cols, thr, margin = meta["features"], meta["threshold"], meta["conformal_margin"]
    clf = lgb.Booster(model_file=str(OUTPUTS / "glucotwin_spike_model.txt"))
    qms = {q: lgb.Booster(model_file=str(OUTPUTS / f"glucotwin_peak_q{q}.txt")) for q in (10, 50, 90)}
    explainer = shap.TreeExplainer(clf)
    ehr = pd.read_csv(DATA_RAW / "ehr_patients.csv").set_index("patient_id")
    twins = json.load(open(OUTPUTS / "twin_params.json"))
    wear = pd.read_csv(DATA_RAW / "wearables_5min.csv.gz", parse_dates=["timestamp"])
    wear = wear.drop(columns=[c for c in wear.columns if c.startswith("_")])
    last_day = int(wear.day.max())

    # choose test patients whose final day has at least one spike, spread across risk levels
    cand = []
    for pid in meta["test"]:
        g = wear[(wear.patient_id == pid) & (wear.day == last_day)].glucose_cgm
        cand.append((pid, float((g > SPIKE_THRESHOLD).mean()), float(ehr.loc[pid, "hba1c_pct"])))
    cand = sorted([c for c in cand if 0.05 < c[1] < 0.6], key=lambda c: c[2])
    pick = [cand[int(i)][0] for i in np.linspace(0, len(cand) - 1, N_SHOW)]

    patients = []
    for pid in pick:
        df = wear[wear.patient_id == pid].sort_values("timestamp").reset_index(drop=True)
        twin = TwinParams(**twins[pid])
        rows = build_patient_features(df, ehr.loc[pid], twin, include_high=True, only_day=last_day)
        f = pd.DataFrame(rows)
        X = f[cols]
        p = clf.predict(X)
        q10, q50, q90 = (qms[q].predict(X) for q in (10, 50, 90))
        sv = explainer.shap_values(X)
        sv = sv[1] if isinstance(sv, list) else sv
        points = []
        for i in range(len(f)):
            order = np.argsort(-np.abs(sv[i]))
            drivers, seen = [], set()
            for j in order:
                name = READABLE.get(cols[j], cols[j])
                if name in seen:
                    continue
                seen.add(name)
                drivers.append({"name": name, "impact": round(float(sv[i, j]), 3),
                                "value": fmt_val(cols[j], float(X.iloc[i, j]))})
                if len(drivers) == 4:
                    break
            points.append({"m": int(f.minute_of_day[i]), "p": round(float(p[i]), 3),
                           "above": bool(f.cgm_now[i] >= SPIKE_THRESHOLD),
                           "q10": round(float(q10[i] - margin)), "q50": round(float(q50[i])),
                           "q90": round(float(q90[i] + margin)), "twin": f.twin_traj[i][::3],
                           "drivers": drivers, "truth_peak": round(float(f.y_peak_2h[i]))})
        day_df = df[df.day == last_day]
        prev_night = df[((df.day == last_day - 1) & (df.minute_of_day >= 18 * 60)) |
                        ((df.day == last_day) & (df.minute_of_day < 12 * 60))]
        sleep_h = nightly_sleep(df)
        week = []
        for d in range(last_day - 6, last_day + 1):
            gd = df[df.day == d].glucose_cgm
            week.append({"day": d, "mean": round(float(gd.mean())),
                         "tir": round(float(((gd >= 70) & (gd <= 180)).mean() * 100)),
                         "sleep": round(float(sleep_h.get(d, np.nan)), 1),
                         "steps": int(df[df.day == d].steps.sum())})
        e = ehr.loc[pid]
        patients.append({
            "id": pid, "name": e.display_name, "city": e.city,
            "date": str(day_df.timestamp.iloc[0].date()),
            "ehr": {k: (e[k].item() if hasattr(e[k], "item") else e[k]) for k in
                    ["age", "sex", "bmi", "waist_cm", "years_since_diagnosis", "hba1c_pct",
                     "fasting_glucose_mgdl", "ldl_mgdl", "triglycerides_mgdl", "egfr", "systolic_bp",
                     "family_history_t2d", "tcf7l2_risk_alleles", "hypertension", "dyslipidemia",
                     "med_metformin", "med_sulfonylurea", "med_insulin", "med_dpp4", "diet_pattern"]},
            "twin": {k: twins[pid][k] for k in ["gb", "k", "cs", "tpk", "beta_sleep", "walk_effect",
                                                 "meal_hist", "meal_carbs_hist"]},
            "series": {"glucose": day_df.glucose_cgm.astype(int).tolist(),
                       "steps": day_df.steps.astype(int).tolist(),
                       "hr": day_df.heart_rate.astype(int).tolist(),
                       "hrv": day_df.hrv_rmssd.round(0).astype(int).tolist(),
                       "meals": [[int(m), int(c)] for m, c in
                                 zip(day_df.minute_of_day, day_df.meal_carbs_logged) if c > 0]},
            "sleep": {"stages": prev_night.sleep_stage.astype(int).tolist(),
                      "hours": round(float(sleep_h.get(last_day, np.nan)), 1)},
            "week": week, "points": points,
        })

    metrics = json.load(open(OUTPUTS / "metrics.json"))
    best = metrics["ablation_test"]["GlucoTwin (fusion + twin)"]
    data = {"threshold": thr, "spike": SPIKE_THRESHOLD, "patients": patients,
            "model": {"auroc": round(best["auroc"], 3), "lead": best["median_lead_min"],
                      "detect": round(best["detection_rate"] * 100)}}
    with open(OUTPUTS / "dashboard_data.json", "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    tpl = (ROOT / "src" / "dashboard" / "template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = ROOT / "dashboard" / "GlucoTwin_Dashboard.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard: {len(patients)} patients -> {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
