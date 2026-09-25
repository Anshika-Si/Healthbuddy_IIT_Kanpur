"""
Train and evaluate the GlucoTwin spike-prediction model.

* Patient-level split: test patients are never seen in training (no leakage).
* Ablation study: proves each data stream adds value (CGM only -> + meal log ->
  + wearables -> + EHR -> + digital twin).
* 5-fold patient-grouped cross-validation for the headline models (mean +/- sd).
* Explainability: SHAP values aggregated per data stream and per prediction.
* Clinical metrics: alert lead time, detection rate, alerts per day.
* Peak-glucose forecast with quantile regression (10th/50th/90th percentile band).
"""
import json
import warnings

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, mean_absolute_error, precision_score, recall_score)
from sklearn.model_selection import GroupKFold

from src.config import DATA_PROCESSED, OUTPUTS, FIGURES, SEED, TEST_FRACTION

warnings.filterwarnings("ignore")
LABELS = ["y_spike", "y_peak_2h", "y_minutes_to_spike"]
META = ["patient_id", "timestamp", "day", "minute_of_day"]
GROUP_NAMES = {"cgm": "CGM history", "meal": "Meal log", "act": "Activity (steps)",
               "hr": "Heart rate / HRV", "slp": "Sleep", "time": "Time of day",
               "ehr": "EHR (clinical record)", "twin": "Digital twin"}
PALETTE = {"ink": "#12343B", "teal": "#0F7C7A", "coral": "#E4572E", "sand": "#E9C46A",
           "grey": "#8FA3A6", "mint": "#6CC5B0"}

LGB_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=SEED, verbose=-1)


def feature_sets(cols):
    f = [c for c in cols if c not in LABELS + META]
    pick = lambda *prefixes: [c for c in f if c.split("_")[0] in prefixes]
    return {
        "CGM only": pick("cgm", "time"),
        "CGM + meal log": pick("cgm", "time", "meal"),
        "CGM + all wearables": pick("cgm", "time", "meal", "act", "hr", "slp"),
        "Wearables + EHR fusion": pick("cgm", "time", "meal", "act", "hr", "slp", "ehr"),
        "GlucoTwin (fusion + twin)": f,
    }


def fit_clf(Xtr, ytr, Xva=None, yva=None):
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    if Xva is not None:
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        m.fit(Xtr, ytr)
    return m


def metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    return {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "sensitivity": recall_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0)}


def lead_time_analysis(test_df, p, thr):
    """For every spike onset, how early did the model first alert (within the 2 h before)?"""
    d = test_df[["patient_id", "timestamp", "y_minutes_to_spike"]].copy()
    d["p"] = p
    d["onset"] = d.timestamp + pd.to_timedelta(d.y_minutes_to_spike, unit="min")
    d = d.dropna(subset=["onset"])
    d["onset"] = d.onset.dt.floor("30min")  # group points that anticipate the same excursion
    leads, detected = [], 0
    events = d.groupby(["patient_id", "onset"])
    for _, e in events:
        alerts = e[e.p >= thr]
        if len(alerts):
            detected += 1
            leads.append(float(alerts.y_minutes_to_spike.max()))
    # Alert episodes: consecutive alerting 15-min points count as ONE notification
    a = test_df[["patient_id", "timestamp", "y_spike"]].copy()
    a["alert"] = p >= thr
    a = a.sort_values(["patient_id", "timestamp"])
    gap = a.groupby("patient_id").timestamp.diff().dt.total_seconds().fillna(1e9) > 15 * 60
    start = a.alert & (~a.alert.shift(fill_value=False) | gap | (a.patient_id != a.patient_id.shift()))
    a["episode"] = start.cumsum()
    eps = a[a.alert].groupby("episode").y_spike.max()
    n_days = test_df.groupby("patient_id").day.nunique().sum()
    return {"n_events": int(events.ngroups), "detection_rate": detected / max(events.ngroups, 1),
            "median_lead_min": float(np.median(leads)) if leads else 0.0,
            "p25_lead_min": float(np.percentile(leads, 25)) if leads else 0.0,
            "alert_episodes_per_patient_day": float(len(eps) / n_days),
            "alert_episode_precision": float(eps.mean()) if len(eps) else 0.0}


def pick_threshold(y, p, target_sens=0.85):
    fpr, tpr, thr = roc_curve(y, p)
    i = int(np.argmax(tpr >= target_sens))
    return float(thr[i])


def main():
    df = pd.read_csv(DATA_PROCESSED / "features.csv.gz", parse_dates=["timestamp"])
    rng = np.random.default_rng(SEED)
    patients = np.array(sorted(df.patient_id.unique()))
    rng.shuffle(patients)
    n_test = int(len(patients) * TEST_FRACTION)
    test_p, val_p, train_p = patients[:n_test], patients[n_test:n_test + 15], patients[n_test + 15:]
    tr, va, te = (df[df.patient_id.isin(s)] for s in (train_p, val_p, test_p))
    sets = feature_sets(df.columns)
    results, models, preds = {}, {}, {}

    # ---- Clinical baseline: linear trend extrapolation (what a CGM arrow does) ----
    trend = te.cgm_now + 60 * te.cgm_slope_30
    results["Trend-arrow rule (baseline)"] = {"auroc": roc_auc_score(te.y_spike, trend),
                                               "auprc": average_precision_score(te.y_spike, trend)}
    preds["Trend-arrow rule (baseline)"] = trend.to_numpy()

    # ---- Ablation ladder ----
    for name, cols in sets.items():
        m = fit_clf(tr[cols], tr.y_spike, va[cols], va.y_spike)
        thr = pick_threshold(va.y_spike, m.predict_proba(va[cols])[:, 1])
        p = m.predict_proba(te[cols])[:, 1]
        results[name] = {**metrics(te.y_spike, p, thr), "threshold": thr, "n_features": len(cols)}
        models[name], preds[name] = m, p
        print(f"{name:32s} AUROC={results[name]['auroc']:.3f}  AUPRC={results[name]['auprc']:.3f}")

    best = "GlucoTwin (fusion + twin)"
    thr = results[best]["threshold"]
    results[best].update(lead_time_analysis(te, preds[best], thr))

    # ---- Patient-grouped 5-fold CV for robustness ----
    cv = {}
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(df, groups=df.patient_id))
    cv["Trend-arrow rule (baseline)"] = {
        "auroc_mean": float(np.mean([roc_auc_score(df.iloc[t].y_spike, df.iloc[t].cgm_now + 60 * df.iloc[t].cgm_slope_30)
                                     for _, t in folds])), "auroc_sd": 0.0, "auprc_mean": None}
    for name, cols in sets.items():
        aucs, aps = [], []
        for tri, tei in folds:
            m = fit_clf(df.iloc[tri][cols], df.iloc[tri].y_spike)
            pp = m.predict_proba(df.iloc[tei][cols])[:, 1]
            aucs.append(roc_auc_score(df.iloc[tei].y_spike, pp))
            aps.append(average_precision_score(df.iloc[tei].y_spike, pp))
        cv[name] = {"auroc_mean": float(np.mean(aucs)), "auroc_sd": float(np.std(aucs)),
                    "auprc_mean": float(np.mean(aps)), "auprc_sd": float(np.std(aps))}
        print(f"CV {name:30s} AUROC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f} | AUPRC {np.mean(aps):.3f}")

    # ---- Peak glucose forecast (quantile regression) ----
    cols = sets[best]
    q_models = {}
    for q in (0.1, 0.5, 0.9):
        qm = lgb.LGBMRegressor(objective="quantile", alpha=q, **{**LGB_PARAMS, "n_estimators": 400})
        qm.fit(tr[cols], tr.y_peak_2h)
        q_models[q] = qm
    # Conformalized quantile regression (CQR): widen the band using validation patients
    v10, v90 = q_models[0.1].predict(va[cols]), q_models[0.9].predict(va[cols])
    conf = np.maximum(v10 - va.y_peak_2h, va.y_peak_2h - v90)
    qhat = float(np.quantile(conf, 0.8 * (1 + 1 / len(va))))
    q10, q50, q90 = (q_models[q].predict(te[cols]) for q in (0.1, 0.5, 0.9))
    q10, q90 = q10 - qhat, q90 + qhat
    forecast = {
        "mae_glucotwin_mgdl": mean_absolute_error(te.y_peak_2h, q50),
        "mae_twin_physics_only_mgdl": mean_absolute_error(te.y_peak_2h, te.twin_peak_habit_2h),
        "mae_persistence_mgdl": mean_absolute_error(te.y_peak_2h, te.cgm_now),
        "interval_80_coverage": float(((te.y_peak_2h >= q10) & (te.y_peak_2h <= q90)).mean()),
        "conformal_margin_mgdl": qhat,
    }
    print("Peak forecast:", {k: round(v, 2) for k, v in forecast.items()})

    # ---- Explainability ----
    clf = models[best]
    sample = te[cols].sample(min(6000, len(te)), random_state=SEED)
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(sample)
    sv = sv[1] if isinstance(sv, list) else sv
    mean_abs = pd.Series(np.abs(sv).mean(0), index=cols)
    by_group = mean_abs.groupby(lambda c: GROUP_NAMES[c.split("_")[0]]).sum().sort_values()
    top_features = mean_abs.sort_values(ascending=False).head(15)

    # ---- Save artefacts ----
    out = {"split": {"train_patients": len(train_p), "val_patients": len(val_p), "test_patients": len(test_p),
                     "train_points": len(tr), "test_points": len(te),
                     "spike_prevalence_test": float(te.y_spike.mean())},
           "ablation_test": results, "cross_validation_5fold": cv, "peak_forecast": forecast,
           "shap_by_stream": by_group.round(4).to_dict(),
           "top_features": top_features.round(4).to_dict()}
    with open(OUTPUTS / "metrics.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    clf.booster_.save_model(str(OUTPUTS / "glucotwin_spike_model.txt"))
    for q, qm in q_models.items():
        qm.booster_.save_model(str(OUTPUTS / f"glucotwin_peak_q{int(q * 100)}.txt"))
    with open(OUTPUTS / "test_patients.json", "w") as f:
        json.dump({"test": list(test_p), "features": cols, "threshold": thr, "conformal_margin": qhat}, f)

    make_figures(te, preds, results, by_group, top_features, thr, cv)
    return out


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=PALETTE["ink"])
    ax.grid(alpha=0.2)


def make_figures(te, preds, results, by_group, top_features, thr, cv):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    # ROC ladder
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = [PALETTE["grey"], "#B8C5C7", "#A7D7C5", PALETTE["mint"], PALETTE["sand"], PALETTE["teal"]]
    for (name, p), c in zip(preds.items(), colors):
        fpr, tpr, _ = roc_curve(te.y_spike, p)
        lw = 3 if "GlucoTwin" in name else 1.6
        ax.plot(fpr, tpr, color=c if "GlucoTwin" not in name else PALETTE["coral"], lw=lw,
                label=f"{name} (AUROC {results[name]['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#ccc")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Fused GlucoTwin vs single-stream baselines\n(2-hour glucose spike, unseen test patients)",
                 color=PALETTE["ink"], fontsize=12)
    ax.legend(fontsize=8.5, loc="lower right", frameon=False); _style(ax)
    fig.tight_layout(); fig.savefig(FIGURES / "roc_ablation.png", dpi=180); plt.close(fig)

    # CV ablation ladder
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    names = list(cv.keys())
    means = [cv[n]["auroc_mean"] for n in names]
    sds = [cv[n]["auroc_sd"] for n in names]
    cols_ = [PALETTE["grey"]] * (len(names) - 1) + [PALETTE["coral"]]
    ax.barh(names[::-1], means[::-1], xerr=sds[::-1], color=cols_[::-1], capsize=3)
    for i, v in enumerate(means[::-1]):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=9, color=PALETTE["ink"])
    ax.set_xlim(0.6, 0.97); ax.set_xlabel("AUROC, 5-fold patient-grouped cross-validation")
    ax.set_title("Ablation: value of each data stream", color=PALETTE["ink"], fontsize=12)
    _style(ax); fig.tight_layout(); fig.savefig(FIGURES / "cv_ablation.png", dpi=180); plt.close(fig)

    # Stream contribution
    fig, ax = plt.subplots(figsize=(7, 4.2))
    cols = [PALETTE["coral"] if n == "Digital twin" else (PALETTE["teal"] if n.startswith("EHR") else PALETTE["grey"])
            for n in by_group.index]
    ax.barh(by_group.index, by_group.values, color=cols)
    ax.set_xlabel("Mean |SHAP| (contribution to prediction)")
    ax.set_title("What drives the prediction, by data stream", color=PALETTE["ink"], fontsize=12)
    _style(ax); fig.tight_layout(); fig.savefig(FIGURES / "shap_by_stream.png", dpi=180); plt.close(fig)

    # Calibration
    fig, ax = plt.subplots(figsize=(5.5, 5))
    pt, pp = calibration_curve(te.y_spike, preds["GlucoTwin (fusion + twin)"], n_bins=10)
    ax.plot([0, 1], [0, 1], ls="--", color="#ccc"); ax.plot(pp, pt, "o-", color=PALETTE["coral"], lw=2)
    ax.set_xlabel("Predicted spike probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration (test patients)", color=PALETTE["ink"], fontsize=12); _style(ax)
    fig.tight_layout(); fig.savefig(FIGURES / "calibration.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
