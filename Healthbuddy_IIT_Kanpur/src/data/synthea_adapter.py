"""
Adapter: map a Synthea CSV export onto the GlucoTwin EHR schema.

Synthea (https://github.com/synthetichealth/synthea) generates realistic synthetic
patients. Run it with CSV export enabled, e.g.

    ./run_synthea -p 500 --exporter.csv.export=true

then point this adapter at the resulting `output/csv/` folder:

    python -m src.data.synthea_adapter path/to/output/csv

It reads patients.csv, observations.csv, conditions.csv and medications.csv, keeps
patients with a Type 2 diabetes diagnosis, and extracts the latest value of each lab
by LOINC code. Fields Synthea does not model (TCF7L2 genotype, diet pattern, waist)
are imputed and flagged so the rest of the pipeline runs unchanged.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_RAW, SEED

T2D_SNOMED = {"44054006"}  # Diabetes mellitus type 2
LOINC = {"4548-4": "hba1c_pct", "2339-0": "fasting_glucose_mgdl", "39156-5": "bmi",
         "18262-6": "ldl_mgdl", "2571-8": "triglycerides_mgdl", "33914-3": "egfr",
         "8480-6": "systolic_bp"}
HTN_SNOMED = {"59621000", "38341003"}
MED_KEYWORDS = {"med_metformin": "metformin", "med_sulfonylurea": "glipizide|glimepiride|glyburide|gliclazide",
                "med_insulin": "insulin", "med_dpp4": "sitagliptin|linagliptin|vildagliptin|saxagliptin"}


def load_synthea(csv_dir) -> pd.DataFrame:
    d = Path(csv_dir)
    pats = pd.read_csv(d / "patients.csv", dtype=str)
    obs = pd.read_csv(d / "observations.csv", dtype=str)
    cond = pd.read_csv(d / "conditions.csv", dtype=str)
    meds = pd.read_csv(d / "medications.csv", dtype=str)
    rng = np.random.default_rng(SEED)

    t2d_ids = set(cond[cond.CODE.isin(T2D_SNOMED)].PATIENT)
    pats = pats[pats.Id.isin(t2d_ids)].copy()
    if pats.empty:
        raise ValueError("No Type 2 diabetes patients found in this Synthea export.")

    obs = obs[obs.CODE.isin(LOINC)].copy()
    obs["VALUE"] = pd.to_numeric(obs.VALUE, errors="coerce")
    obs = obs.sort_values("DATE").groupby(["PATIENT", "CODE"]).VALUE.last().unstack()
    obs = obs.rename(columns=LOINC)

    first_dx = cond[cond.CODE.isin(T2D_SNOMED)].groupby("PATIENT").START.min()
    htn = set(cond[cond.CODE.isin(HTN_SNOMED)].PATIENT)
    active_meds = meds[meds.STOP.isna() | (meds.STOP == "")]

    rows = []
    ref = pd.Timestamp("2026-09-01")
    for _, p in pats.iterrows():
        pid = p.Id
        o = obs.loc[pid] if pid in obs.index else pd.Series(dtype=float)
        get = lambda k, default: float(o[k]) if k in o and pd.notna(o[k]) else default
        birth = pd.to_datetime(p.BIRTHDATE)
        dx = pd.to_datetime(first_dx.get(pid, ref))
        m = active_meds[active_meds.PATIENT == pid].DESCRIPTION.str.lower().fillna("")
        bmi = get("bmi", 26.0)
        rows.append({
            "patient_id": pid[:8], "display_name": f"Synthea {pid[:6]}",
            "age": int((ref - birth).days / 365.25), "sex": p.GENDER, "city": p.get("CITY", "NA"),
            "diet_pattern": "mixed", "bmi": bmi,
            "waist_cm": round((bmi - 25) * 2.4 + (92 if p.GENDER == "M" else 86), 1),  # imputed
            "years_since_diagnosis": round(max((ref - dx).days / 365.25, 0), 1),
            "family_history_t2d": 0, "tcf7l2_risk_alleles": int(rng.binomial(2, 0.3)),  # imputed
            "hypertension": int(pid in htn), "dyslipidemia": int(get("ldl_mgdl", 110) > 130),
            "hba1c_pct": get("hba1c_pct", 7.5), "fasting_glucose_mgdl": get("fasting_glucose_mgdl", 140),
            "ldl_mgdl": get("ldl_mgdl", 115), "triglycerides_mgdl": get("triglycerides_mgdl", 160),
            "egfr": get("egfr", 90), "systolic_bp": get("systolic_bp", 128),
            **{k: int(m.str.contains(v).any()) for k, v in MED_KEYWORDS.items()},
            "imputed_fields": "waist_cm;tcf7l2_risk_alleles;diet_pattern;family_history_t2d",
            # Needed only if you also SIMULATE wearables for these patients: derive a plausible
            # insulin sensitivity from HbA1c (inverse of the relation in generate_ehr.py).
            "_latent_insulin_sensitivity": float(np.clip(1 - (get("hba1c_pct", 7.5) - 6.4) / 1.9, 0.25, 1.4)),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.data.synthea_adapter <synthea output/csv dir>")
    df = load_synthea(sys.argv[1])
    out = DATA_RAW / "ehr_patients_synthea.csv"
    df.to_csv(out, index=False)
    print(f"Mapped {len(df)} Synthea T2D patients -> {out}")
