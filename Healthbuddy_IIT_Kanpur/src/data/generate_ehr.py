"""
Synthetic Electronic Health Record (EHR) generator for an Indian Type 2 Diabetes cohort.

Produces the *static / historical* data stream required by the challenge:
demographics, past diagnoses, lab results, medications and genetic markers.

Distributions are loosely anchored to published Indian population patterns
(earlier onset and lower BMI at diagnosis than Western cohorts, high prevalence
of central obesity, common TCF7L2 risk alleles). No real patient data is used.
To use Synthea-generated records instead, see `synthea_adapter.py`, which maps
Synthea CSV exports onto the same schema.
"""
import json
import numpy as np
import pandas as pd

from src.config import DATA_RAW, N_PATIENTS, SEED

FIRST_NAMES = ["Aarav", "Vihaan", "Ananya", "Diya", "Rohan", "Priya", "Arjun", "Kavya",
               "Rahul", "Sneha", "Karthik", "Meera", "Imran", "Fatima", "Gurpreet",
               "Harleen", "Suresh", "Lakshmi", "Anil", "Pooja", "Vikram", "Nisha"]
CITIES = ["Bengaluru", "Chennai", "Hyderabad", "Mumbai", "Delhi", "Kolkata", "Pune",
          "Lucknow", "Kanpur", "Kochi", "Jaipur", "Ahmedabad"]
DIETS = ["rice_dominant", "wheat_dominant", "mixed"]


def generate_ehr(n=N_PATIENTS, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        sex = str(rng.choice(["M", "F"]))
        age = int(np.clip(rng.normal(50, 11), 28, 78))
        bmi = float(np.clip(rng.normal(26.5, 3.8), 19, 40))
        waist = float(np.clip((bmi - 25) * 2.4 + (92 if sex == "M" else 86) + rng.normal(0, 5), 68, 135))
        years_dx = float(np.clip(rng.gamma(2.0, 3.0), 0, 30))
        family_hx = int(rng.random() < 0.55)
        tcf7l2 = int(rng.binomial(2, 0.30))  # rs7903146 risk alleles (0/1/2)
        hypertension = int(rng.random() < 0.25 + 0.006 * (age - 40))
        dyslipidemia = int(rng.random() < 0.45)
        diet = str(rng.choice(DIETS, p=[0.45, 0.35, 0.20]))

        # Latent insulin sensitivity: the hidden "true physiology" the twin must infer.
        latent_si = (1.0 - 0.025 * (bmi - 24) - 0.004 * (waist - 88) - 0.012 * years_dx
                     - 0.06 * tcf7l2 - 0.05 * family_hx + rng.normal(0, 0.12))
        latent_si = float(np.clip(latent_si, 0.25, 1.4))

        hba1c = float(np.clip(6.4 + 1.9 * (1.0 - latent_si) + 0.05 * years_dx + rng.normal(0, 0.45), 5.7, 12.5))
        fpg = float(np.clip(28.7 * hba1c - 46.7 - 35 + rng.normal(0, 12), 95, 260))

        rows.append({
            "patient_id": f"P{i:04d}",
            "display_name": f"{rng.choice(FIRST_NAMES)} ({sex}, {age})",
            "age": age, "sex": sex, "city": str(rng.choice(CITIES)), "diet_pattern": diet,
            "bmi": round(bmi, 1), "waist_cm": round(waist, 1),
            "years_since_diagnosis": round(years_dx, 1),
            "family_history_t2d": family_hx, "tcf7l2_risk_alleles": tcf7l2,
            "hypertension": hypertension, "dyslipidemia": dyslipidemia,
            "hba1c_pct": round(hba1c, 1), "fasting_glucose_mgdl": round(fpg, 0),
            "ldl_mgdl": round(float(np.clip(rng.normal(118, 30), 55, 220)), 0),
            "triglycerides_mgdl": round(float(np.clip(rng.normal(165, 55) + 40 * (1 - latent_si), 60, 450)), 0),
            "egfr": round(float(np.clip(rng.normal(92 - 0.6 * (age - 40), 12), 35, 125)), 0),
            "systolic_bp": round(float(np.clip(rng.normal(128 + 12 * hypertension, 12), 100, 185)), 0),
            "med_metformin": int(rng.random() < 0.8),
            "med_sulfonylurea": int(rng.random() < (0.15 + 0.03 * years_dx)),
            "med_insulin": int(hba1c > 9.0 and rng.random() < 0.45),
            "med_dpp4": int(rng.random() < 0.25),
            # Ground truth used ONLY by the wearable simulator; never shown to the model.
            "_latent_insulin_sensitivity": round(latent_si, 3),
        })
    return pd.DataFrame(rows)


def to_fhir_like(df: pd.DataFrame) -> list:
    """Export FHIR-flavoured JSON bundles (LOINC / SNOMED coded) for interoperability demos."""
    loinc = {"hba1c_pct": ("4548-4", "%"), "fasting_glucose_mgdl": ("1558-6", "mg/dL"),
             "bmi": ("39156-5", "kg/m2"), "ldl_mgdl": ("18262-6", "mg/dL"),
             "triglycerides_mgdl": ("2571-8", "mg/dL"), "egfr": ("33914-3", "mL/min/1.73m2"),
             "systolic_bp": ("8480-6", "mm[Hg]")}
    bundles = []
    for _, r in df.iterrows():
        entries = [{"resourceType": "Patient", "id": r.patient_id, "gender": r.sex,
                    "extension": [{"url": "age", "valueInteger": int(r.age)}]},
                   {"resourceType": "Condition", "subject": r.patient_id,
                    "code": {"coding": [{"system": "http://snomed.info/sct", "code": "44054006",
                                         "display": "Type 2 diabetes mellitus"}]}}]
        for col, (code, unit) in loinc.items():
            entries.append({"resourceType": "Observation", "subject": r.patient_id,
                            "code": {"coding": [{"system": "http://loinc.org", "code": code}]},
                            "valueQuantity": {"value": float(r[col]), "unit": unit}})
        bundles.append({"resourceType": "Bundle", "type": "collection", "entry": entries})
    return bundles


def main():
    df = generate_ehr()
    df.to_csv(DATA_RAW / "ehr_patients.csv", index=False)
    with open(DATA_RAW / "ehr_fhir_bundles_sample.json", "w") as f:
        json.dump(to_fhir_like(df.head(5)), f, indent=1)
    print(f"EHR: {len(df)} patients -> {DATA_RAW / 'ehr_patients.csv'}")
    return df


if __name__ == "__main__":
    main()
