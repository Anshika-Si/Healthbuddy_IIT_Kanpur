"""Fast unit tests (about 10 s). Run with:  python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.data.generate_ehr import generate_ehr
from src.data.generate_wearables import simulate_patient, STEPS_PER_DAY
from src.config import N_DAYS
from src.twin.glucotwin import TwinParams, meal_shape, simulate, calibrate


@pytest.fixture(scope="module")
def ehr():
    return generate_ehr(n=6, seed=1)


@pytest.fixture(scope="module")
def wear(ehr):
    rng = np.random.default_rng(0)
    return simulate_patient(ehr.iloc[0], rng)


def twin(**kw):
    base = dict(patient_id="T", gb=130, k=0.006, cs=1.3, tpk=55, beta_sleep=0.1, walk_effect=0.25,
                meal_hist=[0.1] * 24, meal_carbs_hist=[50] * 24, n_meals_used=10)
    base.update(kw)
    return TwinParams(**base)


def test_ehr_schema_and_ranges(ehr):
    assert ehr.patient_id.is_unique
    assert ehr.hba1c_pct.between(5.7, 12.5).all()
    assert ehr.bmi.between(19, 40).all()


def test_wearables_physiological(wear):
    assert len(wear) == N_DAYS * STEPS_PER_DAY
    assert wear.glucose_cgm.between(40, 400).all()
    assert wear.heart_rate.between(30, 200).all()
    assert set(wear.sleep_stage.unique()) <= {0, 1, 2, 3}
    daily_steps = wear.groupby("day").steps.sum()
    assert daily_steps.between(1000, 40000).all()


def test_meal_shape_peaks_at_tpk():
    t = np.arange(0, 300)
    assert abs(int(t[np.argmax(meal_shape(t, 55))]) - 55) <= 1
    assert meal_shape(np.array([0.0]), 55)[0] == 0


def test_twin_returns_to_basal_without_meals():
    traj = simulate(twin(), 200, [], horizon_min=600)
    assert abs(traj[-1] - 130) < abs(traj[0] - 130)


def test_walking_lowers_and_short_sleep_raises_peak():
    p = twin()
    usual = simulate(p, 130, [], 180, sleep_h=7, extra_meal=(10, 80)).max()
    walked = simulate(p, 130, [], 180, sleep_h=7, extra_meal=(10, 80), walk_min=20).max()
    tired = simulate(p, 130, [], 180, sleep_h=4.5, extra_meal=(10, 80)).max()
    assert walked < usual < tired


def test_calibration_recovers_sensible_params(wear):
    p = calibrate(wear.drop(columns=["_carbs_true", "_si_day"]))
    assert 60 < p.gb < 300 and 0.3 < p.cs < 5 and 20 < p.tpk < 150
    assert len(p.meal_hist) == 24


def test_synthea_adapter(tmp_path):
    from src.data.synthea_adapter import load_synthea
    pd.DataFrame({"Id": ["abc123456", "zzz999999"], "BIRTHDATE": ["1975-01-01", "1990-01-01"],
                  "GENDER": ["F", "M"], "CITY": ["Pune", "Delhi"]}).to_csv(tmp_path / "patients.csv", index=False)
    pd.DataFrame({"START": ["2015-01-01"], "STOP": [""], "PATIENT": ["abc123456"], "ENCOUNTER": ["e"],
                  "CODE": ["44054006"], "DESCRIPTION": ["Diabetes mellitus type 2"]}).to_csv(tmp_path / "conditions.csv", index=False)
    pd.DataFrame({"DATE": ["2025-01-01", "2026-01-01"], "PATIENT": ["abc123456"] * 2, "ENCOUNTER": ["e"] * 2,
                  "CODE": ["4548-4", "4548-4"], "DESCRIPTION": ["HbA1c"] * 2, "VALUE": ["7.1", "8.2"],
                  "UNITS": ["%"] * 2, "TYPE": ["numeric"] * 2}).to_csv(tmp_path / "observations.csv", index=False)
    pd.DataFrame({"START": ["2016-01-01"], "STOP": [""], "PATIENT": ["abc123456"], "ENCOUNTER": ["e"],
                  "CODE": ["860975"], "DESCRIPTION": ["Metformin 500 MG"]}).to_csv(tmp_path / "medications.csv", index=False)
    df = load_synthea(tmp_path)
    assert len(df) == 1                       # only the T2D patient
    assert df.hba1c_pct.iloc[0] == 8.2        # latest lab value
    assert df.med_metformin.iloc[0] == 1
