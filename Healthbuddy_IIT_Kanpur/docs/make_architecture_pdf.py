"""Generate docs/Architecture_Diagram.pdf (vector, 2 pages). Run: python docs/make_architecture_pdf.py"""
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white

OUT = Path(__file__).resolve().parent / "Architecture_Diagram.pdf"
W, H = 1280, 720
INK, INK2, MUTED, LINE = HexColor("#12343B"), HexColor("#46636A"), HexColor("#7D9296"), HexColor("#C9D6D8")
TEAL, TEAL_S = HexColor("#0F7C7A"), HexColor("#D8EEEC")
CORAL, CORAL_S = HexColor("#E4572E"), HexColor("#FBE3DA")
SAND_S, BG = HexColor("#F8EDCF"), HexColor("#F3F6F6")


def box(c, x, y, w, h, title, lines, fill=white, stroke=LINE, title_col=INK, fs=11.5):
    c.setFillColor(fill); c.setStrokeColor(stroke); c.setLineWidth(1.2)
    c.roundRect(x, y, w, h, 10, fill=1, stroke=1)
    c.setFillColor(title_col); c.setFont("Helvetica-Bold", fs + 1.5)
    c.drawString(x + 12, y + h - 22, title)
    c.setFont("Helvetica", fs); c.setFillColor(INK2)
    yy = y + h - 40
    for ln in lines:
        c.drawString(x + 12, yy, ln); yy -= fs + 4.5


def arrow(c, x1, y1, x2, y2, col=INK2, lw=1.6, label=None):
    import math
    c.setStrokeColor(col); c.setFillColor(col); c.setLineWidth(lw)
    c.line(x1, y1, x2, y2)
    a = math.atan2(y2 - y1, x2 - x1); s = 8
    p = c.beginPath()
    p.moveTo(x2, y2)
    p.lineTo(x2 - s * math.cos(a - 0.4), y2 - s * math.sin(a - 0.4))
    p.lineTo(x2 - s * math.cos(a + 0.4), y2 - s * math.sin(a + 0.4))
    p.close(); c.drawPath(p, fill=1, stroke=0)
    if label:
        c.setFont("Helvetica", 9.5); c.setFillColor(MUTED)
        c.drawCentredString((x1 + x2) / 2, (y1 + y2) / 2 + 5, label)


def column_label(c, x, w, text, n):
    c.setFillColor(INK); c.circle(x + 12, 628, 11, fill=1, stroke=0)
    c.setFillColor(white); c.setFont("Helvetica-Bold", 11); c.drawCentredString(x + 12, 624, str(n))
    c.setFillColor(INK); c.setFont("Helvetica-Bold", 14); c.drawString(x + 30, 623, text)


def page1(c):
    c.setFillColor(BG); c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(INK); c.setFont("Helvetica-Bold", 26)
    c.drawString(40, 670, "GlucoTwin: system architecture")
    c.setFont("Helvetica", 13); c.setFillColor(INK2)
    c.drawString(40, 648, "Type 2 diabetes digital twin that fuses EHR + wearable streams to predict a glucose spike (>180 mg/dL) up to 2 hours ahead")

    cols = [(40, 220, "Data sources"), (285, 210, "Ingest & harmonise"), (520, 250, "Digital twin engine"),
            (795, 225, "Fusion & prediction"), (1045, 200, "Clinician dashboard")]
    for i, (x, w, t) in enumerate(cols, 1):
        column_label(c, x, w, t, i)

    # 1 sources
    box(c, 40, 440, 220, 165, "Static / historical (EHR)", [
        "Demographics, diagnoses", "Labs: HbA1c, FPG, lipids, eGFR", "Medications", "Genetic marker (TCF7L2)",
        "Source: synthetic generator,", "Synthea adapter, MIMIC-IV*"], fill=TEAL_S, stroke=TEAL)
    box(c, 40, 205, 220, 215, "Dynamic / real-time", [
        "CGM glucose (5-min)", "Heart rate, HRV (RMSSD)", "Steps / activity", "Sleep stages (awake/light/deep/REM)",
        "Meal-log app (carbs, ~70% logged)", "Format mimics Apple Health /", "Google Fit / Libre exports"],
        fill=SAND_S, stroke=HexColor("#C9971B"))
    c.setFont("Helvetica-Oblique", 9.5); c.setFillColor(MUTED)
    c.drawString(40, 190, "*credentialed access; adapter pattern shown")

    # 2 ingest
    box(c, 285, 205, 210, 400, "Harmonisation layer", [
        "FHIR-style resources", "LOINC / SNOMED coding", "", "Resample all streams to", "a common 5-min grid",
        "", "Align by patient + time;", "nightly sleep windows", "", "Hide ground truth; only",
        "observable signals pass", "", "Privacy: synthetic / anon.", "data only (DPDP, HIPAA)",
        "Consent + purpose tags"])

    # 3 twin
    box(c, 520, 435, 250, 170, "A. Calibration (onboarding wk)", [
        "Learns from days 1-7 per patient:", "  Gb   basal glucose", "  k    clearance rate",
        "  CS   carb sensitivity (mg/dL/g)", "  tpk  time to meal peak", "  sleep + walking modifiers, meal routine"],
        fill=CORAL_S, stroke=CORAL, title_col=CORAL)
    box(c, 520, 205, 250, 210, "B. Forward simulator", [
        "Mechanistic, explainable ODE-style", "model: basal return + meal", "absorption curves x modifiers",
        "", "Outputs 2-h glucose trajectory", "(logged meals + expected routine)", "",
        "C. What-if: carbs, meal time,", "walk minutes, sleep hours"], fill=CORAL_S, stroke=CORAL, title_col=CORAL)
    arrow(c, 645, 435, 645, 415, CORAL)

    # 4 fusion
    box(c, 795, 435, 225, 170, "Feature fusion (57)", [
        "CGM history (10)", "Meal log (4)  Activity (4)", "HR / HRV (4)  Sleep (3)", "Time of day (2)",
        "EHR (20)", "Digital twin (10)", "(raw streams + twin outputs)"])
    box(c, 795, 205, 225, 210, "Models (LightGBM)", [
        "Spike classifier", "  -> probability in next 2 h", "Quantile regressors q10/50/90",
        "  -> peak glucose range", "Conformal calibration", "  -> 80% band holds 80%", "SHAP explainer",
        "  -> top drivers per alert"])
    arrow(c, 907, 435, 907, 415)

    # 5 dashboard
    box(c, 1045, 205, 200, 400, "Doctor view", [
        "Triage list by live risk", "", "24-h CGM + twin forecast", "with peak range", "",
        "'Why' panel (SHAP)", "", "What-if simulator", "", "Wearables + sleep", "hypnogram", "",
        "7-day trends, EHR", "", "Actions: nudge patient,", "acknowledge alert"], fill=TEAL_S, stroke=TEAL)

    # arrows between columns
    arrow(c, 260, 520, 285, 520); arrow(c, 260, 310, 285, 310)
    arrow(c, 495, 520, 520, 520); arrow(c, 495, 310, 520, 310)
    arrow(c, 770, 310, 795, 310, label=None); arrow(c, 770, 520, 795, 520)
    arrow(c, 1020, 410, 1045, 410)

    # feedback loop
    c.setStrokeColor(TEAL); c.setLineWidth(1.6); c.setDash(5, 4)
    c.line(1145, 205, 1145, 165); c.line(1145, 165, 645, 165); c.setDash()
    arrow(c, 645, 165, 645, 205, TEAL)
    c.setFont("Helvetica", 11); c.setFillColor(TEAL)
    c.drawCentredString(895, 150, "Continuous learning: new CGM + outcomes re-calibrate the twin nightly")

    # bottom strip
    c.setFillColor(white); c.setStrokeColor(LINE); c.roundRect(40, 30, 1205, 95, 10, fill=1, stroke=1)
    c.setFillColor(INK); c.setFont("Helvetica-Bold", 13)
    c.drawString(58, 100, "Evaluation protocol")
    c.drawString(640, 100, "Production path (roadmap)")
    c.setFont("Helvetica", 11); c.setFillColor(INK2)
    for i, t in enumerate(["Patient-level split: test patients never seen in training (no leakage)",
                           "5-fold GroupKFold CV, ablation per stream, trend-arrow clinical baseline",
                           "Clinical metrics: detection rate, lead time, alert episodes/day, calibration"]):
        c.drawString(58, 80 - i * 16, t)
    for i, t in enumerate(["REST API (FastAPI) + message queue for device streams",
                           "ABDM / FHIR R4 integration for hospital EHRs; on-device CGM ingestion",
                           "Prospective validation with a clinical partner before any patient use"]):
        c.drawString(640, 80 - i * 16, t)


def page2(c):
    c.setFillColor(BG); c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(INK); c.setFont("Helvetica-Bold", 26)
    c.drawString(40, 670, "How one prediction is made (every 15 minutes)")
    c.setFont("Helvetica", 13); c.setFillColor(INK2)
    c.drawString(40, 648, "Hybrid design: a physiology twin supplies interpretable structure; ML corrects what the equations miss")

    steps = [
        ("1  Read the last 2 hours", ["CGM level, slopes (15/30/60 min),", "acceleration, variability", "Steps, HR, HRV", "Meals logged in last 4 h"], SAND_S, HexColor("#C9971B")),
        ("2  Add context", ["Last night: sleep hours, deep %,", "REM %, overnight HRV vs baseline", "Time of day", "EHR: HbA1c, BMI, meds, TCF7L2..."], TEAL_S, TEAL),
        ("3  Run the twin", ["Simulate 2 h ahead with this", "patient's calibrated parameters", "Add expected meal from their", "personal routine (probability-weighted)"], CORAL_S, CORAL),
        ("4  Predict + explain", ["P(spike > 180 in 2 h)", "Peak range q10-q90 (+ conformal)", "SHAP top drivers", "Alert if P >= threshold"], white, INK2),
    ]
    x = 40
    for t, lines, f, s in steps:
        box(c, x, 455, 285, 150, t, lines, fill=f, stroke=s, fs=13)
        if x < 900:
            arrow(c, x + 285, 530, x + 305, 530)
        x += 305

    # equation panel
    c.setFillColor(white); c.setStrokeColor(LINE); c.roundRect(40, 60, 600, 360, 10, fill=1, stroke=1)
    c.setFillColor(INK); c.setFont("Helvetica-Bold", 15); c.drawString(60, 390, "Twin forward model")
    c.setFont("Courier", 12.5); c.setFillColor(INK)
    eq = ["G(t) = Gb + R0 * exp(-k t)",
          "       + SUM_meals  CS * carbs * S * W * f(t + ago)",
          "",
          "f(tau) = (tau / tpk) * exp(1 - tau / tpk)",
          "S = 1 + beta_sleep * max(0, 7 - sleep_hours)",
          "W = 1 - walk_effect * min(walk_min, 30) / 30",
          "R0 = unexplained current excess (decays)"]
    for i, ln in enumerate(eq):
        c.drawString(60, 350 - i * 24, ln)
    c.setFont("Helvetica", 11); c.setFillColor(INK2)
    c.drawString(60, 130, "Parameters are fitted per patient from the onboarding week and shrunk")
    c.drawString(60, 114, "toward population priors when data are sparse (empirical-Bayes style).")
    c.drawString(60, 92, "Validation: twin-estimated CS vs hidden true insulin sensitivity, r = -0.87")

    c.setFillColor(white); c.roundRect(660, 60, 585, 360, 10, fill=1, stroke=1)
    c.setFillColor(INK); c.setFont("Helvetica-Bold", 15); c.drawString(680, 390, "Why hybrid, not pure deep learning?")
    c.setFont("Helvetica", 12.5); c.setFillColor(INK2)
    pts = ["Works from 1 week of data per patient (small-data regime)",
           "Every parameter has clinical meaning a doctor can check",
           "Enables what-if simulation via a mechanistic core",
           "Adding the twin lifts CV AUROC 0.928 -> 0.934 over plain fusion",
           "Full model halves 2-h peak error vs persistence (24 -> 12 mg/dL)",
           "Gradient boosting + SHAP: fast on CPU, auditable, easy to deploy",
           "Deep sequence models (e.g. transformers) are a roadmap item",
           "once real multi-month CGM data is available"]
    for i, t in enumerate(pts):
        c.setFillColor(CORAL if i in (3, 4) else INK2)
        c.drawString(680, 350 - i * 30, ("- " if i != 7 else "  ") + t)


def main():
    c = canvas.Canvas(str(OUT), pagesize=(W, H))
    c.setTitle("GlucoTwin - Architecture Diagram"); c.setAuthor("Healthbuddy - IIT Kanpur")
    page1(c); c.showPage(); page2(c); c.showPage(); c.save()
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
