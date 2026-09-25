// Reference script used to build docs/GlucoTwin_Presentation.pptx (pptxgenjs). Paths are from the authoring environment; adjust R and image paths to rebuild locally.
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const md = require("react-icons/md");
const fs = require("fs");

const R = "/home/claude/repo";
const M = JSON.parse(fs.readFileSync(`${R}/outputs/metrics.json`));
const TV = JSON.parse(fs.readFileSync(`${R}/outputs/twin_validation.json`));
const best = M.ablation_test["GlucoTwin (fusion + twin)"];
const cv = M.cross_validation_5fold;

const C = { ink: "12343B", ink2: "46636A", muted: "7D9296", line: "DCE5E6", bg: "F3F6F6", white: "FFFFFF",
  teal: "0F7C7A", tealS: "D8EEEC", coral: "E4572E", coralS: "FBE3DA", sand: "C9971B", sandS: "F8EDCF" };
const F = "Calibri";

async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9"; // 10 x 5.625
  pres.title = "GlucoTwin - Digital Twin Challenge 2026";
  pres.author = "[TEAM NAME]";

  const I = {
    fuse: await icon(md.MdCallMerge, C.white), sim: await icon(md.MdAutoGraph, C.white),
    explain: await icon(md.MdPsychology, C.white), heart: await icon(md.MdMonitorHeart, C.white),
    db: await icon(md.MdStorage, C.white), watch: await icon(md.MdWatch, C.white),
    lock: await icon(md.MdLock, C.white), warn: await icon(md.MdWarningAmber, C.white),
    hosp: await icon(md.MdLocalHospital, C.white), code: await icon(md.MdCode, C.white),
    person: await icon(md.MdPersonOutline, C.white), bolt: await icon(md.MdBolt, C.white),
  };
  const iconCircle = (s, img, x, y, d, fill) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
    s.addImage({ data: img, x: x + d * 0.22, y: y + d * 0.22, w: d * 0.56, h: d * 0.56 });
  };
  const title = (s, t, sub) => {
    s.addText(t, { x: 0.5, y: 0.3, w: 9, h: 0.6, fontFace: F, fontSize: 28, bold: true, color: C.ink, margin: 0, valign: "top", isTextBox: true });
    if (sub) s.addText(sub, { x: 0.5, y: 0.88, w: 9, h: 0.35, fontFace: F, fontSize: 14, color: C.ink2, margin: 0, valign: "top", isTextBox: true });
  };
  const foot = (s, n) => s.addText(`GlucoTwin · Digital Twin Challenge 2026 · ${n}`, { x: 0.5, y: 5.28, w: 9, h: 0.25, fontFace: F, fontSize: 9, color: C.muted, margin: 0, valign: "top", isTextBox: true });
  const card = (s, x, y, w, h, fill) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill || C.white }, line: { color: C.line, width: 0.75 } });

  // 1. Title
  let s = pres.addSlide(); s.background = { color: C.ink };
  s.addShape(pres.shapes.OVAL, { x: 0.5, y: 0.55, w: 0.62, h: 0.62, fill: { color: C.ink }, line: { color: "4DB8B0", width: 3 } });
  s.addShape(pres.shapes.OVAL, { x: 0.85, y: 0.55, w: 0.62, h: 0.62, fill: { type: "none" }, line: { color: "FF7A55", width: 3 } });
  s.addText("GlucoTwin", { x: 0.5, y: 1.45, w: 9, h: 0.9, fontFace: F, fontSize: 48, bold: true, color: C.white, margin: 0, valign: "top", isTextBox: true });
  s.addText("A Type 2 diabetes digital twin that warns of a glucose spike a median 90 minutes before it happens",
    { x: 0.5, y: 2.35, w: 8.2, h: 0.95, fontFace: F, fontSize: 20, color: "CFE3E1", margin: 0, valign: "top", isTextBox: true });
  s.addText([
    { text: "Team: [TEAM NAME]", options: { breakLine: true } },
    { text: "[College / Incubator name]", options: { breakLine: true } },
    { text: "Happiest Health Digital Twin Challenge 2026 · Phase 1 submission" }],
    { x: 0.5, y: 3.9, w: 8, h: 1.0, fontFace: F, fontSize: 14, color: "AFC3C6", margin: 0, valign: "top", isTextBox: true });
  s.addNotes("Introduce the team. One-line pitch: GlucoTwin is a personalised virtual patient for Type 2 diabetes that fuses EHR and wearable data to predict a glucose spike two hours ahead, and lets a doctor test interventions on the twin before advising the real patient.");

  // 2. Problem
  s = pres.addSlide(); s.background = { color: C.white };
  title(s, "Diabetes care in India is still reactive", "Quarterly HbA1c tests average away the daily spikes that drive complications");
  const stats = [["101 M", "Indians living with diabetes (ICMR-INDIAB, 2023)"], ["136 M", "more with prediabetes (same study)"], ["~3 months", "typical gap between HbA1c checks, blind to daily excursions"]];
  stats.forEach(([n, l], i) => {
    const x = 0.5 + i * 3.05;
    card(s, x, 1.55, 2.85, 1.75, i === 2 ? C.coralS : C.tealS);
    s.addText(n, { x: x + 0.2, y: 1.7, w: 2.5, h: 0.8, fontFace: F, fontSize: 40, bold: true, color: i === 2 ? C.coral : C.teal, margin: 0, valign: "top", isTextBox: true });
    s.addText(l, { x: x + 0.2, y: 2.5, w: 2.5, h: 0.7, fontFace: F, fontSize: 13, color: C.ink2, margin: 0, valign: "top", isTextBox: true });
  });
  s.addText([
    { text: "Post-meal glucose spikes are common with carbohydrate-heavy Indian meals and late dinners.", options: { bullet: true, breakLine: true } },
    { text: "Short sleep and stress quietly raise insulin resistance the next day, and nobody connects the dots.", options: { bullet: true, breakLine: true } },
    { text: "CGMs show where glucose is now. Doctors need to know where it is going, and what would change it.", options: { bullet: true } }],
    { x: 0.5, y: 3.55, w: 9, h: 1.5, fontFace: F, fontSize: 15, color: C.ink, paraSpaceAfter: 6, margin: 0, valign: "top", isTextBox: true });
  foot(s, 2);
  s.addNotes("Source for 101 million and 136 million: ICMR-INDIAB national study, Lancet Diabetes & Endocrinology, 2023. Frame the gap: we measure HbA1c every few months, but the damage happens in daily spikes.");

  // 3. Use case
  s = pres.addSlide(); s.background = { color: C.bg };
  title(s, "Our focus: predict the next post-meal spike", "One condition, one outcome, done well, as the challenge brief asks");
  card(s, 0.5, 1.45, 4.3, 3.6);
  s.addText("Prediction target", { x: 0.75, y: 1.6, w: 3.9, h: 0.4, fontFace: F, fontSize: 16, bold: true, color: C.ink, margin: 0, valign: "top", isTextBox: true });
  s.addText([
    { text: "Condition: Type 2 diabetes", options: { bullet: true, breakLine: true } },
    { text: "Event: CGM glucose > 180 mg/dL (ADA post-meal target)", options: { bullet: true, breakLine: true } },
    { text: "Horizon: any time in the next 2 hours", options: { bullet: true, breakLine: true } },
    { text: "Only predicted while glucose is still below 180, so every alert is genuine early warning", options: { bullet: true, breakLine: true } },
    { text: "Refreshed every 15 minutes", options: { bullet: true } }],
    { x: 0.75, y: 2.05, w: 3.9, h: 2.8, fontFace: F, fontSize: 14, color: C.ink2, paraSpaceAfter: 6, margin: 0, valign: "top", isTextBox: true });
  const users = [[I.person, C.teal, "Patient", "Gets a timely nudge: \u201Ca 15-minute walk after lunch will likely keep you in range.\u201D"],
    [I.hosp, C.coral, "Doctor", "Sees a triage list of who is heading out of range, why, and what would help."],
    [I.heart, C.ink, "Care programme", "Tracks time-in-range and targets coaching where it moves outcomes."]];
  users.forEach(([img, col, h, t], i) => {
    const y = 1.45 + i * 1.22;
    card(s, 5.05, y, 4.45, 1.08);
    iconCircle(s, img, 5.22, y + 0.24, 0.6, col);
    s.addText(h, { x: 6.0, y: y + 0.12, w: 3.35, h: 0.35, fontFace: F, fontSize: 15, bold: true, color: C.ink, margin: 0, valign: "top", isTextBox: true });
    s.addText(t, { x: 6.0, y: y + 0.45, w: 3.35, h: 0.58, fontFace: F, fontSize: 12, color: C.ink2, margin: 0, valign: "top", isTextBox: true });
  });
  foot(s, 3);
  s.addNotes("Explain why we narrowed to one outcome. The 180 mg/dL threshold is the ADA post-prandial target. We only predict while glucose is still in range, which is harder than predicting whether it is already high, and it is the clinically useful version.");

  // 4. Solution
  s = pres.addSlide(); s.background = { color: C.white };
  title(s, "GlucoTwin: fuse, simulate, predict", "A hybrid of a personalised physiology model and machine learning");
  const cols3 = [[I.fuse, C.sand, "1  Fuse", "Static EHR (labs, meds, genetics) and live wearables (CGM, HR, HRV, sleep, steps, meals) are aligned on one 5-minute timeline."],
    [I.sim, C.coral, "2  Simulate", "A per-patient twin learns six physiological parameters in one week, then simulates the next 2 hours and answers what-if questions."],
    [I.explain, C.teal, "3  Predict & explain", "Gradient boosting combines raw signals with the twin's simulation, outputs a calibrated risk and peak range, and shows the top reasons."]];
  cols3.forEach(([img, col, h, t], i) => {
    const x = 0.5 + i * 3.05;
    card(s, x, 1.5, 2.85, 3.5, C.bg);
    iconCircle(s, img, x + 0.25, 1.75, 0.8, col);
    s.addText(h, { x: x + 0.25, y: 2.75, w: 2.4, h: 0.45, fontFace: F, fontSize: 18, bold: true, color: C.ink, margin: 0, valign: "top", isTextBox: true });
    s.addText(t, { x: x + 0.25, y: 3.25, w: 2.4, h: 1.6, fontFace: F, fontSize: 13, color: C.ink2, margin: 0, valign: "top", isTextBox: true });
  });
  foot(s, 4);
  s.addNotes("Walk through the three stages. Emphasise the hybrid: pure deep learning needs months of data per person; our twin works from one onboarding week and every parameter means something to a clinician.");

  // 5. Data
  s = pres.addSlide(); s.background = { color: C.bg };
  title(s, "Two data streams, inside the sandbox rules", "Fully synthetic cohort; no real patient data touched (DPDP Act / HIPAA)");
  card(s, 0.5, 1.45, 4.35, 2.55, C.tealS);
  iconCircle(s, I.db, 0.7, 1.62, 0.55, C.teal);
  s.addText("Static / historical EHR", { x: 1.4, y: 1.7, w: 3.3, h: 0.4, fontFace: F, fontSize: 16, bold: true, color: C.ink, margin: 0, valign: "top", isTextBox: true });
  s.addText([
    { text: "120 synthetic Indian T2D patients", options: { bullet: true, breakLine: true } },
    { text: "Demographics, diagnoses, HbA1c, fasting glucose, lipids, eGFR, BP", options: { bullet: true, breakLine: true } },
    { text: "Medications and TCF7L2 genotype", options: { bullet: true, breakLine: true } },
    { text: "FHIR-style export, LOINC / SNOMED codes; Synthea adapter included", options: { bullet: true } }],
    { x: 0.7, y: 2.25, w: 4.0, h: 1.7, fontFace: F, fontSize: 12.5, color: C.ink2, paraSpaceAfter: 4, margin: 0, valign: "top", isTextBox: true });
  card(s, 5.15, 1.45, 4.35, 2.55, C.sandS);
  iconCircle(s, I.watch, 5.35, 1.62, 0.55, C.sand);
  s.addText("Dynamic / real-time wearables", { x: 6.05, y: 1.7, w: 3.3, h: 0.4, fontFace: F, fontSize: 16, bold: true, color: C.ink, margin: 0, valign: "top", isTextBox: true });
  s.addText([
    { text: "CGM glucose every 5 minutes, 14 days", options: { bullet: true, breakLine: true } },
    { text: "Heart rate, HRV, steps, sleep stages", options: { bullet: true, breakLine: true } },
    { text: "Meal-log app; only ~70% of meals logged, with estimation error", options: { bullet: true, breakLine: true } },
    { text: "Physiology: sleep loss, walking, stress, dawn effect, rice vs wheat diets", options: { bullet: true } }],
    { x: 5.35, y: 2.25, w: 4.0, h: 1.7, fontFace: F, fontSize: 12.5, color: C.ink2, paraSpaceAfter: 4, margin: 0, valign: "top", isTextBox: true });
  const dstats = [["483,840", "sensor rows"], ["56,744", "prediction points"], ["57", "fused features"], ["72%", "cohort time-in-range"]];
  dstats.forEach(([n, l], i) => {
    const x = 0.5 + i * 2.3;
    s.addText(n, { x, y: 4.2, w: 2.1, h: 0.55, fontFace: F, fontSize: 28, bold: true, color: C.ink, margin: 0, valign: "top", isTextBox: true });
    s.addText(l, { x, y: 4.75, w: 2.1, h: 0.3, fontFace: F, fontSize: 12, color: C.muted, margin: 0, valign: "top", isTextBox: true });
  });
  foot(s, 5);
  s.addNotes("Key credibility point: the simulator hides each patient's true insulin sensitivity. The twin never sees it and has to infer it, exactly as with a real patient. We also deliberately made the data messy: missing meal logs, logging error, sensor noise.");

  // 6. Architecture
  s = pres.addSlide(); s.background = { color: C.white };
  title(s, "Architecture");
  s.addImage({ path: "/home/claude/deck/arch-1.png", x: 0.5, y: 0.95, w: 9.0, h: 9.0 * 720 / 1280 * 0.97, sizing: { type: "contain", w: 9.0, h: 4.25 } });
  foot(s, 6);
  s.addNotes("Five layers left to right: sources, harmonisation, twin engine, fusion and prediction, dashboard. Point out the nightly recalibration loop. Full two-page diagram is in the repo as Architecture_Diagram.pdf.");

  // 7. Twin
  s = pres.addSlide(); s.background = { color: C.bg };
  title(s, "The twin learns each patient's physiology", "Six interpretable parameters, calibrated from one onboarding week");
  card(s, 0.5, 1.45, 4.2, 3.6);
  const params = [["Basal glucose", "where glucose settles overnight"], ["Carb sensitivity", "mg/dL rise per gram of carbs"], ["Time to peak", "minutes from meal to peak"],
    ["Clearance rate", "how fast glucose returns"], ["Sleep sensitivity", "extra rise per hour of lost sleep"], ["Walking benefit", "peak cut by a post-meal walk"]];
  params.forEach(([a, b], i) => {
    s.addText([{ text: a, options: { bold: true, color: C.ink, breakLine: true } }, { text: b, options: { color: C.ink2, fontSize: 11.5 } }],
      { x: 0.75, y: 1.6 + i * 0.56, w: 3.8, h: 0.52, fontFace: F, fontSize: 13.5, margin: 0, valign: "top", isTextBox: true });
  });
  s.addImage({ path: `${R}/outputs/figures/twin_recovery.png`, x: 4.95, y: 1.45, w: 4.55, h: 4.55 * 4.6 / 6, sizing: { type: "contain", w: 4.55, h: 3.5 } });
  s.addText(`Validated: the twin's carb sensitivity tracks the hidden true insulin sensitivity (r = ${TV.r_carb_sensitivity_vs_true_si.toFixed(2)}).`,
    { x: 4.95, y: 4.95, w: 4.55, h: 0.3, fontFace: F, fontSize: 11, italic: true, color: C.teal, margin: 0, valign: "top", isTextBox: true });
  foot(s, 7);
  s.addNotes("This is the digital twin in the literal sense: a small set of parameters that describe this person. The scatter shows we recover the hidden physiology from observable data alone. Estimates are shrunk to population priors when data are sparse.");

  // 8. Results
  s = pres.addSlide(); s.background = { color: C.white };
  title(s, "Results on unseen patients", `Patient-level split: ${M.split.test_patients} test patients never seen in training`);
  const rs = [[best.auroc.toFixed(3), "AUROC"], [Math.round(best.detection_rate * 100) + "%", "of spikes flagged in advance"],
    [Math.round(best.median_lead_min) + " min", "median warning time"], [M.peak_forecast.mae_glucotwin_mgdl.toFixed(0) + " mg/dL", "error on 2-h peak (vs 24 naive)"]];
  rs.forEach(([n, l], i) => {
    const y = 1.45 + i * 0.93;
    s.addText(n, { x: 0.5, y, w: 2.6, h: 0.55, fontFace: F, fontSize: 32, bold: true, color: i === 2 ? C.coral : C.ink, margin: 0, valign: "top", isTextBox: true });
    s.addText(l, { x: 0.5, y: y + 0.52, w: 2.9, h: 0.3, fontFace: F, fontSize: 12, color: C.muted, margin: 0, valign: "top", isTextBox: true });
  });
  const names = Object.keys(cv);
  s.addChart(pres.charts.BAR, [{ name: "AUROC", labels: names.map(n => n.replace(" (baseline)", "").replace("GlucoTwin (fusion + twin)", "GlucoTwin")), values: names.map(n => +cv[n].auroc_mean.toFixed(3)) }], {
    x: 3.6, y: 1.35, w: 5.9, h: 3.7, barDir: "bar", showTitle: true, title: "Ablation: AUROC, 5-fold patient-grouped CV",
    titleFontSize: 12, titleColor: C.ink, titleFontFace: F, chartColors: [C.muted, C.muted, C.muted, C.muted, C.muted, C.coral],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelColor: C.ink, dataLabelFormatCode: "0.000",
    valAxisMinVal: 0.6, valAxisMaxVal: 1.0, valAxisLabelColor: C.muted, catAxisLabelColor: C.ink, catAxisLabelFontSize: 10.5,
    valGridLine: { color: "E5ECEC", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false, catAxisOrientation: "maxMin",
  });
  foot(s, 8);
  s.addNotes(`The honest story of the ablation: stacking more raw streams helps a little (CGM only ${cv["CGM only"].auroc_mean.toFixed(3)}, fusion ${cv["Wearables + EHR fusion"].auroc_mean.toFixed(3)}), but routing them through the physiology twin gives the biggest gain (${cv["GlucoTwin (fusion + twin)"].auroc_mean.toFixed(3)}). The twin encodes interactions such as sleep loss times carb sensitivity that a tree model struggles to find from raw columns. A standard CGM trend-arrow rule scores ${cv["Trend-arrow rule (baseline)"].auroc_mean.toFixed(3)}. Also mention calibration and the conformal 80% interval which achieves ${(M.peak_forecast.interval_80_coverage*100).toFixed(0)}% coverage.`);

  // 9. Explainability
  s = pres.addSlide(); s.background = { color: C.bg };
  title(s, "Every alert comes with its reasons", "SHAP attributions, grouped by data stream and shown per prediction");
  s.addImage({ path: `${R}/outputs/figures/shap_by_stream.png`, x: 0.5, y: 1.45, w: 5.2, h: 5.2 * 4.2 / 7, sizing: { type: "contain", w: 5.2, h: 3.2 } });
  card(s, 6.0, 1.45, 3.5, 3.6);
  s.addText("Fatima, 13:00: top reasons", { x: 6.2, y: 1.6, w: 3.1, h: 0.4, fontFace: F, fontSize: 15, bold: true, color: C.ink, margin: 0, valign: "top", isTextBox: true });
  s.addText([
    { text: "Twin-simulated peak: 252 mg/dL", options: { bullet: true, breakLine: true } },
    { text: "Usual lunch window (time of day)", options: { bullet: true, breakLine: true } },
    { text: "High personal basal glucose: 171", options: { bullet: true, breakLine: true } },
    { text: "56% chance of a meal in next 2 h", options: { bullet: true } }],
    { x: 6.2, y: 2.05, w: 3.1, h: 1.7, fontFace: F, fontSize: 12.5, color: C.ink2, paraSpaceAfter: 5, margin: 0, valign: "top", isTextBox: true });
  s.addText("Her lunch was never logged, yet she peaked at 259, inside the predicted 193\u2013275 range. The twin knew her routine.",
    { x: 6.2, y: 3.95, w: 3.1, h: 0.9, fontFace: F, fontSize: 12.5, italic: true, color: C.teal, margin: 0, valign: "top", isTextBox: true });
  foot(s, 9);
  s.addNotes("The digital twin carries the most weight because it compresses EHR, sleep and meal routine into one physiologically meaningful number. Clinicians do not have to trust a black box: they see the drivers.");

  // 10. What-if
  s = pres.addSlide(); s.background = { color: C.white };
  title(s, "Ask the twin before advising the patient", `Same 85 g rice dinner, simulated on patient ${TV.whatif_patient}'s twin`);
  s.addImage({ path: `${R}/outputs/figures/whatif.png`, x: 0.5, y: 1.35, w: 5.9, h: 3.75, sizing: { type: "contain", w: 5.9, h: 3.75 } });
  const wp = TV.whatif_peaks, keys = Object.keys(wp);
  const lines = [["Usual", keys[0], C.ink], ["After a 4.5 h night", keys[1], C.coral], ["15-min walk after", keys[2], C.teal], ["Smaller portion + walk", keys[3], C.teal]];
  lines.forEach(([lab, k, col], i) => {
    const y = 1.5 + i * 0.88;
    s.addText(Math.round(wp[k]) + " mg/dL", { x: 6.7, y, w: 2.8, h: 0.48, fontFace: F, fontSize: 26, bold: true, color: col, margin: 0, valign: "top", isTextBox: true });
    s.addText("peak · " + lab, { x: 6.7, y: y + 0.46, w: 2.8, h: 0.3, fontFace: F, fontSize: 12, color: C.muted, margin: 0, valign: "top", isTextBox: true });
  });
  foot(s, 10);
  s.addNotes("This is what makes it a twin rather than just a predictor: counterfactuals. For this sleep-sensitive patient, one short night pushes the peak far above target, while a smaller portion plus a short walk keeps them in range. The same sliders are live in the dashboard.");

  // 11. Dashboard
  s = pres.addSlide(); s.background = { color: C.bg };
  title(s, "The clinician dashboard");
  s.addImage({ path: "/home/claude/deck/dash_main.png", x: 0.5, y: 0.95, w: 6.7, h: 6.7 * 820 / 1440 });
  s.addText([
    { text: "Triage list re-sorts by live risk", options: { bullet: true, breakLine: true } },
    { text: "24-h CGM with the twin's forecast and peak range", options: { bullet: true, breakLine: true } },
    { text: "Risk, reasons and one-tap patient nudge", options: { bullet: true, breakLine: true } },
    { text: "What-if sliders on the twin", options: { bullet: true, breakLine: true } },
    { text: "Replay the day to see alerts arrive in time", options: { bullet: true } }],
    { x: 7.4, y: 1.0, w: 2.15, h: 3.8, fontFace: F, fontSize: 12.5, color: C.ink, paraSpaceAfter: 8, margin: 0, valign: "top", isTextBox: true });
  s.addText("Single HTML file, works offline", { x: 0.5, y: 4.9, w: 6.7, h: 0.3, fontFace: F, fontSize: 11, italic: true, color: C.muted, margin: 0, valign: "top", isTextBox: true });
  foot(s, 11);
  s.addNotes("Demo this live in the video: drag the clinic clock, watch a patient move up the triage list before their spike, open the reasons, and use the what-if sliders.");

  // 12. Honest limits + roadmap
  s = pres.addSlide(); s.background = { color: C.white };
  title(s, "What this proves, what it doesn't, what's next");
  card(s, 0.5, 1.15, 4.35, 3.9, C.coralS);
  iconCircle(s, I.warn, 0.7, 1.32, 0.5, C.coral);
  s.addText("Limitations", { x: 1.35, y: 1.38, w: 3.3, h: 0.4, fontFace: F, fontSize: 16, bold: true, color: C.ink, margin: 0, valign: "top", isTextBox: true });
  s.addText([
    { text: "Synthetic data: metrics show the method works, not clinical accuracy", options: { bullet: true, breakLine: true } },
    { text: "The simulator and twin share a physiological family, which flatters results", options: { bullet: true, breakLine: true } },
    { text: "Relies on a CGM; meal logging is imperfect", options: { bullet: true, breakLine: true } },
    { text: "Not a medical device; decision support only", options: { bullet: true } }],
    { x: 0.7, y: 1.95, w: 4.0, h: 3.0, fontFace: F, fontSize: 13, color: C.ink2, paraSpaceAfter: 6, margin: 0, valign: "top", isTextBox: true });
  card(s, 5.15, 1.15, 4.35, 3.9, C.tealS);
  iconCircle(s, I.bolt, 5.35, 1.32, 0.5, C.teal);
  s.addText("Roadmap", { x: 6.0, y: 1.38, w: 3.3, h: 0.4, fontFace: F, fontSize: 16, bold: true, color: C.ink, margin: 0, valign: "top", isTextBox: true });
  s.addText([
    { text: "Validate on open CGM + wearable datasets, then a prospective pilot with a diabetologist", options: { bullet: true, breakLine: true } },
    { text: "ABDM / FHIR R4 integration for hospital records", options: { bullet: true, breakLine: true } },
    { text: "Add hypoglycaemia risk for insulin and sulfonylurea users", options: { bullet: true, breakLine: true } },
    { text: "Sequence models once multi-month data exists", options: { bullet: true, breakLine: true } },
    { text: "Regional-language nudges via WhatsApp", options: { bullet: true } }],
    { x: 5.35, y: 1.95, w: 4.0, h: 3.0, fontFace: F, fontSize: 13, color: C.ink2, paraSpaceAfter: 6, margin: 0, valign: "top", isTextBox: true });
  foot(s, 12);
  s.addNotes("Being upfront about limits builds trust with a clinical jury. The key limitation: on synthetic data we can validate the method but not clinical accuracy. The obvious next step is real CGM plus wearable data under proper ethics approval.");

  // 13. Close
  s = pres.addSlide(); s.background = { color: C.ink };
  s.addText("Thank you", { x: 0.5, y: 1.2, w: 9, h: 0.9, fontFace: F, fontSize: 44, bold: true, color: C.white, margin: 0, valign: "top", isTextBox: true });
  s.addText("GlucoTwin sees the spike coming, explains why, and shows what would prevent it.",
    { x: 0.5, y: 2.1, w: 8.5, h: 0.6, fontFace: F, fontSize: 18, color: "CFE3E1", margin: 0, valign: "top", isTextBox: true });
  s.addText([
    { text: "Stack: Python · LightGBM · SHAP · scikit-learn · pandas · HTML/SVG dashboard", options: { breakLine: true } },
    { text: "Code, data generator, models, dashboard: [GitHub repository link]", options: { breakLine: true } },
    { text: "Open source under the MIT License", options: { breakLine: true } },
    { text: "[Team leader name] · [email]" }],
    { x: 0.5, y: 3.2, w: 9, h: 1.5, fontFace: F, fontSize: 14, color: "AFC3C6", paraSpaceAfter: 4, margin: 0, valign: "top", isTextBox: true });
  s.addNotes("Close with the one-liner and the repo link. Invite questions.");

  await pres.writeFile({ fileName: `${R}/docs/GlucoTwin_Presentation.pptx` });
  console.log("ok");
})();
