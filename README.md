# GlucoTwin: Healthbuddy, IIT Kanpur

**Happiest Health Digital Twin Challenge 2026 · Phase 1 submission**

A Type 2 diabetes digital twin that fuses health-record and wearable data to predict a glucose spike (above 180 mg/dL) up to 2 hours ahead. It explains every alert and lets a doctor test "what if?" changes before advising the patient.

**Team:** Anshika Singh (Team Leader) and Raj Dipak Kamble, IIT Kanpur

## Quick links

| | |
|---|---|
| **Full project README** | [Healthbuddy_IIT_Kanpur/README.md](Healthbuddy_IIT_Kanpur/README.md) |
| **Demo video** | [UNLISTED YOUTUBE LINK] |
| **Presentation** | [PPTX](Healthbuddy_IIT_Kanpur/docs/GlucoTwin_Presentation.pptx) · [PDF](Healthbuddy_IIT_Kanpur/docs/GlucoTwin_Presentation.pdf) |
| **Architecture diagram** | [PDF](Healthbuddy_IIT_Kanpur/docs/Architecture_Diagram.pdf) |
| **Live dashboard** | [Open the dashboard](https://anshika-si.github.io/Healthbuddy_IIT_Kanpur/Healthbuddy_IIT_Kanpur/dashboard/GlucoTwin_Dashboard.html) |
| **Source code** | [Healthbuddy_IIT_Kanpur/src](Healthbuddy_IIT_Kanpur/src) |
| **License** | MIT ([LICENSE](Healthbuddy_IIT_Kanpur/LICENSE)) |

## Key results (unseen test patients)

- AUROC **0.934**
- **95%** of glucose spikes flagged in advance, with a median warning time of **90 minutes**
- 2-hour peak glucose predicted within **12 mg/dL** on average

All data is synthetic, in line with the challenge's data rules.

## How to run

```bash
cd Healthbuddy_IIT_Kanpur
pip install -r requirements.txt
python run_pipeline.py
```
