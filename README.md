# Retail Emotion Sensing

> A privacy-preserving computer-vision pipeline that reads how shoppers feel in a store and turns it into merchandising signals — without recording who they are.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)

![Demo](docs/demo.gif)

*Detection, multi-person tracking, and a per-person emotion read on in-store footage, with every face anonymized. The track IDs and emotion labels are the system's live output; the appraisal behind them runs in the cognition layer, off the frame.*

---

## What it does

Most retail analytics count footfall. This reads **affect**. For each shopper it tracks, it reads an emotion and attributes it to the merchandising zone the shopper is in, then surfaces a recommendation for staff. Faces never leave the camera in recognizable form: they are replaced with a synthetic proxy on-site, and only anonymous per-track summaries are stored.

It is grounded in the **S-A-O-R** (Stimulus–Appraisal–Organism–Response) framework, an embodied perception–cognition–behavior model for retail emotion sensing (paper under submission to *IEEE Access*, 2026).

## Architecture

```mermaid
flowchart LR
    A[RTSP camera / video] --> B[YOLOv8-face detect + ByteTrack track]
    B --> C[DeepFace emotion per face]
    B --> D[FaceAnonymizer synthetic-proxy swap]
    C --> E[(SQLite: detections / sessions / embeddings)]
    D --> F[Annotated frames / live feed]
    E --> G[Rust verifier: DB integrity]
    E --> H[classifier: emotion to signal to zone score]
    H --> I[Streamlit dashboard + recommendation]
    E --> J[PySpark to Delta Lake: dwell / emotion / revisit]
```

## Pipeline components

| Stage | What it does | Tech |
|---|---|---|
| Perception | Face detection, multi-person tracking, re-ID for revisits | YOLOv8-face, ByteTrack, DeepFace |
| Emotion | Per-face expression read | DeepFace (7-class FER) |
| Privacy | Irreversible synthetic-proxy face swap at capture | OpenCV seamlessClone |
| Store | Per-track detections, sessions, embeddings | SQLite |
| Integrity | Dataset verification over the DB | Rust |
| Cognition | Emotion to signal to zone score to recommendation | Python |
| Analytics | Dwell, emotion distribution, revisits by zone | PySpark, Delta Lake |
| Interface | Live dashboard, video upload, RTSP streams | Streamlit |

## Getting started

### Prerequisites
- Python 3.10+
- A YOLOv8-face model at `models/yolov8n-face-lindevs.pt`
- `ffmpeg` on your PATH (RTSP ingest)
- DeepFace downloads its own weights on first run
- *(optional)* Rust toolchain, to build the DB verifier

### Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run the dashboard
```bash
streamlit run app.py
```

### Render an annotated clip
```bash
python render_demo.py -i input.mp4 -o demo_annotated.mp4
python render_demo.py -i input.mp4 -o out.mp4 --no-anonymize   # faces visible
python render_demo.py -i input.mp4 -o out.mp4 --show-category   # add zone labels
```

### Live RTSP (credentials via environment, never hard-coded)
```bash
export CAM1_URL="rtsp://user:password@host:554/stream"
export CAM2_URL="rtsp://user:password@host:554/stream"
streamlit run app.py
```

## Privacy by design
- **Synthetic-proxy substitution at capture.** Detected faces are overwritten with a non-existent AI proxy before any frame is displayed or saved.
- **On-premise processing.** Inference runs locally; no video is sent off-site.
- **Anonymous aggregates only.** The database stores per-track summaries (emotion, dwell, zone, revisit), not identities.

*Known limit:* masking is detection-gated — a face too small or turned away may be missed. Oblique CCTV angles cut face-detection recall; a person-based head-blur covers the gap in the demo, and tightening on-camera coverage is on the roadmap.

## Status & limitations
A **research prototype**, not a production system.
- Emotion uses DeepFace's 7-class FER; on low-resolution CCTV it skews toward *neutral* and *sad*.
- The emotion-to-signal mapping is a heuristic, not yet the full appraisal model.
- The S-A-O-R appraisal model and its uncertainty-gated agent are the research direction (the paper), computed off the video — not the current production path.
- No empirical accuracy claims yet; validation is the next phase.

## Roadmap
- [ ] Validate the appraisal model on real in-store footage
- [ ] Incremental-value test: appraisal vs. a direct classifier vs. a plain S-O-R baseline
- [ ] Replace the heuristic emotion-to-signal map with the S-A-O-R appraisal + uncertainty gate
- [ ] Tighten anonymization coverage (lower detection threshold, head-blur fallback)
- [ ] Fairness and calibration across groups before any deployment

## Tech stack
Python · OpenCV · Ultralytics YOLOv8 · ByteTrack · DeepFace · SQLite · Rust · PySpark · Delta Lake · Streamlit

## License
MIT — see [`LICENSE`](LICENSE).

## Author
**Caleb A. Wossen** — MS, Computer Science & Engineering, Sejong University, Seoul.
Open to applied computer-vision and forward-deployed engineering roles.
`cwossen2@gmail.com`
