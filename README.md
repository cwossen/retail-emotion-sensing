# Retail Emotion Sensing

> A privacy-preserving computer-vision pipeline that reads how shoppers feel in a store and turns it into merchandising signals — without recording who they are.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)

![Demo](demo.gif)

*Live detection, multi-person tracking, and a per-person emotion read on in-store footage. Each face is swapped for a synthetic proxy at capture; the track IDs and emotion labels are the system's own output.*

---

## What it does

Most retail analytics count footfall. This reads **affect**. For each shopper it tracks, it estimates an emotion, maps that to an engagement signal, and attributes the signal to the merchandising zone the shopper is standing in — then surfaces a recommendation for staff. Faces never leave the camera in recognizable form: they are replaced with a synthetic proxy on-site, and only anonymous per-track summaries are stored.

It is grounded in the **S-A-O-R** (Stimulus–Appraisal–Organism–Response) framework, an embodied perception–cognition–behavior model for retail emotion sensing (paper under submission to *IEEE Access*, 2026).

## Architecture

```mermaid
flowchart LR
    A[RTSP camera / video] --> B[YOLOv8-face detect<br/>+ ByteTrack track]
    B --> C[DeepFace emotion<br/>per face]
    B --> D[FaceAnonymizer<br/>synthetic-proxy swap]
    C --> E[(SQLite<br/>detections · sessions · embeddings)]
    D --> F[Annotated frames / live feed]
    E --> G[Rust verifier<br/>DB integrity]
    E --> H[classifier<br/>emotion → signal → zone score]
    H --> I[Streamlit dashboard<br/>+ recommendation]
    E --> J[PySpark → Delta Lake<br/>dwell · emotion · revisit]
```

The loop is the point: perception (who is here, how do they look), cognition (what does it mean for engagement), behavior (what should staff do), with each stage feeding the store's picture of itself.

## Pipeline components

| Stage | What it does | Tech |
|---|---|---|
| Perception | Face detection, multi-person tracking, re-identification for revisits | YOLOv8-face, ByteTrack, DeepFace embeddings |
| Emotion | Per-face expression read | DeepFace (7-class FER) |
| Privacy | Irreversible synthetic-proxy face swap at capture | OpenCV `seamlessClone` |
| Store | Per-track detections, sessions, embeddings | SQLite |
| Integrity | Dataset verification pass over the DB | Rust |
| Cognition | Emotion → engagement signal → per-zone score → recommendation | Python |
| Analytics | Dwell, emotion distribution, and revisits by zone | PySpark, Delta Lake |
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
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -r requirements.txt
```

### Run the dashboard
```bash
streamlit run app.py
```
Use **Run Analysis** for a simulated session, **Upload a test video** to process a clip, or the **Live** controls to stream from cameras.

### Render an annotated clip
`render_demo.py` re-runs detection + tracking on a video and writes an annotated MP4 (boxes, track IDs, emotion labels, proxy-masked faces). Useful for demos and for eyeballing the pipeline:
```bash
python render_demo.py -i input.mp4 -o demo_annotated.mp4
python render_demo.py -i input.mp4 -o out.mp4 --no-anonymize   # faces visible
python render_demo.py -i input.mp4 -o out.mp4 --show-category   # add zone labels
```

### Live RTSP (credentials via environment, never hard-coded)
```bash
export CAM1_URL="rtsp://user:pass@host:554/stream"
export CAM2_URL="rtsp://user:pass@host:554/stream"
streamlit run app.py
```

## Privacy by design
- **Synthetic-proxy substitution at capture.** Detected faces are overwritten with a non-existent AI proxy before any frame is displayed or saved.
- **On-premise processing.** Inference runs locally; no video is sent off-site.
- **Anonymous aggregates only.** The database stores per-track summaries (emotion, dwell, zone, revisit), not identities.

*Known limit:* masking is detection-gated. A face too small (far from the camera) or turned away may not be detected, and therefore is not masked. Tightening this coverage is on the roadmap.

## Status & limitations
This is a **research prototype**, not a production system.
- Emotion uses DeepFace's 7-class FER; on low-resolution CCTV it skews toward *neutral* and *sad*.
- The emotion → signal mapping is a heuristic, not yet the full appraisal model.
- The S-A-O-R appraisal model and its uncertainty-gated agent are the research direction (the paper), not the current production path.
- No empirical accuracy claims yet — validation is the next phase.

## Roadmap
- [ ] Validate the appraisal model on real in-store footage
- [ ] Incremental-value test: appraisal vs. a direct classifier vs. a plain S-O-R baseline
- [ ] Replace the heuristic emotion→signal map with the S-A-O-R appraisal + uncertainty gate
- [ ] Tighten anonymization coverage (lower detection threshold, blur fallback for small faces)
- [ ] Fairness and calibration across groups before any deployment

## Tech stack
Python · OpenCV · Ultralytics YOLOv8 · ByteTrack · DeepFace · SQLite · Rust · PySpark · Delta Lake · Streamlit

## License
MIT — see [`LICENSE`](LICENSE).

## Author
**Caleb A. Wossen** — MS, Computer Science & Engineering, Sejong University, Seoul.
Open to applied computer-vision and forward-deployed engineering roles.
`cwossen2@gmail.com`
