#!/usr/bin/env python3
"""
render_demo.py - produce an ANNOTATED demo video from the retail pipeline.

Fills the gap in face_detector.py: run_face_detector() writes rows to SQLite
but never renders an annotated video. This re-runs YOLOv8-face + ByteTrack on
an input video, reads emotion per track (DeepFace, cached for speed), optionally
swaps faces for the synthetic proxy, draws boxes + track id + emotion (+ optional
zone), and writes an MP4. It only renders; it does not touch the database.

Run inside your cv-env, from the folder that has models/ and assets/:
    python render_demo.py -i annotated_retail.mp4 -o demo_annotated.mp4
    python render_demo.py -i store_clip.mp4 -o out.mp4 --show-category
    python render_demo.py -i clip.mp4 -o out.mp4 --no-anonymize   # faces visible
"""
import argparse
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "models", "yolov8n-face-lindevs.pt")
DEFAULT_PROXY = os.path.join(HERE, "assets", "privacy", "neutral_proxy.jpeg")

# Four equal-width planogram zones (generalized from your ZONES, which were
# hard-coded for 1280px frames; here they scale to the actual frame width).
ZONE_NAMES = ["DC", "Marvel", "Indie", "Manga"]

# BGR colors per emotion, chosen to stay readable over CCTV.
EMO_COLOR = {
    "happy":    (80, 200, 80),
    "surprise": (0, 200, 255),
    "neutral":  (200, 200, 200),
    "sad":      (200, 120, 60),
    "fear":     (180, 80, 200),
    "angry":    (60, 60, 230),
    "disgust":  (60, 160, 160),
}


def load_proxy(path):
    img = cv2.imread(path) if path and os.path.exists(path) else None
    if img is None:
        print("WARNING: proxy image not found, using gray placeholder:", path)
        img = np.full((224, 224, 3), 128, dtype=np.uint8)
    return img


def apply_mask(frame, proxy, x1, y1, x2, y2):
    """Your FaceAnonymizer.apply_mask, standalone."""
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return frame
    rp = cv2.resize(proxy, (w, h))
    mask = 255 * np.ones(rp.shape, rp.dtype)
    center = (x1 + w // 2, y1 + h // 2)
    try:
        return cv2.seamlessClone(rp, frame, mask, center, cv2.NORMAL_CLONE)
    except Exception:
        frame[y1:y2, x1:x2] = rp
        return frame


def get_emotion(face_crop):
    try:
        face = cv2.resize(face_crop, (224, 224))
        res = DeepFace.analyze(face, actions=["emotion"], enforce_detection=False)
        emo = res[0]["emotion"]
        return max(emo, key=emo.get)
    except Exception:
        return "neutral"


def category_for(x_center, width):
    i = int(x_center / max(width, 1) * len(ZONE_NAMES))
    return ZONE_NAMES[min(i, len(ZONE_NAMES) - 1)]


def draw_label(frame, x1, y1, text, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_top = max(0, y1 - th - 8)
    cv2.rectangle(frame, (x1, y_top), (x1 + tw + 8, y_top + th + 8), color, -1)
    cv2.putText(frame, text, (x1 + 4, y_top + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Render an annotated demo clip.")
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--output", "-o", default="demo_annotated.mp4")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--proxy", default=DEFAULT_PROXY)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--emotion-every", type=int, default=12,
                    help="re-run DeepFace every N frames per track; cache between")
    ap.add_argument("--no-anonymize", dest="anonymize", action="store_false",
                    help="leave real faces visible (default: swap for proxy)")
    ap.add_argument("--show-category", action="store_true",
                    help="append the zone label (only meaningful on your store footage)")
    ap.set_defaults(anonymize=True)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"input not found: {args.input}")
    if not os.path.exists(args.model):
        sys.exit(f"model not found: {args.model}  (pass --model)")

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    proxy = load_proxy(args.proxy)
    model = YOLO(args.model)

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        sys.exit("could not open VideoWriter - check codec/output path")

    emo_cache = {}  # track_id -> (emotion, frame_index_last_computed)
    n = 0
    print(f"rendering {args.input}  ({width}x{height} @ {fps:.0f}fps, "
          f"anonymize={args.anonymize}) ...")

    results = model.track(args.input, stream=True, persist=True,
                          tracker="bytetrack.yaml", verbose=False)
    for r in results:
        frame = r.orig_img.copy()
        boxes = r.boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().tolist()
            for b, tid in zip(boxes, ids):
                if float(b.conf[0]) < args.conf:
                    continue
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Emotion is read from the REAL face, then cached per track.
                prev = emo_cache.get(tid)
                if prev is None or (n - prev[1]) >= args.emotion_every:
                    emo = get_emotion(crop)
                    emo_cache[tid] = (emo, n)
                else:
                    emo = prev[0]

                # Swap the face for the proxy before drawing (emotion already read).
                if args.anonymize:
                    frame = apply_mask(frame, proxy, x1, y1, x2, y2)

                color = EMO_COLOR.get(emo, (200, 200, 200))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"#{tid}  {emo}"
                if args.show_category:
                    label += f"  |  {category_for((x1 + x2) // 2, width)}"
                draw_label(frame, x1, y1, label, color)

        writer.write(frame)
        n += 1
        if n % 50 == 0:
            print(f"  {n} frames...")

    writer.release()
    print(f"done: {n} frames -> {args.output}  (anonymize={args.anonymize})")


if __name__ == "__main__":
    main()
