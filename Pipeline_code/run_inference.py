"""
Rooftop Solar Detection – Inference Pipeline
EcoInnovators Ideathon 2026

Author: NVP
"""

import os
import sys
import json
import time
import getpass
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
import cv2
import numpy as np
from ultralytics import YOLO


# =========================
# SAFE DIRECTORY CREATION
# =========================
BASE_OUTPUT_DIR = Path("outputs")
IMAGES_DIR = BASE_OUTPUT_DIR / "images"
OVERLAYS_DIR = BASE_OUTPUT_DIR / "overlays"
JSON_DIR = BASE_OUTPUT_DIR / "json"

for d in [IMAGES_DIR, OVERLAYS_DIR, JSON_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =========================
# USER INPUTS (INTERACTIVE)
# =========================
print("\n=== Rooftop Solar Inference ===\n")

excel_path = input("Enter path to Excel file (.xlsx): ").strip().strip('"')
if not Path(excel_path).exists():
    print("ERROR: Excel file not found.")
    sys.exit(1)

api_key = getpass.getpass("Enter Google Static Maps API key (hidden): ").strip()
if not api_key:
    print("ERROR: API key cannot be empty.")
    sys.exit(1)

model_path = Path("Trained_model/best.pt")
if not model_path.exists():
    print("ERROR: Trained model not found at Trained_model/best.pt")
    sys.exit(1)


# =========================
# LOAD MODEL
# =========================
print("\nLoading YOLO model...")
model = YOLO(str(model_path))


# =========================
# LOAD EXCEL
# =========================
df = pd.read_excel(excel_path)

required_cols = {"latitude", "longitude"}
if not required_cols.issubset(df.columns):
    print("ERROR: Excel must contain 'latitude' and 'longitude' columns.")
    sys.exit(1)

if "sample_id" not in df.columns:
    df["sample_id"] = range(1, len(df) + 1)


# =========================
# GOOGLE STATIC MAPS FETCH
# =========================
def fetch_satellite_image(lat, lon, save_path):
    url = (
        "https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}&zoom=20&size=640x640&maptype=satellite&key={api_key}"
    )
    r = requests.get(url, timeout=30)
    if r.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(r.content)
        return True
    return False


# =========================
# MAIN LOOP
# =========================
results = []

print("\nRunning inference...\n")

for _, row in df.iterrows():
    sid = row["sample_id"]
    lat = row["latitude"]
    lon = row["longitude"]

    image_path = IMAGES_DIR / f"{sid}.jpg"
    overlay_path = OVERLAYS_DIR / f"{sid}_overlay.jpg"

    ok = fetch_satellite_image(lat, lon, image_path)
    if not ok:
        qc = "NOT_VERIFIABLE"
        results.append({
            "sample_id": sid,
            "lat": lat,
            "lon": lon,
            "has_solar": False,
            "confidence": 0.0,
            "pv_area_sqm_est": 0.0,
            "buffer_radius_sqft": 1200,
            "qc_status": qc,
            "bbox_or_mask": None,
            "image_metadata": {"source": "Google Static Maps", "capture_date": None}
        })
        continue

    preds = model(str(image_path), conf=0.25)[0]

    has_solar = len(preds.boxes) > 0
    confidence = float(preds.boxes.conf.max()) if has_solar else 0.0

    # Approximate area calculation (pixel proxy)
    area_px = 0
    boxes_out = []
    for b in preds.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = b
        area_px += (x2 - x1) * (y2 - y1)
        boxes_out.append([float(x1), float(y1), float(x2), float(y2)])

    pv_area_sqm = round(area_px * 0.0004, 2)  # conservative proxy
    qc = "VERIFIABLE" if has_solar else "NOT_VERIFIABLE"

    annotated = preds.plot()
    cv2.imwrite(str(overlay_path), annotated)

    results.append({
        "sample_id": sid,
        "lat": lat,
        "lon": lon,
        "has_solar": has_solar,
        "confidence": round(confidence, 3),
        "pv_area_sqm_est": pv_area_sqm,
        "buffer_radius_sqft": 1200,
        "qc_status": qc,
        "bbox_or_mask": boxes_out,
        "image_metadata": {
            "source": "Google Static Maps",
            "capture_date": datetime.utcnow().strftime("%Y-%m-%d")
        }
    })


# =========================
# SAVE JSON
# =========================
json_path = JSON_DIR / "inference_results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n=== INFERENCE COMPLETED SUCCESSFULLY ===")
print(f"Images saved to     : {IMAGES_DIR.resolve()}")
print(f"Overlay images to   : {OVERLAYS_DIR.resolve()}")
print(f"JSON results saved  : {json_path.resolve()}")
print("\nYou may now review outputs for audit and submission.\n")
