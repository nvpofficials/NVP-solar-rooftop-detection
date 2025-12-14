"""
Offline Inference Pipeline
EcoInnovators Ideathon – Solar Rooftop Detection
Author: NVP
"""

import os
import json
import argparse
import requests
import pandas as pd
import cv2
from ultralytics import YOLO
from datetime import datetime

# ==============================
# CONFIGURATION
# ==============================

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "AIzaSyAF71xKeFF13D1A8ZHV8foB1upZhRPR7oE")

IMAGE_SIZE = "640x640"
ZOOM_LEVEL = 20

BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
OVERLAY_DIR = os.path.join(OUTPUT_DIR, "overlays")
JSON_DIR = os.path.join(OUTPUT_DIR, "json")

COMBINED_JSON_PATH = os.path.join(JSON_DIR, "combined_predictions.json")

MODEL_PATH = os.path.join(BASE_DIR, "Trained_model", "best.pt")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

# ==============================
# HELPERS
# ==============================

def download_satellite_image(lat, lon, out_path):
    url = (
        "https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}&zoom={ZOOM_LEVEL}&size={IMAGE_SIZE}"
        f"&maptype=satellite&key={GOOGLE_MAPS_API_KEY}"
    )
    r = requests.get(url)
    with open(out_path, "wb") as f:
        f.write(r.content)

def run_detection(model, image_path, overlay_path):
    results = model(image_path, conf=0.25)
    annotated = results[0].plot()
    cv2.imwrite(overlay_path, annotated)
    return results[0]

def build_json_record(site_id, lat, lon, detections):
    has_solar = len(detections.boxes) > 0

    record = {
        "site_id": site_id,
        "latitude": lat,
        "longitude": lon,
        "has_solar": has_solar,
        "buffer_radius_sqft": 1200 if has_solar else 2400,
        "panel_count": len(detections.boxes),
        "detections": [],
        "timestamp": datetime.utcnow().isoformat()
    }

    for box in detections.boxes:
        record["detections"].append({
            "bbox_xyxy": box.xyxy.tolist()[0],
            "confidence": float(box.conf[0])
        })

    return record

# ==============================
# MAIN PIPELINE
# ==============================

def main(excel_path):
    df = pd.read_excel(excel_path)

    model = YOLO(MODEL_PATH)

    combined_results = []

    for idx, row in df.iterrows():
        site_id = row.get("site_id", idx)
        lat = row["latitude"]
        lon = row["longitude"]

        image_path = os.path.join(IMAGE_DIR, f"{site_id}.png")
        overlay_path = os.path.join(OVERLAY_DIR, f"{site_id}_overlay.png")
        json_path = os.path.join(JSON_DIR, f"{site_id}.json")

        download_satellite_image(lat, lon, image_path)

        detections = run_detection(model, image_path, overlay_path)

        record = build_json_record(site_id, lat, lon, detections)

        with open(json_path, "w") as f:
            json.dump(record, f, indent=2)

        combined_results.append(record)

    with open(COMBINED_JSON_PATH, "w") as f:
        json.dump(combined_results, f, indent=2)

    print("Inference complete.")
    print(f"Combined JSON saved to: {COMBINED_JSON_PATH}")

# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", required=True, help="Path to Excel file with latitude & longitude")
    args = parser.parse_args()

    main(args.excel)

