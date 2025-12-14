import os
import sys
import json
import time
from pathlib import Path

import pandas as pd
import requests
from ultralytics import YOLO


# ---------------------------
# SAFE DIRECTORY HANDLING
# ---------------------------
def ensure_dir(path: Path):
    if path.exists():
        if path.is_file():
            raise RuntimeError(f"Path exists as a FILE, expected directory: {path}")
    else:
        path.mkdir(parents=True, exist_ok=True)


# ---------------------------
# USER INPUT HELPERS
# ---------------------------
def ask_excel_path():
    while True:
        excel_path = input("Enter path to Excel file (.xlsx): ").strip().strip('"')
        if not excel_path:
            print("❌ Excel path cannot be empty.")
            continue
        p = Path(excel_path)
        if p.exists() and p.suffix.lower() == ".xlsx":
            return p
        print("❌ Invalid Excel file. Please provide a valid .xlsx file.")


def ask_api_key():
    while True:
        key = input("Enter Google Static Maps API key: ").strip()
        if key:
            return key
        print("❌ API key cannot be empty.")


# ---------------------------
# LOAD & VALIDATE EXCEL
# ---------------------------
def load_coordinates(excel_path: Path):
    df = pd.read_excel(excel_path)

    df.columns = [c.lower().strip() for c in df.columns]

    if "lat" not in df.columns or "lon" not in df.columns:
        raise RuntimeError("Excel must contain columns named 'lat' and 'lon'")

    coords = list(zip(df["lat"], df["lon"]))
    if not coords:
        raise RuntimeError("Excel file contains no coordinates.")

    return coords


# ---------------------------
# DOWNLOAD SATELLITE IMAGE
# ---------------------------
def fetch_satellite_image(lat, lon, api_key, save_path):
    url = (
        "https://maps.googleapis.com/maps/api/staticmap"
        f"?center={lat},{lon}"
        "&zoom=20"
        "&size=640x640"
        "&maptype=satellite"
        f"&key={api_key}"
    )

    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        raise RuntimeError("Failed to download satellite image")

    with open(save_path, "wb") as f:
        f.write(r.content)


# ---------------------------
# MAIN INFERENCE PIPELINE
# ---------------------------
def main():
    print("\n=== Rooftop Solar Detection Inference ===\n")

    excel_path = ask_excel_path()
    api_key = ask_api_key()

    model_path = Path("Trained_model/best.pt")
    if not model_path.exists():
        raise RuntimeError("Model file not found at Trained_model/best.pt")

    # Output structure
    output_dir = Path("outputs")
    overlays_dir = output_dir / "overlays"
    json_dir = output_dir / "json"
    images_dir = output_dir / "images"

    ensure_dir(output_dir)
    ensure_dir(overlays_dir)
    ensure_dir(json_dir)
    ensure_dir(images_dir)

    print("✔ Output directories ready")

    coords = load_coordinates(excel_path)
    print(f"✔ Loaded {len(coords)} locations from Excel")

    print("✔ Loading YOLO model...")
    model = YOLO(str(model_path))

    results_json = []

    for idx, (lat, lon) in enumerate(coords, start=1):
        print(f"→ Processing location {idx}/{len(coords)} ({lat}, {lon})")

        image_path = images_dir / f"input_{idx}.png"
        overlay_path = overlays_dir / f"overlay_{idx}.png"

        fetch_satellite_image(lat, lon, api_key, image_path)

        results = model.predict(
            source=str(image_path),
            conf=0.25,
            save=False
        )

        detections = []
        for box in results[0].boxes:
            detections.append({
                "x1": float(box.xyxy[0][0]),
                "y1": float(box.xyxy[0][1]),
                "x2": float(box.xyxy[0][2]),
                "y2": float(box.xyxy[0][3]),
                "confidence": float(box.conf[0])
            })

        # Save overlay
        results[0].save(filename=str(overlay_path))

        results_json.append({
            "latitude": lat,
            "longitude": lon,
            "detections": detections
        })

    json_path = json_dir / "inference_results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)

    print("\n✅ Inference completed successfully!")
    print("\n📁 Outputs saved to:")
    print(f"  Overlays : {overlays_dir.resolve()}")
    print(f"  JSON     : {json_path.resolve()}")
    print(f"  Images   : {images_dir.resolve()}")


# ---------------------------
# ENTRY POINT
# ---------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
