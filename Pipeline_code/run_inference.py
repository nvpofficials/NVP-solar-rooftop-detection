import argparse
import sys
import json
import requests
from pathlib import Path

import pandas as pd
import cv2
from ultralytics import YOLO


# -----------------------------
# Helper functions
# -----------------------------

def ask_for_excel(default_path: Path) -> Path:
    print("\n=== Rooftop Solar Inference ===")
    print("Press ENTER to use the sample Excel file")
    print(f"Sample: {default_path}\n")

    user_input = input("Enter path to Excel file (.xlsx): ").strip()

    if user_input == "":
        if not default_path.exists():
            print("❌ Sample Excel not found.")
            sys.exit(1)
        return default_path

    excel_path = Path(user_input)
    if not excel_path.exists():
        print("❌ Provided Excel file does not exist.")
        sys.exit(1)

    return excel_path


def ask_for_api_key() -> str:
    print("\nGoogle Static Maps API key is required.")
    print("Your key is NOT stored anywhere.\n")

    api_key = input("Enter Google Maps API Key: ").strip()
    if api_key == "":
        print("❌ API key cannot be empty.")
        sys.exit(1)

    return api_key


def load_excel(excel_path: Path):
    df = pd.read_excel(excel_path)
    df.columns = [c.lower().strip() for c in df.columns]

    if "lat" not in df.columns or "lon" not in df.columns:
        print("❌ Excel must contain columns: lat, lon")
        sys.exit(1)

    return df


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
        print(f"⚠️ Failed to fetch image for {lat},{lon}")
        return False

    with open(save_path, "wb") as f:
        f.write(r.content)

    return True


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Rooftop Solar Detection Inference")
    parser.add_argument(
        "--excel",
        help="Path to Excel file containing lat/lon",
        required=False
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    model_path = root / "Trained_model" / "best.pt"
    sample_excel = root / "Prediction_files" / "sample_input_lat_long.xlsx"
    output_dir = root / "outputs"
    images_dir = output_dir / "images"
    overlays_dir = output_dir / "overlays"
    json_dir = output_dir / "json"

    # Safe directory creation
    for d in [output_dir, images_dir, overlays_dir, json_dir]:
        if not d.exists():
            d.mkdir(parents=True)

    if not model_path.exists():
        print("❌ Model file not found:", model_path)
        sys.exit(1)

    excel_path = Path(args.excel) if args.excel else ask_for_excel(sample_excel)
    api_key = ask_for_api_key()

    print("\nLoading model...")
    model = YOLO(str(model_path))

    print("Reading Excel...")
    df = load_excel(excel_path)

    all_results = []

    print("\nRunning inference...\n")

    for idx, row in df.iterrows():
        lat, lon = row["lat"], row["lon"]

        img_path = images_dir / f"loc_{idx}.png"
        overlay_path = overlays_dir / f"overlay_{idx}.png"

        ok = fetch_satellite_image(lat, lon, api_key, img_path)
        if not ok:
            continue

        results = model(str(img_path))[0]

        detections = []
        img = cv2.imread(str(img_path))

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(conf, 3)
                })

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite(str(overlay_path), img)

        all_results.append({
            "index": idx,
            "latitude": lat,
            "longitude": lon,
            "detections": detections
        })

    json_path = json_dir / "inference_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n✅ Inference completed successfully!")
    print(f"\n📂 Overlay images saved at:\n   {overlays_dir}")
    print(f"\n📄 JSON results saved at:\n   {json_path}")
    print("\nJudges can now inspect visual and structured outputs.")


if __name__ == "__main__":
    main()
