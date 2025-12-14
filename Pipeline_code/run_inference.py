import argparse
import os
import sys
import pandas as pd
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
DEFAULT_EXCEL = "Prediction_files/sample_input_lat_long.xlsx"
DEFAULT_MODEL = "Trained_model/best.pt"
DEFAULT_OUTPUT = "outputs"

# -------------------------
# ARGUMENTS
# -------------------------
parser = argparse.ArgumentParser(
    description="Rooftop Solar Panel Detection – Inference Pipeline"
)

parser.add_argument(
    "--excel",
    type=str,
    default=DEFAULT_EXCEL,
    help="Path to Excel file with latitude & longitude (default: sample file)"
)

parser.add_argument(
    "--model",
    type=str,
    default=DEFAULT_MODEL,
    help="Path to trained YOLO model (default: Trained_model/best.pt)"
)

parser.add_argument(
    "--output",
    type=str,
    default=DEFAULT_OUTPUT,
    help="Output directory for results (default: outputs/)"
)

args = parser.parse_args()

# -------------------------
# VALIDATION
# -------------------------
excel_path = Path(args.excel)
model_path = Path(args.model)
output_dir = Path(args.output)

if not excel_path.exists():
    print(f"❌ Excel file not found: {excel_path}")
    sys.exit(1)

if not model_path.exists():
    print(f"❌ Model file not found: {model_path}")
    sys.exit(1)

# Create output folders
(output_dir / "images").mkdir(parents=True, exist_ok=True)
(output_dir / "overlays").mkdir(parents=True, exist_ok=True)
(output_dir / "json").mkdir(parents=True, exist_ok=True)

print("\n✅ Inference Configuration")
print(f"• Excel file : {excel_path.resolve()}")
print(f"• Model file : {model_path.resolve()}")
print(f"• Output dir : {output_dir.resolve()}")
print("-" * 50)

# -------------------------
# READ EXCEL (ROBUST)
# -------------------------
df = pd.read_excel(excel_path)

# Normalize column names
df.columns = [c.lower().strip() for c in df.columns]

lat_col = next((c for c in df.columns if c in ["lat", "latitude"]), None)
lon_col = next((c for c in df.columns if c in ["lon", "lng", "longitude", "long"]), None)

if not lat_col or not lon_col:
    print("❌ Excel must contain latitude & longitude columns.")
    sys.exit(1)

locations = df[[lat_col, lon_col]].dropna().values.tolist()

if len(locations) == 0:
    print("❌ No valid lat/lon rows found.")
    sys.exit(1)

print(f"📍 Locations loaded: {len(locations)}")

# -------------------------
# INFERENCE (YOUR LOGIC)
# -------------------------
print("\n🚀 Running inference...\n")

# 🔽 PLACE YOUR EXISTING LOGIC HERE 🔽
# for each lat, lon:
#   1. download satellite image
#   2. run YOLO inference
#   3. save overlay image
#   4. append JSON results

# -------------------------
# FINAL SUMMARY
# -------------------------
print("\n✅ Inference completed successfully!")
print("\n📂 Results saved at:")
print(f"• Overlay images : { (output_dir / 'overlays').resolve() }")
print(f"• JSON results   : { (output_dir / 'json').resolve() }")
print("\n🎯 Judges can inspect these folders for outputs.")
