# Rooftop Solar Panel Detection – Judge Execution Guide

This repository contains a complete offline inference pipeline to detect rooftop solar panels
from satellite imagery using latitude and longitude input.

The trained YOLOv8 model is already provided. Judges only need to follow the steps below.

--------------------------------------------------------------------

STEP 1: Clone the repository and move into it

git clone https://github.com/nvpofficials/NVP-solar-rooftop-detection.git
cd NVP-solar-rooftop-detection

--------------------------------------------------------------------

STEP 2: Create a Python virtual environment (recommended)

python -m venv venv

Activate it:

Windows:
venv\Scripts\activate

Linux / macOS:
source venv/bin/activate

--------------------------------------------------------------------

STEP 3: Install required dependencies

pip install -r Pipeline_code/requirements.txt

--------------------------------------------------------------------

STEP 4: Prepare input Excel file

The inference script accepts ANY .xlsx file containing latitude and longitude.

Required columns (case-insensitive):
lat or latitude
lon or lng or longitude

Example format:
lat,lon
18.5204,73.8567
19.0760,72.8777

A ready sample file is already provided:
Prediction_files/sample_input_lat_long.xlsx

Judges may use this sample OR replace it with their own Excel file.

--------------------------------------------------------------------

STEP 5: Run inference (sample Excel)

python Pipeline_code/run_inference.py

--------------------------------------------------------------------

STEP 6: Run inference (custom Excel file)

python Pipeline_code/run_inference.py --excel path/to/your_excel_file.xlsx

--------------------------------------------------------------------

STEP 7: What happens during inference

For each latitude and longitude:
1. Satellite image is downloaded
2. YOLOv8 trained model is applied
3. Rooftop solar panels are detected
4. Outputs are saved automatically

--------------------------------------------------------------------

STEP 8: Output locations (printed in terminal after run)

All outputs are saved under the outputs/ folder:

outputs/images/    -> raw satellite images
outputs/overlays/  -> annotated images with detection boxes
outputs/json/      -> structured prediction results (JSON)

The script prints the exact save locations at the end.

--------------------------------------------------------------------

STEP 9: Trained model used

The trained YOLOv8 model is located at:
Trained_model/best.pt

This file is tracked using Git LFS and loads automatically.

--------------------------------------------------------------------

STEP 10: How judges can verify results

1. Open images in outputs/overlays/ to visually confirm detections
2. Inspect JSON files in outputs/json/ for confidence scores and coordinates
3. Match detections with rooftop locations in satellite imagery

If outputs are generated successfully, the pipeline is working correctly.

--------------------------------------------------------------------

END OF GUIDE
