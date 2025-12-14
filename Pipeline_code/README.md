# Inference Pipeline – Rooftop Solar Detection

This folder contains the final, judge-ready inference pipeline used to validate the trained YOLOv8 rooftop solar detection model.

The pipeline is designed to be fully reproducible, interactive, robust to any valid Excel input, and safe to run multiple times without errors.

---------------------------------------------------------------------

STEP 1: Clone the Repository

git clone https://github.com/nvpofficials/NVP-solar-rooftop-detection
cd NVP-solar-rooftop-detection

---------------------------------------------------------------------

STEP 2: Create and Activate Virtual Environment (Recommended)

Create virtual environment:
python -m venv venv

Activate it:

Windows:
venv\Scripts\activate

Linux / macOS:
source venv/bin/activate

---------------------------------------------------------------------

STEP 3: Install Required Dependencies

pip install -r Environment_details/requirements.txt

---------------------------------------------------------------------

STEP 4: Prepare Input Excel File

The inference script accepts ANY .xlsx file containing latitude and longitude.

Required columns (case-insensitive):
lat OR latitude  
lon OR lng OR longitude  

Example format:
latitude,longitude  
18.5204,73.8567  
19.0760,72.8777  

A ready sample file is already provided:
Prediction_files/sample_input_lat_long.xlsx

Judges may use this sample OR replace it with their own Excel file.

---------------------------------------------------------------------

STEP 5: Run Inference (Interactive Mode)

Run the inference script:

python Pipeline_code/run_inference.py

The script will ask interactively:

1) Enter path to Excel file (.xlsx)
Example:
Prediction_files/sample_input_lat_long.xlsx

2) Enter Google Static Maps API key
(The input is hidden for security)

---------------------------------------------------------------------

STEP 6: What Happens During Inference

For each latitude and longitude:

1. Satellite image is downloaded using Google Static Maps
2. YOLOv8 trained model is applied
3. Rooftop solar panels are detected
4. Bounding boxes and confidence scores are computed
5. Annotated overlay images are generated
6. Structured JSON results are saved automatically

---------------------------------------------------------------------

STEP 7: Output Locations

All outputs are saved under the outputs/ directory:

outputs/images/   → Raw satellite images  
outputs/overlays/ → Annotated images with detection boxes  
outputs/json/     → Structured detection results (JSON)  

At the end of execution, the script prints the exact output paths in the terminal.

---------------------------------------------------------------------

STEP 8: Trained Model Used

Model architecture: YOLOv8  
Weights file: Trained_model/best.pt  

The model file is tracked using Git LFS and loads automatically during inference.

---------------------------------------------------------------------

STEP 9: How Judges Can Verify Results

Judges can verify results by:

1. Opening images in outputs/overlays/ to visually confirm detections
2. Inspecting JSON files in outputs/json/ for confidence scores and coordinates
3. Matching detected rooftops with satellite imagery

If outputs are generated successfully, the pipeline is functioning correctly.

---------------------------------------------------------------------

END OF GUIDE
