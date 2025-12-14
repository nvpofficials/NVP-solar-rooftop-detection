# NVP-solar-rooftop-detection

## Problem Statement

Accurate estimation of rooftop solar adoption is critical for
urban energy planning, policy formulation, and sustainability analysis.
Manual surveys are slow, expensive, and non-scalable.

This project presents an offline AI-based pipeline that automatically
detects rooftop solar panels from satellite imagery using deep learning.

Offline AI pipeline for rooftop solar panel detection using satellite imagery. Developed for the EcoInnovators Ideathon (2026). Includes trained model, inference pipeline, predictions, artefacts, and documentation.

## Solution Overview

The system uses a YOLO-based object detection model trained on
satellite imagery to identify rooftop solar panels.

Key capabilities:
- Takes latitude & longitude as input
- Fetches corresponding satellite imagery
- Detects rooftop solar panels
- Produces structured JSON outputs
- Generates annotated visual overlays for validation

## End-to-End Workflow

1. Input locations provided via Excel file (latitude & longitude)
2. Satellite images processed in Google Colab
3. YOLO model (`best.pt`) performs inference
4. Predictions exported as JSON
5. Bounding boxes rendered on satellite images

## Model Details

- Architecture: YOLO (object detection)
- Task: Rooftop solar panel detection
- Training environment: Google Colab
- Trained weights: `Trained_model/best.pt` (tracked using Git LFS)

## Model Validation (Post-Training)

After training, the model was validated in Google Colab
using real satellite imagery.

Validation artefacts provided in this repository include:
- Sample input Excel file with latitude & longitude
- Prediction results in JSON format
- Annotated satellite images with detected solar panels

Relevant folders:
- `Prediction_files/`
- `Artefacts/`

These results confirm that the trained model performs
real-world rooftop solar detection.

## Repository Structure

NVP-solar-rooftop-detection/
│
├── Trained_model/          # YOLO trained weights (best.pt)
├── Pipeline_code/          # Inference and processing scripts
├── Prediction_files/       # XLSX inputs and JSON outputs
├── Artefacts/              # Annotated result images
├── Model_Training_Logs/    # Training metrics and summaries
├── Model_card/             # Model documentation
├── Environment_details/   # Environment & dependency info
└── README.md               # Project documentation

## Impact & Applications

- Urban solar adoption mapping
- Smart city energy planning
- Renewable energy policy assessment
- Scalable analysis without physical surveys
- Applicable to developing and dense urban regions

## Limitations & Future Scope

- Detection accuracy depends on satellite image resolution
- Performance may vary across geographies

Future improvements:
- Higher-resolution imagery support
- Panel capacity estimation
- Temporal analysis for adoption trends
- Deployment as a web-based dashboard

### Sample Detection Results

![Solar Detection Example](Artefacts/overlay_1.jpg)

## Acknowledgement

Developed as part of the EcoInnovators Ideathon (2026).


This confirms the trained model performs real-world detection.
