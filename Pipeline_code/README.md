# Inference Pipeline – Rooftop Solar Detection

This folder contains the runnable inference pipeline used to validate the trained YOLOv8 model.

The model was trained in Google Colab and validated on real satellite imagery.
This pipeline reproduces the validation step.

---

## Requirements

Python 3.9+

Install dependencies:

```bash
pip install -r requirements.txt

lat | lon
----|----
18.5204 | 73.8567

Prediction_files/sample_input_lat_long.xlsx

python Pipeline_code/run_inference.py \
  --model Trained_model/best.pt \
  --input Prediction_files/sample_input_lat_long.xlsx \
  --output Prediction_files/

