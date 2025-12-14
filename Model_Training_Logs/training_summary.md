# Model Training Summary

## Model
- Architecture: YOLOv8
- Task: Rooftop Solar Panel Detection
- Framework: Ultralytics YOLO
- Backbone: CSPDarknet

## Dataset
- Source: Satellite imagery (mixed public datasets)
- Total images: ~9,000
- Train / Val split: 80 / 20
- Classes: Solar Panel (1 class)

## Training Configuration
- Image size: 640x640
- Batch size: 16
- Epochs: 50
- Optimizer: AdamW
- Learning rate: 0.001
- Hardware: Google Colab ( T4 GPU)

## Results
- Best mAP@0.5: ~80%
- Precision: ~75%
- Recall: ~67%

## Notes
- Model under-detects small panels in dense rooftops.
- Accuracy expected to improve with more rooftop diversity.
