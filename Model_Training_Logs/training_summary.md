# Model Training Summary

## Model
- Architecture: YOLOv8
- Task: Rooftop Solar Panel Detection
- Framework: Ultralytics YOLOv8
- Training Environment: Google Colab (GPU)

## Datasets Used
- solar-panels-detection-master (MIT License)
- PVP Dataset (PVNet – academic)
- LSGI547 (CC BY 4.0)
- Custom Workflow Dataset (CC BY 4.0)
- Solar Panels v1 (CC BY 4.0)
- roofData (CC BY 4.0)

## Training Configuration
- Image size: 640 × 640
- Batch size: 16
- Epochs: <your epochs>
- Optimizer: AdamW
- Pretrained weights: YOLOv8 pretrained
- Fine-tuning: Yes (multi-stage)

## Key Metrics (Final Epoch)
- Precision: ~85%
- Recall: ~80%
- mAP@0.5: ~90%
- mAP@0.5:0.95: ~87%

## Validation Method
- Validation performed on unseen rooftop imagery
- Inference tested using latitude–longitude inputs
- Outputs verified via JSON predictions and overlay images

## Notes
- Full training artefacts retained offline due to size
- Final model weights (`best.pt`) included in repository
