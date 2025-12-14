# Model Card — Rooftop Solar Panel Detection

## Model Overview
This model detects rooftop-mounted solar panels from satellite imagery using YOLOv8.

## Intended Use
- Rooftop solar potential analysis
- Urban energy planning
- GIS-based solar estimation

## Not Intended Use
- Legal land assessment
- Real-time drone navigation
- Non-rooftop solar farms

## Training Data
- Satellite images from mixed public datasets
- Urban and semi-urban rooftops
- Resolution: 0.3–0.6 m per pixel

## Performance
- mAP@0.5: ~60%
- Precision: ~65%
- Recall: ~55%

## Limitations
- Small panels are sometimes missed
- Shadows and blue rooftops cause confusion
- Performance drops in rural areas

## Ethical Considerations
- No personal data used
- Satellite imagery only
- No human identification
