# SAR Ship Detection: Deterministic vs. Deep Learning

An exploratory remote sensing project investigating **vessel detection in Synthetic Aperture Radar (SAR) imagery** using deterministic image processing and deep learning.

This project compares a traditional brightness-based detection algorithm against a fine-tuned **YOLO11n** object detection model, examining the strengths, limitations, and practical applications of both methods.

## Project Overview

Synthetic Aperture Radar provides valuable capabilities for maritime monitoring, including imaging through cloud cover and during nighttime conditions.

This project explores how SAR backscatter can be used to identify vessels in ocean imagery through two different approaches:

1. **Deterministic Detection:** Identify bright radar returns using statistical thresholding, connected-component labeling, and minimum bounding-box dimensions.
2. **Deep Learning:** Fine-tune a pretrained YOLO11n model to detect vessels using learned visual features.

## Data and Methodology

The project uses two primary data sources:

- **Sentinel-1 SAR:** VV and VH polarization data accessed through Microsoft Planetary Computer's STAC API for exploratory backscatter analysis.
- **SSDD (SAR Ship Detection Dataset):** Annotated SAR image chips used for model development, training, and validation.

The deterministic model identifies candidate vessels using a brightness threshold of μ + 2σ and filters connected regions by bounding-box dimensions.

The YOLO11n model is fine-tuned using labeled SSDD imagery and evaluated against the same validation samples.

Both methods are evaluated using bounding-box intersection over union (IoU), precision, recall, and F1 scores.

## Results

The models were evaluated on **139 SSDD validation images containing 285 annotated vessels**, using an IoU threshold of 0.30.

| Metric | Deterministic | YOLO11n |
|---|---:|---:|
| Correctly detected vessels | 217 | 274 |
| Overall vessel recall | 76.1% | 96.1% |
| Average image precision | 58.6% | 94% |
| Average image F1 | 61.8% | 95% |

YOLO significantly outperformed the deterministic approach, particularly in distinguishing vessels from bright backgrounds and ship wakes. However, the deterministic model demonstrated the value of transparent, customizable detection logic without requiring training.

These results are based on the validation dataset rather than a final held-out test evaluation.

## Repository Contents

- `main.ipynb` — Primary analysis notebook, including methodology, visualizations, and experimental results.
- `data_loader.py` — SSDD ingestion, annotation parsing, and model evaluation utilities.
- `deterministic_model.py` — Statistical thresholding and connected-component detection.
- `model.py` — YOLO model loading and inference.
- `model_comparison.py` — Performance comparison visualizations.
- `sar_multichannel_analysis.py` — Sentinel-1 polarization and raster analysis.
- `download_ssdd.py` — SSDD dataset download utility.
- `checkpoint_inv.py` — Model checkpoint inspection.

## Getting Started

Install the required Python dependencies, including NumPy, SciPy, Pandas, Matplotlib, Rasterio, PyTorch, Ultralytics, and the Microsoft Planetary Computer libraries.

Download the [SSDD dataset](https://huggingface.co/datasets/dronefreak/SSDD) using the included `download_ssdd.py` utility and open `main.ipynb` in Jupyter or VS Code.

The notebook documents the analysis from initial SAR exploration through training and model evaluation. YOLO training requires a pretrained checkpoint and the SSDD dataset.

## Technologies

Python · NumPy · SciPy · Matplotlib · Rasterio · GeoPandas · STAC · Sentinel-1 · PyTorch · Ultralytics YOLO

## Future Work

Potential extensions include evaluating performance on held-out test imagery, exploring multichannel SAR inputs, and integrating vessel detections with geospatial tracking or maritime monitoring pipelines.