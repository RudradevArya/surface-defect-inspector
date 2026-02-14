# DefectVision AI

Multi-domain AI surface defect inspection system supporting **metals**, **PCBs**, and **buildings**.

## Features

- **Phase 1: Supervised Detection** -- YOLOv8 models trained on labeled defect datasets. Identifies specific defect types with bounding boxes and confidence scores.
- **Phase 2: Anomaly Detection** -- PatchCore model trained only on normal/good images. Flags any abnormality with a heatmap overlay.
- **Multi-Domain** -- Supports metal surface defects, PCB manufacturing defects, and building/structural damage.
- **Web Interface** -- Gradio-based UI with image upload, webcam support, and PDF report generation.

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train Models (Google Colab)

Open the notebooks in `notebooks/` in Google Colab and follow the instructions:

1. `01_train_metal_yolo.ipynb` -- Metal surface defect detection
2. `02_train_pcb_yolo.ipynb` -- PCB defect detection
3. `03_train_building_yolo.ipynb` -- Building/structural defect detection
4. `04_train_anomaly.ipynb` -- Anomaly detection (Phase 2)

Download the trained `.pt` weight files and place them in `models/`.

### 3. Run the App

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

## Project Structure

```
defect-vision/
├── app.py                  # Gradio web application
├── detector.py             # Phase 1 YOLO detection module
├── anomaly_detector.py     # Phase 2 anomaly detection module
├── report_gen.py           # PDF report generator
├── config.py               # Domain configurations
├── requirements.txt        # Python dependencies
├── notebooks/              # Training notebooks (run in Colab)
├── models/                 # Trained model weights
├── sample_images/          # Demo images per domain
└── presentation/           # Hackathon PPT
```

## Tech Stack

- **Ultralytics YOLOv8** -- Object detection
- **Anomalib** -- Anomaly detection (PatchCore)
- **Gradio** -- Web interface
- **OpenCV** -- Image processing
- **FPDF2** -- PDF report generation

## Datasets

- **Metal:** NEU-DET (Northeastern University Surface Defect Dataset)
- **PCB:** PCB Defect Dataset (Roboflow)
- **Building:** Concrete Crack / Structural Damage (Kaggle)
- **Anomaly:** MVTec AD (for Phase 2 training)

---
*DefectVision AI -- Hackathon 2026*
