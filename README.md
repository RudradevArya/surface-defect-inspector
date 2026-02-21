# DefectVision AI

Multi-domain AI surface defect inspection system with two detection phases: **supervised object detection** (YOLOv8) and **unsupervised anomaly detection** (PatchCore). Supports **metal surfaces**, **PCB manufacturing**, and **building/structural** inspection domains.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Running Locally](#running-locally)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Starting the App](#starting-the-app)
  - [Clearing Port 7860 if Congested](#clearing-port-7860-if-congested)
- [Training Models](#training-models)
  - [Option A: Google Colab / Kaggle (Recommended)](#option-a-google-colab--kaggle-recommended)
  - [Option B: Local Training](#option-b-local-training)
  - [Getting Your Trained Weights](#getting-your-trained-weights)
  - [GPU Auto-Configuration](#gpu-auto-configuration)
- [Testing the App](#testing-the-app)
- [Defect Types by Domain](#defect-types-by-domain)
- [Datasets](#datasets)
- [Statistics and Metrics](#statistics-and-metrics)
  - [Training Metrics](#training-metrics)
  - [Severity Classification](#severity-classification)
  - [Anomaly Scoring](#anomaly-scoring)
- [Glossary](#glossary)
- [Theory and Further Reading](#theory-and-further-reading)
- [Troubleshooting](#troubleshooting)

---

## Features

- **Phase 1: Supervised Detection** -- YOLOv8 models trained on labeled defect datasets. Detects specific defect types with bounding boxes, class labels, and confidence scores.
- **Phase 2: Anomaly Detection** -- PatchCore model trained only on normal/good images. Flags any abnormality with a heatmap overlay and anomaly score, no defect labels needed.
- **Multi-Domain** -- Separate models for metal surface defects, PCB manufacturing defects, and building/structural damage.
- **Web Interface** -- Gradio-based UI with image upload, webcam capture, URL loading, confidence threshold slider, and downloadable PDF inspection reports.
- **PDF Reports** -- Auto-generated one-page inspection reports with annotated images, defect summary tables, severity verdicts, and metadata.

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Object Detection | [Ultralytics YOLOv8](https://docs.ultralytics.com/) | Phase 1 supervised defect detection (bounding boxes) |
| Anomaly Detection | [Anomalib](https://github.com/openvinotoolkit/anomalib) (PatchCore) | Phase 2 unsupervised anomaly detection (heatmaps) |
| Deep Learning | [PyTorch](https://pytorch.org/) + TorchVision | Underlying neural network framework |
| Web Framework | [Gradio](https://www.gradio.app/) | Interactive web UI with image I/O |
| Image Processing | [OpenCV](https://opencv.org/) (headless), [Pillow](https://pillow.readthedocs.io/) | Image manipulation, drawing, color conversion |
| Numerical | [NumPy](https://numpy.org/) | Array operations |
| Visualization | [Plotly](https://plotly.com/python/) | Charts and plots |
| PDF Generation | [FPDF2](https://py-pdf.github.io/fpdf2/) | Inspection report PDFs |
| Dataset Management | [Roboflow](https://roboflow.com/) | Downloading annotated datasets for training |
| Training Environments | Google Colab, Kaggle, Local GPU/CPU | Model training |

---

## Project Structure

```
surface-defect-inspector/
├── app.py                  # Main Gradio web application (UI + orchestration)
├── detector.py             # Phase 1: YOLOv8 detection wrapper class
├── anomaly_detector.py     # Phase 2: PatchCore anomaly detection wrapper
├── report_gen.py           # PDF inspection report generator
├── config.py               # Domain configs (model paths, class names, thresholds)
├── train_local.py          # CLI script for local GPU/CPU training
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore
├── notebooks/              # Training notebooks (for Colab/Kaggle)
│   ├── 01_train_metal_yolo.ipynb
│   ├── 02_train_pcb_yolo.ipynb
│   ├── 03_train_building_yolo.ipynb
│   └── 04_train_anomaly.ipynb
├── models/                 # Trained model weights (not tracked in git)
│   ├── metal_yolo_best.pt
│   ├── pcb_yolo_best.pt
│   ├── building_yolo_best.pt
│   └── anomaly_model/      # PatchCore weights directory
├── sample_images/          # Demo images for each domain
├── datasets/               # Downloaded training datasets (gitignored)
├── runs/                   # YOLO training output runs (gitignored)
└── presentation/           # Hackathon presentation files
```

---

## Running Locally

### Prerequisites

- **Python 3.10+**
- **pip** (Python package manager)
- **GPU (optional but recommended):** NVIDIA GPU with CUDA support for faster inference. CPU works but is slower.
- **Trained model weights** in the `models/` directory (see [Training Models](#training-models)).

### Installation

**1. Clone the repository:**

```bash
git clone https://github.com/YOUR_USERNAME/surface-defect-inspector.git
cd surface-defect-inspector
```

**2. Create and activate a virtual environment:**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. (Optional) Install PyTorch with CUDA for GPU acceleration:**

If the default PyTorch install doesn't include CUDA, reinstall with CUDA support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Check your CUDA version with `nvidia-smi` and match the wheel URL accordingly (`cu118`, `cu121`, `cu126`, etc.).

**5. Place model weights:**

Ensure your trained `.pt` files are in the `models/` directory:

```
models/
├── metal_yolo_best.pt
├── pcb_yolo_best.pt
├── building_yolo_best.pt
└── anomaly_model/
    └── (model weights here)
```

The app will skip any domain whose weights are missing and still work for the rest.

### Starting the App

```bash
python app.py
```

The app launches at **http://localhost:7860**. Open this URL in your browser.

You can change the port via an environment variable:

```bash
# Windows (PowerShell)
$env:GRADIO_SERVER_PORT = 7870; python app.py

# Linux / macOS
GRADIO_SERVER_PORT=7870 python app.py
```

### Clearing Port 7860 if Congested

If you get an error like `OSError: [Errno 98] Address already in use` or `port 7860 is already in use`, another process is occupying the port.

**Windows (PowerShell):**

```powershell
# Find the process using port 7860
netstat -ano | findstr :7860

# Kill the process by PID (replace 12345 with the actual PID from above)
taskkill /PID 12345 /F
```

**Windows (CMD):**

```cmd
netstat -ano | findstr :7860
taskkill /PID 12345 /F
```

**Linux / macOS:**

```bash
# Find and kill the process using port 7860
lsof -ti:7860 | xargs kill -9

# Or step by step:
lsof -i :7860          # find the PID
kill -9 <PID>          # kill it
```

**Alternative: just use a different port** (see above).

---

## Training Models

There are **4 models** to train -- 3 YOLOv8 models (one per domain) and 1 PatchCore anomaly model.

### Option A: Google Colab / Kaggle (Recommended)

Best for free GPU access. Each notebook auto-detects the environment and configures itself.

| Notebook | Domain | Dataset | GPU Time |
|----------|--------|---------|----------|
| `01_train_metal_yolo.ipynb` | Metal | NEU-DET (~9.7k images, 6 classes) | ~30-45 min |
| `02_train_pcb_yolo.ipynb` | PCB | PCB Defects (6 classes) | ~30-45 min |
| `03_train_building_yolo.ipynb` | Building | Structural Defects (4 classes) | ~20-30 min |
| `04_train_anomaly.ipynb` | Anomaly (Phase 2) | MVTec AD metal_nut | ~15-20 min |

**Steps:**

1. Open the notebook in [Google Colab](https://colab.research.google.com/) or [Kaggle](https://www.kaggle.com/).
2. Set the runtime to **GPU** (Colab: Runtime > Change runtime type > T4 GPU).
3. Enter your **Roboflow API key** when prompted (free at [roboflow.com](https://roboflow.com/)).
4. Run all cells.
5. Download the trained weight file (the notebook will print the path and offer a download link).
6. Place the `.pt` file in the `models/` directory locally.

### Option B: Local Training

Use `train_local.py` for training on your own machine. Requires a Roboflow API key.

```bash
# Train a single domain
python train_local.py --domain building --api-key YOUR_ROBOFLOW_KEY

# Train with specific model size and batch size
python train_local.py --domain metal --api-key YOUR_KEY --model yolov8s.pt --batch 8

# Train all three YOLO domains
python train_local.py --domain all --api-key YOUR_KEY

# Custom epoch count
python train_local.py --domain pcb --api-key YOUR_KEY --epochs 150
```

**CLI Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--domain` | Yes | -- | `metal`, `pcb`, `building`, or `all` |
| `--api-key` | Yes | -- | Roboflow API key |
| `--model` | No | Auto | `yolov8n.pt` (nano), `yolov8s.pt` (small), `yolov8m.pt` (medium) |
| `--epochs` | No | 100 | Number of training epochs |
| `--batch` | No | Auto | Batch size (auto-detected from VRAM) |

### Getting Your Trained Weights

After training completes (via notebook or local script):

**From Colab/Kaggle notebooks:**
- The notebook's final cells will show the path to `best.pt`.
- Use the file browser sidebar to download it, or the notebook will provide a `files.download()` call.
- Rename and place in `models/`:
  - Metal: `models/metal_yolo_best.pt`
  - PCB: `models/pcb_yolo_best.pt`
  - Building: `models/building_yolo_best.pt`

**From local training (`train_local.py`):**
- The script automatically copies `best.pt` from `runs/detect/` into `models/`.
- Weights are saved to `models/{domain}_yolo_best.pt`.
- The validation metrics (mAP50, mAP50-95, Precision, Recall) are printed at the end.

**Anomaly model (Phase 2):**
- Notebook `04_train_anomaly.ipynb` exports to OpenVINO format (`.bin` + `.xml`).
- Place the entire exported folder in `models/anomaly_model/`.
- Alternatively, a `.ckpt` checkpoint file also works.

### GPU Auto-Configuration

The training scripts auto-detect your GPU and choose optimal settings:

| VRAM | Model Size | Batch Size | Example GPUs |
|------|-----------|------------|--------------|
| >= 12 GB | YOLOv8s (small) | 16 | RTX 3060 12GB, RTX 4070, A100 |
| >= 6 GB | YOLOv8s (small) | 12 | RTX 3060 8GB, GTX 1660 Ti |
| >= 4 GB | YOLOv8n (nano) | 8 | GTX 1650 Ti, MX450 |
| CPU | YOLOv8n (nano) | 4 | Any (very slow, ~3-6 hrs/domain) |

You can override these with `--model` and `--batch` flags.

---

## Testing the App

**1. Phase 1 (Supervised Detection):**

- Open `http://localhost:7860` in your browser.
- Go to the **Phase 1** tab.
- Upload an image (or use webcam / paste a URL).
- Select the domain: Metal, PCB, or Building.
- Adjust the confidence threshold slider (default: 0.25). Lower values show more detections, higher values show only high-confidence ones.
- Click **Run Inspection**.
- View: annotated image with bounding boxes, defect summary, severity verdict.
- Download the PDF inspection report.

**2. Phase 2 (Anomaly Detection):**

- Go to the **Phase 2** tab.
- Upload an image of a metal surface (the anomaly model is trained on metal_nut from MVTec AD).
- Click **Detect Anomalies**.
- View: heatmap overlay showing anomalous regions, anomaly score (0-1), and a normal/anomalous verdict.

**3. Using sample images:**

The `sample_images/` directory contains demo images for each domain. Use these to quickly verify the models are working.

**4. Verifying model loading:**

When `app.py` starts, check the console output. It will print:

```
[OK] Loaded metal model from models/metal_yolo_best.pt
[OK] Loaded pcb model from models/pcb_yolo_best.pt
[OK] Loaded building model from models/building_yolo_best.pt
[OK] Loaded anomaly model (OpenVINO): model.bin
```

Any missing models will show `[INFO] ... model not found ... (train it first)`.

---

## Defect Types by Domain

### Metal Surface Defects (NEU-DET)

| Defect | Description |
|--------|-------------|
| **Crazing** | Fine network of surface cracks caused by thermal stress or aging |
| **Inclusion** | Foreign particles trapped in the metal during manufacturing |
| **Patches** | Irregular discolored areas from uneven surface treatment |
| **Pitted Surface** | Small pits/holes from corrosion or gas entrapment during casting |
| **Rolled-in Scale** | Oxide scale pressed into the surface during rolling |
| **Scratches** | Linear marks from mechanical contact or handling |

### PCB Manufacturing Defects

| Defect | Description |
|--------|-------------|
| **Missing Hole** | A through-hole or via that should exist but is absent |
| **Mouse Bite** | Irregular nibbled edge caused by breakout tab removal |
| **Open Circuit** | A broken trace where the copper path is interrupted |
| **Short** | Unintended copper connection between two traces |
| **Spur** | Small unwanted protrusion extending from a copper trace |
| **Spurious Copper** | Leftover copper that should have been etched away |

### Building / Structural Defects

| Defect | Description |
|--------|-------------|
| **Crack** | Fracture lines in concrete, masonry, or plaster |
| **Spalling** | Flaking/chipping of concrete surface, often exposing aggregate |
| **Corrosion** | Rust/oxidation stains, typically from reinforcement steel |
| **Exposed Rebar** | Reinforcement bars visible through deteriorated concrete cover |

---

## Datasets

| Domain | Dataset | Source | Size |
|--------|---------|--------|------|
| Metal | NEU-DET (Northeastern University Surface Defect Dataset) | [Roboflow](https://universe.roboflow.com/) | ~9,700 images, 6 classes |
| PCB | FICS-PCB Defect Dataset | [Roboflow](https://universe.roboflow.com/) | 6 classes |
| Building | Structural Crack/Damage Dataset | [Roboflow](https://universe.roboflow.com/) | 4 classes |
| Anomaly | MVTec AD (metal_nut category) | [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad) | Normal + anomalous images |

All YOLO datasets are downloaded automatically via the Roboflow API during training. You need a free Roboflow API key.

---

## Statistics and Metrics

### Training Metrics

These are computed during and after model training:

| Metric | Formula | What it Means |
|--------|---------|---------------|
| **Precision** | TP / (TP + FP) | Of all detections the model made, what fraction are actually correct. High precision = few false alarms. |
| **Recall** | TP / (TP + FN) | Of all actual defects present, what fraction did the model find. High recall = few missed defects. |
| **mAP50** | Mean AP at IoU=0.5 | Average precision across all classes, where a detection is "correct" if its bounding box overlaps the ground truth by at least 50%. The primary YOLO metric. |
| **mAP50-95** | Mean AP at IoU 0.5 to 0.95 | Stricter version -- averages AP across IoU thresholds from 0.5 to 0.95 (step 0.05). Rewards tighter bounding boxes. |
| **IoU** | Area of Overlap / Area of Union | Measures how well a predicted bounding box aligns with the ground truth box. 1.0 = perfect overlap, 0.0 = no overlap. |
| **Loss** | Various (box, cls, dfl) | Training loss values. Lower = better. Box loss measures localization error, cls loss measures classification error, dfl (distribution focal loss) measures bounding box refinement quality. |

### Severity Classification

Phase 1 detections are classified into severity levels:

| Severity | Rule | Meaning |
|----------|------|---------|
| **Pass** | 0 defects detected | Surface is defect-free |
| **Low** | <= 2 defects AND average confidence < 0.4 | Minor, uncertain detections -- likely acceptable |
| **Medium** | <= 5 defects AND average confidence < 0.7 | Moderate number of defects, needs review |
| **High** | Everything else (>5 defects or high confidence) | Serious quality issue, likely reject |

### Anomaly Scoring

Phase 2 anomaly detection produces:

| Output | Range | Meaning |
|--------|-------|---------|
| **Anomaly Score** | 0.0 - 1.0 | Overall anomaly level. Higher = more anomalous. |
| **Anomaly Map** | Per-pixel heatmap | Spatial map showing where anomalies are located. Visualized as a JET colormap overlay. |
| **Verdict** | Normal / Anomalous | Binary classification based on model's learned threshold. |

---

## Glossary

| Term | Meaning |
|------|---------|
| **YOLO** | "You Only Look Once" -- a family of real-time object detection models. The image is processed in a single forward pass (unlike older two-stage detectors). YOLOv8 is the version used here. |
| **YOLOv8n / YOLOv8s / YOLOv8m** | YOLOv8 model sizes: nano (~3.2M params), small (~11.2M params), medium (~25.9M params). Bigger = more accurate but slower and needs more VRAM. |
| **PatchCore** | An anomaly detection algorithm that extracts patch-level feature embeddings from a pretrained backbone (e.g., Wide ResNet-50) and stores them in a memory bank. At inference, it compares new image patches to this memory bank -- patches that differ significantly from "normal" patches are flagged as anomalous. |
| **Anomalib** | An open-source library by Intel/OpenVINO for anomaly detection. Provides implementations of PatchCore, PaDiM, STFPM, and other algorithms. |
| **Bounding Box** | A rectangle drawn around a detected defect, defined by (x1, y1, x2, y2) coordinates representing the top-left and bottom-right corners. |
| **Confidence Score** | A value between 0 and 1 indicating how certain the model is about a detection. Higher = more confident. The confidence threshold filters out low-confidence detections. |
| **Epoch** | One complete pass through the entire training dataset. Training for 100 epochs means the model sees every training image 100 times. |
| **Batch Size** | Number of images processed together in one training step. Larger batches are faster but need more GPU memory. |
| **VRAM** | Video RAM -- the memory on your GPU. Determines what model size and batch size you can use for training. |
| **Inference** | Running a trained model on new images to get predictions (as opposed to training). |
| **IoU (Intersection over Union)** | A metric measuring bounding box accuracy: the overlap area between predicted and ground truth boxes divided by their combined area. |
| **mAP (Mean Average Precision)** | The primary object detection metric. Averages the precision-recall curve area across all classes. |
| **Precision** | Of all predictions made, the fraction that are correct (true positives vs false positives). |
| **Recall** | Of all actual objects, the fraction the model detected (true positives vs false negatives). |
| **TP / FP / FN** | True Positive (correct detection), False Positive (false alarm), False Negative (missed defect). |
| **Backbone** | The feature extraction network inside a model. PatchCore uses Wide ResNet-50 as its backbone to extract image features. |
| **Memory Bank** | In PatchCore, a stored set of feature vectors from normal training images. Test images are compared against this bank to find anomalies. |
| **Feature Embedding** | A numerical vector representation of an image or image patch, learned by a neural network. Similar images produce similar embeddings. |
| **Heatmap** | A color-coded visualization where colors represent intensity (here, anomaly severity). Blue = normal, red = highly anomalous. |
| **Augmentation** | Artificially expanding the training dataset by applying transforms like flips, rotations, color shifts, and mosaic tiling to existing images. Helps the model generalize. |
| **Patience** | Early stopping parameter: training stops if the metric doesn't improve for this many consecutive epochs (set to 20 here). Prevents overfitting. |
| **Transfer Learning** | Starting training from a model already trained on a large dataset (like COCO) instead of from scratch. YOLOv8 models are pretrained on COCO, then fine-tuned on our defect datasets. |
| **OpenVINO** | Intel's toolkit for optimizing and deploying deep learning models. The anomaly model can be exported to OpenVINO format for faster inference. |
| **Supervised Learning** | Training with labeled data (images + annotations). Phase 1 uses this -- every defect is manually labeled with a class and bounding box. |
| **Unsupervised / Self-Supervised Learning** | Training without explicit defect labels. Phase 2 (PatchCore) only uses "normal" images and learns what normal looks like, flagging anything different as anomalous. |
| **Gradio** | A Python library for quickly building web-based ML demos with input/output components (image upload, sliders, text, etc.). |
| **Roboflow** | A platform for managing, annotating, and hosting computer vision datasets. Used here to download pre-annotated defect datasets. |
| **NEU-DET** | Northeastern University Surface Defect Dataset -- a benchmark dataset with 1,800 grayscale images of 6 types of hot-rolled steel strip defects. The Roboflow version used here is augmented to ~9,700 images. |
| **MVTec AD** | MVTec Anomaly Detection Dataset -- a benchmark for unsupervised anomaly detection containing 5,354 high-resolution images across 15 categories. We use the metal_nut category. |
| **Mosaic Augmentation** | A YOLO-specific augmentation that combines 4 training images into a single mosaic, helping the model learn to detect objects at different scales and contexts. |
| **DFL (Distribution Focal Loss)** | A loss function used in YOLOv8 for bounding box regression that models the box coordinates as a probability distribution rather than a single value, improving localization accuracy. |

---

## Theory and Further Reading

### Object Detection (Phase 1)

- **YOLO overview:** [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/) -- official documentation covering architecture, training, and inference.
- **YOLO paper lineage:** The original YOLO paper [You Only Look Once (Redmon et al., 2016)](https://arxiv.org/abs/1506.02640) introduced single-shot detection. YOLOv8 is a modern evolution with anchor-free detection and decoupled head design.
- **Object detection fundamentals:** [Stanford CS231n: Detection and Segmentation](http://cs231n.stanford.edu/) -- lectures on how CNNs are used for localization and detection.
- **mAP explained:** [mAP (Mean Average Precision) for Object Detection](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173) -- detailed breakdown of how mAP is calculated.
- **Data augmentation for detection:** [Albumentations library docs](https://albumentations.ai/docs/) and the [YOLO augmentation guide](https://docs.ultralytics.com/guides/hyperparameter-tuning/).

### Anomaly Detection (Phase 2)

- **PatchCore paper:** [Towards Total Recall in Industrial Anomaly Detection (Roth et al., 2022)](https://arxiv.org/abs/2106.08265) -- the algorithm used in Phase 2. Achieves near-perfect recall on MVTec AD by using a coreset-subsampled memory bank of patch features.
- **Anomalib library:** [Anomalib Documentation](https://anomalib.readthedocs.io/) -- the library used to train and deploy PatchCore.
- **MVTec AD benchmark:** [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) -- the standard benchmark for industrial anomaly detection.
- **Feature extraction for anomaly detection:** PatchCore uses a Wide ResNet-50 backbone pretrained on ImageNet. Features from intermediate layers capture texture and structural patterns. See [Deep Residual Learning (He et al., 2015)](https://arxiv.org/abs/1512.03385).

### General ML Concepts

- **Transfer learning:** [A Survey on Transfer Learning (Pan & Yang, 2010)](https://ieeexplore.ieee.org/document/5288526) -- why starting from pretrained weights is more effective than training from scratch.
- **Precision-Recall tradeoff:** [Scikit-learn: Precision-Recall](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html) -- interactive explanation of the tradeoff.
- **Convolutional Neural Networks:** [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) -- excellent visual introduction to neural networks.

### Surface Defect Detection Specifically

- **NEU surface defect dataset paper:** [A surface defect detection method based on neural network (Song & Yan, 2013)](https://www.sciencedirect.com/science/article/pii/S0169743913000476)
- **Industrial visual inspection survey:** [A Survey of Visual Inspection Technology in Industry](https://arxiv.org/abs/2211.12304) -- comprehensive overview of AI-powered quality control.

---

## Troubleshooting

### "No module named 'ultralytics'" or similar import errors

Make sure your virtual environment is activated and dependencies are installed:

```bash
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

### "CUDA out of memory" during training

Reduce the batch size:

```bash
python train_local.py --domain metal --api-key YOUR_KEY --batch 4
```

Or use a smaller model:

```bash
python train_local.py --domain metal --api-key YOUR_KEY --model yolov8n.pt --batch 4
```

### Port 7860 already in use

See [Clearing Port 7860 if Congested](#clearing-port-7860-if-congested) above.

### Models not loading / "[INFO] model not found"

Verify your `models/` directory has the correct files:

```bash
ls models/
# Expected: metal_yolo_best.pt, pcb_yolo_best.pt, building_yolo_best.pt, anomaly_model/
```

The app will still start even if some models are missing -- it just won't offer those domains.

### Anomaly model loads but gives poor results

- The anomaly model is trained on MVTec AD `metal_nut` category. It works best on similar metallic surfaces.
- Ensure the input image is reasonably well-lit and centered.
- The anomaly score threshold is determined by the model during training -- it's not manually adjustable in the UI.

### PyTorch not detecting GPU

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

If this prints `False`, reinstall PyTorch with the correct CUDA version:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Gradio app crashes on startup

Check if all required packages are installed. Anomalib in particular has many transitive dependencies. Try:

```bash
pip install anomalib --upgrade
```

---

*DefectVision AI -- Pratiti Hackathon 2026*
