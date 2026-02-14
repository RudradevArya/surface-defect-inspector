"""
DefectVision AI - Local Training Script
Train YOLOv8n models on your local GPU.

Usage:
    python train_local.py --domain metal --api-key YOUR_ROBOFLOW_KEY
    python train_local.py --domain pcb --api-key YOUR_ROBOFLOW_KEY
    python train_local.py --domain building --api-key YOUR_ROBOFLOW_KEY
    python train_local.py --domain all --api-key YOUR_ROBOFLOW_KEY

For GTX 1650 Ti (4GB VRAM): uses batch=8, imgsz=640, YOLOv8n (nano).
Training takes ~30-45 min per domain.
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO


# Dataset configs for Roboflow download
DATASET_CONFIGS = {
    "metal": {
        "workspace": "harit-yadav-u3zph",
        "project": "neu-det-jkimb",
        "version": 1,
        "description": "NEU-DET Metal Surface Defects (6 classes)",
        "output_name": "metal_yolo_best.pt",
    },
    "pcb": {
        "workspace": "fics-pcb",
        "project": "fics-pcb",
        "version": 2,
        "description": "PCB Manufacturing Defects (6 classes)",
        "output_name": "pcb_yolo_best.pt",
    },
    "building": {
        "workspace": "university-bswxt",
        "project": "crack-bphdr",
        "version": 2,
        "description": "Building/Structural Defects",
        "output_name": "building_yolo_best.pt",
    },
}


def check_gpu():
    """Check GPU availability and VRAM."""
    print("\n=== GPU Check ===")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_mem / 1e9
        vram_free = (torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_allocated(0)) / 1e9
        print(f"  GPU: {gpu_name}")
        print(f"  VRAM: {vram_total:.1f} GB total")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA: {torch.version.cuda}")
        return True
    else:
        print("  WARNING: No CUDA GPU detected. Training will use CPU (much slower).")
        return False


def download_dataset(domain: str, api_key: str) -> str:
    """Download dataset from Roboflow and return the data.yaml path."""
    from roboflow import Roboflow

    config = DATASET_CONFIGS[domain]
    print(f"\n=== Downloading {config['description']} ===")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(config["workspace"]).project(config["project"])
    version = project.version(config["version"])
    dataset = version.download("yolov8", location=f"datasets/{domain}")

    data_yaml = Path(dataset.location) / "data.yaml"
    print(f"  Dataset downloaded to: {dataset.location}")
    print(f"  data.yaml: {data_yaml}")
    return str(data_yaml)


def train_model(domain: str, data_yaml: str, epochs: int = 50, batch: int = 8):
    """Train YOLOv8n model on the dataset."""
    config = DATASET_CONFIGS[domain]
    print(f"\n=== Training YOLOv8n for {domain.upper()} ===")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch}")
    print(f"  Image size: 640")
    print(f"  Model: yolov8n.pt (nano - 3.2M params)")
    print(f"  This will take ~30-45 minutes on GTX 1650 Ti...")
    print()

    # Load pretrained YOLOv8 nano
    model = YOLO("yolov8n.pt")

    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        name=f"{domain}_defect_detector",
        patience=10,          # early stopping
        save=True,
        plots=True,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=4,
        exist_ok=True,        # overwrite previous run
    )

    # Evaluate
    print(f"\n=== Validation Results ({domain.upper()}) ===")
    metrics = model.val()
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")

    # Copy best weights to models/ directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    best_weights = Path(f"runs/detect/{domain}_defect_detector/weights/best.pt")
    output_path = models_dir / config["output_name"]

    if best_weights.exists():
        shutil.copy2(best_weights, output_path)
        print(f"\n  Best weights saved to: {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")
    else:
        print(f"\n  WARNING: Could not find best.pt at {best_weights}")
        # Try to find it
        for pt_file in Path("runs/detect").rglob("best.pt"):
            print(f"  Found: {pt_file}")
            shutil.copy2(pt_file, output_path)
            print(f"  Copied to: {output_path}")
            break

    return output_path


def main():
    parser = argparse.ArgumentParser(description="DefectVision AI - Local Training")
    parser.add_argument(
        "--domain",
        choices=["metal", "pcb", "building", "all"],
        required=True,
        help="Which domain to train (or 'all' for all three)",
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="Roboflow API key (free at roboflow.com)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8, safe for 4GB VRAM)",
    )

    args = parser.parse_args()

    # Check GPU
    has_gpu = check_gpu()
    if not has_gpu:
        print("\nTraining will proceed on CPU. This will be very slow.")
        response = input("Continue? [y/N]: ").strip().lower()
        if response != "y":
            sys.exit(0)

    # Determine which domains to train
    domains = list(DATASET_CONFIGS.keys()) if args.domain == "all" else [args.domain]

    for domain in domains:
        print(f"\n{'='*60}")
        print(f"  TRAINING: {domain.upper()}")
        print(f"{'='*60}")

        # Download dataset
        data_yaml = download_dataset(domain, args.api_key)

        # Train
        output_path = train_model(domain, data_yaml, args.epochs, args.batch)

        print(f"\n  {domain.upper()} training complete!")
        print(f"  Weights: {output_path}")

    print(f"\n{'='*60}")
    print("  ALL TRAINING COMPLETE!")
    print(f"{'='*60}")
    print("\nTrained models in models/:")
    for pt_file in Path("models").glob("*.pt"):
        print(f"  {pt_file} ({pt_file.stat().st_size / 1e6:.1f} MB)")
    print("\nYou can now run the app: python app.py")


if __name__ == "__main__":
    main()
