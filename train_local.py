"""
DefectVision AI - Local Training Script
Train YOLOv8 models on your local GPU or CPU.

Usage:
    python train_local.py --domain building --api-key YOUR_ROBOFLOW_KEY
    python train_local.py --domain building --api-key YOUR_KEY --model yolov8s.pt
    python train_local.py --domain all --api-key YOUR_ROBOFLOW_KEY
    python train_local.py --domain metal --api-key YOUR_KEY --epochs 100

GPU auto-detection:
    >= 12GB VRAM  ->  yolov8s, batch=16
    >= 6GB VRAM   ->  yolov8s, batch=12
    >= 4GB VRAM   ->  yolov8n, batch=8   (GTX 1650 Ti)
    CPU fallback  ->  yolov8n, batch=4
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO


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
    """Check GPU availability, return (has_gpu, vram_gb)."""
    print("\n=== GPU Check ===")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1e9
        print(f"  GPU:      {gpu_name}")
        print(f"  VRAM:     {vram_gb:.1f} GB")
        print(f"  PyTorch:  {torch.__version__}")
        print(f"  CUDA:     {torch.version.cuda}")
        return True, vram_gb
    else:
        print("  WARNING: No CUDA GPU detected. Training will use CPU (much slower).")
        print("  Make sure you installed PyTorch with CUDA support:")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126")
        return False, 0


def auto_config(vram_gb: float):
    """Return (model_size, batch_size) based on available VRAM."""
    if vram_gb >= 12:
        return "yolov8s.pt", 16
    elif vram_gb >= 6:
        return "yolov8s.pt", 12
    elif vram_gb >= 4:
        return "yolov8n.pt", 8
    else:
        return "yolov8n.pt", 4


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


def train_model(
    domain: str,
    data_yaml: str,
    model_size: str,
    epochs: int,
    batch: int,
    device,
):
    """Train a YOLOv8 model on the dataset."""
    config = DATASET_CONFIGS[domain]

    print(f"\n=== Training {model_size} for {domain.upper()} ===")
    print(f"  Model:      {model_size}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch size: {batch}")
    print(f"  Image size: 640")
    print(f"  Device:     {device}")
    print()

    model = YOLO(model_size)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        name=f"{domain}_defect_detector",
        patience=20,
        save=True,
        plots=True,
        device=device,
        workers=2,
        exist_ok=True,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.5,
        mosaic=1.0,
        scale=0.5,
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
    output_path = models_dir / config["output_name"]

    # YOLO may save to a resolved path (e.g. OneDrive Desktop redirect),
    # so we check multiple candidate locations.
    candidates = [
        Path(f"runs/detect/{domain}_defect_detector/weights/best.pt"),
        Path(f"runs/detect/{domain}_defect_detector2/weights/best.pt"),
    ]

    # Also check the path YOLO actually printed (resolve cwd for OneDrive)
    resolved_cwd = Path.cwd().resolve()
    candidates.append(resolved_cwd / f"runs/detect/{domain}_defect_detector/weights/best.pt")

    best_weights = None
    for candidate in candidates:
        if candidate.exists():
            best_weights = candidate
            break

    # Fallback: search recursively
    if best_weights is None:
        for search_root in [Path("runs/detect"), resolved_cwd / "runs/detect"]:
            if search_root.exists():
                for pt_file in search_root.rglob("best.pt"):
                    best_weights = pt_file
                    break
            if best_weights:
                break

    if best_weights:
        shutil.copy2(best_weights, output_path)
        print(f"\n  Best weights saved to: {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")
    else:
        print(f"\n  ERROR: Could not find best.pt anywhere in runs/detect/")
        print(f"  Check these paths manually:")
        print(f"    {Path.cwd() / 'runs/detect'}")
        print(f"    {resolved_cwd / 'runs/detect'}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="DefectVision AI - Local Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_local.py --domain building --api-key YOUR_KEY
  python train_local.py --domain building --api-key YOUR_KEY --model yolov8s.pt --batch 4
  python train_local.py --domain all --api-key YOUR_KEY --epochs 100
        """,
    )
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
        "--model",
        default=None,
        help="Model size: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium). "
             "Auto-detected from VRAM if not specified.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size. Auto-detected from VRAM if not specified.",
    )

    args = parser.parse_args()

    # Check GPU and auto-configure
    has_gpu, vram_gb = check_gpu()
    auto_model, auto_batch = auto_config(vram_gb)

    model_size = args.model or auto_model
    batch_size = args.batch or auto_batch
    device = 0 if has_gpu else "cpu"

    print(f"\n=== Training Configuration ===")
    print(f"  Model:      {model_size} {'(auto)' if not args.model else '(manual)'}")
    print(f"  Batch size: {batch_size} {'(auto)' if not args.batch else '(manual)'}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Device:     {'GPU' if has_gpu else 'CPU'}")

    if not has_gpu:
        print("\nTraining will proceed on CPU. This will be VERY slow (~3-6 hours per domain).")
        response = input("Continue? [y/N]: ").strip().lower()
        if response != "y":
            sys.exit(0)

    # Determine which domains to train
    domains = list(DATASET_CONFIGS.keys()) if args.domain == "all" else [args.domain]

    for domain in domains:
        print(f"\n{'='*60}")
        print(f"  TRAINING: {domain.upper()}")
        print(f"{'='*60}")

        data_yaml = download_dataset(domain, args.api_key)

        output_path = train_model(
            domain=domain,
            data_yaml=data_yaml,
            model_size=model_size,
            epochs=args.epochs,
            batch=batch_size,
            device=device,
        )

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
