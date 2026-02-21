"""
DefectVision AI - Phase 2: Anomaly Detection
Detects surface anomalies using PatchCore trained only on good/normal images.
Generates heatmaps showing where anomalies are located.
"""

import cv2
import numpy as np
from pathlib import Path


class AnomalyDetector:
    """Anomaly-based defect detector using PatchCore via Anomalib."""

    def __init__(self, model_path: str = "models/anomaly_model"):
        self.model_path = Path(model_path)
        self.model = None
        self.inferencer_type = None
        self._load_model()

    def _load_model(self):
        """Load the trained anomaly detection model (OpenVINO or checkpoint)."""
        if not self.model_path.exists():
            print(f"[INFO] Anomaly model not found at {self.model_path}")
            return

        openvino_model = self._find_file("*.bin")
        ckpt_model = self._find_file("*.ckpt")

        if openvino_model:
            self._load_openvino(openvino_model)
        elif ckpt_model:
            self._load_checkpoint(ckpt_model)
        else:
            print(f"[WARN] No model weights found in {self.model_path}")

    def _find_file(self, pattern: str):
        """Find first file matching glob pattern in model directory."""
        matches = list(self.model_path.rglob(pattern))
        return matches[0] if matches else None

    def _load_openvino(self, model_bin: Path):
        """Load OpenVINO exported model."""
        try:
            from anomalib.deploy import OpenVINOInferencer
            self.model = OpenVINOInferencer(path=model_bin)
            self.inferencer_type = "openvino"
            print(f"[OK] Loaded anomaly model (OpenVINO): {model_bin.name}")
        except Exception as e:
            print(f"[WARN] Failed to load OpenVINO model: {e}")

    def _load_checkpoint(self, ckpt_path: Path):
        """Load Lightning checkpoint for inference via Engine."""
        try:
            from anomalib.models import Patchcore
            from anomalib.engine import Engine
            self.model = {"engine": Engine(), "model": Patchcore(), "ckpt": str(ckpt_path)}
            self.inferencer_type = "checkpoint"
            print(f"[OK] Loaded anomaly model (checkpoint): {ckpt_path.name}")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint model: {e}")

    def is_available(self):
        """Check if anomaly model is loaded and ready."""
        return self.model is not None

    def detect_anomaly(self, image: np.ndarray):
        """
        Run anomaly detection on an image.

        Args:
            image: Input image as numpy array (RGB, HWC)

        Returns:
            dict with keys:
                - anomaly_map: Heatmap array (H, W) normalized 0-1
                - anomaly_score: Overall anomaly score (0-1)
                - is_anomalous: Boolean verdict
                - visualization: Image with heatmap overlay (RGB numpy array)
        """
        if not self.is_available():
            raise RuntimeError("Anomaly model not loaded. Train Phase 2 model first.")

        if self.inferencer_type == "openvino":
            return self._predict_openvino(image)
        elif self.inferencer_type == "checkpoint":
            return self._predict_checkpoint(image)
        else:
            raise RuntimeError(f"Unknown inferencer type: {self.inferencer_type}")

    def _predict_openvino(self, image: np.ndarray):
        """Run inference using OpenVINO model."""
        predictions = self.model.predict(image=image)

        if isinstance(predictions, list):
            pred = predictions[0]
        else:
            pred = predictions

        anomaly_map = pred.anomaly_map.squeeze()
        if hasattr(anomaly_map, 'cpu'):
            anomaly_map = anomaly_map.cpu().numpy()
        anomaly_map = anomaly_map.astype(np.float32)

        pred_score = float(pred.pred_score) if pred.pred_score is not None else 0.0
        pred_label = bool(pred.pred_label) if pred.pred_label is not None else False

        visualization = self._create_heatmap_overlay(image, anomaly_map)

        return {
            "anomaly_map": anomaly_map,
            "anomaly_score": pred_score,
            "is_anomalous": pred_label,
            "visualization": visualization,
        }

    def _predict_checkpoint(self, image: np.ndarray):
        """Run inference using Lightning checkpoint."""
        import torch
        from anomalib.data import PredictDataset
        import tempfile, os

        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, "input.png")
        cv2.imwrite(tmp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        dataset = PredictDataset(path=tmp_path, image_size=(256, 256))
        engine = self.model["engine"]
        model = self.model["model"]
        ckpt = self.model["ckpt"]

        preds = engine.predict(model=model, dataset=dataset, ckpt_path=ckpt)

        os.remove(tmp_path)
        os.rmdir(tmp_dir)

        if preds and len(preds) > 0:
            pred = preds[0]
            anomaly_map = pred.anomaly_map.squeeze().cpu().numpy().astype(np.float32)
            pred_score = float(pred.pred_score) if pred.pred_score is not None else 0.0
            pred_label = bool(pred.pred_label) if pred.pred_label is not None else False
        else:
            h, w = image.shape[:2]
            anomaly_map = np.zeros((h, w), dtype=np.float32)
            pred_score = 0.0
            pred_label = False

        visualization = self._create_heatmap_overlay(image, anomaly_map)

        return {
            "anomaly_map": anomaly_map,
            "anomaly_score": pred_score,
            "is_anomalous": pred_label,
            "visualization": visualization,
        }

    def _create_heatmap_overlay(self, image: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.5):
        """Create a visualization by overlaying anomaly heatmap on the original image."""
        h, w = image.shape[:2]

        amap = anomaly_map.copy()
        if amap.max() > amap.min():
            amap = (amap - amap.min()) / (amap.max() - amap.min())

        amap_resized = cv2.resize(amap, (w, h))
        heatmap = cv2.applyColorMap((amap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        return overlay
