"""
DefectVision AI - Phase 1: Supervised Defect Detection
Unified detection class wrapping YOLOv8 models for all domains.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from config import MODEL_PATHS, CLASS_NAMES, DEFAULT_CONFIDENCE, INFERENCE_IMG_SIZE


class DefectDetector:
    """Unified defect detector supporting multiple domains (metal, PCB, building)."""

    def __init__(self):
        self.models = {}
        self._load_models()

    def _load_models(self):
        """Load available YOLO models for each domain."""
        for domain, path in MODEL_PATHS.items():
            if Path(path).exists():
                try:
                    self.models[domain] = YOLO(path)
                    print(f"[OK] Loaded {domain} model from {path}")
                except Exception as e:
                    print(f"[WARN] Failed to load {domain} model: {e}")
            else:
                print(f"[INFO] {domain} model not found at {path} (train it first)")

    def get_available_domains(self):
        """Return list of domains with loaded models."""
        return list(self.models.keys())

    def detect(self, image: np.ndarray, domain: str, confidence: float = DEFAULT_CONFIDENCE):
        """
        Run defect detection on an image.

        Args:
            image: Input image as numpy array (RGB)
            domain: One of 'metal', 'pcb', 'building'
            confidence: Minimum confidence threshold (0-1)

        Returns:
            dict with keys:
                - annotated_image: Image with bounding boxes drawn (RGB numpy array)
                - detections: List of dicts with {class_name, confidence, bbox}
                - summary: Dict with counts per defect type
                - severity: 'low', 'medium', or 'high'
        """
        if domain not in self.models:
            raise ValueError(
                f"No model loaded for domain '{domain}'. "
                f"Available: {self.get_available_domains()}"
            )

        model = self.models[domain]

        # Run inference
        results = model(image, conf=confidence, imgsz=INFERENCE_IMG_SIZE, verbose=False)
        result = results[0]

        # Parse detections
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Get class name from model or config
            if hasattr(result, "names") and cls_id in result.names:
                class_name = result.names[cls_id]
            elif domain in CLASS_NAMES and cls_id < len(CLASS_NAMES[domain]):
                class_name = CLASS_NAMES[domain][cls_id]
            else:
                class_name = f"defect_{cls_id}"

            detections.append({
                "class_name": class_name,
                "confidence": round(conf, 3),
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
            })

        # Generate annotated image
        annotated_image = result.plot()
        # Convert BGR to RGB if needed
        if annotated_image.shape[2] == 3:
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # Build summary (count per defect type)
        summary = {}
        for det in detections:
            name = det["class_name"]
            summary[name] = summary.get(name, 0) + 1

        # Determine severity
        severity = self._compute_severity(detections)

        return {
            "annotated_image": annotated_image,
            "detections": detections,
            "summary": summary,
            "severity": severity,
            "total_defects": len(detections),
        }

    def _compute_severity(self, detections):
        """Classify inspection severity based on defect count and confidence."""
        if len(detections) == 0:
            return "pass"

        avg_conf = sum(d["confidence"] for d in detections) / len(detections)
        count = len(detections)

        if count <= 2 and avg_conf < 0.4:
            return "low"
        elif count <= 5 and avg_conf < 0.7:
            return "medium"
        else:
            return "high"
