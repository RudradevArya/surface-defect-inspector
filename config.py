"""
DefectVision AI - Configuration
Domain-specific settings for defect detection models.
"""

# Model weight paths (downloaded from Colab after training)
MODEL_PATHS = {
    "metal": "models/metal_yolo_best.pt",
    "pcb": "models/pcb_yolo_best.pt",
    "building": "models/building_yolo_best.pt",
}

# Anomaly detection model path (Phase 2)
ANOMALY_MODEL_PATH = "models/anomaly_model"

# Class names per domain
CLASS_NAMES = {
    "metal": [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches",
    ],
    "pcb": [
        "missing_hole",
        "mouse_bite",
        "open_circuit",
        "short",
        "spur",
        "spurious_copper",
    ],
    "building": [
        "crack",
        "spalling",
        "corrosion",
        "exposed_rebar",
    ],
}

# Colors for bounding boxes (BGR for OpenCV)
DOMAIN_COLORS = {
    "metal": (0, 165, 255),     # orange
    "pcb": (0, 255, 0),         # green
    "building": (0, 0, 255),    # red
}

# Severity thresholds
SEVERITY_THRESHOLDS = {
    "low": {"max_defects": 2, "max_avg_confidence": 0.4},
    "medium": {"max_defects": 5, "max_avg_confidence": 0.7},
    "high": {},  # anything above medium
}

# Default confidence threshold for detection
DEFAULT_CONFIDENCE = 0.25

# Image size for inference
INFERENCE_IMG_SIZE = 640

# Report settings
REPORT_TITLE = "DefectVision AI - Inspection Report"
REPORT_PASS_THRESHOLD = 0  # 0 defects = pass
