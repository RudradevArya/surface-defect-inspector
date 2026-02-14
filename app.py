"""
DefectVision AI - Gradio Web Application
Multi-domain surface defect inspection with Phase 1 (YOLO) and Phase 2 (Anomaly) detection.
"""

import os
import numpy as np
import gradio as gr
from PIL import Image
from pathlib import Path

from detector import DefectDetector
from anomaly_detector import AnomalyDetector
from report_gen import generate_report
from config import DEFAULT_CONFIDENCE, CLASS_NAMES

# Initialize detectors
print("Loading models...")
yolo_detector = DefectDetector()
anomaly_detector = AnomalyDetector()

# Collect sample images for examples
def get_sample_images(domain: str):
    """Get sample image paths for a domain."""
    sample_dir = Path(f"sample_images/{domain}")
    if sample_dir.exists():
        return sorted([str(p) for p in sample_dir.glob("*.jpg")] +
                      [str(p) for p in sample_dir.glob("*.png")])
    return []


# ========== Phase 1: YOLO Detection ==========

def run_yolo_detection(image, domain, confidence):
    """Run Phase 1 YOLO defect detection."""
    if image is None:
        return None, "Please upload an image.", None

    if not domain:
        return None, "Please select a domain.", None

    domain_key = domain.lower()
    available = yolo_detector.get_available_domains()

    if domain_key not in available:
        return (
            None,
            f"No model loaded for '{domain}'. Available domains: {available}. "
            f"Train the model first using the Colab notebook.",
            None,
        )

    try:
        results = yolo_detector.detect(image, domain_key, confidence)

        # Build text summary
        lines = [f"**Total Defects Found: {results['total_defects']}**"]
        lines.append(f"**Severity: {results['severity'].upper()}**")
        lines.append("")

        if results["summary"]:
            lines.append("| Defect Type | Count |")
            lines.append("|-------------|-------|")
            for defect_type, count in results["summary"].items():
                lines.append(f"| {defect_type} | {count} |")
        else:
            lines.append("No defects detected - surface looks good!")

        summary_text = "\n".join(lines)

        # Generate PDF report
        pdf_bytes = generate_report(
            original_image=image,
            annotated_image=results["annotated_image"],
            detections=results["detections"],
            summary=results["summary"],
            severity=results["severity"],
            domain=domain_key,
            confidence_threshold=confidence,
        )

        # Save PDF to temp file for download
        pdf_path = f"/tmp/defectvision_report_{domain_key}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        return results["annotated_image"], summary_text, pdf_path

    except Exception as e:
        return None, f"Error during detection: {str(e)}", None


# ========== Phase 2: Anomaly Detection ==========

def run_anomaly_detection(image):
    """Run Phase 2 anomaly detection."""
    if image is None:
        return None, "Please upload an image."

    if not anomaly_detector.is_available():
        return (
            None,
            "**Anomaly model not loaded.**\n\n"
            "Train the Phase 2 model first using notebook `04_train_anomaly.ipynb` in Google Colab.\n\n"
            "Phase 2 uses PatchCore to learn what 'normal' looks like, "
            "then flags anything unusual with a heatmap.",
        )

    try:
        results = anomaly_detector.detect_anomaly(image)
        score = results["anomaly_score"]
        verdict = "ANOMALOUS" if results["is_anomalous"] else "NORMAL"

        summary = (
            f"**Anomaly Score: {score:.3f}**\n\n"
            f"**Verdict: {verdict}**\n\n"
            f"{'The heatmap shows regions the model considers abnormal.' if results['is_anomalous'] else 'No anomalies detected.'}"
        )

        return results["visualization"], summary

    except Exception as e:
        return None, f"Error during anomaly detection: {str(e)}"


# ========== Build Gradio Interface ==========

def create_app():
    """Create the Gradio application."""

    with gr.Blocks(
        title="DefectVision AI",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="orange"),
    ) as app:

        # Header
        gr.Markdown(
            """
            # DefectVision AI
            ### Multi-Domain Surface Defect Inspection System
            Powered by YOLOv8 (Phase 1) and PatchCore Anomaly Detection (Phase 2)
            """
        )

        with gr.Tabs():

            # ===== Tab 1: Phase 1 - YOLO Detection =====
            with gr.Tab("Phase 1: Defect Detection", id="phase1"):
                gr.Markdown(
                    """
                    **Supervised Detection** -- Identifies and locates specific defect types
                    using YOLOv8 models trained on labeled datasets.
                    """
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        domain_selector = gr.Radio(
                            choices=["Metal", "PCB", "Building"],
                            label="Select Inspection Domain",
                            value="Metal",
                        )
                        confidence_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=DEFAULT_CONFIDENCE,
                            step=0.05,
                            label="Confidence Threshold",
                        )
                        input_image = gr.Image(
                            label="Upload Image for Inspection",
                            type="numpy",
                            sources=["upload", "webcam", "clipboard"],
                        )
                        detect_btn = gr.Button(
                            "Run Inspection",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=1):
                        output_image = gr.Image(
                            label="Detection Results",
                            type="numpy",
                        )
                        summary_output = gr.Markdown(label="Summary")
                        report_download = gr.File(label="Download PDF Report")

                detect_btn.click(
                    fn=run_yolo_detection,
                    inputs=[input_image, domain_selector, confidence_slider],
                    outputs=[output_image, summary_output, report_download],
                )

                # Show available models status
                available = yolo_detector.get_available_domains()
                if available:
                    status_text = f"Loaded models: {', '.join(d.upper() for d in available)}"
                else:
                    status_text = (
                        "No models loaded yet. Train models using the Colab notebooks first, "
                        "then place the .pt files in the models/ folder."
                    )
                gr.Markdown(f"*{status_text}*")

            # ===== Tab 2: Phase 2 - Anomaly Detection =====
            with gr.Tab("Phase 2: Anomaly Detection", id="phase2"):
                gr.Markdown(
                    """
                    **Unsupervised Anomaly Detection** -- Trained only on normal/good images.
                    Detects *any* abnormality and generates a heatmap showing where.
                    No labeled defect data needed -- mimics real manufacturing scenarios.
                    """
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        anomaly_input = gr.Image(
                            label="Upload Image",
                            type="numpy",
                            sources=["upload", "webcam", "clipboard"],
                        )
                        anomaly_btn = gr.Button(
                            "Detect Anomalies",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=1):
                        anomaly_output = gr.Image(
                            label="Anomaly Heatmap",
                            type="numpy",
                        )
                        anomaly_summary = gr.Markdown(label="Anomaly Results")

                anomaly_btn.click(
                    fn=run_anomaly_detection,
                    inputs=[anomaly_input],
                    outputs=[anomaly_output, anomaly_summary],
                )

                if not anomaly_detector.is_available():
                    gr.Markdown(
                        "*Phase 2 model not loaded. "
                        "Train it using notebook `04_train_anomaly.ipynb`.*"
                    )

            # ===== Tab 3: About =====
            with gr.Tab("About", id="about"):
                gr.Markdown(
                    """
                    ## How It Works

                    ### Phase 1: Supervised Detection (YOLOv8)
                    - Trained on labeled datasets with bounding box annotations
                    - Identifies **specific defect types** (cracks, scratches, solder bridges, etc.)
                    - Works across 3 domains: **Metal**, **PCB**, **Building**
                    - Outputs: bounding boxes, class labels, confidence scores

                    ### Phase 2: Anomaly Detection (PatchCore)
                    - Trained on **only normal/good images** -- no defect labels needed
                    - Learns what "normal" looks like, flags anything different
                    - Outputs: anomaly heatmap + anomaly score
                    - Better for real manufacturing where defects are rare

                    ### Supported Domains
                    | Domain | Defect Types |
                    |--------|-------------|
                    | Metal | Crazing, inclusion, patches, pitted surface, rolled-in scale, scratches |
                    | PCB | Missing hole, mouse bite, open circuit, short, spur, spurious copper |
                    | Building | Crack, spalling, corrosion, exposed rebar |

                    ---
                    *DefectVision AI -- Hackathon 2026*
                    """
                )

    return app


# ========== Main ==========
if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
