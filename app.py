"""
DefectVision AI - Gradio Web Application
Multi-domain surface defect inspection with Phase 1 (YOLO) and Phase 2 (Anomaly) detection.
"""

import os
import tempfile
import urllib.request
import io
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


def get_sample_images(domain: str):
    """Get sample image paths for a domain."""
    sample_dir = Path(f"sample_images/{domain}")
    if sample_dir.exists():
        return sorted([str(p) for p in sample_dir.glob("*.jpg")] +
                      [str(p) for p in sample_dir.glob("*.png")])
    return []


def load_image_from_url(url: str):
    """
    Fetch an image from a URL, save to a temp file, return gr.update so
    visibility can be restored when URL mode is active.
    """
    if not url or not url.strip():
        return gr.update(value=None, visible=False), "⚠️ Please enter a URL."
    url = url.strip()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        tmp_path = os.path.join(tempfile.gettempdir(), "defectvision_url_image.png")
        img.save(tmp_path)
        return gr.update(value=tmp_path, visible=True), "✅ Image loaded from URL."
    except Exception as e:
        return gr.update(value=None, visible=False), f"❌ Failed to load image: {e}"


# ========== Result HTML Builders ==========

P1_PLACEHOLDER = '<div class="result-placeholder">⚡ Run an inspection to see results here…</div>'
P2_PLACEHOLDER = '<div class="result-placeholder">🔬 Run anomaly detection to see results here…</div>'
PDF_PLACEHOLDER = '<div class="pdf-beam-placeholder">📄 PDF report will appear here after inspection</div>'


def _build_yolo_result_html(results, domain):
    total     = results["total_defects"]
    severity  = results["severity"].lower()
    summary   = results["summary"]

    sev_class = f"sev-{severity}"

    stats_html = (
        f'<div class="result-stats-row">'
        f'  <div class="stat-chip">'
        f'    <span class="stat-number">{total}</span>'
        f'    <span class="stat-label">Defects Found</span>'
        f'  </div>'
        f'  <div class="stat-chip {sev_class}">'
        f'    <span class="sev-dot"></span>'
        f'    <span class="sev-label">{severity.upper()}</span>'
        f'    <span class="stat-label">Severity</span>'
        f'  </div>'
        f'</div>'
    )

    if summary:
        rows = "".join(
            f'<div class="defect-row">'
            f'<span class="defect-name">{k.replace("_", " ")}</span>'
            f'<span class="defect-count-badge">{v}</span>'
            f'</div>'
            for k, v in summary.items()
        )
        breakdown = f'<div class="defect-list">{rows}</div>'
    else:
        breakdown = (
            '<div class="good-state">'
            '<span class="good-icon">✅</span>'
            '<span class="good-text">No defects detected — surface looks great!</span>'
            '</div>'
        )

    return (
        f'<div class="result-summary">'
        f'<div class="domain-badge">🔍 {domain.upper()} Inspection</div>'
        f'{stats_html}'
        f'<div class="result-divider"></div>'
        f'{breakdown}'
        f'</div>'
    )

def _build_pdf_beam_html(pdf_path: str) -> str:
    """Build styled download beam HTML for a PDF file.
    Uses a base64 data-URI so the download works on any OS without
    relying on Gradio's /file= route (which breaks on Windows paths).
    """
    import base64
    try:
        size_bytes = os.path.getsize(pdf_path)
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes/1024:.1f} KB"
        else:
            size_str = f"{size_bytes/(1024*1024):.1f} MB"
        with open(pdf_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        data_url = f"data:application/pdf;base64,{b64}"
    except Exception as e:
        return f'<div class="pdf-beam-placeholder">❌ Could not prepare PDF: {e}</div>'

    filename = os.path.basename(pdf_path)
    return (
        f'<div class="pdf-beam-wrap">'
        f'  <a class="pdf-beam" href="{data_url}" download="{filename}">'
        f'    <div class="pdf-icon-circle">📄</div>'
        f'    <div class="pdf-meta">'
        f'      <span class="pdf-filename">{filename}</span>'
        f'      <span class="pdf-filesize">{size_str} &nbsp;·&nbsp; PDF Report</span>'
        f'    </div>'
        f'    <div class="pdf-dl-badge">↓ Download</div>'
        f'  </a>'
        f'</div>'
    )


    score    = results["anomaly_score"]
    is_anom  = results["is_anomalous"]
    verdict  = "ANOMALOUS" if is_anom else "NORMAL"
    vclass   = "verdict-anom" if is_anom else "verdict-normal"
    icon     = "🚨" if is_anom else "✅"
    detail   = (
        "Heatmap highlights abnormal regions detected on the surface."
        if is_anom else
        "Surface appears normal — no anomalies detected."
    )
    pct      = min(100, int(score * 100))
    bar_col  = "#ef4444" if is_anom else "#22c55e"

    return (
        f'<div class="result-summary">'
        f'  <div class="anomaly-score-row">'
        f'    <div class="anomaly-score-chip">'
        f'      <span class="anomaly-score-num">{score:.3f}</span>'
        f'      <span class="stat-label">Anomaly Score</span>'
        f'      <div class="score-bar-track">'
        f'        <div class="score-bar-fill" style="width:{pct}%;background:{bar_col}"></div>'
        f'      </div>'
        f'    </div>'
        f'    <div class="verdict-chip {vclass}">'
        f'      <span class="verdict-icon">{icon}</span>'
        f'      <span class="verdict-text">{verdict}</span>'
        f'    </div>'
        f'  </div>'
        f'  <p class="anomaly-detail">{detail}</p>'
        f'</div>'
    )


# ========== Phase 1: YOLO Detection ==========

def _ensure_numpy(image):
    """Accept numpy arrays, PIL Images, or file-path strings (Gradio 5.x)."""
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, str):  # file path returned by load_image_from_url
        return np.array(Image.open(image).convert("RGB"))
    # PIL Image
    try:
        return np.array(image)
    except Exception:
        return None


def run_yolo_detection(image, domain, confidence):
    image = _ensure_numpy(image)
    if image is None:
        return None, "Please upload an image.", None
    if not domain:
        return None, "Please select a domain.", None

    domain_key = domain.lower()
    available = yolo_detector.get_available_domains()

    if domain_key not in available:
        return (
            None,
            f"No model loaded for '{domain}'. Available: {available}. Train first via Colab.",
            None,
        )

    try:
        results = yolo_detector.detect(image, domain_key, confidence)

        pdf_bytes = generate_report(
            original_image=image,
            annotated_image=results["annotated_image"],
            detections=results["detections"],
            summary=results["summary"],
            severity=results["severity"],
            domain=domain_key,
            confidence_threshold=confidence,
        )
        pdf_path = os.path.join(tempfile.gettempdir(), f"defectvision_report_{domain_key}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        return results["annotated_image"], _build_yolo_result_html(results, domain_key), _build_pdf_beam_html(pdf_path)

    except Exception as e:
        return None, f"Error during detection: {str(e)}", None


# ========== Phase 2: Anomaly Detection ==========

def run_anomaly_detection(image):
    image = _ensure_numpy(image)
    if image is None:
        return None, "Please upload an image."
    if not anomaly_detector.is_available():
        return (
            None,
            "**Anomaly model not loaded.**\n\nTrain using `04_train_anomaly.ipynb` in Google Colab.",
        )
    try:
        results = anomaly_detector.detect_anomaly(image)
        return results["visualization"], _build_anomaly_result_html(results)
    except Exception as e:
        return None, f"<p style='color:#dc2626;font-weight:600'>Error: {str(e)}</p>"


# ========== CSS ==========

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── PAGE SHELL ── */
html, body {
    margin: 0 !important;
    padding: 0 !important;
    background: #f5f3ff !important;
    font-family: 'Inter', sans-serif !important;
    min-height: 100vh !important;
}

/* animated soft mesh background */
body::before {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background:
        radial-gradient(ellipse 65% 55% at 10% 5%,  rgba(167,139,250,0.30) 0%, transparent 55%),
        radial-gradient(ellipse 50% 45% at 90% 10%,  rgba(236,72,153,0.18) 0%, transparent 55%),
        radial-gradient(ellipse 55% 50% at 80% 90%,  rgba(99,102,241,0.18) 0%, transparent 55%),
        radial-gradient(ellipse 45% 40% at 5%  85%,  rgba(124,58,237,0.14) 0%, transparent 55%);
    animation: bgPulse 14s ease-in-out infinite alternate;
}
@keyframes bgPulse {
    from { opacity: .80; }
    to   { opacity: 1;   }
}

/* ── GRADIO CONTAINER ── */
.gradio-container {
    position: relative;
    z-index: 1;
    max-width: 1200px !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 0 24px 48px !important;
    background: transparent !important;
    font-family: 'Inter', sans-serif !important;
    color: #1e1b4b !important;
    box-sizing: border-box !important;
}

/* ── HERO BANNER (HTML) ── */
#hero-wrap {
    background: rgba(255, 255, 255, 0.6) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(124, 58, 237, 0.1) !important;
    border-radius: 32px !important;
    padding: 64px 48px !important;
    margin: 32px 0 0 !important;
    text-align: center;
    box-shadow: 0 10px 40px rgba(124, 58, 237, 0.04) !important;
    position: relative;
    overflow: hidden;
}
#hero-wrap::before {
    content: '';
    position: absolute;
    top: -50px; left: -50px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(167, 139, 250, 0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    padding: 6px 16px;
    background: rgba(124, 58, 237, 0.08);
    color: #7c3aed;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 20px;
}
.hero-title {
    font-size: 3.5rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 12px;
    background: linear-gradient(135deg, #7c3aed, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 1.1rem;
    color: #4c4f7a;
    font-weight: 500;
    margin-bottom: 32px;
}
.hero-chips {
    display: flex;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
}
.chip {
    padding: 8px 20px;
    background: #ffffff;
    border: 1px solid rgba(124, 58, 237, 0.12);
    border-radius: 100px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #4c4f7a;
    box-shadow: 0 2px 8px rgba(124, 58, 237, 0.04);
}

/* ── SUPREME GLASS CARDS (NUCLEAR PADDING) ── */
.glass-card {
    background: #f7f7fb !important;
    backdrop-filter: blur(40px) !important;
    -webkit-backdrop-filter: blur(40px) !important;
    border: 1.5px solid #ebebf5 !important;
    border-radius: 28px !important;
    padding: 36px !important;
    margin: 12px !important;
    box-shadow:
        0 1px 3px rgba(0, 0, 0, 0.04),
        0 8px 24px rgba(124, 58, 237, 0.07) !important;
    overflow: hidden !important;
    box-sizing: border-box !important;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease !important;
}

.glass-card:hover {
    transform: translateY(-3px) !important;
    border-color: rgba(124, 58, 237, 0.18) !important;
    box-shadow:
        0 4px 8px rgba(0, 0, 0, 0.05),
        0 20px 40px rgba(124, 58, 237, 0.12) !important;
}

/* ── CARD HEADER — wrapper layout ── */
.gradio-container .card-header-main,
.gradio-container .card-header-main .prose {
    padding: 1px 3px !important;
    border-bottom: 1.5px solid rgba(124, 58, 237, 0.10) !important;
    width: 100% !important;
    background: transparent !important;
    box-shadow: none !important;
    border-left: none !important;
    border-top: none !important;
    border-right: none !important;
}

/* ── CARD HEADER — actual <h3> text inside .prose ── */
.gradio-container .card-header-main .prose h3,
.gradio-container .card-header-main h3 {
    font-size: 1.12rem !important;
    font-weight: 800 !important;
    color: #1e1b4b !important;
    -webkit-text-fill-color: #1e1b4b !important;
    background: none !important;
    -webkit-background-clip: unset !important;
    background-clip: unset !important;
    margin: 0 !important;
    border: none !important;
    line-height: 1.4 !important;
}

/* Force all widgets inside the glass card to have breathing room */
.glass-card .block, .glass-card .gr-form, .glass-card .gr-group {
    margin-bottom: 32px !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* ── TABS (CENTERED PILL) ── */
div[role="tablist"] {
    display: flex !important;
    justify-content: center !important;
    background: rgba(255, 255, 255, 0.5) !important;
    backdrop-filter: blur(20px) !important;
    padding: 8px !important;
    border-radius: 999px !important;
    border: 1px solid rgba(124, 58, 237, 0.15) !important;
    width: fit-content !important;
    margin: 48px auto !important;
    gap: 8px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04) !important;
}
div[role="tablist"]::after { display: none !important; }
button[role="tab"] {
    border: none !important;
    background: transparent !important;
    border-radius: 999px !important;
    padding: 12px 32px !important;
    font-weight: 800 !important;
    color: #64748b !important;
    transition: all 0.3s ease !important;
}
button[role="tab"][aria-selected="true"] {
    background: #ffffff !important;
    color: #7c3aed !important;
    box-shadow: 0 8px 16px rgba(124, 58, 237, 0.15) !important;
}

/* ── DESCRIPTION BANNER ── */
#phase1-desc, #phase2-desc {
    background: rgba(255, 255, 255, 0.5) !important;
    border-left: 6px solid #7c3aed !important;
    border-radius: 12px 32px 32px 12px !important;
    padding: 32px 40px !important;
    margin-bottom: 48px !important;
}
#phase1-desc p, #phase2-desc p {
    font-size: 1.05rem !important;
    line-height: 1.8 !important;
    color: #312e81 !important;
}

/* ── DOMAIN SELECTOR — segmented pill control ── */
#domain-selector > .wrap {
    display: flex !important;
    flex-direction: row !important;
    gap: 0 !important;
    background: #f1eeff !important;
    border-radius: 14px !important;
    padding: 4px !important;
    border: 1px solid rgba(124,58,237,0.12) !important;
    box-shadow: inset 0 1px 3px rgba(124,58,237,0.07) !important;
    width: 100% !important;
    box-sizing: border-box !important;
}
#domain-selector > .wrap label {
    flex: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px !important;
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 0 !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    color: #7b7fa8 !important;
    cursor: pointer !important;
    transition: all 0.22s ease !important;
    white-space: nowrap !important;
    box-shadow: none !important;
    user-select: none !important;
}
#domain-selector > .wrap label:hover {
    color: #7c3aed !important;
    background: rgba(124,58,237,0.06) !important;
    transform: none !important;
}
/* active/selected pill */
#domain-selector > .wrap label:has(input:checked),
#domain-selector > .wrap label.selected {
    background: #ffffff !important;
    color: #7c3aed !important;
    font-weight: 800 !important;
    box-shadow: 0 2px 10px rgba(124,58,237,0.15), 0 1px 3px rgba(0,0,0,0.06) !important;
    border-radius: 10px !important;
}
/* hide the actual radio dot */
#domain-selector > .wrap label input[type="radio"] {
    display: none !important;
}
/* domain icons via ::before */
#domain-selector > .wrap label:nth-child(1)::before { content: '🔩 '; }
#domain-selector > .wrap label:nth-child(2)::before { content: '💻 '; }
#domain-selector > .wrap label:nth-child(3)::before { content: '🏗️ '; }

/* ── OTHER RADIO BUTTONS (generic) ── */
.glass-card .wrap { gap: 10px !important; }
.glass-card .wrap label {
    background: #ffffff !important;
    border: 1px solid rgba(124, 58, 237, 0.15) !important;
    border-radius: 14px !important;
    padding: 10px 18px !important;
    font-weight: 700 !important;
    color: #4c4f7a !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02) !important;
}
.glass-card .wrap label:hover {
    border-color: #7c3aed !important;
    color: #7c3aed !important;
}
.glass-card .selected {
    border-color: #7c3aed !important;
    color: #7c3aed !important;
    background: rgba(124, 58, 237, 0.05) !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1) !important;
}

/* ── INTERACTION ELEMENTS ── */
.glass-card label span {
    font-size: 0.75rem !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #8b8fb5 !important;
    margin-bottom: 8px !important;
}

input[type="range"] { accent-color: #7c3aed !important; }

/* ── URL INPUT ROW ── */
.url-input-row {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    margin-top: 10px !important;
}
#url-input-p1 textarea, #url-input-p2 textarea,
#url-input-p1 input,    #url-input-p2 input {
    background: #ffffff !important;
    border: 1.5px solid #e0dff5 !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
    font-size: 13px !important;
    color: #1e1b4b !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    box-shadow: none !important;
}
#url-input-p1 textarea:focus, #url-input-p2 textarea:focus,
#url-input-p1 input:focus,    #url-input-p2 input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.10) !important;
    outline: none !important;
}
#url-btn {
    background: #ffffff !important;
    border: 1.5px solid rgba(124,58,237,0.25) !important;
    border-radius: 12px !important;
    color: #7c3aed !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 10px 18px !important;
    white-space: nowrap !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
    width: auto !important;
}
#url-btn:hover {
    background: rgba(124,58,237,0.07) !important;
    border-color: #7c3aed !important;
    transform: none !important;
    box-shadow: 0 2px 8px rgba(124,58,237,0.15) !important;
}
#url-status .prose p {
    font-size: 12px !important;
    margin: 4px 0 0 2px !important;
    color: #6b7280 !important;
}

/* ── PRIMARY ACTION BUTTON ── */
#run-btn, #anomaly-btn {
    margin-top: 16px !important;
    background: linear-gradient(135deg, #7c3aed 0%, #ec4899 100%) !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 18px !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    box-shadow: 0 8px 32px rgba(124, 58, 237, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
}
#run-btn:hover, #anomaly-btn:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 12px 40px rgba(124, 58, 237, 0.45) !important;
}

/* ── INPUT METHOD SELECTOR (tab-strip above image) ── */
#input-method-p1 > .wrap,
#input-method-p2 > .wrap {
    display: flex !important;
    flex-direction: row !important;
    gap: 0 !important;
    background: #f1eeff !important;
    border-radius: 14px 14px 0 0 !important;
    padding: 4px !important;
    border: 1.5px solid rgba(124,58,237,0.14) !important;
    border-bottom: none !important;
    box-sizing: border-box !important;
    margin-bottom: 0 !important;
}
#input-method-p1 > .wrap label,
#input-method-p2 > .wrap label {
    flex: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 8px 4px !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    color: #7b7fa8 !important;
    cursor: pointer !important;
    transition: all 0.22s ease !important;
    user-select: none !important;
    box-shadow: none !important;
}
#input-method-p1 > .wrap label:hover,
#input-method-p2 > .wrap label:hover {
    color: #7c3aed !important;
    background: rgba(124,58,237,0.06) !important;
}
#input-method-p1 > .wrap label:has(input:checked),
#input-method-p2 > .wrap label:has(input:checked) {
    background: #ffffff !important;
    color: #7c3aed !important;
    font-weight: 800 !important;
    box-shadow: 0 2px 10px rgba(124,58,237,0.15), 0 1px 3px rgba(0,0,0,0.06) !important;
    border-radius: 10px !important;
}
#input-method-p1 > .wrap label input[type="radio"],
#input-method-p2 > .wrap label input[type="radio"] {
    display: none !important;
}
/* attach image-area border to selector so they look joined */
#input-method-p1 + div > .svelte-1ipelgc,
#input-method-p2 + div > .svelte-1ipelgc {
    border-top-left-radius: 0 !important;
    border-top-right-radius: 0 !important;
}

/* ── DISABLED PRIMARY BUTTON ── */
#run-btn:disabled, #run-btn[disabled],
#anomaly-btn:disabled, #anomaly-btn[disabled] {
    background: linear-gradient(135deg, #c4b5fd 0%, #f9a8d4 100%) !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
    transform: none !important;
    opacity: 0.65 !important;
}

/* ── HIDE FOOTER ── */
footer { display: none !important; }

/* ── PDF DOWNLOAD BEAM ── */
.pdf-beam-wrap { margin-top: 14px; }
.pdf-beam {
    display: flex;
    align-items: center;
    gap: 14px;
    width: 100%;
    box-sizing: border-box;
    background: linear-gradient(135deg, #fdf4ff 0%, #f5f3ff 100%);
    border: 1.5px solid rgba(124,58,237,0.18);
    border-radius: 18px;
    padding: 16px 20px;
    text-decoration: none;
    transition: all 0.22s ease;
    cursor: pointer;
    position: relative;
    overflow: hidden;
}
.pdf-beam::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(124,58,237,0.06), rgba(236,72,153,0.04));
    opacity: 0;
    transition: opacity 0.22s ease;
}
.pdf-beam:hover { border-color: #7c3aed; box-shadow: 0 6px 24px rgba(124,58,237,0.15); transform: translateY(-2px); }
.pdf-beam:hover::before { opacity: 1; }
.pdf-icon-circle {
    flex-shrink: 0;
    width: 46px; height: 46px;
    border-radius: 14px;
    background: linear-gradient(135deg, #7c3aed, #ec4899);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.35rem;
    box-shadow: 0 4px 12px rgba(124,58,237,0.25);
}
.pdf-meta { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 2px; }
.pdf-filename {
    font-weight: 800;
    font-size: 0.88rem;
    color: #1e1b4b;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.pdf-filesize {
    font-size: 0.72rem;
    font-weight: 600;
    color: #8b8fb5;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.pdf-dl-badge {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: 5px;
    background: rgba(124,58,237,0.10);
    color: #7c3aed;
    font-weight: 800;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    padding: 6px 12px;
    border-radius: 100px;
}
.pdf-beam-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    width: 100%;
    box-sizing: border-box;
    border: 1.5px dashed rgba(124,58,237,0.18);
    border-radius: 18px;
    padding: 16px 20px;
    margin-top: 14px;
    color: #a0a3c4;
    font-size: 0.82rem;
    font-weight: 600;
}

/* hide the raw gr.File component */
#report-download-hidden { display: none !important; }

/* ── OUTPUT IMAGE CHIP LABEL ── */
.img-chip-label {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: linear-gradient(135deg, rgba(124,58,237,0.08), rgba(236,72,153,0.06));
    border: 1.5px solid rgba(124,58,237,0.13);
    border-radius: 12px 12px 0 0;
    padding: 7px 10px 7px 16px;
    width: 100%;
    box-sizing: border-box;
    margin-bottom: -2px;
}
.img-chip-icon { font-size: 1rem; line-height: 1; }
.img-chip-text {
    font-size: 0.75rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #6d28d9;
    flex: 1;
}
.img-chip-toolbar {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-left: auto;
    flex-shrink: 0;
}
.img-chip-btn {
    background: rgba(124,58,237,0.07);
    border: 1px solid rgba(124,58,237,0.15);
    border-radius: 8px;
    color: #7c3aed;
    font-size: 0.8rem;
    width: 28px; height: 28px;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer;
    transition: all 0.18s ease;
    padding: 0;
    line-height: 1;
}
.img-chip-btn:hover {
    background: rgba(124,58,237,0.15);
    border-color: #7c3aed;
    transform: scale(1.08);
}
/* fullscreen modal */
#dv-modal {
    position: fixed;
    inset: 0;
    z-index: 99999;
    background: rgba(0,0,0,0.90);
    backdrop-filter: blur(6px);
    display: none;
    align-items: center;
    justify-content: center;
    cursor: zoom-out;
}
#dv-modal.active { display: flex !important; }
#dv-modal-img {
    max-width: 92vw;
    max-height: 90vh;
    border-radius: 16px;
    object-fit: contain;
    box-shadow: 0 30px 80px rgba(0,0,0,0.6);
    cursor: default;
}
#dv-modal-close {
    position: fixed;
    top: 20px; right: 24px;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18);
    color: #fff;
    border-radius: 50%;
    width: 42px; height: 42px;
    font-size: 1.1rem;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: background 0.18s;
}
#dv-modal-close:hover { background: rgba(255,255,255,0.22); }
#dv-modal-hint {
    position: fixed;
    bottom: 22px;
    left: 50%;
    transform: translateX(-50%);
    color: rgba(255,255,255,0.45);
    font-size: 0.75rem;
    font-family: Inter, sans-serif;
    pointer-events: none;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.7); }
}
/* connect image block border flush to chip label above */
#output-image-p1 > div,
#output-image-p2 > div {
    border-radius: 0 0 14px 14px !important;
    border: 1.5px solid rgba(124,58,237,0.13) !important;
    border-top: none !important;
    overflow: hidden !important;
}
/* fullscreen trigger button */
#fs-btn-p1, #fs-btn-p2 {
    margin-top: 6px !important;
    background: rgba(124,58,237,0.06) !important;
    border: 1.5px solid rgba(124,58,237,0.15) !important;
    border-radius: 12px !important;
    color: #7c3aed !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    padding: 7px 14px !important;
    width: auto !important;
    box-shadow: none !important;
    transition: all 0.18s ease !important;
}
#fs-btn-p1:hover, #fs-btn-p2:hover {
    background: rgba(124,58,237,0.13) !important;
    border-color: #7c3aed !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── MODEL STATUS CHIPS ── */
.model-status-bar {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 18px;
    padding-bottom: 16px;
    border-bottom: 1.5px solid rgba(124,58,237,0.08);
}
.model-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    border: 1.5px solid;
}
.model-chip-on  { background: #f0fdf4; border-color: #86efac; color: #15803d; }
.model-chip-off { background: #fafafa; border-color: #e5e7eb; color: #9ca3af; }
.model-chip-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.model-chip-on  .model-chip-dot { background: #22c55e; }
.model-chip-off .model-chip-dot { background: #d1d5db; }
.model-status-warn {
    font-size: 0.75rem;
    font-weight: 600;
    color: #d97706;
    background: #fffbeb;
    border: 1.5px solid #fcd34d;
    border-radius: 12px;
    padding: 8px 14px;
    margin-bottom: 18px;
}

#about-content {
    background: rgba(255, 255, 255, 0.6) !important;
    border-radius: 32px !important;
    padding: 60px !important;
    margin: 40px auto !important;
    max-width: 900px !important;
    border: 1px solid rgba(124, 58, 237, 0.1) !important;
}

/* ── RESULTS CARD ENHANCED ── */
.result-placeholder {
    text-align: center;
    padding: 40px 16px;
    color: #a0a3c4;
    font-size: 0.9rem;
    font-style: italic;
}
.result-summary { padding: 2px 0; }

/* stats row */
.result-stats-row {
    display: flex;
    gap: 12px;
    margin-bottom: 18px;
}
.stat-chip {
    flex: 1;
    background: #f8f6ff;
    border: 1.5px solid rgba(124,58,237,0.12);
    border-radius: 16px;
    padding: 14px 12px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 3px;
}
.stat-number {
    font-size: 2rem;
    font-weight: 900;
    color: #7c3aed;
    line-height: 1;
}
.stat-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8b8fb5;
}

/* severity chip variants */
.sev-none, .sev-low   { background: #f0fdf4 !important; border-color: #bbf7d0 !important; }
.sev-medium           { background: #fffbeb !important; border-color: #fcd34d !important; }
.sev-high             { background: #fef2f2 !important; border-color: #fecaca !important; }
.sev-none .sev-label, .sev-low .sev-label { color: #15803d !important; }
.sev-medium .sev-label { color: #d97706 !important; }
.sev-high   .sev-label { color: #dc2626 !important; }
.sev-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    margin-bottom: 1px;
}
.sev-none .sev-dot, .sev-low .sev-dot { background: #22c55e; }
.sev-medium .sev-dot { background: #f59e0b; }
.sev-high   .sev-dot { background: #ef4444; }
.sev-label {
    font-size: 1rem;
    font-weight: 900;
    letter-spacing: 0.04em;
}

/* defect breakdown list */
.defect-list { display: flex; flex-direction: column; gap: 7px; margin-top: 2px; }
.defect-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    border: 1px solid rgba(124,58,237,0.10);
    border-radius: 12px;
    padding: 9px 14px;
}
.defect-name {
    font-weight: 600;
    font-size: 0.85rem;
    color: #312e81;
    text-transform: capitalize;
}
.defect-count-badge {
    background: rgba(124,58,237,0.10);
    color: #7c3aed;
    font-weight: 800;
    font-size: 0.78rem;
    padding: 3px 10px;
    border-radius: 100px;
}

/* good/clean state */
.good-state {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #f0fdf4;
    border: 1.5px solid #bbf7d0;
    border-radius: 14px;
    padding: 14px 18px;
    margin-top: 2px;
}
.good-icon { font-size: 1.3rem; }
.good-text { font-weight: 700; color: #15803d; font-size: 0.88rem; }

/* domain badge */
.domain-badge {
    display: inline-block;
    padding: 4px 14px;
    background: rgba(124,58,237,0.08);
    color: #7c3aed;
    border-radius: 100px;
    font-size: 0.68rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 14px;
}

/* divider */
.result-divider {
    height: 1px;
    background: rgba(124,58,237,0.08);
    margin: 14px 0;
}

/* anomaly-specific */
.anomaly-score-row {
    display: flex;
    gap: 12px;
    margin-bottom: 14px;
}
.anomaly-score-chip {
    flex: 2;
    background: #f8f6ff;
    border: 1.5px solid rgba(124,58,237,0.15);
    border-radius: 16px;
    padding: 14px 12px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
}
.anomaly-score-num {
    font-size: 2rem;
    font-weight: 900;
    color: #7c3aed;
    line-height: 1;
}
.score-bar-track {
    width: 100%;
    height: 6px;
    background: #e5e7eb;
    border-radius: 100px;
    margin-top: 6px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 100px;
}
.verdict-chip {
    flex: 1;
    border-radius: 16px;
    padding: 14px 10px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 6px;
    border: 1.5px solid;
}
.verdict-normal { background: #f0fdf4; border-color: #bbf7d0; }
.verdict-anom   { background: #fef2f2; border-color: #fecaca; }
.verdict-icon   { font-size: 1.5rem; line-height: 1; }
.verdict-text {
    font-weight: 900;
    font-size: 0.85rem;
    letter-spacing: 0.04em;
}
.verdict-normal .verdict-text { color: #15803d; }
.verdict-anom   .verdict-text { color: #dc2626; }
.anomaly-detail {
    font-size: 0.83rem;
    color: #6b7280;
    font-weight: 500;
    margin: 2px 2px 0;
    line-height: 1.5;
    padding: 10px 14px;
    background: #f9fafb;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
}
"""

# ========== Hero HTML ==========

HERO_HTML = """
<div id="hero-wrap">
  <div class="hero-badge">✦ AI-Powered Inspection</div>
  <h1 class="hero-title">DefectVision AI</h1>
  <p class="hero-sub">Multi-Domain Surface Defect Inspection System</p>
  <div class="hero-chips">
    <span class="chip">⚡ YOLOv8 Detection</span>
    <span class="chip">🧠 PatchCore Anomaly</span>
    <span class="chip">📄 PDF Reports</span>
    <span class="chip">🏆 Hackathon 2026</span>
  </div>
</div>
"""

# ========== Gradio Theme ==========

LIGHT_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#f8faff",
    body_text_color="#2d2b57",
    block_background_fill="transparent",
    block_border_color="transparent",
    block_title_text_weight="700",
    block_label_text_size="0.75rem",
    block_label_text_weight="800",
    input_background_fill="#ffffff",
    input_border_color="rgba(124,58,237,0.12)",
    input_border_color_focus="rgba(124,58,237,0.4)",
    button_primary_background_fill="linear-gradient(135deg, #7c3aed, #ec4899)",
    button_primary_text_color="white",
    button_primary_border_color="transparent",
    button_large_padding="16px 32px",
    checkbox_label_background_fill="rgba(255,255,255,0.8)",
)

# ========== Input Method Helpers ==========

def _switch_input_method(method):
    """Returns updates for (image_component, url_row, url_status, run_btn)."""
    if method == "📁 Upload":
        return (
            gr.update(sources=["upload"], value=None, interactive=True, visible=True),
            gr.update(visible=False),
            gr.update(value=""),
            gr.update(interactive=False),
        )
    elif method == "📷 Webcam":
        return (
            gr.update(sources=["webcam"], value=None, interactive=True, visible=True),
            gr.update(visible=False),
            gr.update(value=""),
            gr.update(interactive=False),
        )
    else:  # URL
        return (
            gr.update(sources=["upload"], value=None, interactive=True, visible=False),
            gr.update(visible=True),
            gr.update(value=""),
            gr.update(interactive=False),
        )


def _toggle_run_btn(image):
    return gr.update(interactive=(image is not None))


def _clear_results_p1(image):
    """Wipe Phase-1 results when the image is removed."""
    if image is not None:
        return gr.update(), gr.update(), gr.update()
    return (
        gr.update(value=None),
        gr.update(value=P1_PLACEHOLDER),
        gr.update(value=PDF_PLACEHOLDER),
    )


def _clear_results_p2(image):
    """Wipe Phase-2 results when the image is removed."""
    if image is not None:
        return gr.update(), gr.update()
    return (
        gr.update(value=None),
        gr.update(value=P2_PLACEHOLDER),
    )


MODAL_AND_JS_HTML = """
<div id="dv-modal" onclick="dvCloseModal()">
  <img id="dv-modal-img" src="" alt="" onclick="event.stopPropagation()">
  <button id="dv-modal-close" onclick="event.stopPropagation();dvCloseModal()">&#x2715;</button>
  <div id="dv-modal-hint">Click anywhere outside to close &nbsp;&nbsp; ESC to dismiss</div>
</div>
<script>
function dvGetImg(elemId) {
    var c = document.getElementById(elemId);
    if (!c) return null;
    // Find the actual result image - skip tiny placeholders and gifs
    var imgs = c.querySelectorAll('img');
    for (var i = 0; i < imgs.length; i++) {
        var s = imgs[i].src || '';
        if (s && !s.startsWith('data:image/gif') && !s.endsWith('.gif') && s !== window.location.href) {
            return imgs[i];
        }
    }
    return null;
}
function dvMaximize(elemId) {
    var img = dvGetImg(elemId);
    if (!img) {
        alert('No result image yet — run an inspection first.');
        return;
    }
    document.getElementById('dv-modal-img').src = img.src;
    document.getElementById('dv-modal').classList.add('active');
    document.body.style.overflow = 'hidden';
}
function dvShare(elemId) {
    var img = dvGetImg(elemId);
    if (!img) {
        alert('No result image yet — run an inspection first.');
        return;
    }
    var src = img.src;
    if (navigator.share && navigator.canShare) {
        fetch(src)
          .then(function(r){ return r.blob(); })
          .then(function(blob){
              var file = new File([blob], 'defectvision-result.png', {type: blob.type || 'image/png'});
              if (navigator.canShare({files:[file]})) {
                  navigator.share({files:[file], title:'DefectVision AI Result'})
                    .catch(function(){ window.open(src,'_blank'); });
              } else {
                  window.open(src,'_blank');
              }
          })
          .catch(function(){ window.open(src,'_blank'); });
    } else {
        window.open(src, '_blank');
    }
}
function dvCloseModal() {
    document.getElementById('dv-modal').classList.remove('active');
    document.body.style.overflow = '';
}
document.addEventListener('keydown', function(e){ if(e.key==='Escape') dvCloseModal(); });
</script>
"""


# ========== Build Gradio Interface ==========

def create_app():
    available = yolo_detector.get_available_domains()

    _all_domains = [
        ("Metal",    "🔩"),
        ("PCB",      "💻"),
        ("Building", "🏗️"),
    ]
    _default_domain = next((d for d,_ in _all_domains if d.lower() in available), _all_domains[0][0])
    _available_json = '[' + ','.join(f'"{d.lower()}"' for d,_ in _all_domains if d.lower() in available) + ']'

    # Build compact pill-strip buttons (matches Upload/Webcam/URL style)
    _card_items = []
    for d, icon in _all_domains:
        is_on = d.lower() in available
        dot_html = '<span class="dv-live-dot"></span>' if is_on else ''
        dis_attr = '' if is_on else 'disabled'
        sel_cls  = ' dv-card-selected' if d == _default_domain else ''
        off_cls  = ' dv-card-off' if not is_on else ''
        _card_items.append(
            f'<button class="dv-card{sel_cls}{off_cls}" data-val="{d}" '
            f'data-active="{str(is_on).lower()}" {dis_attr}>'
            f'{icon}&nbsp;{d}{dot_html}'
            f'</button>'
        )
    _cards_html = ''.join(_card_items)

    domain_picker_html = f"""
<style>
@keyframes dvpulse {{
  0%   {{ box-shadow:0 0 0 0   rgba(34,197,94,.85); }}
  65%  {{ box-shadow:0 0 0 8px rgba(34,197,94,0);   }}
  100% {{ box-shadow:0 0 0 0   rgba(34,197,94,0);   }}
}}
.dv-seg-track {{
  display: flex !important;
  flex-direction: row !important;
  gap: 0 !important;
  background: #f1eeff !important;
  border-radius: 14px !important;
  padding: 4px !important;
  border: 1.5px solid rgba(124,58,237,0.14) !important;
  box-sizing: border-box !important;
  margin-bottom: 12px !important;
}}
.dv-card {{
  flex: 1 !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  background: transparent !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 8px 4px !important;
  font-weight: 700 !important;
  font-size: 12px !important;
  line-height: 1 !important;
  color: #7b7fa8 !important;
  cursor: pointer !important;
  transition: all 0.22s ease !important;
  user-select: none !important;
  white-space: nowrap !important;
  box-shadow: none !important;
}}
.dv-card:hover:not([disabled]) {{
  color: #7c3aed !important;
  background: rgba(124,58,237,0.06) !important;
}}
.dv-card.dv-card-selected {{
  background: #ffffff !important;
  color: #7c3aed !important;
  font-weight: 800 !important;
  box-shadow: 0 2px 10px rgba(124,58,237,0.15), 0 1px 3px rgba(0,0,0,0.06) !important;
}}
.dv-card.dv-card-off {{
  opacity: 0.38 !important;
  cursor: not-allowed !important;
}}
.dv-live-dot {{
  display: inline-block;
  width: 6px; height: 6px;
  border-radius: 50%;
  background: #22c55e;
  margin-left: 5px;
  vertical-align: middle;
  position: relative; top: -1px;
  flex-shrink: 0;
  animation: dvpulse 1.8s ease-in-out infinite;
}}
/* strip out gr.HTML block wrapper padding */
#domain-picker {{
  padding: 0 !important;
  margin: 0 !important;
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
}}
#domain-picker > div {{
  padding: 0 !important;
  margin: 0 !important;
}}
.dv-dom-section-label {{
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}}
.dv-dom-section-label::before {{
  content: '';
  display: inline-block;
  width: 3px;
  height: 16px;
  border-radius: 2px;
  background: linear-gradient(180deg, #7c3aed, #a78bfa);
  flex-shrink: 0;
}}
.dv-dom-section-label span {{
  font-size: 10.5px;
  font-weight: 700;
  color: #7c3aed;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  line-height: 1;
}}
</style>
<div class="dv-dom-section-label"><span>Inspection Domain</span></div>
<div class="dv-seg-track" id="dv-seg-track">{_cards_html}</div>

<script>
(function(){{
  var DEFAULT = '{_default_domain}';
  function dvSetDomain(val) {{
    document.querySelectorAll('.dv-card').forEach(function(b) {{
      b.classList.toggle('dv-card-selected', b.dataset.val === val);
    }});
    var wrap = document.getElementById('domain-val');
    if (!wrap) return;
    var inp = wrap.querySelector('textarea,input');
    if (!inp) return;
    var desc = Object.getOwnPropertyDescriptor(
      inp.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype, 'value');
    if (desc && desc.set) desc.set.call(inp, val); else inp.value = val;
    inp.dispatchEvent(new Event('input',  {{bubbles:true}}));
    inp.dispatchEvent(new Event('change', {{bubbles:true}}));
  }}
  document.addEventListener('click', function(e) {{
    var btn = e.target.closest('.dv-card');
    if (!btn || btn.disabled || btn.dataset.active !== 'true') return;
    dvSetDomain(btn.dataset.val);
  }});
  [200, 600, 1500].forEach(function(t) {{ setTimeout(function(){{ dvSetDomain(DEFAULT); }}, t); }});
}})();
</script>
"""

    with gr.Blocks(title="DefectVision AI") as app:

        # ── Hero ──
        gr.HTML(HERO_HTML)
        gr.HTML(MODAL_AND_JS_HTML)

        with gr.Tabs(elem_id="main-tabs"):

            # ── Tab 1: Phase 1 ──
            with gr.Tab("⚡  Phase 1 — Defect Detection", id="phase1"):

                gr.Markdown(
                    "**Supervised Detection** &nbsp;·&nbsp; Identifies and localizes specific defect types "
                    "using YOLOv8 models trained on domain-specific labeled datasets. Outputs bounding boxes, "
                    "class labels, confidence scores, and a downloadable PDF inspection report.",
                    elem_id="phase1-desc",
                )

                with gr.Row(equal_height=False):

                    with gr.Column(scale=1):
                        with gr.Group(elem_classes="glass-card"):
                            gr.Markdown("### 🎯 Inspection Setup", elem_classes="card-header-main")

                            domain_selector = gr.Textbox(
                                value=_default_domain,
                                visible=False,
                                elem_id="domain-val",
                                label="domain-val",
                            )
                            gr.HTML(domain_picker_html, elem_id="domain-picker")
                            confidence_slider = gr.Slider(
                                minimum=0.1, maximum=1.0,
                                value=DEFAULT_CONFIDENCE, step=0.05,
                                label="Confidence Threshold",
                                info="Lower values = more sensitivity.",
                            )

                            # ── Input method tab-strip ──
                            input_method_p1 = gr.Radio(
                                choices=["📁 Upload", "📷 Webcam", "🔗 URL"],
                                value="📁 Upload",
                                label="Image Source",
                                show_label=False,
                                elem_id="input-method-p1",
                            )
                            gr.HTML(
                                '<div class="img-chip-label">'
                                '<span class="img-chip-icon">🖼️</span>'
                                '<span class="img-chip-text">Surface Image</span>'
                                '</div>'
                            )
                            input_image = gr.Image(
                                label="Surface Image",
                                show_label=False,
                                type="numpy",
                                sources=["upload"],
                                height=260,
                                buttons=[],
                            )

                            # ── URL row (hidden by default) ──
                            with gr.Row(visible=False, elem_classes="url-input-row") as url_row_p1:
                                url_input_p1 = gr.Textbox(
                                    placeholder="🔗  Paste image URL and click Load…",
                                    show_label=False,
                                    elem_id="url-input-p1",
                                    scale=4,
                                )
                                url_btn_p1 = gr.Button(
                                    "Load",
                                    size="sm",
                                    elem_id="url-btn",
                                    scale=1,
                                )
                            url_status_p1 = gr.Markdown("", elem_id="url-status")

                            detect_btn = gr.Button(
                                "⚡  Run Inspection",
                                variant="primary", size="lg",
                                elem_id="run-btn",
                                interactive=False,
                            )

                    with gr.Column(scale=1):
                        with gr.Group(elem_classes="glass-card"):
                            gr.Markdown("### 📊 Analysis Results", elem_classes="card-header-main")

                            gr.HTML(
                                '<div class="img-chip-label">'
                                '<span class="img-chip-icon">🔍</span>'
                                '<span class="img-chip-text">Annotated Detection Output</span>'
                                '</div>'
                            )
                            output_image = gr.Image(
                                label="Annotated Detection Output",
                                show_label=False,
                                type="numpy", height=300,
                                elem_id="output-image-p1",
                                buttons=["download"],
                            )
                            summary_output = gr.HTML(
                                value=P1_PLACEHOLDER,
                                elem_id="summary-output",
                            )
                            report_download = gr.HTML(
                                value=PDF_PLACEHOLDER,
                                elem_id="report-download",
                            )

                detect_btn.click(
                    fn=run_yolo_detection,
                    inputs=[input_image, domain_selector, confidence_slider],
                    outputs=[output_image, summary_output, report_download],
                )
                # input-method switcher wires
                input_method_p1.change(
                    fn=_switch_input_method,
                    inputs=[input_method_p1],
                    outputs=[input_image, url_row_p1, url_status_p1, detect_btn],
                )
                # enable/disable run button based on image presence
                input_image.change(
                    fn=_toggle_run_btn,
                    inputs=[input_image],
                    outputs=[detect_btn],
                )
                # silently clear results when image is removed
                input_image.change(
                    fn=_clear_results_p1,
                    inputs=[input_image],
                    outputs=[output_image, summary_output, report_download],
                )
                url_btn_p1.click(
                    fn=load_image_from_url,
                    inputs=[url_input_p1],
                    outputs=[input_image, url_status_p1],
                )
                url_input_p1.submit(
                    fn=load_image_from_url,
                    inputs=[url_input_p1],
                    outputs=[input_image, url_status_p1],
                )

            # ── Tab 2: Phase 2 ──
            with gr.Tab("🔬  Phase 2 — Anomaly Detection", id="phase2"):

                gr.Markdown(
                    "**Unsupervised Anomaly Detection** &nbsp;·&nbsp; Trained exclusively on normal images "
                    "using PatchCore. Detects *any* abnormality without requiring labeled defect data. "
                    "Outputs an anomaly heatmap and a confidence score.",
                    elem_id="phase2-desc",
                )

                with gr.Row(equal_height=False):

                    with gr.Column(scale=1):
                        with gr.Group(elem_classes="glass-card"):
                            gr.Markdown("### 🔍 Image Input", elem_classes="card-header-main")

                            # ── Input method tab-strip ──
                            input_method_p2 = gr.Radio(
                                choices=["📁 Upload", "📷 Webcam", "🔗 URL"],
                                value="📁 Upload",
                                label="Image Source",
                                show_label=False,
                                elem_id="input-method-p2",
                            )
                            gr.HTML(
                                '<div class="img-chip-label">'
                                '<span class="img-chip-icon">🖼️</span>'
                                '<span class="img-chip-text">Surface Image</span>'
                                '</div>'
                            )
                            anomaly_input = gr.Image(
                                label="Surface Image",
                                show_label=False,
                                type="numpy",
                                sources=["upload"],
                                height=280,
                                buttons=[],
                            )

                            # ── URL row (hidden by default) ──
                            with gr.Row(visible=False, elem_classes="url-input-row") as url_row_p2:
                                url_input_p2 = gr.Textbox(
                                    placeholder="🔗  Paste image URL and click Load…",
                                    show_label=False,
                                    elem_id="url-input-p2",
                                    scale=4,
                                )
                                url_btn_p2 = gr.Button(
                                    "Load",
                                    size="sm",
                                    elem_id="url-btn",
                                    scale=1,
                                )
                            url_status_p2 = gr.Markdown("", elem_id="url-status")

                            anomaly_btn = gr.Button(
                                "🧠  Detect Anomalies",
                                variant="primary", size="lg",
                                elem_id="anomaly-btn",
                                interactive=False,
                            )

                    with gr.Column(scale=1):
                        with gr.Group(elem_classes="glass-card"):
                            gr.Markdown("### 🌡️ Anomaly Heatmap", elem_classes="card-header-main")

                            gr.HTML(
                                '<div class="img-chip-label">'
                                '<span class="img-chip-icon">🌡️</span>'
                                '<span class="img-chip-text">Anomaly Visualization</span>'
                                '</div>'
                            )
                            anomaly_output = gr.Image(
                                label="Anomaly Visualization",
                                show_label=False,
                                type="numpy", height=340,
                                elem_id="output-image-p2",
                                buttons=["download"],
                            )
                            anomaly_summary = gr.HTML(
                                value=P2_PLACEHOLDER,
                                elem_id="anomaly-summary",
                            )

                anomaly_btn.click(
                    fn=run_anomaly_detection,
                    inputs=[anomaly_input],
                    outputs=[anomaly_output, anomaly_summary],
                )
                # input-method switcher wires
                input_method_p2.change(
                    fn=_switch_input_method,
                    inputs=[input_method_p2],
                    outputs=[anomaly_input, url_row_p2, url_status_p2, anomaly_btn],
                )
                # enable/disable detect button based on image presence
                anomaly_input.change(
                    fn=_toggle_run_btn,
                    inputs=[anomaly_input],
                    outputs=[anomaly_btn],
                )
                # silently clear results when image is removed
                anomaly_input.change(
                    fn=_clear_results_p2,
                    inputs=[anomaly_input],
                    outputs=[anomaly_output, anomaly_summary],
                )
                url_btn_p2.click(
                    fn=load_image_from_url,
                    inputs=[url_input_p2],
                    outputs=[anomaly_input, url_status_p2],
                )
                url_input_p2.submit(
                    fn=load_image_from_url,
                    inputs=[url_input_p2],
                    outputs=[anomaly_input, url_status_p2],
                )

                if not anomaly_detector.is_available():
                    gr.Markdown(
                        "⚠️ *Phase 2 model not loaded. "
                        "Train it using `04_train_anomaly.ipynb` in Google Colab.*"
                    )

            # ── Tab 3: About ──
            with gr.Tab("ℹ️  About", id="about"):
                gr.Markdown(
                    """
## How DefectVision AI Works

### ⚡ Phase 1 — Supervised Detection (YOLOv8)
- Trained on labeled datasets with bounding box annotations
- Identifies **specific defect types** (cracks, scratches, solder bridges, etc.)
- Works across 3 domains: **Metal**, **PCB**, **Building**
- Outputs: bounding boxes, class labels, confidence scores, PDF report

### 🧠 Phase 2 — Anomaly Detection (PatchCore)
- Trained on **only normal/good images** — no defect labels needed
- Learns what "normal" looks like, then flags anything different
- Outputs: anomaly heatmap + anomaly score

---

### 🗂️ Supported Domains & Defect Types

| Domain | Defect Types |
|--------|-------------|
| **Metal** | Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches |
| **PCB** | Missing Hole, Mouse Bite, Open Circuit, Short, Spur, Spurious Copper |
| **Building** | Crack, Spalling, Corrosion, Exposed Rebar |

---

### 🏗️ System Architecture
- **Frontend**: Gradio Blocks with custom light glassmorphism UI
- **Phase 1 Engine**: Ultralytics YOLOv8 (object detection)
- **Phase 2 Engine**: PatchCore (memory-bank anomaly detection)
- **Report Generation**: ReportLab PDF with annotated images

---
*DefectVision AI — Hackathon 2026 · Built with ❤️ and Python*
                    """,
                    elem_id="about-content",
                )

    return app


# ========== Main ==========
if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        share=False,
        theme=LIGHT_THEME,
        css=CUSTOM_CSS,
        allowed_paths=[tempfile.gettempdir()],
    )