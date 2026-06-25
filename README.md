# DecodeLabs Industrial Training: AI Image & Text Recognition Pipeline

[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)

A modular, production-grade AI recognition pipeline designed for the **DecodeLabs Industrial Training Kit (Project 4)**. This system features two discrete, high-performance processing branches: **Path 1 (OCR)** using Google Tesseract for localized text extraction, and **Path 2 (Object Detection)** using MobileNet-SSD via OpenCV's DNN module for real-time item detection. 

The repository includes a robust CLI engine, strict data contract schemas, a 136-test suite validation framework, and a modern, cyber-industrial web dashboard for local pipeline visualization.

---

## 📸 Dashboard Overview
The visual interface represents a state-of-the-art diagnostic dashboard utilizing the **Deep Sonar** design system. It includes:
* **Interactive Control Hub**: Switch pipeline modes, upload images, select Tesseract Page Segmentation Modes (PSM), and audit the 80% confidence threshold gate.
* **Dual-Mode Visualizer**: Active canvas showing bounding-box annotations (tiered by confidence) and structural layouts.
* **Live Console Telemetry**: Replicates pipeline terminal logs line-by-line with timestamps and severity indicators (`OK`, `INFO`, `WARN`, `ERROR`).
* **Milestone Gate Audit**: Verifies compliance with DecodeLabs validation standards in real time.

---

## 🛠️ Technology Stack
* **Language**: Python 3.8+ (Core engine) & Vanilla ES6+ HTML5/CSS3 (Dashboard)
* **Core Libraries**: 
  * `opencv-python`: Image ingestion, preprocessing (grayscale, blur, deskew, adaptive thresholding), and DNN inference.
  * `pytesseract`: Wrapped OCR text extraction with structure-preserving metadata.
  * `numpy` & `pandas`: Vectorized geometric operations and DataFrame filtration.
  * `imutils`: Structural orientation and deskew helpers.
  * `Pillow`: Image asset fallback loader.
* **Testing Framework**: `pytest`, `pytest-cov` (Target coverage: >90% code lines).

---

## 📂 Project Directory Structure

```directory
.
├── config.py                 # Configuration parameters & single source of truth
├── main.py                   # Unified CLI Entry point for pipeline execution
├── setup_models.py           # Downloader script for Caffe proto and weights
├── requirements.txt          # Pinned project dependencies
├── LICENSE                   # MIT License
├── README.md                 # Production-grade user documentation
├── docs/                     # Design Contracts and Specs
│   ├── ARCHITECTURE.md       # Dataflow, Module contracts and constraints
│   ├── PRD.md                # System requirements & confidence policies
│   ├── ALGORITHM_SPEC.md     # Preprocessing and network engineering math
│   └── TEST_PLAN.md          # 136-case verification matrix
├── src/                      # Source Code
│   ├── __init__.py
│   ├── config.py             # Internal configuration mappings
│   ├── schemas.py            # Dataclasses & Custom exception hierarchy
│   ├── utils.py              # Math, scaling, and file validators
│   ├── preprocessor.py       # Grayscale -> Blur -> Deskew -> Thresholding
│   ├── ocr_pipeline.py       # Tesseract wrapper & 5-step DataFrame filter
│   ├── detection_pipeline.py # MobileNet-SSD engine & NMS coordinate scaling
│   └── postprocessor.py      # Console logger & annotated image exporter
├── dashboard/                # Web Dashboard
│   ├── index.html            # Core DOM structure (semantic & accessible)
│   ├── style.css             # Cyberpunk theme (Deep Sonar palettes)
│   └── app.js                # State-machine & processing simulations
├── models/                   # Pre-trained Model weights (generated)
│   ├── MobileNetSSD_deploy.caffemodel
│   ├── MobileNetSSD_deploy.prototxt
│   └── coco_labels.txt       # VOC 20-class labels mappings
├── output/                   # Directory for annotated export images
└── tests/                    # 136-Case Verification Suite
    ├── conftest.py           # Shared fixtures (synthetic images, mocked dfs)
    ├── test_input_validation.py
    ├── test_preprocessing.py
    ├── test_confidence_filter.py
    ├── test_coordinate_scaling.py
    ├── test_bounding_box.py
    ├── test_text_reconstruction.py
    ├── test_output_save.py
    ├── test_config_validation.py
    ├── test_gates.py         # Milestone certification gates checker
    └── test_adversarial.py   # System anti-gaming/anti-cheat checks
```

---

## 🚀 Setup Instructions

### 1. Prerequisite Installations
Ensure your OS has Python 3.8+ and the Tesseract OCR binary engine installed:
```bash
# Ubuntu / Debian
sudo apt update
sudo apt install python3-pip python3-venv tesseract-ocr -y

# macOS (using Homebrew)
brew install tesseract
```

### 2. Clone and Setup Environment
Navigate to the directory, initialize a Python virtual environment, and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Fetch Pre-trained Weights
The Caffe model files for MobileNet-SSD are not committed to source control due to size constraints. Download them using the automated setup script:
```bash
python setup_models.py
```
This fetches the `.prototxt` definition and the `.caffemodel` weights, writing them directly to the `models/` directory.

---

## 💻 Running the Backend Pipeline via CLI

Execute the pipeline via the CLI entrypoint `src/main.py`. The pipeline accepts local images and exports annotated results to the `output/` directory alongside a structured JSON report.

```bash
# General usage structure
python src/main.py <input-image-path> [--mode {ocr,detection}] [--psm {0..13}] [--save-report]

# Example 1: Run OCR on an invoice image with PSM 3 (auto)
python src/main.py data/invoice.jpg --mode ocr --save-report

# Example 2: Run Object Detection on a warehouse snapshot
python src/main.py data/warehouse.png --mode detection
```

---

## 🧪 Verification & Testing Suite
The repository includes a comprehensive 136-test suite that validates all pipeline modules, data schemas, mathematical invariants, security boundaries, and milestone compliance rules.

Run the test suite using `pytest`:
```bash
# Run all tests with standard verbose logging
python -m pytest tests/ -v

# Run tests and generate an HTML code coverage report
python -m pytest tests/ --cov=src --cov-report=html
```

### Key Verified Areas:
* **`test_gates.py`**: Ensures all four Milestone Gates (Library Integration, Preprocessing, Benchmarking, Visual Confirmation) pass certification requirements.
* **`test_adversarial.py`**: Anti-gaming checks targeting coordinate scale integrity, type validation bypasses, Tesseract function workarounds, and confidence threshold manipulation.
* **`test_preprocessing.py`**: Asserts mathematical soundness of the 4-step image preprocessing chain.

---

## 🌐 Launching the Visual Dashboard
The dashboard is built entirely with modern vanilla HTML5, CSS3, and JavaScript, meaning it runs locally without any runtime dependencies.

1. Navigate to the dashboard directory:
   ```bash
   cd dashboard
   ```
2. Start a lightweight local web server:
   ```bash
   python3 -m http.server 8080
   ```
3. Open your browser and navigate to:
   [http://localhost:8080](http://localhost:8080)

---

## 🛡️ Algorithm & Safety Invariants

This system is configured with several strict engineering guardrails:
1. **The 80% Threshold Gate**: All detections (words or objects) with confidence scores `< 0.80` are rejected immediately. This threshold is controlled via a central constant in `src/config.py` and validated at build time.
2. **Coordinate Validation**: Bound coordinates ($x_1, y_1, x_2, y_2$) are enforced as integers. D degenerate ($width=0$ or $height=0$) and inverted ($x_1 \ge x_2$, $y_1 \ge y_2$) bounding boxes are rejected immediately at the schema level.
3. **No-Hardcoding Rule**: Algorithmic constants (e.g. NMS IoU threshold, Gaussian blur kernel size) are mapped configuration parameters.

---

## 📝 License
Distributed under the MIT License. See [LICENSE](LICENSE) for more details.
