# Project 4: AI Image & Text Recognition Pipeline
**DecodeLabs Industrial Training Kit | Batch 2026**

> Optional Mastery Milestone — Model Implementation Track

---

## What This Is

A Python-based computer vision pipeline that uses pre-trained AI models to either:
- **Extract text from images** using Google's Tesseract OCR engine (Path 1)
- **Detect and locate objects in images** using MobileNet-SSD (Path 2)

Choose one path, implement the pipeline, pass the four milestone gates, and submit for certification credit.

---

## Quick Start

```bash
# 1. Set up environment
python -m venv venv && source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate                            # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download model files (Path 2 only)
python setup_models.py

# 4. Install Tesseract (Path 1 only — see SETUP.md)

# 5. Configure your path
# Edit src/config.py → PIPELINE_MODE = "ocr" or "detection"

# 6. Run
python src/main.py --image sample_images/ocr_sample.jpg
```

---

## Documentation Index

| File | Purpose |
|------|---------|
| [`PRD.md`](PRD.md) | Full product requirements, acceptance criteria, risk register |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System design, data flows, module contracts |
| [`DESIGN.md`](DESIGN.md) | Visual design spec — colors, typography, animations, output style |
| [`SETUP.md`](SETUP.md) | Step-by-step environment setup for Windows/macOS/Linux |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | Common errors and their fixes |

---

## Milestone Gates (All 4 Required for Certification)

| Gate | Requirement |
|------|-------------|
| ✅ Gate 1 | Libraries import without errors; model files load |
| ✅ Gate 2 | Grayscale conversion + Gaussian Blur + Thresholding all applied |
| ✅ Gate 3 | `CONFIDENCE_THRESHOLD = 0.80` constant present; filters correctly |
| ✅ Gate 4 | Annotated output image saved to disk with labels and confidence values |

---

## Project Structure

```
project4-recognition/
├── src/
│   ├── main.py               ← Entry point
│   ├── config.py             ← All constants (change mode here)
│   ├── preprocessor.py       ← Image pre-processing pipeline
│   ├── ocr_pipeline.py       ← Path 1: OCR logic
│   ├── detection_pipeline.py ← Path 2: Object detection logic
│   ├── postprocessor.py      ← Output formatting and image annotation
│   └── utils.py              ← File validation, logging, I/O helpers
├── models/                   ← MobileNet-SSD weights (downloaded by setup_models.py)
├── sample_images/            ← Test images for both paths
├── output/                   ← Annotated results saved here
└── requirements.txt
```

---

## Choosing Your Path

| | Path 1: OCR | Path 2: Object Detection |
|--|-------------|-------------------------|
| **Task** | Read text from images | Find and locate objects |
| **Library** | `pytesseract` | `cv2.dnn` + MobileNet-SSD |
| **Output** | Extracted text strings | Bounding boxes with labels |
| **Best test image** | Invoice, sign, document | Street scene, room photo |
| **Extra install** | Tesseract binary | Run `setup_models.py` |

Set your choice in `src/config.py`:
```python
PIPELINE_MODE = "ocr"        # or "detection"
```

---

## Contact

**DecodeLabs**
- Email: decodelabs.tech@gmail.com
- Website: www.decodelabs.tech
- Phone: +91 89330 06408
- Location: Greater Lucknow, India
