# SETUP.md
## Project 4: Environment Setup Guide
**DecodeLabs | Batch 2026**

---

## Prerequisites

Before starting, confirm you have:
- Python 3.8 or higher (`python --version`)
- pip (`pip --version`)
- ~500MB free disk space (for model weights and libraries)
- An internet connection (one-time setup only — runtime is fully offline)

---

## Step 1: Clone / Download the Project

```bash
# If using Git
git clone https://github.com/decodelabs/project4-recognition.git
cd project4-recognition

# Or unzip the downloaded folder
unzip project4-recognition.zip
cd project4-recognition
```

---

## Step 2: Create a Virtual Environment (Strongly Recommended)

```bash
# Create the environment
python -m venv venv

# Activate it:
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt. All subsequent commands must be run inside this environment.

---

## Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` contents:
```
opencv-python==4.8.1.78
pytesseract==0.3.10
Pillow==10.0.1
numpy==1.24.4
imutils==0.5.4
```

If you encounter issues on Windows with `opencv-python`, try:
```bash
pip install opencv-python-headless==4.8.1.78
```

---

## Step 4: Install Tesseract OCR Binary (Path 1 Only)

`pytesseract` is only a Python wrapper — the actual Tesseract engine must be installed separately as a system binary.

### Windows

1. Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer. During installation, select **"Add to PATH"** checkbox.
3. Verify: open a new Command Prompt and run `tesseract --version`
4. If PATH wasn't set automatically, add this to `config.py`:
   ```python
   import pytesseract
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

### macOS

```bash
brew install tesseract
tesseract --version  # Verify
```

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install tesseract-ocr
tesseract --version  # Verify
```

### Verify the language pack (English)

```bash
tesseract --list-langs
# Should include: eng
```

If `eng` is missing:
- Windows: Re-run installer, select language data
- Ubuntu: `sudo apt install tesseract-ocr-eng`
- macOS: `brew install tesseract-lang`

---

## Step 5: Download Model Files (Path 2 Only)

Run the included setup script:

```bash
python setup_models.py
```

This script will:
1. Download `MobileNetSSD_deploy.prototxt` (~28KB)
2. Download `MobileNetSSD_deploy.caffemodel` (~23MB)
3. Download `coco_labels.txt` (91 class labels)
4. Place all files in `/models/`
5. Verify file integrity (MD5 checksum)

**If the script fails** (network issue), manually download:
- Prototxt: https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt
- Caffemodel: https://drive.google.com/open?id=0B3gersZ2cHIxRm5PMWRoTkdHdHc
- Labels: https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

Place them manually in the `models/` directory.

---

## Step 6: Verify the Installation

Run the verification script:

```bash
python src/utils.py --verify
```

Expected output:
```
✓ opencv-python: 4.8.1.78
✓ pytesseract: 0.3.10
✓ numpy: 1.24.4
✓ Pillow: 10.0.1
✓ imutils: 0.5.4
✓ Tesseract binary: 5.x.x (eng language pack found)
✓ MobileNetSSD_deploy.prototxt: found
✓ MobileNetSSD_deploy.caffemodel: found (23.1 MB)
✓ coco_labels.txt: found (91 labels)

All checks passed. You are ready to run Project 4.
```

---

## Step 7: Run the Pipeline

```bash
# Using the command-line argument
python src/main.py --image sample_images/ocr_sample.jpg

# Or run without argument — you will be prompted:
python src/main.py
# > Enter image path: sample_images/detection_sample.jpg
```

Output will appear in the `/output/` directory.

---

## Changing the Pipeline Mode

Open `src/config.py` and change:

```python
PIPELINE_MODE = "ocr"        # ← Change to "detection" for Path 2
```

Save the file and re-run.

---

## Common Installation Errors

See `TROUBLESHOOTING.md` for a full list. Quick reference:

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: cv2` | Run `pip install opencv-python` inside venv |
| `TesseractNotFoundError` | Tesseract binary not installed or not in PATH — see Step 4 |
| `FileNotFoundError: MobileNetSSD_deploy.caffemodel` | Run `python setup_models.py` |
| `error: (-215) !_src.empty()` | Image path is wrong or file is corrupted |

---

*End of SETUP.md*
