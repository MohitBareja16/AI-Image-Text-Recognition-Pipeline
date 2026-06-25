#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# setup_models.py — One-Time MobileNet-SSD Model Download Script
# Downloads MobileNetSSD_deploy.prototxt and .caffemodel to /models/
# Also creates a VOC-compatible 21-class label file used by MobileNet-SSD.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import urllib.request
import hashlib

# ── Target directory ──────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ── File definitions ──────────────────────────────────────────────────────────
# Official sources: OpenCV GitHub / PyImageSearch CDN
FILES = [
    {
        "name": "MobileNetSSD_deploy.prototxt",
        "url": (
            "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/"
            "master/MobileNetSSD_deploy.prototxt"
        ),
        "min_size_bytes": 5_000,    # ~29KB
        "required": True,
    },
    {
        "name": "MobileNetSSD_deploy.caffemodel",
        "url": (
            "https://drive.usercontent.google.com/download?"
            "id=0B3gersZ2cHIxRm5PMWRoTkdHdHc&export=download"
        ),
        "min_size_bytes": 20 * 1024 * 1024,  # Must be > 20MB
        "required": True,
    },
]

# MobileNet-SSD VOC labels (21 classes: background + 20 PASCAL VOC classes)
# These are the labels the .caffemodel was trained on.
VOC_LABELS = [
    "background",    # class 0 — always skip
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def download_file(url: str, dest_path: str, name: str) -> bool:
    """Downloads a file from URL to dest_path with a progress indicator."""
    print(f"  Downloading {name}...")
    print(f"  URL: {url}")

    try:
        # Use urllib with a browser-like User-Agent to avoid 403s
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 64  # 64KB chunks

            with open(dest_path, "wb") as out_file:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        mb = downloaded / (1024 * 1024)
                        print(f"\r  Progress: {mb:.1f} MB ({pct:.0f}%)", end="", flush=True)
        print()  # Newline after progress
        return True

    except Exception as e:
        print(f"\n  ERROR downloading {name}: {e}", file=sys.stderr)
        # Clean up partial download
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def create_labels_file(labels_path: str) -> None:
    """Writes the VOC label file to disk."""
    with open(labels_path, "w", encoding="utf-8") as f:
        for label in VOC_LABELS:
            f.write(label + "\n")
    print(f"  Labels file created: {labels_path} ({len(VOC_LABELS)} classes)")


def main() -> int:
    """Downloads model files and creates labels file."""
    print("=" * 60)
    print("DecodeLabs Project 4 — Model Setup")
    print("=" * 60)
    print(f"Models directory: {MODELS_DIR}")
    print()

    # Create models directory
    os.makedirs(MODELS_DIR, exist_ok=True)

    all_ok = True

    # ── Download model files ──────────────────────────────────────────────────
    for file_info in FILES:
        dest_path = os.path.join(MODELS_DIR, file_info["name"])

        # Check if file already exists and meets minimum size
        if os.path.isfile(dest_path):
            size = os.path.getsize(dest_path)
            if size >= file_info["min_size_bytes"]:
                print(f"✓ {file_info['name']} already exists ({size / (1024*1024):.1f} MB). Skipping.")
                continue
            else:
                print(f"✗ {file_info['name']} exists but appears incomplete ({size} bytes). Re-downloading...")
                os.remove(dest_path)

        # Download
        success = download_file(file_info["url"], dest_path, file_info["name"])

        if success:
            size = os.path.getsize(dest_path)
            if size >= file_info["min_size_bytes"]:
                print(f"✓ {file_info['name']} downloaded successfully ({size / (1024*1024):.1f} MB)")
            else:
                print(
                    f"✗ {file_info['name']} download may be incomplete "
                    f"({size / (1024*1024):.1f} MB, expected > "
                    f"{file_info['min_size_bytes'] / (1024*1024):.0f} MB)",
                    file=sys.stderr,
                )
                if file_info["required"]:
                    all_ok = False
        else:
            if file_info["required"]:
                all_ok = False

    # ── Create labels file ────────────────────────────────────────────────────
    labels_path = os.path.join(MODELS_DIR, "coco_labels.txt")
    if not os.path.isfile(labels_path):
        print(f"\nCreating labels file...")
        create_labels_file(labels_path)
    else:
        print(f"✓ Labels file already exists: {labels_path}")

    print()
    if all_ok:
        print("✅ Setup complete. You can now run detection mode:")
        print("   python src/main.py --image sample_images/detection_sample.jpg --mode detection")
    else:
        print("⚠️  Some files could not be downloaded automatically.", file=sys.stderr)
        print("   Please download them manually — see docs/SETUP.md", file=sys.stderr)

    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
