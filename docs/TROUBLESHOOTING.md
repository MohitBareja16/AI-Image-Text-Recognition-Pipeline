# TROUBLESHOOTING.md
## Project 4: Common Issues & Fixes
**DecodeLabs | Batch 2026**

---

## 1. Tesseract Issues (OCR Path)

### Issue: `TesseractNotFoundError: tesseract is not installed or it's not in your PATH`

**Cause:** The Tesseract binary is either not installed, or Python can't find it.

**Fix (Windows):**
1. Confirm Tesseract is installed: Search "Tesseract" in Windows Start.
2. Find the install path — usually `C:\Program Files\Tesseract-OCR\tesseract.exe`.
3. Add this to the top of `config.py`:
   ```python
   import pytesseract
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

**Fix (macOS/Linux):**
```bash
which tesseract
# Should print: /usr/local/bin/tesseract or /usr/bin/tesseract
# If nothing prints: re-install using brew install tesseract or sudo apt install tesseract-ocr
```

---

### Issue: OCR returns empty string or gibberish

**Cause:** PSM mode is wrong for the image layout, or pre-processing degraded the text.

**Fix — Try different PSM modes** in `config.py`:

| PSM Value | Best For |
|-----------|----------|
| `3` | Mixed layouts, general purpose (default) |
| `6` | Single block of uniform text (book pages, articles) |
| `7` | Single line (number plates, banners, headers) |
| `11` | Sparse text, varied positions (invoices, receipts) |
| `4` | Single column of text (narrow newspaper columns) |

```python
TESSERACT_PSM = 6  # Change this in config.py
```

**Fix — Verify pre-processing quality:**
Add this to `preprocessor.py` temporarily to save and inspect the intermediate image:
```python
cv2.imwrite("debug_preprocessed.jpg", processed_image)
```
Open `debug_preprocessed.jpg`. The text should appear as sharp black characters on a white background. If it looks muddy or inverted, try `cv2.THRESH_BINARY_INV` instead of `cv2.THRESH_BINARY`.

---

### Issue: All words have confidence = -1

**Cause:** `image_to_data()` returns `-1` for layout-level rows (not actual words). You must filter these.

**Fix:**
```python
df = df[df['conf'] != -1]   # Remove layout rows before filtering by confidence
df = df[df['conf'] >= 80]
```

---

## 2. Object Detection Issues (Path 2)

### Issue: `FileNotFoundError: MobileNetSSD_deploy.caffemodel`

**Cause:** Model files weren't downloaded.

**Fix:**
```bash
python setup_models.py
```
If that fails, download manually (see SETUP.md Step 5) and place in `models/` directory.

---

### Issue: All detections cluster in the top-left corner of the output image

**Cause:** Bounding box coordinates were not scaled from normalized [0,1] to pixel values.

**Fix:** Your coordinate scaling code must multiply by image dimensions:
```python
h, w = image.shape[:2]
x_start = max(0, int(detection[3] * w))
y_start = max(0, int(detection[4] * h))
x_end   = min(w, int(detection[5] * w))
y_end   = min(h, int(detection[6] * h))
```

---

### Issue: Zero detections on an image that clearly has objects

**Cause 1:** Confidence threshold filtering out everything — your image may be genuinely hard for MobileNet-SSD (unusual angle, heavy occlusion, non-COCO-class objects).

**Fix:** Try `detection_sample.jpg` first to confirm the model works. Use images with clear, front-facing, well-lit objects.

**Cause 2:** Blob not constructed correctly.

**Fix:** Verify your blob construction exactly matches:
```python
blob = cv2.dnn.blobFromImage(
    image,
    scalefactor=1/127.5,
    size=(300, 300),
    mean=(127.5, 127.5, 127.5),
    swapRB=True,    # Important: OpenCV loads BGR, model expects RGB
    crop=False
)
```

---

### Issue: `cv2.error: (-215:Assertion failed) !_src.empty()`

**Cause:** `cv2.imread()` returned `None` — the image path is wrong, or the file is corrupted.

**Fix:**
```python
image = cv2.imread(path)
if image is None:
    raise FileNotFoundError(f"Could not load image: {path}. Check the path and file format.")
```
Also verify: the path uses correct slashes for your OS. On Windows, either use raw strings (`r"C:\path\to\img.jpg"`) or forward slashes (`"C:/path/to/img.jpg"`).

---

## 3. General Pipeline Issues

### Issue: Output image not saved — only a window appears

**Cause:** `cv2.imwrite()` call is missing or the output path is invalid.

**Fix:**
```python
output_path = os.path.join(OUTPUT_DIR, f"{base_name}_output.jpg")
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output dir if it doesn't exist
cv2.imwrite(output_path, annotated_image)
print(f"Output saved to: {output_path}")
```

This is required for Gate 4.

---

### Issue: `cv2.imshow()` crashes on headless server (SSH, GitHub Codespaces)

**Fix:** Wrap `imshow` in a try/except and always save to disk as backup:
```python
try:
    cv2.imshow("Project 4 Output", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except cv2.error:
    print("Note: Display window unavailable in this environment. Output saved to disk.")
```

---

### Issue: Script runs but no console output appears

**Cause:** The `print_summary()` function isn't being called, or results are empty and the zero-results branch isn't reached.

**Fix:** Add debug prints at each stage during development:
```python
print(f"[DEBUG] Pre-processing complete. Shape: {processed.shape}")
print(f"[DEBUG] Raw detections: {len(raw_results)}")
print(f"[DEBUG] After threshold filter: {len(filtered_results)}")
```
Remove debug prints before final submission.

---

## 4. Confidence Score Questions

### Q: My best detection is 79.8% — can I lower the threshold?

**No.** The 80% threshold is a Milestone Gate 3 requirement. Lowering it will fail validation.

Instead:
- Use a better-quality input image (higher resolution, better lighting)
- For OCR: try a different PSM mode
- For Detection: ensure the object is clearly visible and centrally framed
- Use the provided `sample_images/` which are pre-validated to produce ≥ 80% detections

---

### Q: Can I apply NMS (Non-Maximum Suppression) to get cleaner results?

Yes — and it's encouraged. NMS removes duplicate boxes for the same object:
```python
# After filtering by confidence
boxes = [(x, y, w, h) for each detection]
confidences = [conf for each detection]
indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.80, nms_threshold=0.4)
```
Use NMS threshold of 0.4 — this is the IoU overlap threshold, not the confidence threshold.

---

## 5. Evaluation / Submission Issues

### Q: Mentor says my output doesn't show confidence values

**Fix:** Ensure your output image labels include the percentage:
```python
label = f"{class_name}: {confidence * 100:.1f}%"
cv2.putText(image, label, ...)
```

And ensure your console output block includes per-detection confidence as specified in the Output Contract in `PRD.md`.

### Q: Script works on my test image but fails on evaluator's image

**Fix:** Your pre-processing pipeline is too aggressive (over-blurring, wrong threshold type) or your PSM is hard-coded for a specific layout. Review:
1. Use `TESSERACT_PSM = 3` (Auto) as the default — it's most robust
2. Use `cv2.adaptiveThreshold` instead of `cv2.threshold` — it handles uneven lighting better
3. Test on multiple images before submitting

---

*End of TROUBLESHOOTING.md*
