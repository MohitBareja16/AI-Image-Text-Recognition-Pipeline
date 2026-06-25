# DESIGN.md
## Project 4: Visual Design Specification
**DecodeLabs AI Recognition Pipeline | Batch 2026**

---

## 1. Design Philosophy

This is a technical tool used by engineering trainees and evaluated by mentors. The aesthetic language should communicate **precision, intelligence, and machine perception** — not polish for its own sake. Every visual decision should reinforce the idea that we are looking at the world through a machine's eyes.

**The Guiding Metaphor:** A terminal with a neural pulse. The interface feels like a hybrid of a hardware oscilloscope readout and a modern AI dashboard — structured grids, tight monospaced data, with flashes of color that represent the machine "seeing" something in the noise.

**The Risk Taken:** Instead of the standard dark-with-green-accent terminal aesthetic (the default for AI tools), this design uses a **deep navy + bioluminescent cyan** palette with amber accents — referencing deep-sea sonar displays. It feels technical but distinctively non-generic.

---

## 2. Color System

### 2.1 Palette — "Deep Sonar"

| Token | Hex | Usage |
|-------|-----|-------|
| `--color-bg-primary` | `#050D1A` | Main background — near-black navy |
| `--color-bg-secondary` | `#0A1628` | Card/panel backgrounds |
| `--color-bg-tertiary` | `#0F1E38` | Hover states, input fields |
| `--color-surface-border` | `#1A3050` | Panel borders, dividers |
| `--color-accent-primary` | `#00D4FF` | Primary cyan — detections, active states |
| `--color-accent-secondary` | `#00FFB3` | Success/confirmed — high confidence badge |
| `--color-accent-amber` | `#FFB347` | Warning — medium confidence, caution states |
| `--color-accent-red` | `#FF4F6B` | Reject — below threshold, errors |
| `--color-text-primary` | `#E8F4FD` | Body text — slightly blue-tinted white |
| `--color-text-secondary` | `#7BA8CC` | Labels, captions, metadata |
| `--color-text-mono` | `#00D4FF` | Monospaced data output — confidence scores |
| `--color-glow-cyan` | `rgba(0, 212, 255, 0.15)` | Glow effect on active panels |
| `--color-glow-amber` | `rgba(255, 179, 71, 0.12)` | Glow effect on warning panels |

### 2.2 Confidence Score Color Mapping

This is the key visual language of the interface. Every confidence score renders in a color that corresponds to its value:

```
≥ 90%   →  --color-accent-secondary  (#00FFB3)  ← Verified / Certain
80–89%  →  --color-accent-primary    (#00D4FF)  ← Accepted / Confident  
60–79%  →  --color-accent-amber      (#FFB347)  ← Borderline (filtered out, shown greyed)
< 60%   →  --color-accent-red        (#FF4F6B)  ← Rejected (never shown in output panel)
```

This color mapping applies to: confidence badges, progress bars, bounding box colors on the output image, and the terminal log text color.

---

## 3. Typography

### 3.1 Type Roles

| Role | Font | Weight | Size | Usage |
|------|------|--------|------|-------|
| Display | `Space Grotesk` | 700 | 32–48px | Page title, mode headline |
| Body | `Inter` | 400/500 | 14–16px | Descriptions, labels, paragraphs |
| Mono / Data | `JetBrains Mono` | 400/600 | 12–14px | Confidence scores, coordinates, console output, code |
| Label / Eyebrow | `Space Grotesk` | 600 | 10–11px | Section labels, metadata, uppercase tracked |

### 3.2 Type Scale

```
Display XL:    48px / 700 / Space Grotesk   → Page title only
Display L:     32px / 700 / Space Grotesk   → Mode header (OCR / DETECTION)
Heading M:     20px / 600 / Inter           → Panel headers
Body L:        16px / 400 / Inter           → Primary readable content
Body S:        14px / 400 / Inter           → Secondary content, descriptions
Label:         11px / 600 / Space Grotesk   → Uppercase tracking 0.1em
Mono L:        14px / 600 / JetBrains Mono → Confidence values, bbox coords
Mono S:        12px / 400 / JetBrains Mono → Console log lines
```

### 3.3 Loading State Typography

During inference (while the model processes): display a pulsing mono font console with streaming text — characters appear one at a time to simulate real computation. This is purely CSS animation, no timing fakery that misleads the user about actual processing time.

---

## 4. Layout System

### 4.1 Grid

- **Base unit:** 8px
- **Container max-width:** 1200px, centered
- **Column system:** 12-column CSS Grid with 24px gutters
- **Breakpoints:**
  - Mobile: < 640px (single column)
  - Tablet: 640–1024px (two column)
  - Desktop: > 1024px (full 12-column)

### 4.2 Panel Layout (Desktop)

```
┌──────────────────────────────────────────────────────────────┐
│  HEADER: DecodeLabs Logo + "Project 4" + mode badge          │
├────────────────────────┬─────────────────────────────────────┤
│                        │                                     │
│  CONTROL PANEL         │  OUTPUT VIEWER                      │
│  (4 cols)              │  (8 cols)                           │
│                        │                                     │
│  [Image Upload]        │  [Annotated Image Display]          │
│  [Mode Toggle]         │                                     │
│  [PSM Selector]        │  ─────────────────────────          │
│  [Threshold Display]   │                                     │
│  [Run Button]          │  [Results Panel]                    │
│                        │  Confidence table + text output     │
├────────────────────────┴─────────────────────────────────────┤
│  CONSOLE LOG PANEL (full width, 5 lines visible, scrollable) │
└──────────────────────────────────────────────────────────────┘
```

### 4.3 Spacing Conventions

- Panel padding: 24px
- Between panel sections: 16px
- Between label and value: 8px
- Border radius: 8px (panels), 4px (badges), 2px (progress bars)

---

## 5. Component Specifications

### 5.1 Confidence Badge

```
╔══════════════╗
║  ●  91.3%   ║   ← Color corresponds to score tier
╚══════════════╝

- Shape: Pill (border-radius: 999px)
- Padding: 4px 10px
- Dot: 6px circle, same color as text
- Font: JetBrains Mono 600 12px
- Background: 10% opacity of the badge color
- Border: 1px solid, 30% opacity of badge color
- Animation: Fade-in on appear (200ms ease-out)
```

### 5.2 Detection Card (per result)

```
┌──────────────────────────────────────────┐
│  ┌──────┐  Person                        │
│  │ BBox │  ████████████░░░░  91.3%       │
│  │ img  │  Coords: (124, 88) → (340, 412)│
│  └──────┘                                │
└──────────────────────────────────────────┘
```

- Thumbnail of the cropped detection region (left, 60×60px)
- Label text: Inter 600 16px
- Confidence bar: full-width, 6px tall, animated fill on mount
- Coordinates: JetBrains Mono 12px, --color-text-secondary
- Card border: 1px solid --color-surface-border
- On hover: background lifts to --color-bg-tertiary, left border gains 2px accent color

### 5.3 Console Log Panel

```
┌─────────────────────────────────────────────────────────────┐
│  ▶  CONSOLE                                            [↕]   │
├─────────────────────────────────────────────────────────────┤
│  [00:00.001] Loading MobileNet-SSD weights...               │
│  [00:00.043] Blob constructed: (1, 3, 300, 300)             │
│  [00:00.891] Forward pass complete. 7 raw detections.       │
│  [00:00.892] Filtering: threshold = 0.80                    │
│  [00:00.893] ✓ Person: 91.3% — retained                    │
│  [00:00.894] ✗ Chair: 67.2% — rejected (below threshold)   │
└─────────────────────────────────────────────────────────────┘
```

- Background: `#030810` (darker than primary bg)
- Font: JetBrains Mono 12px
- Timestamp: --color-text-secondary
- ✓ lines: --color-accent-secondary
- ✗ lines: --color-accent-red at 60% opacity
- Log lines stream in — each new line slides up from bottom (transform: translateY(8px) → 0, 150ms)

### 5.4 Mode Toggle (OCR vs Detection)

```
┌────────────────────────────────────┐
│  [  OCR  ]   [ DETECTION ]        │
└────────────────────────────────────┘
```

- Segmented control, not two separate buttons
- Active state: `--color-accent-primary` background at 15% + bottom border 2px accent
- Inactive: transparent, text at 50% opacity
- Transition: 200ms ease, background slides between options
- Selecting a mode updates the console panel header and reconfigures visible options

### 5.5 Confidence Threshold Display

```
  THRESHOLD       80%
  ─────────────────────────────────
  ████████████████░░░░░░░░░░░  (locked at 80%)
```

- Display-only (read-only). Not adjustable by the user in the UI.
- This reinforces the Gate 3 requirement — the threshold is not negotiable.
- Lock icon beside the value: 🔒
- Tooltip on hover: "Minimum confidence required by DecodeLabs Milestone Gate 3"

---

## 6. Bounding Box Visual Language (on Output Image)

When drawing bounding boxes on the annotated output image (via OpenCV):

| Confidence | Box Color (BGR for cv2) | Thickness |
|-----------|------------------------|-----------|
| ≥ 90% | `(255, 179, 0)` → Cyan/Teal | 3px |
| 80–89% | `(255, 212, 0)` → Blue | 2px |

Label format on image:
```
┌─────────────────┐
│ Person  91.3%   │
└─────────────────┘
[bounding box lines]
```
- Label background: filled rectangle, same color as box, 70% opacity
- Label text: white, `cv2.FONT_HERSHEY_SIMPLEX`, scale 0.6, thickness 2

---

## 7. Animation Catalog

### 7.1 Page Load Sequence (orchestrated, 1.2s total)

```
t=0ms    Header fades in + logo pulse (opacity 0→1, 300ms)
t=150ms  Control panel slides in from left (translateX(-20px)→0, 350ms, ease-out)
t=200ms  Output viewer slides in from right (translateX(20px)→0, 350ms, ease-out)
t=400ms  Console panel fades up from bottom (translateY(10px)→0, 250ms)
t=500ms  "READY" pulse on run button (box-shadow glow, 400ms, repeating)
```

No animation lasts longer than 400ms. All animations respect `prefers-reduced-motion`.

```css
@media (prefers-reduced-motion: reduce) {
  * { animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }
}
```

### 7.2 Inference Running State

When the pipeline is actively running:

1. **Run button** transforms to a pulsing "PROCESSING..." state with a spinner (CSS border-radius animation, 1s loop)
2. **Output viewer** dims to 40% opacity with a scanning line animation (a 2px horizontal line sweeping top-to-bottom, 1.5s loop, `--color-accent-primary` at 60% opacity)
3. **Console log** streams new lines in real time
4. **Control panel** inputs are disabled (opacity 50%, pointer-events: none)

### 7.3 Result Reveal Sequence

When results arrive (staggered, 80ms delay per card):

```
t=0ms       Annotated image fades in (opacity 0→1, 400ms)
t=100ms     First detection card slides up (translateY(12px)→0, 250ms)
t=180ms     Confidence bar fills left→right (width 0→actual%, 600ms, ease-out)
t=180ms+80n Each subsequent card, staggered 80ms
```

### 7.4 Micro-interactions

- **Badge hover:** Scale 1.0 → 1.05, 100ms
- **Card hover:** Background transitions 150ms, border-left accent extends 150ms
- **Console line appear:** translateY(8px)→0, opacity 0→1, 150ms per line
- **Logo pulse (idle):** Glow box-shadow breathes in and out on 3s loop at 0.3 opacity

---

## 8. Output Image Annotation Style Guide (cv2 specific)

When annotating images with OpenCV, follow these drawing conventions:

### 8.1 Color Constants (BGR format for cv2)

```python
# In postprocessor.py
COLOR_HIGH_CONFIDENCE    = (255, 200, 0)   # Cyan — ≥ 90%
COLOR_MED_CONFIDENCE     = (255, 150, 50)  # Blue — 80–89%
COLOR_TEXT_ON_BOX        = (255, 255, 255) # White
COLOR_LABEL_BG_ALPHA     = 0.7             # Label background opacity
```

### 8.2 Text Rendering on Output Image

```python
# Standard label rendering
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE    = 0.6
FONT_THICKNESS = 2
FONT_LINE     = cv2.LINE_AA   # Anti-aliased — always use this

# Label box padding
LABEL_PADDING = 4  # pixels above/below text in label background rect
```

### 8.3 Minimum Legibility Rules

- Never draw text smaller than scale 0.5 — unreadable at most output resolutions
- Never place a label box outside the image boundary — clamp Y coordinate to max(0, y - label_height)
- If two bounding boxes overlap > 70%, use Non-Maximum Suppression (NMS) to retain only the highest-confidence box

---

## 9. Font Loading (Web Dashboard)

```html
<!-- In HTML head — load from Google Fonts -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
```

Fallback stack:
```css
--font-display: 'Space Grotesk', 'Helvetica Neue', Arial, sans-serif;
--font-body:    'Inter', 'Segoe UI', system-ui, sans-serif;
--font-mono:    'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
```

---

## 10. Accessibility Checklist

- [ ] All colors meet WCAG AA contrast ratio (4.5:1 for body text, 3:1 for large text) — verified against `#050D1A` background
- [ ] Focus rings: visible 2px outline in `--color-accent-primary` on all interactive elements
- [ ] `prefers-reduced-motion` respected — all animations disabled
- [ ] All images have `alt` attributes; annotated output has descriptive alt text generated from detection results
- [ ] Confidence badge colors are not the **only** indicator — also uses text labels and percentage numbers (colorblind safe)
- [ ] Console log is an `aria-live="polite"` region so screen readers announce new log lines

---

*End of DESIGN.md*
