[README.md](https://github.com/user-attachments/files/35314844/README.md)
<img width="1064" height="1243" alt="7EE53DC1-2811-4E44-B6A8-B4F64B24D1AF_1_201_a" src="https://github.com/user-attachments/assets/aca7472a-3472-49ab-9177-dce0c0545529" />

# OMR Sheet Scanner

Upload a filled-in student answer sheet and a marked key sheet (same
printed template) and get an automatic score, even from an imperfect
phone photo.

## What's new in this version

1. **Perspective correction** — `omr/preprocess.py` finds the sheet's
   four corners as a true quadrilateral (`cv2.approxPolyDP`) and warps
   it flat with `cv2.warpPerspective`, correcting real keystone/tilt
   distortion, not just rotation. Sheets that print four solid corner
   registration squares can opt into explicit fiducial-marker detection
   (`corner_markers: true` in the template) for even more reliable
   alignment.
2. **Adaptive layout reconstruction** — `omr/cluster_structure.py` no
   longer relies on a fixed `x_eps` pixel constant. Rows are grouped by
   an elbow-detected gap threshold (or exact KMeans counts when the
   template states `num_questions`/`columns`), and column-blocks are
   split from each row using the template's declared options-per-
   question. Works for single-column and multi-column (column-major
   numbered) sheets alike.
3. **Multi-feature bubble classification** — `omr/features.py` +
   `omr/classifier.py` score each bubble on Otsu fill-ratio, central
   mass density, and edge variance, and require a clear confidence
   margin between the top two candidates before committing to an
   answer. Light erasures, stray pencil marks, and smudges are
   correctly read as unanswered/ambiguous instead of a false answer.
4. **Flexible template config** — `omr/template.py` is a plain JSON-
   serializable object describing question count, option letters
   (A–D, A–E, ...), column-block count, corner-marker mode, and header/
   footer crop fractions. Nothing about sheet geometry is hardcoded;
   every field also has a sane auto-detect default (`null`/`0`) so an
   unfamiliar sheet still works out of the box.
5. **Annotated result image** — `omr/annotate.py` draws a green/red/
   yellow overlay directly on the student's deskewed sheet (correct /
   incorrect / unanswered-ambiguous) and the frontend shows it in a
   modal via "View annotated sheet".

## Running it

```bash
pip install -r requirements.txt
python3 app.py
# open http://localhost:5000
```

## Template configuration

The "Advanced" panel in the UI exposes the common fields directly
(options per question, column count, total questions, corner-marker
checkbox, header/footer crop). For full control, paste JSON into the
"custom template" box — it overrides the simple fields:

```json
{
  "num_questions": 60,
  "options": "ABCDE",
  "columns": 3,
  "corner_markers": true,
  "header_fraction": 0.12,
  "footer_fraction": 0.0
}
```

| Field             | Meaning                                                                 | Default (auto)          |
|--------------------|--------------------------------------------------------------------------|--------------------------|
| `num_questions`   | Total question count.                                                   | `null` → detected        |
| `options`         | Option letters, e.g. `"ABCD"` or `["A","B","C","D","E"]`.               | `"ABCD"`                 |
| `columns`         | Side-by-side column blocks (numbered column-major).                     | `null` → detected        |
| `corner_markers`  | Sheet prints 4 solid square registration marks near its corners.        | `false`                  |
| `header_fraction` | Fraction of the top of the sheet to exclude from bubble detection.      | `0.0`                    |
| `footer_fraction` | Fraction of the bottom of the sheet to exclude from bubble detection.   | `0.0`                    |

Same template must be used for both the key and the student sheet in a
given scan (they're assumed to be the same printed layout).

## Pipeline modules (`omr/`)

| Module                | Responsibility                                            |
|------------------------|-------------------------------------------------------------|
| `template.py`          | Sheet layout config object                                 |
| `preprocess.py`        | Corner detection + perspective warp + adaptive threshold   |
| `detect_bubbles.py`    | Locate bubble-shaped contours, size-adaptive per sheet     |
| `cluster_structure.py` | Group bubbles into rows/columns/questions                  |
| `features.py`          | Per-bubble fill-ratio / central-density / edge-variance    |
| `classifier.py`        | Confidence-margin decision: filled / blank / ambiguous     |
| `key_extractor.py`     | Wires the above into one `extract_sheet()` call             |
| `scorer.py`            | Compares student vs key results                            |
| `annotate.py`          | Draws the color-coded result overlay image                 |
