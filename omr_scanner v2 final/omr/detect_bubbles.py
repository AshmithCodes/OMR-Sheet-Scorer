"""
Module 2: Bubble Detection

Given a thresholded (binary) image of a deskewed OMR sheet, find every
bubble-like shape on the page, regardless of layout. We don't know the
question count or options-per-question ahead of time -- we just look
for things that are roughly circular and roughly bubble-sized, then let
the geometry (module 3) figure out rows/columns/questions.
"""
import cv2
import numpy as np


def detect_bubbles(thresh_img, min_circularity=0.55, min_aspect=0.6,
                    size_tolerance=(0.35, 2.8), roi_y_range=None):
    """
    Find candidate bubble contours in a binary threshold image.

    Bubble size (in pixels) varies a lot between sheets depending on scan
    resolution and layout density -- a fixed area threshold that works
    for one sheet can completely miss bubbles on another. So instead of
    a fixed size range, we do this in two passes:

      Pass 1: filter contours by shape only (circularity + aspect
              ratio), ignoring size, to get a rough set of "round-ish"
              candidates.
      Pass 2: take the MEDIAN area of that round-ish set as an estimate
              of this sheet's actual bubble size, then keep only
              contours whose area falls within `size_tolerance` of that
              median. This adapts automatically to each sheet's
              resolution/layout without any hardcoded pixel size.

    Args:
        thresh_img: binary image (bubbles = white regions on black),
                    e.g. output of cv2.adaptiveThreshold
        min_circularity: 4*pi*area/perimeter^2, 1.0 = perfect circle
        min_aspect: bounding box width/height ratio (and its inverse)
                    must exceed this, to reject elongated shapes (text,
                    lines, checkmarks)
        size_tolerance: (low, high) multipliers applied to the estimated
                    median bubble area to define the accepted size range
        roi_y_range: optional (y0, y1) pixel band (in this image's own
                    coordinates) to restrict detection to -- used to
                    exclude a header/footer region defined by the sheet
                    template instead of hardcoding a fractional cutoff
                    here.

    Returns:
        list of dicts: {'x': cx, 'y': cy, 'r': radius, 'contour': c}
        one per detected bubble, unsorted.
    """
    # Use RETR_LIST to catch every candidate contour, including bubbles
    # that sit as "holes" inside a larger connected ink blob (which
    # RETR_EXTERNAL would incorrectly discard). This means a single
    # hollow bubble can produce two nested contours (outer + inner ring
    # edge); we deduplicate those explicitly below rather than relying
    # on contour retrieval mode to sort it out.
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    y0, y1 = roi_y_range if roi_y_range else (0, thresh_img.shape[0])

    # Pass 1: shape-only filter (no size constraint yet)
    round_candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 4:  # ignore pure noise specks
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < min_circularity:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        aspect = w / float(h)
        if aspect < min_aspect or aspect > (1.0 / min_aspect):
            continue

        cy = y + h / 2.0
        if cy < y0 or cy > y1:
            continue

        round_candidates.append((c, area, x, y, w, h))

    if not round_candidates:
        return []

    # Pass 2: strip out tiny noise specks (paper grain, JPEG artifacts).
    # This is a fixed pixel floor rather than a fraction of image area,
    # which is safe because preprocess() always warps to a fixed target
    # canvas area -- so noise stays a roughly constant tiny pixel size
    # regardless of sheet, while real bubbles (drawn to be visibly
    # fillable by a pen) are reliably larger than this floor.
    noise_floor_area = 100
    survivors = [rc for rc in round_candidates if rc[1] >= noise_floor_area]

    if not survivors:
        survivors = round_candidates  # fall back rather than return nothing

    # Estimate this sheet's typical bubble size from the survivors, then
    # keep only contours within a tolerance band around that estimate --
    # this trims any remaining oversized outliers (logos, headers,
    # borders) that happened to pass the circularity/aspect checks.
    areas = np.array([a for _, a, _, _, _, _ in survivors])
    median_area = np.median(areas)
    low_mult, high_mult = size_tolerance
    min_area = median_area * low_mult
    max_area = median_area * high_mult

    bubbles = []
    for c, area, x, y, w, h in survivors:
        if area < min_area or area > max_area:
            continue
        cx, cy = x + w / 2.0, y + h / 2.0
        radius = (w + h) / 4.0
        bubbles.append({'x': cx, 'y': cy, 'r': radius, 'contour': c, 'area': area})

    return _deduplicate(bubbles)


def _deduplicate(bubbles, center_dist_ratio=0.5):
    """
    A hollow (unfilled) bubble drawn as a ring often produces two nested
    contours -- the outer edge and inner edge of the ring -- which would
    otherwise be counted as two separate bubbles despite being the same
    physical bubble. Both contours share nearly the same center, so we
    merge any two detections whose centers are closer than
    `center_dist_ratio * radius`, keeping the larger (outer) one.
    """
    if len(bubbles) <= 1:
        return bubbles

    # process larger contours first so we keep the outer ring when merging
    bubbles_sorted = sorted(bubbles, key=lambda b: b['area'], reverse=True)
    kept = []
    for b in bubbles_sorted:
        is_duplicate = False
        for k in kept:
            dist = np.hypot(b['x'] - k['x'], b['y'] - k['y'])
            if dist < center_dist_ratio * k['r']:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(b)

    return kept


def draw_bubbles(image, bubbles, color=(0, 255, 0), thickness=2):
    """Debug helper: draw circles on a copy of `image` for every detected bubble."""
    out = image.copy()
    for b in bubbles:
        cv2.circle(out, (int(b['x']), int(b['y'])), int(b['r']), color, thickness)
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from preprocess import preprocess

    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    warped, gray, thresh = preprocess(path)
    bubbles = detect_bubbles(thresh)
    print(f"Detected {len(bubbles)} candidate bubbles")

    annotated = draw_bubbles(warped, bubbles)
    cv2.imwrite("debug_bubbles.jpg", annotated)
    print("Saved debug_bubbles.jpg")
