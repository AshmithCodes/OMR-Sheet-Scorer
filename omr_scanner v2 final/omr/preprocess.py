"""
Module 1: Preprocessing & Perspective Correction

Takes a raw phone photo of an OMR sheet -- tilted, rotated, keystoned by
perspective, unevenly lit -- and returns a clean, flat, top-down image
that bubble detection can rely on.

Corner-finding strategy (in order of preference):

  1. Fiducial markers: if the template says the sheet prints four solid
     square registration marks near its corners, look for those
     specifically. They're the most reliable signal because they're
     designed for exactly this job.
  2. Sheet-boundary quadrilateral: fit a true 4-point polygon
     (cv2.approxPolyDP) to the sheet's own outer contour. This corrects
     genuine keystone/perspective distortion (opposite edges are NOT
     required to stay parallel), unlike a rotated bounding rectangle.
  3. Rotated bounding rectangle (cv2.minAreaRect): handles simple
     rotation/tilt when the contour is too noisy for a clean 4-point
     fit. Doesn't correct keystoning, but is a safe fallback.
  4. No confident boundary at all: just resize the original frame and
     flag the fallback so the caller/UI can warn the user, rather than
     failing outright.
"""
import cv2
import numpy as np


# ---------------------------------------------------------------------
# Corner ordering / warping
# ---------------------------------------------------------------------

def order_corners(pts):
    """
    Given 4 corner points in any order, return them ordered as:
    top-left, top-right, bottom-right, bottom-left.
    This ordering is required for getPerspectiveTransform.
    """
    pts = pts.reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]  # top-left has smallest x+y
    ordered[2] = pts[np.argmax(s)]  # bottom-right has largest x+y

    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]  # top-right has smallest y-x
    ordered[3] = pts[np.argmax(diff)]  # bottom-left has largest y-x

    return ordered


def warp_sheet(image, corners, target_area=1_300_000):
    """
    Perspective-warp the sheet so it fills the frame as a flat rectangle.

    The output aspect ratio is derived from the corners themselves (the
    actual measured width/height of the detected sheet) rather than
    forced to a fixed portrait canvas -- forcing e.g. a landscape crop
    into a portrait canvas squishes circles into ellipses and breaks
    circularity-based bubble detection. `target_area` keeps resolution
    roughly constant regardless of aspect ratio / input photo size.
    """
    ordered = order_corners(corners)
    tl, tr, br, bl = ordered

    measured_w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    measured_h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    aspect = max(measured_w, 1) / max(measured_h, 1)

    h = int(round((target_area / aspect) ** 0.5))
    w = int(round(h * aspect))

    dst = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped


# ---------------------------------------------------------------------
# Corner-finding strategies
# ---------------------------------------------------------------------

def find_fiducial_markers(gray):
    """
    Look for four solid, square, dark blobs -- one per corner quadrant
    of the image. Real fiducial markers are printed specifically to be
    easy to segment: filled (high solidity), roughly square (aspect
    ratio near 1), and small relative to the page.

    Returns a (4, 1, 2) float32 array of corner points, or None if a
    confident set of four matching markers wasn't found.
    """
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    page_area = float(w * h)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        # a fiducial is a small, deliberate mark -- not a speck of noise,
        # not a large chunk of the page (e.g. the answer grid itself)
        if area < page_area * 0.0004 or area > page_area * 0.02:
            continue

        x, y, cw, ch = cv2.boundingRect(c)
        if ch == 0 or cw == 0:
            continue
        aspect = cw / float(ch)
        if aspect < 0.7 or aspect > 1.4:
            continue

        rect_area = cw * ch
        solidity = area / rect_area if rect_area > 0 else 0
        if solidity < 0.65:
            continue

        candidates.append((x + cw / 2.0, y + ch / 2.0, area))

    if len(candidates) < 4:
        return None

    # Bucket candidates into the 4 quadrants of the page and keep, per
    # quadrant, whichever candidate sits closest to that page corner.
    quadrants = {"tl": [], "tr": [], "bl": [], "br": []}
    for cx, cy, area in candidates:
        if cx < w / 2 and cy < h / 2:
            quadrants["tl"].append((cx, cy, area))
        elif cx >= w / 2 and cy < h / 2:
            quadrants["tr"].append((cx, cy, area))
        elif cx < w / 2 and cy >= h / 2:
            quadrants["bl"].append((cx, cy, area))
        else:
            quadrants["br"].append((cx, cy, area))

    if any(len(v) == 0 for v in quadrants.values()):
        return None

    corner_targets = {"tl": (0, 0), "tr": (w, 0), "bl": (0, h), "br": (w, h)}
    chosen = {}
    for key, pts in quadrants.items():
        tx, ty = corner_targets[key]
        chosen[key] = min(pts, key=lambda p: (p[0] - tx) ** 2 + (p[1] - ty) ** 2)

    # sanity check: real fiducials are printed at (roughly) the same
    # size -- if the "matches" vary wildly in area they're probably
    # unrelated blobs, not a real marker set
    areas = [chosen[k][2] for k in chosen]
    if max(areas) > 4.0 * max(min(areas), 1e-6):
        return None

    pts = np.array([
        chosen["tl"][:2], chosen["tr"][:2], chosen["br"][:2], chosen["bl"][:2]
    ], dtype=np.float32).reshape(4, 1, 2)
    return pts


def find_sheet_quad(gray):
    """
    Find the sheet's outer boundary and return it as a true 4-point
    quadrilateral when possible (corrects real keystone/perspective
    distortion), falling back to a rotated bounding rectangle
    (rotation-only correction) when the contour is too messy for a
    clean 4-point fit.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / (gray.shape[0] * gray.shape[1])

    # sanity check: the sheet should dominate the frame; if the largest
    # contour is small, we probably didn't find the sheet at all
    if area_ratio < 0.15:
        return None

    peri = cv2.arcLength(largest, True)
    for eps_frac in (0.01, 0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(largest, eps_frac * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.astype(np.float32).reshape(4, 1, 2)

    # Couldn't land on a clean 4-point polygon (folds, torn edges, noisy
    # contour). Fall back to a rotated rectangle around the convex hull
    # -- this still corrects simple rotation/tilt even though it can't
    # correct true keystoning.
    hull = cv2.convexHull(largest)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    return np.array(box, dtype=np.float32).reshape(4, 1, 2)


def find_corners(gray, use_fiducials=False):
    """
    Run the corner-finding strategy chain and return whichever succeeds
    first, along with a label describing which method was used (useful
    for debug/telemetry).
    """
    if use_fiducials:
        corners = find_fiducial_markers(gray)
        if corners is not None:
            return corners, "fiducial_markers"

    corners = find_sheet_quad(gray)
    if corners is not None:
        return corners, "sheet_quad"

    return None, "none"


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def preprocess(image_path, output_size=None, use_fiducials=False, debug=False):
    """
    Loads an image, finds the sheet, and returns:
      - warped: the deskewed / perspective-corrected color image
      - warped_gray: grayscale version
      - warped_thresh: binary thresholded version (bubbles = white on black)

    The output canvas shape is derived automatically from the detected
    sheet's own aspect ratio unless an explicit output_size (w, h) is
    given. If no confident sheet boundary can be found at all (e.g. a
    tightly cropped photo, torn edges), falls back to just resizing the
    original image rather than failing outright.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, method = find_corners(gray, use_fiducials=use_fiducials)

    if corners is not None:
        warped = warp_sheet(image, corners)
        if output_size is not None:
            warped = cv2.resize(warped, output_size)
        used_fallback = False
    else:
        # No confident sheet boundary found. Just resize, preserving the
        # original image's own aspect ratio unless the caller explicitly
        # asked for a specific output_size.
        if output_size is not None:
            warped = cv2.resize(image, output_size)
        else:
            h0, w0 = image.shape[:2]
            aspect = w0 / h0
            target_area = 1_300_000
            h = int(round((target_area / aspect) ** 0.5))
            w = int(round(h * aspect))
            warped = cv2.resize(image, (w, h))
        used_fallback = True

    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_blur = cv2.GaussianBlur(warped_gray, (3, 3), 0)

    # Adaptive threshold handles uneven lighting across the sheet much
    # better than a single global threshold value would.
    warped_thresh = cv2.adaptiveThreshold(
        warped_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25,
        C=10
    )

    if debug:
        return warped, warped_gray, warped_thresh, used_fallback, method

    return warped, warped_gray, warped_thresh


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    warped, gray, thresh, fallback, method = preprocess(path, debug=True)
    cv2.imwrite("debug_warped.jpg", warped)
    cv2.imwrite("debug_thresh.jpg", thresh)
    print(f"Corner method used: {method} (fallback={fallback})")
    print("Saved debug_warped.jpg and debug_thresh.jpg")
