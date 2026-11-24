"""
Module 4: Bubble Feature Extraction

For a single bubble, simple mean-intensity thresholding can't tell a
confidently-filled bubble apart from a light erasure smudge, a stray
pencil graze, or a partially-rubbed-out mark -- all of those can push
average darkness past a single global cutoff. Instead we compute three
complementary signals from each bubble's image patch and let the
classifier (module 5) combine them with an explicit confidence margin:

  * fill_ratio        -- what fraction of the patch is "ink" by Otsu's
                          method (which picks its own threshold per
                          patch, adapting to local contrast/lighting
                          instead of using one global cutoff).
  * central_density    -- fill_ratio computed only inside a smaller
                          central disk. A genuine pen/pencil fill covers
                          the *middle* of the bubble solidly; a thin
                          stray mark, a tick alongside the bubble, or a
                          ring-shaped smudge from a partial erasure
                          tends to miss the center, so this feature is
                          what mainly separates those cases from a real
                          answer.
  * edge_variance      -- standard deviation of the Laplacian inside
                          the patch. A solid, deliberate fill is close
                          to a uniform dark blob (low internal texture
                          variance). A smudge, cross-out, or scribbled
                          correction has much rougher, higher-variance
                          internal texture.
"""
import cv2
import numpy as np


def _central_mask(shape, fraction=0.55):
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (max(int(w * fraction / 2), 1), max(int(h * fraction / 2), 1))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def extract_bubble_features(gray_image, bubble, pad_ratio=0.15):
    """
    Crop the region around one bubble and compute its fill features.

    Args:
        gray_image: full-page grayscale (or single-channel) image, NOT
                    thresholded -- Otsu needs real intensity values.
        bubble: dict with 'x', 'y', 'r'
        pad_ratio: extra margin around the bubble radius to include in
                    the crop, so the whole bubble outline is captured
                    even if the detected radius slightly undershoots.

    Returns:
        dict: {fill_ratio, central_density, edge_variance}
        All three are already ~normalized to comparable [0, 1]-ish
        ranges (edge_variance is min-max clipped) so the classifier can
        combine them with simple fixed weights.
    """
    h_img, w_img = gray_image.shape[:2]
    r = max(bubble['r'], 3)
    pad = r * (1 + pad_ratio)

    x0 = max(int(bubble['x'] - pad), 0)
    x1 = min(int(bubble['x'] + pad), w_img)
    y0 = max(int(bubble['y'] - pad), 0)
    y1 = min(int(bubble['y'] + pad), h_img)

    patch = gray_image[y0:y1, x0:x1]
    if patch.size == 0 or patch.shape[0] < 3 or patch.shape[1] < 3:
        return {'fill_ratio': 0.0, 'central_density': 0.0, 'edge_variance': 0.0}

    patch = cv2.GaussianBlur(patch, (3, 3), 0)

    # Otsu picks a threshold specific to THIS patch's own contrast, so a
    # bubble in a shadowed part of the sheet is judged on its own local
    # darkness range rather than a single sheet-wide cutoff.
    _, otsu = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fill_ratio = float(np.count_nonzero(otsu)) / otsu.size

    central_mask = _central_mask(patch.shape, fraction=0.55)
    central_ink = cv2.bitwise_and(otsu, otsu, mask=central_mask)
    central_area = np.count_nonzero(central_mask)
    central_density = float(np.count_nonzero(central_ink)) / central_area if central_area else 0.0

    laplacian = cv2.Laplacian(patch, cv2.CV_64F)
    raw_edge_var = float(np.std(laplacian))
    # normalize against the patch's own intensity range so this isn't
    # sensitive to absolute brightness/exposure differences between photos
    intensity_range = max(float(patch.max()) - float(patch.min()), 1.0)
    edge_variance = min(raw_edge_var / intensity_range, 1.0)

    return {
        'fill_ratio': fill_ratio,
        'central_density': central_density,
        'edge_variance': edge_variance,
    }
