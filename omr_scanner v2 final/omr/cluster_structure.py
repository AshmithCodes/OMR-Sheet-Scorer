"""
Module 3: Structure Reconstruction

We have a flat list of detected bubble centers with no idea which
question or option each belongs to. This module figures that out from
geometry, adaptively -- no fixed pixel constants baked in for a
particular sheet's DPI or layout density.

Approach (row-first, which is robust for both single- and multi-column
sheets because rows are horizontally aligned across the *whole* sheet
width, while column-block boundaries are only meaningful *within* a
row):

  1. Group all bubbles into horizontal rows by y-coordinate. If the
     template tells us the total question count and column count, the
     exact expected row count is known, so we use KMeans with that
     fixed k -- the most robust option. Otherwise we estimate a
     row-separation distance from the data's own y-coordinate gap
     histogram (row spacing is normally far more uniform than column
     spacing, since it's dictated by a single repeated table row
     height) and run DBSCAN with that adaptive eps.
  2. Within each row, sort bubbles left-to-right and split them into
     one or more equal-size chunks of `options_per_question` bubbles --
     one chunk per side-by-side question column-block. The chunk count
     per row is the template's declared `columns` if given, otherwise
     inferred from that row's own bubble count.
  3. Assemble chunks into final question order: column-major (top-to-
     bottom within block 1, then block 2, ...) by default, matching the
     numbering convention nearly all printed multi-column bubble sheets
     use; row-major is also supported via `column_major=False`.
  4. Rows whose bubble count isn't a clean multiple of
     `options_per_question` are dropped as stray noise (roll-number
     grids, instruction icons, etc.) rather than guessed at.
"""
from collections import Counter

import numpy as np
from sklearn.cluster import DBSCAN, KMeans


def _estimate_gap_eps(sorted_values, fallback=15):
    """
    Estimate a reasonable "same group" distance from the data itself
    via elbow detection, rather than a fixed pixel constant tuned for
    one sheet's layout/resolution/DPI.

    Consecutive gaps between sorted values naturally fall into two
    populations on a real grid: small "within-group" gaps (bubbles in
    the same row) and large "between-group" gaps (the jump to the next
    row). We sort the gap values themselves and look for the single
    biggest relative jump between consecutive sorted gaps -- that jump
    is the boundary between the two populations, so eps is set right
    in the middle of it.

    A plain median-based ratio breaks on a perfectly regular grid where
    many within-group gaps are exactly (or nearly) zero -- filtering
    those out can leave the between-group gap as the median itself.
    Elbow detection is robust to that because it only cares about the
    *relative* jump between consecutive sorted gap magnitudes, not
    their absolute distribution.
    """
    if len(sorted_values) < 3:
        return fallback
    diffs = np.diff(sorted_values)
    diffs = diffs[diffs > 1e-6]
    if len(diffs) < 2:
        # zero or one nonzero gap: nothing to split on, so don't split
        span = float(sorted_values[-1] - sorted_values[0])
        return max(span + 1.0, fallback)

    diffs_sorted = np.sort(diffs)
    jumps = np.diff(diffs_sorted)
    split_idx = int(np.argmax(jumps))
    small_val = float(diffs_sorted[split_idx])
    large_val = float(diffs_sorted[split_idx + 1])

    jump_ratio = large_val / max(small_val, 1e-6)
    if jump_ratio < 2.0:
        # no clear bimodal split in the gaps -- treat everything as one
        # group rather than risk over-splitting on noise
        return max(float(diffs_sorted[-1] * 1.3), fallback)

    eps = (small_val + large_val) / 2.0
    return max(eps, fallback)


def _group_into_rows(bubbles, expected_rows=None):
    """
    Group bubbles into horizontal rows by y-coordinate.

    If `expected_rows` is known (derivable from the template's declared
    num_questions and columns), KMeans with that exact k is used --
    deterministic and immune to noisy spacing. Otherwise rows are
    discovered adaptively via gap-based DBSCAN on the y-coordinates.

    Returns a list of rows (each a list of bubble dicts), ordered top
    to bottom. Bubbles within each row are NOT yet sorted by x.
    """
    n = len(bubbles)
    ys = np.array([b['y'] for b in bubbles], dtype=float)

    if expected_rows and 1 <= expected_rows <= n:
        km = KMeans(n_clusters=expected_rows, n_init=8, random_state=0)
        labels = km.fit_predict(ys.reshape(-1, 1))
    else:
        eps = _estimate_gap_eps(np.sort(ys))
        labels = DBSCAN(eps=eps, min_samples=1).fit_predict(ys.reshape(-1, 1))

    row_ids = sorted(set(labels), key=lambda lbl: np.mean(ys[labels == lbl]))
    return [
        [bubbles[i] for i in range(n) if labels[i] == lbl]
        for lbl in row_ids
    ]


def _split_row_into_blocks(row_bubbles, options_per_question, columns=None):
    """
    Split one row's bubbles (sorted left-to-right) into one or more
    chunks of exactly `options_per_question` bubbles -- one chunk per
    side-by-side question column-block in this row.
    """
    row_sorted = sorted(row_bubbles, key=lambda b: b['x'])
    n = len(row_sorted)

    if columns and columns >= 1:
        n_blocks = columns
    else:
        n_blocks = max(1, round(n / options_per_question))

    blocks = []
    for i in range(n_blocks):
        chunk = row_sorted[i * options_per_question:(i + 1) * options_per_question]
        if chunk:
            blocks.append(chunk)
    return blocks


def reconstruct_structure(bubbles, options_per_question=4, columns=None,
                           num_questions=None, column_major=True):
    """
    Args:
        bubbles: list of dicts with 'x', 'y', 'r' (output of detect_bubbles)
        options_per_question: number of options per question (from the
               sheet Template) -- the one constant nearly every OMR
               sheet fixes in advance, used to make block-splitting
               deterministic.
        columns: expected number of side-by-side column blocks, from the
               Template. None = infer per-row from each row's own
               bubble count.
        num_questions: total expected question count, from the
               Template. Combined with `columns`, gives an exact
               expected row-per-column count for a much more reliable
               row grouping. None = discover row count adaptively.
        column_major: True = number questions top-to-bottom within the
               first column-block, then the next column-block, etc.
               (the convention almost all printed multi-column bubble
               sheets use). False = number left-to-right across a row
               first, then move down (row-major).

    Returns:
        List of questions in reading order. Each question is a list of
        bubble dicts sorted left-to-right (= option order A, B, C...):
        [
          [bubble_A, bubble_B, bubble_C, bubble_D],   # question 1
          [bubble_A, bubble_B, bubble_C, bubble_D],   # question 2
          ...
        ]
    """
    if not bubbles:
        return []

    expected_rows = None
    if num_questions and columns:
        expected_rows = max(1, round(num_questions / columns))

    rows = _group_into_rows(bubbles, expected_rows=expected_rows)

    # blocks_by_column[k] = list of question-blocks (each `options_per_question`
    # bubbles) belonging to column-block k, in top-to-bottom row order
    blocks_by_column = {}
    row_major_questions = []
    for row_bubbles in rows:
        blocks = _split_row_into_blocks(row_bubbles, options_per_question, columns=columns)
        for col_idx, block in enumerate(blocks):
            blocks_by_column.setdefault(col_idx, []).append(block)
        row_major_questions.extend(blocks)

    if column_major:
        questions = []
        for col_idx in sorted(blocks_by_column.keys()):
            questions.extend(blocks_by_column[col_idx])
    else:
        questions = row_major_questions

    # Final cleanup: real answer rows overwhelmingly share the exact
    # options_per_question option count; anything else (roll-number
    # boxes, ID grids, instruction icons that slipped through bubble
    # detection) gets dropped here rather than guessed at.
    questions = [q for q in questions if len(q) == options_per_question]

    if num_questions and len(questions) > num_questions:
        questions = questions[:num_questions]

    return questions


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from preprocess import preprocess
    from detect_bubbles import detect_bubbles
    import cv2

    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    warped, gray, thresh = preprocess(path)
    bubbles = detect_bubbles(thresh)
    questions = reconstruct_structure(bubbles)

    print(f"Reconstructed {len(questions)} questions")
    option_counts = [len(q) for q in questions]
    if option_counts:
        print("Options-per-question distribution:", Counter(option_counts))

    annotated = warped.copy()
    for qi, q in enumerate(questions):
        for oi, b in enumerate(q):
            cv2.circle(annotated, (int(b['x']), int(b['y'])), int(b['r']), (0, 255, 0), 2)
        if q:
            first = q[0]
            cv2.putText(annotated, str(qi + 1), (int(first['x']) - 25, int(first['y']) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.imwrite("debug_structure.jpg", annotated)
    print("Saved debug_structure.jpg")
