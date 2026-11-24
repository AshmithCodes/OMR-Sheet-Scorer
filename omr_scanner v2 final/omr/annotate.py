"""
Module 8: Annotated Result Image

Draws a color-coded overlay directly on the student's deskewed sheet so
a teacher can visually audit exactly how each bubble was read, without
digging through raw coordinates:

  * Green circle  = answered, and correct
  * Red circle    = answered, and incorrect
  * Yellow circle = unanswered or ambiguous (multiple marks / smudge
                    with no confident winner)

Every option bubble in a question gets a faint outline so the whole
grid stays legible; the option the pipeline actually judged as "the
student's mark" (or, for an ambiguous question, every option that
crossed the fill threshold) gets the bold, colored ring plus a small
status glyph, and the question number is labeled beside the row.
"""
import base64

import cv2
import numpy as np

GREEN = (60, 180, 75)
RED = (50, 50, 220)
YELLOW = (0, 200, 235)
GRAY = (190, 190, 190)
DARK = (60, 60, 60)


def annotate_result_image(warped_bgr, questions, score_details):
    """
    Args:
        warped_bgr: the student's deskewed sheet (BGR)
        questions: list of question bubble-lists (geometry), same order
            used for classification -- i.e. `SheetExtraction.questions`
        score_details: list[QuestionScore] from scorer.score_sheet,
            same order/length as `questions`

    Returns:
        annotated BGR image (a copy; input is not modified)
    """
    out = warped_bgr.copy()
    if warped_bgr.shape[0] == 0 or warped_bgr.shape[1] == 0:
        return out

    scale = max(warped_bgr.shape[1] / 1200.0, 0.6)
    thin = max(int(round(1 * scale)), 1)
    thick = max(int(round(3 * scale)), 2)
    font_scale = 0.45 * scale

    for q, score in zip(questions, score_details):
        if not q:
            continue

        color = {
            "correct": GREEN,
            "incorrect": RED,
            "unanswered": YELLOW,
            "key_unavailable": GRAY,
        }.get(score.result, GRAY)

        # faint outline on every option bubble in the row
        for b in q:
            cv2.circle(out, (int(b['x']), int(b['y'])), int(b['r']), GRAY, thin)

        if score.student_status == "filled" and score.student_answer:
            idx = _option_index(score.student_answer, len(q))
            if idx is not None:
                b = q[idx]
                cv2.circle(out, (int(b['x']), int(b['y'])), int(b['r'] * 1.25), color, thick)
        else:
            # unanswered / ambiguous: highlight the whole row region
            # rather than a single bubble, since there's no single
            # confident mark to point at
            xs = [b['x'] for b in q]
            ys = [b['y'] for b in q]
            rs = [b['r'] for b in q]
            x0 = int(min(xs) - rs[0] * 1.4)
            x1 = int(max(xs) + rs[-1] * 1.4)
            y0 = int(min(ys) - max(rs) * 1.4)
            y1 = int(max(ys) + max(rs) * 1.4)
            cv2.rectangle(out, (x0, y0), (x1, y1), YELLOW, thin)

        # question number label to the left of the row
        first = q[0]
        label_x = max(int(first['x'] - first['r'] * 3.2), 2)
        cv2.putText(out, str(score.number), (label_x, int(first['y'] + first['r'] * 0.4)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, DARK, thin, cv2.LINE_AA)

    legend_y = 24
    for text, color in (("Correct", GREEN), ("Incorrect", RED), ("Unanswered/Ambiguous", YELLOW)):
        cv2.putText(out, text, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        legend_y += 22

    return out


def _option_index(letter, n_options):
    idx = ord(letter.upper()) - ord('A')
    return idx if 0 <= idx < n_options else None


def encode_image_base64(bgr_image, ext=".png"):
    """Encode a BGR image as a base64 data URI string for embedding in JSON."""
    ok, buf = cv2.imencode(ext, bgr_image)
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"
