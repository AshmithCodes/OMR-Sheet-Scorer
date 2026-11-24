"""
Module 6: Sheet Extraction Pipeline

Wires modules 1-5 together into a single call: given an image path and
a Template, produce the deskewed image plus a fully classified answer
list for that sheet.
"""
from dataclasses import dataclass
from typing import List

import numpy as np

from .preprocess import preprocess
from .detect_bubbles import detect_bubbles
from .cluster_structure import reconstruct_structure
from .classifier import classify_sheet, QuestionResult
from .template import Template


@dataclass
class SheetExtraction:
    warped_image: np.ndarray            # BGR deskewed sheet image
    warped_gray: np.ndarray
    questions: List[list]               # geometry: list of bubble-lists
    results: List[QuestionResult]       # classification per question
    used_fallback_corners: bool
    corner_method: str


def extract_sheet(image_path, template: Template) -> SheetExtraction:
    warped, gray, thresh, used_fallback, corner_method = preprocess(
        image_path, use_fiducials=template.corner_markers, debug=True
    )

    header_px = int(template.header_fraction * warped.shape[0])
    footer_px = int((1 - template.footer_fraction) * warped.shape[0]) if template.footer_fraction else warped.shape[0]
    roi_y_range = (header_px, footer_px)

    bubbles = detect_bubbles(thresh, roi_y_range=roi_y_range)

    questions = reconstruct_structure(
        bubbles,
        options_per_question=template.options_per_question,
        columns=template.columns,
        num_questions=template.num_questions,
    )

    results = classify_sheet(gray, questions, template.options)

    return SheetExtraction(
        warped_image=warped,
        warped_gray=gray,
        questions=questions,
        results=results,
        used_fallback_corners=used_fallback,
        corner_method=corner_method,
    )
