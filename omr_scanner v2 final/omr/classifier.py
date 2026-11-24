"""
Module 5: Bubble Classification

Turns each question's per-option features into a single decision:
which option (if any) was actually marked, and how confidently.

Rather than a single mean-intensity threshold (which misreads light
erasures, smudges, and stray marks as valid fills, or clean fills as
blanks under uneven lighting), each option gets one combined
`fill_score` from the three features in `features.py`:

    fill_score = 0.45 * fill_ratio
               + 0.45 * central_density
               - 0.10 * edge_variance

A genuine fill scores high on both fill_ratio and central_density (the
whole bubble, especially its center, is solidly dark) and low on
edge_variance (fairly uniform blob). A stray tick, light erasure ghost,
or partial smudge typically fails at least one of those -- it might
darken *some* of the patch (nonzero fill_ratio) without actually
covering the center (low central_density), or it covers area
unevenly (high edge_variance).

The classifier then requires a clear *confidence margin* between the
best-scoring option and the runner-up before committing to an answer:

  * best score too low                       -> "blank"    (unanswered)
  * best score high enough, but runner-up is
    too close behind (double-mark, indecisive
    erase-and-reselect, heavy smudge on two
    bubbles)                                  -> "ambiguous"
  * best score high enough AND a clear margin
    over the runner-up                        -> "filled", that option
"""
from dataclasses import dataclass, field
from typing import List, Optional

from .features import extract_bubble_features

# A bubble needs at least this combined score to be considered "possibly marked"
FILL_MIN = 0.30
# The winning option must beat the runner-up by at least this much to be
# treated as an unambiguous single answer
MARGIN_MIN = 0.14


@dataclass
class BubbleResult:
    option: str
    fill_score: float
    fill_ratio: float
    central_density: float
    edge_variance: float


@dataclass
class QuestionResult:
    answer: Optional[str]       # e.g. "B", or None if blank/ambiguous
    status: str                 # "filled" | "blank" | "ambiguous"
    confidence: float           # margin-based confidence, 0-1
    bubble_results: List[BubbleResult] = field(default_factory=list)
    marked_options: List[str] = field(default_factory=list)  # for ambiguous: all options above FILL_MIN


def _fill_score(features: dict) -> float:
    return (
        0.45 * features['fill_ratio']
        + 0.45 * features['central_density']
        - 0.10 * features['edge_variance']
    )


def classify_question(gray_image, question_bubbles, option_labels):
    """
    Args:
        gray_image: full-page grayscale image (not thresholded)
        question_bubbles: list of bubble dicts, already sorted left to
            right (i.e. in option order)
        option_labels: list of option letters matching question_bubbles
            1:1, e.g. ["A", "B", "C", "D"]

    Returns:
        QuestionResult
    """
    results = []
    for bubble, label in zip(question_bubbles, option_labels):
        feats = extract_bubble_features(gray_image, bubble)
        score = _fill_score(feats)
        results.append(BubbleResult(
            option=label,
            fill_score=score,
            fill_ratio=feats['fill_ratio'],
            central_density=feats['central_density'],
            edge_variance=feats['edge_variance'],
        ))

    ranked = sorted(results, key=lambda r: r.fill_score, reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    second_score = second.fill_score if second else 0.0

    marked_options = [r.option for r in results if r.fill_score >= FILL_MIN]

    if best.fill_score < FILL_MIN:
        return QuestionResult(
            answer=None, status="blank", confidence=1.0 - best.fill_score,
            bubble_results=results, marked_options=marked_options,
        )

    margin = best.fill_score - second_score
    if margin < MARGIN_MIN:
        return QuestionResult(
            answer=None, status="ambiguous", confidence=max(margin, 0.0),
            bubble_results=results, marked_options=marked_options,
        )

    confidence = min(margin / max(best.fill_score, 1e-6), 1.0)
    return QuestionResult(
        answer=best.option, status="filled", confidence=confidence,
        bubble_results=results, marked_options=marked_options,
    )


def classify_sheet(gray_image, questions, option_labels):
    """
    Args:
        gray_image: full-page grayscale image
        questions: list of questions, each a list of bubble dicts
            (output of reconstruct_structure)
        option_labels: option letters in order, e.g. ["A","B","C","D"]

    Returns:
        list of QuestionResult, one per question, in the same order as
        `questions`.
    """
    return [classify_question(gray_image, q, option_labels) for q in questions]
