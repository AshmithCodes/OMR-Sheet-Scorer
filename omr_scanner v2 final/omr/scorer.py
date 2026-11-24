"""
Module 7: Scoring

Compares a student's classified answers against a key's classified
answers, question by question. Distinguishes three student outcomes
per question -- correct, incorrect, and unanswered -- where "unanswered"
covers both a truly blank bubble row and an ambiguous mark (multiple
bubbles filled, or a smudge with no clear winner), since neither of
those is a scorable definite answer.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QuestionScore:
    number: int
    student_answer: Optional[str]
    student_status: str              # "filled" | "blank" | "ambiguous"
    key_answer: Optional[str]
    key_status: str
    result: str                      # "correct" | "incorrect" | "unanswered" | "key_unavailable"


@dataclass
class ScoreSummary:
    total: int
    correct: int
    incorrect: int
    unanswered: int
    percent: float
    details: List[QuestionScore] = field(default_factory=list)


def score_sheet(student_results, key_results) -> ScoreSummary:
    """
    Args:
        student_results: list[QuestionResult] for the student's sheet
        key_results: list[QuestionResult] for the key sheet

    Returns:
        ScoreSummary
    """
    total = min(len(student_results), len(key_results)) if key_results else len(student_results)
    if len(student_results) != len(key_results):
        total = min(len(student_results), len(key_results))

    correct = incorrect = unanswered = 0
    details = []

    n = max(len(student_results), len(key_results))
    for i in range(n):
        s = student_results[i] if i < len(student_results) else None
        k = key_results[i] if i < len(key_results) else None

        student_answer = s.answer if s else None
        student_status = s.status if s else "blank"
        key_answer = k.answer if k else None
        key_status = k.status if k else "blank"

        if k is None or key_status != "filled":
            result = "key_unavailable"
        elif student_status != "filled":
            result = "unanswered"
            unanswered += 1
        elif student_answer == key_answer:
            result = "correct"
            correct += 1
        else:
            result = "incorrect"
            incorrect += 1

        details.append(QuestionScore(
            number=i + 1,
            student_answer=student_answer,
            student_status=student_status,
            key_answer=key_answer,
            key_status=key_status,
            result=result,
        ))

    scorable = correct + incorrect + unanswered
    percent = round((correct / scorable) * 100, 1) if scorable else 0.0

    return ScoreSummary(
        total=scorable,
        correct=correct,
        incorrect=incorrect,
        unanswered=unanswered,
        percent=percent,
        details=details,
    )
