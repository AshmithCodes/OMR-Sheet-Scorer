"""
Module 0: Sheet Template Configuration

Decouples all "geometry knowledge" about a specific answer sheet layout
(how many questions, how many options, how many side-by-side column
blocks, whether the sheet carries printed corner registration markers,
how much of the top of the sheet is header/instructions that should be
ignored) into a single plain-data object.

Nothing downstream hardcodes a question count, an option-letter set, a
column count, or a header cutoff fraction -- every one of those is read
from a Template instance. The default Template is fully auto-detecting
(every field left as None/0 means "figure it out from the image"), so
existing behaviour for unseen sheets keeps working out of the box. A
caller who *does* know their sheet's layout (e.g. a school running the
same printed template every time) can pin down some or all fields for
much more reliable results, either by constructing a Template directly
or by POSTing a JSON blob that maps onto one.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional


DEFAULT_OPTIONS = ["A", "B", "C", "D"]


@dataclass
class Template:
    # Total number of questions on the sheet. None = auto-detect from
    # however many bubble rows are found.
    num_questions: Optional[int] = None

    # Answer option letters, in left-to-right order, e.g. ["A","B","C","D"]
    # or ["A","B","C","D","E"]. Determines options_per_question.
    options: List[str] = field(default_factory=lambda: list(DEFAULT_OPTIONS))

    # Number of side-by-side column blocks (e.g. 3 for a Q1-20/21-40/41-60
    # layout). None = auto-detect via adaptive gap clustering.
    columns: Optional[int] = None

    # Whether the sheet has four printed solid-square registration
    # markers near its corners. When True, preprocessing looks for them
    # explicitly and uses them for perspective correction; when False
    # (default) it falls back to detecting the sheet's own outer
    # boundary as a quadrilateral.
    corner_markers: bool = False

    # Fraction (0.0-1.0) of the top of the *warped* sheet to exclude
    # from bubble detection -- for sheets with a large header / student
    # info block that could otherwise contain circular logos or ID
    # bubbles that get mistaken for answer bubbles. 0 = don't crop
    # anything (default; safe for sheets without such a header).
    header_fraction: float = 0.0

    # Symmetric option for a footer block (instructions, signature box)
    # at the bottom of the sheet. 0 = don't crop anything.
    footer_fraction: float = 0.0

    @property
    def options_per_question(self) -> int:
        return len(self.options) if self.options else 4

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Template":
        if not d:
            return cls()

        options = d.get("options")
        if isinstance(options, str):
            # allow the convenience shorthand "ABCD" / "ABCDE"
            options = list(options.strip().upper())
        if not options:
            options = list(DEFAULT_OPTIONS)
        options = [str(o).strip().upper()[:1] for o in options if str(o).strip()]
        if len(options) < 2:
            options = list(DEFAULT_OPTIONS)

        def _int_or_none(v):
            if v in (None, "", "auto", "null"):
                return None
            try:
                iv = int(v)
                return iv if iv > 0 else None
            except (TypeError, ValueError):
                return None

        def _float(v, default):
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return default
            return min(max(fv, 0.0), 0.9)

        return cls(
            num_questions=_int_or_none(d.get("num_questions")),
            options=options,
            columns=_int_or_none(d.get("columns")),
            corner_markers=bool(d.get("corner_markers", False)),
            header_fraction=_float(d.get("header_fraction"), 0.0),
            footer_fraction=_float(d.get("footer_fraction"), 0.0),
        )

    def to_dict(self) -> dict:
        return asdict(self)
