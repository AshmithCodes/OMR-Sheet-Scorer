"""
Flask backend for the OMR Sheet Scanner.

Accepts a student answer-sheet photo and a key/answer-sheet photo (same
printed template), runs both through the OMR pipeline
(preprocess -> detect_bubbles -> cluster_structure -> classifier),
scores the student against the key, and returns:

  - a per-question results table
  - an annotated overlay image of the student's sheet
    (green = correct, red = incorrect, yellow = unanswered/ambiguous)

Sheet layout (question count, options A-D/A-E, column count, whether
the sheet has printed corner registration markers, header/footer crop)
is entirely configurable per-request via a Template, instead of being
hardcoded -- see omr/template.py.
"""
import os
import traceback
import uuid

from flask import Flask, jsonify, render_template, request

from omr.annotate import annotate_result_image, encode_image_base64
from omr.key_extractor import extract_sheet
from omr.scorer import score_sheet
from omr.template import Template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _save_upload(file_storage, prefix):
    ext = file_storage.filename.rsplit(".", 1)[1].lower()
    name = f"{prefix}_{uuid.uuid4().hex}.{ext}"
    path = os.path.join(UPLOAD_DIR, name)
    file_storage.save(path)
    return path


def _build_template_from_form(form):
    """
    Builds a Template from the simple per-field form controls the UI
    exposes (options-per-question, column count, corner markers,
    header/footer crop). If the user pasted a full custom JSON template
    into the advanced box, that takes precedence.
    """
    import json

    raw_json = (form.get("template_json") or "").strip()
    if raw_json:
        try:
            return Template.from_dict(json.loads(raw_json))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid template JSON: {exc}")

    options_letters = form.get("options_per_question", "ABCD")
    columns_raw = form.get("columns", "").strip()
    num_questions_raw = form.get("num_questions", "").strip()

    return Template.from_dict({
        "options": options_letters,
        "columns": columns_raw or None,
        "num_questions": num_questions_raw or None,
        "corner_markers": form.get("corner_markers") == "on",
        "header_fraction": form.get("header_fraction", 0) or 0,
        "footer_fraction": form.get("footer_fraction", 0) or 0,
    })


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scan", methods=["POST"])
def scan():
    student_file = request.files.get("student_sheet")
    key_file = request.files.get("key_sheet")

    if not student_file or student_file.filename == "":
        return jsonify({"error": "Please choose the student's answer sheet."}), 400
    if not key_file or key_file.filename == "":
        return jsonify({"error": "Please choose the key/answer sheet."}), 400
    if not _allowed_file(student_file.filename) or not _allowed_file(key_file.filename):
        return jsonify({"error": "Unsupported file type. Please upload an image (jpg/png/webp/bmp)."}), 400

    try:
        template = _build_template_from_form(request.form)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    student_path = key_path = None
    try:
        student_path = _save_upload(student_file, "student")
        key_path = _save_upload(key_file, "key")

        key_extraction = extract_sheet(key_path, template)
        student_extraction = extract_sheet(student_path, template)

        if not key_extraction.questions:
            return jsonify({
                "error": "Couldn't detect any answer bubbles on the key sheet. "
                         "Try a clearer photo, or adjust the template settings "
                         "(options per question / columns) in Advanced."
            }), 422
        if not student_extraction.questions:
            return jsonify({
                "error": "Couldn't detect any answer bubbles on the student sheet. "
                         "Try a clearer photo, or adjust the template settings "
                         "(options per question / columns) in Advanced."
            }), 422

        summary = score_sheet(student_extraction.results, key_extraction.results)

        annotated = annotate_result_image(
            student_extraction.warped_image, student_extraction.questions, summary.details
        )
        annotated_data_uri = encode_image_base64(annotated)

        warnings = []
        if student_extraction.used_fallback_corners:
            warnings.append(
                "Couldn't confidently find the student sheet's corners/edges -- "
                "results may be less reliable. Try a photo with the full sheet visible "
                "and good contrast against the background."
            )
        if key_extraction.used_fallback_corners:
            warnings.append(
                "Couldn't confidently find the key sheet's corners/edges -- "
                "results may be less reliable."
            )
        if len(key_extraction.questions) != len(student_extraction.questions):
            warnings.append(
                f"Key sheet has {len(key_extraction.questions)} detected questions but "
                f"student sheet has {len(student_extraction.questions)}. Scoring only the "
                f"{summary.total} questions present on both."
            )

        details = [{
            "number": d.number,
            "student_answer": d.student_answer,
            "student_status": d.student_status,
            "key_answer": d.key_answer,
            "result": d.result,
        } for d in summary.details]

        return jsonify({
            "score": {
                "percent": summary.percent,
                "correct": summary.correct,
                "incorrect": summary.incorrect,
                "unanswered": summary.unanswered,
                "total": summary.total,
            },
            "details": details,
            "annotated_image": annotated_data_uri,
            "warnings": warnings,
            "template_used": template.to_dict(),
        })

    except Exception as exc:  # noqa: BLE001 - surface a readable error to the UI
        traceback.print_exc()
        return jsonify({"error": f"Something went wrong while scanning: {exc}"}), 500

    finally:
        for p in (student_path, key_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
