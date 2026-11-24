(() => {
  const form = document.getElementById("scan-form");
  const submitBtn = document.getElementById("submit-btn");
  const btnLabel = submitBtn.querySelector(".btn-label");
  const btnSpinner = submitBtn.querySelector(".btn-spinner");
  const formError = document.getElementById("form-error");

  const resultsSection = document.getElementById("results");
  const scorePercent = document.getElementById("score-percent");
  const tallyCorrect = document.getElementById("tally-correct");
  const tallyIncorrect = document.getElementById("tally-incorrect");
  const tallyUnanswered = document.getElementById("tally-unanswered");
  const resultsBody = document.getElementById("results-body");
  const warningsBox = document.getElementById("warnings");

  const viewAnnotatedBtn = document.getElementById("view-annotated-btn");
  const modal = document.getElementById("annotated-modal");
  const modalImage = document.getElementById("annotated-image");
  const modalClose = document.getElementById("modal-close");
  const modalBackdrop = document.getElementById("modal-backdrop");

  let lastAnnotatedImage = null;

  // ---- dropzone filename feedback ----
  document.querySelectorAll(".dropzone").forEach((zone) => {
    const input = zone.querySelector("input[type=file]");
    const textEl = zone.querySelector(".dropzone-text");
    input.addEventListener("change", () => {
      if (input.files && input.files[0]) {
        textEl.textContent = input.files[0].name;
        zone.classList.add("has-file");
      } else {
        textEl.textContent = textEl.dataset.default;
        zone.classList.remove("has-file");
      }
    });
  });

  const RESULT_LABELS = {
    correct: "Correct",
    incorrect: "Incorrect",
    unanswered: "Unanswered",
    key_unavailable: "No key",
  };

  const STATUS_LABELS = {
    filled: "",
    blank: "&mdash;",
    ambiguous: "multiple marks",
  };

  function renderAnswerCell(answer, status) {
    if (status === "filled" && answer) return answer;
    if (status === "ambiguous") return '<span title="Multiple bubbles marked, or unclear">ambiguous</span>';
    return "&mdash;";
  }

  function renderResults(data) {
    scorePercent.textContent = `${data.score.percent}%`;
    tallyCorrect.textContent = data.score.correct;
    tallyIncorrect.textContent = data.score.incorrect;
    tallyUnanswered.textContent = data.score.unanswered;

    resultsBody.innerHTML = data.details.map((d) => `
      <tr>
        <td>${d.number}</td>
        <td>${renderAnswerCell(d.student_answer, d.student_status)}</td>
        <td>${d.key_answer ? d.key_answer : "&mdash;"}</td>
        <td><span class="result-pill ${d.result}">${RESULT_LABELS[d.result] || d.result}</span></td>
      </tr>
    `).join("");

    if (data.warnings && data.warnings.length) {
      warningsBox.innerHTML = data.warnings.map((w) => `<p>&#9888; ${w}</p>`).join("");
      warningsBox.hidden = false;
    } else {
      warningsBox.hidden = true;
    }

    lastAnnotatedImage = data.annotated_image;
    resultsSection.hidden = false;
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    btnSpinner.hidden = !isLoading;
    btnLabel.textContent = isLoading ? "Scanning\u2026" : "Scan & Score";
  }

  function showError(message) {
    formError.textContent = message;
    formError.hidden = false;
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    formError.hidden = true;
    setLoading(true);
    resultsSection.hidden = true;

    try {
      const formData = new FormData(form);
      const res = await fetch("/scan", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok) {
        showError(data.error || "Something went wrong while scanning.");
        return;
      }

      renderResults(data);
    } catch (err) {
      showError("Network error while scanning. Please try again.");
    } finally {
      setLoading(false);
    }
  });

  viewAnnotatedBtn.addEventListener("click", () => {
    if (!lastAnnotatedImage) return;
    modalImage.src = lastAnnotatedImage;
    modal.hidden = false;
  });

  function closeModal() { modal.hidden = true; }
  modalClose.addEventListener("click", closeModal);
  modalBackdrop.addEventListener("click", closeModal);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
  });
})();
