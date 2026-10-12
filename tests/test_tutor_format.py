from src.generation.tutor_format import (
    bind_numbered_citations,
    drop_unverified_citations,
    infer_grounding,
    polish_tutor_answer,
    setext_to_atx,
)


def test_setext_headings_become_atx():
    raw = "Abraham Lincoln\n================\n\nExam contrast\n-------------\nHe preserved the Union."
    polished = setext_to_atx(raw)
    assert polished.startswith("## Abraham Lincoln")
    assert "### Exam contrast" in polished
    assert "===" not in polished


def test_polish_strips_emoji_and_fake_blog_citation():
    raw = (
        "Lincoln led the Union. [Source: https://santaclausind.org/blog/lincoln, Page: 1]\n"
        "Did you know? 🏛\n"
        "Core idea\n==========\n"
    )
    polished = polish_tutor_answer(
        raw,
        allowed_sources=["https://en.wikipedia.org/wiki/Abraham_Lincoln"],
        grounding="web",
    )
    assert "santaclausind.org" not in polished
    assert "## Core idea" in polished
    assert "🏛" not in polished


def test_general_mode_removes_every_citation():
    raw = "He signed it. [Source: lecture.pdf, Page: 4] See https://example.com/lincoln"
    polished = polish_tutor_answer(raw, grounding="general")
    assert "lecture.pdf" not in polished
    assert "http" not in polished
    assert "He signed it." in polished


def test_infer_grounding():
    assert infer_grounding([]) == "general"
    assert infer_grounding([{"metadata": {"source_type": "web"}}]) == "web"
    assert infer_grounding([{"metadata": {"source_type": "document"}}]) == "course"


def test_numbered_citations_match_retrieved_chunks_only():
    chunks = [
        {"text": "Planets orbit in ellipses.", "metadata": {"source": "kepler.pdf", "page": 4, "file_path": "/tmp/kepler.pdf"}},
        {"text": "Equal areas.", "metadata": {"source": "kepler.pdf", "page": 6, "file_path": "/tmp/kepler.pdf"}},
    ]
    answer = (
        "The orbit is an ellipse [Source: kepler.pdf, Page: 4]. "
        "A fake blog agrees [Source: https://santaclausind.org/lincoln, Page: 1]. "
        "The second law is [2]."
    )
    text, refs = bind_numbered_citations(answer, chunks)
    assert "[1]" in text
    assert "[2]" in text
    assert "santaclausind" not in text
    assert refs[0]["page"] == 4
    assert refs[1]["label"].endswith("page 6")


def test_verified_course_citation_is_kept():
    raw = "Inertia is resistance. [Source: physics_lecture1.pdf, Page: 4]"
    polished = drop_unverified_citations(raw, ["physics_lecture1.pdf"])
    assert "physics_lecture1.pdf" in polished
