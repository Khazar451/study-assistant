from src.generation.tutor_format import (
    bind_numbered_citations,
    drop_unverified_citations,
    infer_grounding,
    normalize_math,
    polish_tutor_answer,
    setext_to_atx,
)


def test_broken_integral_becomes_display_math():
    raw = (
        r"The formula is (\displaystyle\int_a^b f(x)g'(x),dx = "
        r"f(x)g(x)\Big|_a^b-\int_a^b f'(x)g(x),dx) [1]."
    )
    polished = normalize_math(raw)
    assert r"\displaystyle" not in polished
    assert r"g'(x)\,dx" in polished
    assert r"g(x)\,dx" in polished
    assert r"\int_a^b" in polished
    assert polished.count("$$") >= 2


def test_bracketed_partial_derivative_becomes_katex():
    raw = (
        "The partial derivative of (f(x,y)) with respect to (x) is\n"
        r"[ \frac{\partial f}{\partial x}(x_0,y_0)=\lim_{h\to0}\frac{f(x_0+h,y_0)-f(x_0,y_0)}{h}. ]"
        "\n1\n"
        r"Fix (y=y_0). The slope is (\partial f/\partial x)."
    )
    polished = normalize_math(raw)
    assert r"\frac{\partial f}{\partial x}" in polished
    assert "$$" in polished
    assert "$f(x,y)$" in polished
    assert "$y=y_0$" in polished
    assert r"$(\partial f/\partial x)$" not in polished
    assert r"\partial f/\partial x" in polished


def test_partial_formula_stays_one_math_block():
    raw = (
        r"\frac{\partial f}{\partial x}(x_0,y_0)=\lim_{h\to0}"
        r"\frac{f(x_0+h,y_0)-f(x_0,y_0)}{h}."
    )
    polished = normalize_math(raw)
    assert polished.count("$$") == 2
    assert "$x_0,y_0$" not in polished
    assert "(x_0,y_0)" in polished
    assert r"\frac{f(x_0+h,y_0)-f(x_0,y_0)}{h}" in polished


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
