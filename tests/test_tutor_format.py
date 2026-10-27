import re

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


def test_inline_integrals_stay_inline_and_unbroken():
    raw = (
        "Integration by parts rewrites an integral of a product "
        r"\int u,dy into another integral \int y,du that is often simpler to evaluate [1]."
    )
    polished = normalize_math(raw)
    assert r"$\int u\,dy$" in polished
    assert r"$\int y\,du$" in polished
    assert "into another integral" in polished
    assert polished.count("$$") == 0


def test_duplicate_display_dollars_collapsed():
    raw = "Formula:\n$$\n\n$$\n\\frac{d}{dx}[f(x)]\n$$\n\n$$\n[2]."
    polished = normalize_math(raw)
    assert polished.count("$$") == 2
    assert r"\frac{d}{dx}[f(x)]" in polished


def test_duplicate_heading_hashes_collapsed():
    raw = "## ## Derivation of the Formula\nSome text.\n### ### Deep dive\nMore text."
    polished = setext_to_atx(raw)
    assert "## Derivation of the Formula" in polished
    assert "## ##" not in polished
    assert "### Deep dive" in polished
    assert "### ###" not in polished


def test_plain_headings_keep_their_level():
    polished = setext_to_atx("## Statement of the rule\nText.\n### Exam contrast\nMore.\n#### Detail")
    assert re.search(r"(?m)^## Statement of the rule$", polished)
    assert re.search(r"(?m)^### Exam contrast$", polished)
    assert re.search(r"(?m)^#### Detail$", polished)


def test_math_citation_extracted_outside_display_math():
    raw = r"$$\int f(x)g'(x)\,dx = f(x)g(x)-\int f'(x)g(x)\,dx \quad\text{[3]} .$$"
    polished = normalize_math(raw)
    assert r"\text{[3]}" not in polished
    assert "[3]" in polished
    assert "$$" in polished
    # The formula itself is clean LaTeX without \text{[3]}
    assert r"\int f(x)g'(x)\,dx = f(x)g(x)-\int f'(x)g(x)\,dx" in polished


def test_math_citation_extracted_outside_inline_math():
    raw = r"$\int u\,dy = u y-\int y\,du \quad\text{[1]} .$"
    polished = normalize_math(raw)
    assert r"\text{[1]}" not in polished
    assert "[1]" in polished
    assert r"$\int u\,dy = u y-\int y\,du$" in polished


def test_bind_numbered_citations_cleans_math_mode_source_tags():
    chunks = [
        {"text": "Integration by parts.", "metadata": {"source": "calc.pdf", "page": 475}},
    ]
    raw = r"$$\int u\,dy = uy-\int y\,du \quad\text{[Source: calc.pdf, Page: 475]} .$$"
    text, refs = bind_numbered_citations(raw, chunks)
    assert r"\text" not in text
    assert "[1]" in text
    assert "$$" in text
    assert r"\int u\,dy = uy-\int y\,du" in text


def test_nested_bracketed_integral_does_not_break_bigl():
    raw = r"Substitute back:\n\n\int e^x\cos x\,dx = e^x\sin x - \bigl[ \int e^x\sin x\,dx \bigr] [3]"
    polished = normalize_math(raw)
    assert r"\bigl[ \int e^x\sin x\,dx \bigr]" in polished
    assert "[3]" in polished
    assert "$$" in polished
    # Ensure \bigl is never left severed on its own line
    assert r"\bigl" + "\n" not in polished


def test_unicode_math_symbols_converted_to_latex():
    raw = "Start from ∫ u dv = uv - ∫ v du [1]."
    polished = normalize_math(raw)
    assert "∫" not in polished
    assert r"\int" in polished
    assert "[1]" in polished


def test_dangling_sizing_command_stripped():
    raw = r"$$\int e^x\,dx = e^x - \bigl$$"
    polished = normalize_math(raw)
    assert r"\bigl" not in polished
    assert r"\int e^x\,dx = e^x -" in polished


def test_math_spacing_between_inline_math_and_words():
    raw = "The goal is to replace an integral $\\int u\\,dy$that is difficult with pick$dy$as something and find$y$) and be$u$ [4]."
    polished = normalize_math(raw)
    assert r"$\int u\,dy$ that" in polished
    assert "pick $dy$ as" in polished
    assert "find $y$)" in polished
    assert "be $u$ [4]" in polished

    # Punctuation tests
    punct_raw = "Consider ($x$, $y$) where value is $x$. Next step."
    punct_polished = normalize_math(punct_raw)
    assert "($x$, $y$)" in punct_polished
    assert "value is $x$. Next" in punct_polished


def test_consecutive_display_equations_do_not_invert_parity():
    raw = (
        "Substitute back:\n"
        r"\int e^x \cos x dx = e^x \sin x + e^x \cos x - \int e^x \cos x dx" "\n\n"
        r"\int e^{x}\cos x,dx = \frac{e^{x}}{2}(\sin x+\cos x)+C" "\n\n"
        "(The textbook notes the same result.)# Reduction formula\n[3]"
    )
    polished = normalize_math(raw)
    # Both formulas are inside math, but English commentary is outside math
    assert r"\int e^x \cos x dx" in polished
    assert r"\frac{e^{x}}{2}" in polished
    assert "(The textbook notes the same result.)" in polished
    # The commentary must NOT be inside $$
    parts = polished.split("$$")
    # In parts: even indices (0, 2, 4...) are outside $$, odd indices (1, 3, 5...) are inside $$
    outside_math = " ".join([parts[i] for i in range(0, len(parts), 2)])
    assert "The textbook notes the same result." in outside_math
    assert "[3]" in outside_math


def test_glued_headings_without_whitespace_separated():
    raw = "(result.)# Heading 1\nSome text.\n(contrast.)## Heading 2"
    polished = setext_to_atx(raw)
    assert "(result.)\n\n# Heading 1" in polished
    assert "(contrast.)\n\n## Heading 2" in polished




