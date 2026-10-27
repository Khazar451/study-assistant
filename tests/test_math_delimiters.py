"""Answers written with \\( \\) and \\[ \\] render, and a stray delimiter cannot turn the prose after it into math."""

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.generation.generator import CUT_SHORT_NOTE, StudyGenerator, cut_short_ending
from src.generation.tutor_format import normalize_math, open_math_closer, polish_tutor_answer

# Rebuilt from a real Nemotron 3 Super answer whose rendered page showed red KaTeX text:
# five stray $$ marks, each of which the old cleanup paired with the next formula.
STRAY_DOLLARS_ANSWER = """The Power Rule tells us how to differentiate any power of $x$.

## How the rule is obtained

### From the definition using logarithms

For $x>0$ we define $x^{n}=e^{n\\ln x}$ [5].

Differentiating with the chain rule yields the power rule for *any* real exponent $n$ $$ [2].

### Proof for positive integer exponents

Starting from the limit definition of the derivative,

$$
f'(x)=\\lim_{z\\to x}\\frac{z^{n}-x^{n}}{z-x}
$$

This establishes the rule for all positive integers $n$ $$.

## Examples illustrating the rule

$$
\\frac{d}{dx}\\,x^{3}=3x^{2}
$$

(constant multiple rule combined with the power rule) $$

These examples show the pattern.

## Exam contrast

| Aspect | Integer proof | General proof |
| --- | --- | --- |
| Starting point | Limit definition with factorization $$ | Definition $x^n=e^{n\\ln x}$ |
| Scope | Positive integers only | Any real $n$ |

Remembering the logarithmic argument is essential $$ [2] [4] [5].

## Check-in

Are you preparing for a Calculus I course?"""

BRACKET_ANSWER = """For \\(x>0\\) we define \\(x^{n}=e^{n\\ln x}\\) [5].

### Proof for positive integer exponents

\\[
f'(x)=\\lim_{z\\to x}\\frac{z^{n}-x^{n}}{z-x}
\\]
[2]

This holds for every positive integer \\(n\\)."""


def _math_blocks(text: str):
    """(math, prose) split on $$; asserts the marks pair up."""
    parts = text.split("$$")
    assert len(parts) % 2 == 1, "unpaired $$"
    return parts[1::2], parts[0::2]


def _assert_prose_outside_math(text: str):
    blocks, _ = _math_blocks(text)
    for block in blocks:
        assert not re.search(r"(?m)^\s*#{1,6}\s", block), f"heading inside math: {block!r}"
        assert not re.search(r"(?m)^\s*\|", block), f"table row inside math: {block!r}"
        assert "[2]" not in block and "Check-in" not in block, f"prose inside math: {block!r}"


def test_real_stray_dollars_answer_keeps_its_prose_out_of_math():
    cleaned = polish_tutor_answer(STRAY_DOLLARS_ANSWER)
    _assert_prose_outside_math(cleaned)
    blocks, _ = _math_blocks(cleaned)
    assert any("\\lim_{z\\to x}" in block for block in blocks)
    assert any("3x^{2}" in block for block in blocks)
    assert "Check-in\n" in cleaned and "essential [2] [4] [5]." in cleaned


def test_real_bracket_answer_with_cited_math_keeps_its_layout():
    # Raw Nemotron 3 Super answer in the new format; it wraps citations in math, like \([3][4][5]\)
    raw = (Path(__file__).parent / "fixtures" / "super_power_rule_raw.md").read_text(encoding="utf-8")
    cleaned = polish_tutor_answer(raw)
    _assert_prose_outside_math(cleaned)
    blocks, _ = _math_blocks(cleaned)
    assert len(blocks) == 2
    assert re.search(r"(?m)^#+ How to Apply the Rule\s*$", cleaned)
    assert re.search(r"(?m)^1\. \*\*Identify the exponent\*\* \$n\$", cleaned)
    assert "irrational exponents [3][4][5]." in cleaned
    assert "$[" not in cleaned


@pytest.mark.parametrize("cited", ["$[2]$", "$$[2]$$", "\\([2]\\)", "\\[ [2] \\]"])
def test_math_holding_only_a_citation_becomes_a_plain_citation(cited):
    cleaned = normalize_math(f"Holds for any real $n$ {cited}.\n\n## Proof\n\n$$\nx^2\n$$")
    _assert_prose_outside_math(cleaned)
    blocks, _ = _math_blocks(cleaned)
    assert [b.strip() for b in blocks] == ["x^2"]
    assert re.search(r"\$n\$ +\[2\]", cleaned)


def test_stray_double_dollar_before_a_citation_is_dropped():
    cleaned = normalize_math("Holds for any real exponent $n$ $$ [2].\n\n## Proof\n\n$$\nx^2\n$$")
    _assert_prose_outside_math(cleaned)
    assert "$n$ [2]." in cleaned


def test_stray_double_dollar_in_a_table_cell_is_dropped():
    table = "| A | B |\n| --- | --- |\n| Limit with factorization $$ | Chain rule |\n\n$$\nx^2\n$$"
    cleaned = normalize_math(table)
    _assert_prose_outside_math(cleaned)
    assert "| Limit with factorization | Chain rule |" in cleaned


def test_stray_double_dollar_at_line_end_before_a_paragraph_is_dropped():
    cleaned = normalize_math("(constant multiple rule) $$\n\nThese examples show it.\n\n$$\nx^2\n$$")
    blocks, _ = _math_blocks(cleaned)
    assert blocks and all("examples" not in block for block in blocks)


def test_bracket_delimiters_become_dollars_for_the_page():
    cleaned = polish_tutor_answer(BRACKET_ANSWER)
    assert "\\(" not in cleaned and "\\[" not in cleaned
    assert "$x>0$" in cleaned and "$x^{n}=e^{n\\ln x}$" in cleaned
    blocks, _ = _math_blocks(cleaned)
    assert blocks == ["\nf'(x)=\\lim_{z\\to x}\\frac{z^{n}-x^{n}}{z-x}\n"]
    assert "integer $n$." in cleaned
    _assert_prose_outside_math(cleaned)


def test_unmatched_bracket_opener_is_dropped_not_extended():
    cleaned = normalize_math("Since \\[ the rule holds.\n\n## Next step\n\nFor \\(n=2\\) we get \\(2x\\).")
    _assert_prose_outside_math(cleaned)
    assert "$n=2$" in cleaned and "$2x$" in cleaned
    assert "\\[" not in cleaned


def test_unmatched_bracket_closer_is_dropped():
    cleaned = normalize_math("The rule \\] holds for \\(n\\).")
    assert "\\]" not in cleaned and "$n$" in cleaned


def test_inline_bracket_cannot_span_a_paragraph():
    cleaned = normalize_math("Let \\(x be real.\n\nThen \\(y\\) follows.")
    assert "$y$" in cleaned
    assert "be real" not in "".join(re.findall(r"\$[^$]*\$", cleaned))


def test_line_break_spacing_inside_aligned_is_left_alone():
    cleaned = normalize_math("\\[\n\\begin{aligned} a &= b \\\\[4pt] c &= d \\end{aligned}\n\\]")
    blocks, _ = _math_blocks(cleaned)
    assert len(blocks) == 1 and "\\\\[4pt]" in blocks[0]


def test_legitimate_dollar_formulas_still_render():
    cleaned = normalize_math("The result is $$\nx = 1\n$$\nand $y$ follows.")
    blocks, _ = _math_blocks(cleaned)
    assert [b.strip() for b in blocks] == ["x = 1"] and "$y$" in cleaned


def test_blank_lines_inside_a_bracket_display_are_tolerated():
    cleaned = normalize_math("\\[\n\nx = 1\n\n\\]")
    blocks, _ = _math_blocks(cleaned)
    assert [b.strip() for b in blocks] == ["x = 1"]


def test_formula_open_at_the_very_end_is_closed():
    cleaned = normalize_math("Starting from\n$$\n\\lim_{z\\to x} z^n")
    blocks, _ = _math_blocks(cleaned)
    assert len(blocks) == 1 and "\\lim" in blocks[0]


@pytest.mark.parametrize(
    "text, closer",
    [
        ("Using \\[ \\frac{d}{dx} x^n", "\n\\]"),
        ("so \\(x^{n", "\\)"),
        ("we get $$ x^n", "\n$$"),
        ("Done \\(x\\) and \\[y\\].", ""),
    ],
)
def test_open_math_closer(text, closer):
    assert open_math_closer(text) == closer


def test_cut_off_bracket_formula_ends_cleanly_after_cleanup():
    raw = "Starting from the limit:\n\\[\n\\lim_{z\\to x} \\frac{z^n"
    cleaned = polish_tutor_answer(raw + cut_short_ending(raw))
    blocks, prose = _math_blocks(cleaned)
    assert len(blocks) == 1 and "\\lim" in blocks[0]
    assert cleaned.rstrip().endswith(CUT_SHORT_NOTE)


def test_all_prompts_ask_for_bracket_delimiters(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "neb-test-key")
    generator = StudyGenerator(client=MagicMock())
    for grounding in ("course", "web", "general"):
        prompt = generator._prompt_for(grounding)
        assert "\\(" in prompt and "\\[" in prompt
        assert "Never use dollar signs" in prompt
