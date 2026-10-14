import pytest
from src.verification.symbolic_verifier import (
    extract_equations,
    verify_equation,
    verify_math_solution,
    generate_wolfram_url,
    generate_python_snippet,
)


def test_extract_equations_display_math():
    text = r"""
Here is the derivation:
$$ \int x e^x dx = x e^x - \int e^x dx = x e^x - e^x + C $$
And that is the answer.
"""
    eqs = extract_equations(text)
    assert len(eqs) >= 1
    assert "x e^x" in eqs[0]["lhs"]
    assert "x e^x - e^x + C" in eqs[0]["rhs"]


def test_extract_equations_inline_math():
    text = r"Recall that \( (x + 1)^2 = x^2 + 2x + 1 \) which is a standard identity."
    eqs = extract_equations(text)
    assert len(eqs) == 1
    assert eqs[0]["lhs"] == "(x + 1)^2"
    assert "x^2 + 2x + 1" in eqs[0]["rhs"]


def test_verify_equation_integral():
    lhs = r"\int x e^{x} dx"
    rhs = r"x e^{x} - e^{x} + C"
    res = verify_equation(lhs, rhs)
    assert res["status"] == "verified"
    assert res["method"] == "derivative_check"


def test_verify_equation_algebra():
    lhs = r"(x + 1)^2"
    rhs = r"x^2 + 2*x + 1"
    res = verify_equation(lhs, rhs)
    assert res["status"] == "verified"
    assert res["method"] == "symbolic_simplify"


def test_verify_math_solution_full_pipeline():
    sample_answer = r"""
### 1. Core Idea
We use integration by parts:
$$ \int u \, dv = u v - \int v \, du $$

### 2. Derivation
Let $u = x$ and $dv = e^x dx$. Then $du = dx$ and $v = e^x$.
$$ \int x e^{x} \, dx = x e^{x} - e^{x} + C $$

### 3. Check
The derivative matches.
"""
    result = verify_math_solution(sample_answer)
    assert result is not None
    assert result["has_math"] is True
    assert result["status"] == "verified"
    assert "wolframalpha.com" in result["wolfram_url"]
    assert "sympy" in result["python_code"]
    assert len(result["boundary_checks"]) > 0


def test_generate_wolfram_url():
    url = generate_wolfram_url(r"\int x e^{x} dx", r"x e^{x} - e^{x} + C")
    assert "wolframalpha.com/input?i=" in url
    assert "integrate" in url


def test_abstract_equation_returns_none():
    abstract_answer = r"""
Here is the product rule derivation:
$$ \frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x) $$
"""
    result = verify_math_solution(abstract_answer)
    assert result is None

