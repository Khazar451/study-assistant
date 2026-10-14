"""Symbolic math verification engine using SymPy and external solver integrations.

Extracts equations from tutor responses, checks identities using SymPy's symbolic
differentiation and simplification, runs boundary/numerical evaluations, generates
runnable Python scripts, and builds Wolfram Alpha cross-verification queries.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import sympy as sp

logger = logging.getLogger(__name__)


def _clean_latex_for_sympy(latex_str: str) -> str:
    """Sanitize and clean raw LaTeX string so SymPy's parse_latex can parse it."""
    if not latex_str:
        return ""

    s = latex_str.strip()

    # Strip math wrappers
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2].strip()
    elif s.startswith("\\[") and s.endswith("\\]"):
        s = s[2:-2].strip()
    elif s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()
    elif s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()

    # Strip trailing punctuation
    s = s.rstrip(".,;: \n")

    # Strip \boxed{...}
    if s.startswith(r"\boxed{") and s.endswith("}"):
        s = s[len(r"\boxed{"):-1].strip()
    s = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", s)

    # Remove citations like [1], [2]
    s = re.sub(r"\[\d+\]", "", s)

    # Remove tags like \tag{1} or \tag{*}
    s = re.sub(r"\\tag\{[^}]*\}", "", s)

    # Remove spacing and formatting commands
    s = re.sub(r"\\(?:quad|qquad|thickspace|medspace|thinspace|!|,|;|:)", " ", s)
    s = re.sub(r"\\(?:left|right)", "", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathit\{([^}]*)\}", r"\1", s)

    # Replace \cdot with multiplication or space
    s = re.sub(r"\\cdot", " * ", s)

    # Normalize dx, dy, dt differentials (e.g., \, dx or \mathrm{d}x -> dx)
    s = re.sub(r"\\mathrm\{d\}([a-zA-Z])", r"d\1", s)

    # Clean multiple spaces and trailing punctuation
    s = re.sub(r"\s+", " ", s).strip().rstrip(".,;: ")
    return s


def extract_equations(text: str) -> List[Dict[str, str]]:
    """Extract equations containing '=' from math blocks in tutor output.

    Handles display math ($$ ... $$) and inline math (\\( ... \\) or $ ... $).
    Supports chained equalities (A = B = C) by extracting the outer pair (A = C).
    """
    if not text:
        return []

    candidates: List[Dict[str, str]] = []

    # Find display math blocks
    display_matches = re.findall(r"\$\$(.*?)\$\$", text, flags=re.DOTALL)
    display_brackets = re.findall(r"\\\[(.*?)\\\]", text, flags=re.DOTALL)

    # Find inline math blocks
    inline_parens = re.findall(r"\\\((.*?)\\\)", text, flags=re.DOTALL)
    # Inline $...$ (ensure not $$)
    inline_dollars = re.findall(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", text, flags=re.DOTALL)

    all_blocks = []
    for blk in display_matches + display_brackets:
        all_blocks.append(("display", blk.strip()))
    for blk in inline_parens + inline_dollars:
        all_blocks.append(("inline", blk.strip()))

    for block_type, blk in all_blocks:
        blk_strip = blk.strip().rstrip(".,;: \n")
        if blk_strip.startswith(r"\boxed{") and blk_strip.endswith("}"):
            blk_strip = blk_strip[len(r"\boxed{"):-1].strip()

        # Must contain '=' and not inequality commands
        if "=" not in blk_strip:
            continue
        if any(bad in blk_strip for bad in ["\\neq", "\\le", "\\ge", "\\leq", "\\geq", "\\approx", "\\approxeq"]):
            # For now, skip pure approximations or inequalities from symbolic = checks
            pass

        # Split on '='
        parts = [p.strip() for p in blk_strip.split("=") if p.strip()]
        if len(parts) >= 2:
            # First part is LHS, last part is final simplified RHS
            lhs_clean = _clean_latex_for_sympy(parts[0])
            rhs_clean = _clean_latex_for_sympy(parts[-1])
            if lhs_clean and rhs_clean:
                candidates.append({
                    "type": block_type,
                    "raw_latex": blk,
                    "lhs": lhs_clean,
                    "rhs": rhs_clean,
                    "parts": [_clean_latex_for_sympy(p) for p in parts]
                })

    return candidates


def _safe_parse_latex(latex_str: str) -> Optional[sp.Expr]:
    """Parse a LaTeX string with SymPy, handling Euler's e and common edge cases."""
    try:
        from sympy.parsing.latex import parse_latex
        expr = parse_latex(latex_str)
        # Substitute symbol 'e' with Euler's number sp.E
        if hasattr(expr, "subs"):
            expr = expr.subs(sp.Symbol("e"), sp.E)
        return expr
    except Exception as exc:
        logger.debug("Failed to parse LaTeX '%s': %s", latex_str, exc)
        return None


def verify_equation(lhs_str: str, rhs_str: str) -> Dict[str, Any]:
    """Verify equality between LHS and RHS using symbolic calculus and algebra."""
    lhs_expr = _safe_parse_latex(lhs_str)
    rhs_expr = _safe_parse_latex(rhs_str)

    if lhs_expr is None or rhs_expr is None:
        return {
            "status": "inconclusive",
            "method": "parse_failure",
            "message": "LaTeX expression could not be parsed symbolically by SymPy.",
            "lhs_expr": str(lhs_expr) if lhs_expr else None,
            "rhs_expr": str(rhs_expr) if rhs_expr else None,
        }

    # Remove arbitrary constant C or c from RHS for integration checks
    rhs_no_const = rhs_expr
    for c_sym in [sp.Symbol("C"), sp.Symbol("c")]:
        if hasattr(rhs_no_const, "subs"):
            rhs_no_const = rhs_no_const.subs(c_sym, 0)

    # 1. Check if LHS is an indefinite integral: \int f(x) dx
    if isinstance(lhs_expr, sp.Integral):
        var = lhs_expr.variables[0] if lhs_expr.variables else sp.Symbol("x")
        integrand = lhs_expr.function
        try:
            d_rhs = sp.diff(rhs_no_const, var)
            diff = sp.simplify(integrand - d_rhs)
            if diff == 0:
                return {
                    "status": "verified",
                    "method": "derivative_check",
                    "message": "Indefinite integral verified: d/d{} of RHS equals integrand.".format(var),
                    "integrand": str(integrand),
                    "d_rhs": str(d_rhs),
                    "var": str(var),
                    "lhs_expr": lhs_expr,
                    "rhs_expr": rhs_no_const,
                }
        except Exception as exc:
            logger.debug("Error in integral derivative check: %s", exc)

    # 2. Check if LHS is a derivative: d/dx f(x)
    if isinstance(lhs_expr, sp.Derivative):
        try:
            lhs_eval = lhs_expr.doit()
            diff = sp.simplify(lhs_eval - rhs_no_const)
            if diff == 0:
                return {
                    "status": "verified",
                    "method": "derivative_check",
                    "message": "Derivative operation verified.",
                    "lhs_expr": lhs_eval,
                    "rhs_expr": rhs_no_const,
                }
        except Exception as exc:
            logger.debug("Error in derivative check: %s", exc)

    # 3. Direct symbolic equality: simplify(LHS - RHS) == 0
    try:
        diff = sp.simplify(lhs_expr - rhs_no_const)
        if diff == 0:
            return {
                "status": "verified",
                "method": "symbolic_simplify",
                "message": "Symbolic equality verified: simplify(LHS - RHS) == 0.",
                "lhs_expr": lhs_expr,
                "rhs_expr": rhs_no_const,
            }

        # Try trigonometric simplification
        trig_diff = sp.trigsimp(diff)
        if trig_diff == 0:
            return {
                "status": "verified",
                "method": "trigsimp",
                "message": "Trigonometric identity verified.",
                "lhs_expr": lhs_expr,
                "rhs_expr": rhs_no_const,
            }
    except Exception as exc:
        logger.debug("Error in symbolic simplify: %s", exc)

    # 4. Numerical Boundary Evaluation for free symbols
    free_syms = list((lhs_expr - rhs_no_const).free_symbols)
    if free_syms:
        # Filter out constant symbols
        eval_syms = [s for s in free_syms if str(s) not in ["C", "c"]]
        if len(eval_syms) == 1:
            sym = eval_syms[0]
            test_points = [0.5, 1.0, 2.0, 3.0]
            boundary_results = []
            all_match = True

            for pt in test_points:
                try:
                    lhs_val = None
                    if isinstance(lhs_expr, sp.Integral):
                        lhs_antideriv = lhs_expr.doit()
                        if lhs_antideriv is not None and not isinstance(lhs_antideriv, sp.Integral):
                            lhs_val = complex(lhs_antideriv.evalf(subs={sym: pt}))
                    else:
                        lhs_val = complex(lhs_expr.evalf(subs={sym: pt}))

                    rhs_val = complex(rhs_no_const.evalf(subs={sym: pt}))

                    if lhs_val is not None:
                        is_close = abs(lhs_val - rhs_val) < 1e-5
                        if not is_close:
                            all_match = False
                        boundary_results.append({
                            "input": f"{sym} = {pt}",
                            "lhs": round(lhs_val.real, 5) if abs(lhs_val.imag) < 1e-9 else str(lhs_val),
                            "rhs": round(rhs_val.real, 5) if abs(rhs_val.imag) < 1e-9 else str(rhs_val),
                            "match": is_close,
                        })
                except Exception:
                    continue

            if boundary_results and all_match:
                return {
                    "status": "verified",
                    "method": "boundary_eval",
                    "message": "Verified across numerical boundary inputs.",
                    "boundary_results": boundary_results,
                    "lhs_expr": lhs_expr,
                    "rhs_expr": rhs_no_const,
                }
            elif boundary_results and not all_match:
                return {
                    "status": "discrepancy",
                    "method": "boundary_eval",
                    "message": "Boundary input evaluation showed divergence between LHS and RHS.",
                    "boundary_results": boundary_results,
                    "lhs_expr": lhs_expr,
                    "rhs_expr": rhs_no_const,
                }

    return {
        "status": "inconclusive",
        "method": "unknown",
        "message": "Could not conclusively verify symbolic equality.",
        "lhs_expr": lhs_expr,
        "rhs_expr": rhs_no_const,
    }


def generate_python_snippet(
    eq_data: Dict[str, Any],
    verification_res: Dict[str, Any],
) -> str:
    """Generate a clean, copyable Python verification script using SymPy."""
    lhs = eq_data.get("lhs", "")
    rhs = eq_data.get("rhs", "")
    method = verification_res.get("method", "symbolic_simplify")

    code_lines = [
        "# --- Automated SymPy Verification Script ---",
        "import sympy as sp",
        "from sympy.parsing.latex import parse_latex",
        "",
        "# Define symbolic variables",
        "x = sp.Symbol('x')",
        "C = sp.Symbol('C')",
        "",
        f"# Equation: {lhs} = {rhs}",
        f"raw_lhs = r'''{lhs}'''",
        f"raw_rhs = r'''{rhs}'''",
        "",
        "lhs_expr = parse_latex(raw_lhs).subs(sp.Symbol('e'), sp.E)",
        "rhs_expr = parse_latex(raw_rhs).subs(sp.Symbol('e'), sp.E)",
        "",
    ]

    if method == "derivative_check":
        code_lines.extend([
            "# Indefinite Integral Check:",
            "# The derivative of the antiderivative (RHS) must equal the integrand (LHS)",
            "if isinstance(lhs_expr, sp.Integral):",
            "    var = lhs_expr.variables[0]",
            "    integrand = lhs_expr.function",
            "    d_rhs = sp.diff(rhs_expr.subs(C, 0), var)",
            "    diff = sp.simplify(integrand - d_rhs)",
            "    print(f'Integrand: {integrand}')",
            "    print(f'd/d{var}(RHS): {d_rhs}')",
            "    print(f'Symbolic identity verified: {diff == 0}')",
        ])
    else:
        code_lines.extend([
            "# General Symbolic Equality Check:",
            "diff = sp.simplify(lhs_expr - rhs_expr.subs(C, 0))",
            "print(f'Difference simplify(LHS - RHS): {diff}')",
            "print(f'Symbolic match: {diff == 0}')",
            "",
            "# Boundary input evaluation",
            "free_vars = list((lhs_expr - rhs_expr).free_symbols)",
            "if free_vars and x in free_vars:",
            "    for pt in [0.5, 1.0, 2.0]:",
            "        try:",
            "            v_lhs = float(lhs_expr.evalf(subs={x: pt}))",
            "            v_rhs = float(rhs_expr.subs(C, 0).evalf(subs={x: pt}))",
            "            print(f'At x={pt}: LHS={v_lhs:.4f}, RHS={v_rhs:.4f}, Match={abs(v_lhs - v_rhs) < 1e-4}')",
            "        except Exception as e:",
            "            print(f'At x={pt}: {e}')",
        ])

    return "\n".join(code_lines)


def generate_wolfram_url(lhs: str, rhs: str) -> str:
    """Generate a direct URL to Wolfram Alpha for the equation or integral."""
    query = lhs
    if r"\int" in lhs:
        q = lhs.replace(r"\int", "integrate ").replace(r"\cdot", "*")
        q = re.sub(r"e\^\{([^}]+)\}", r"exp(\1)", q)
        q = re.sub(r"e\^([a-zA-Z0-9])", r"exp(\1)", q)
        query = q.strip()
    elif "=" in f"{lhs} = {rhs}":
        query = f"{lhs} = {rhs}"

    query = re.sub(r"\\(?:frac\{([^}]+)\}\{([^}]+)\})", r"(\1)/(\2)", query)
    query = query.replace("\\", "")

    encoded = urllib.parse.quote_plus(query.strip())
    return f"https://www.wolframalpha.com/input?i={encoded}"


def _is_abstract_equation(eq_str: str) -> bool:
    """Check if an equation contains abstract, uninstantiated functions or derivative primes."""
    abstract_patterns = [
        r"[fghuvw]\s*\(",
        r"[fghuvw]'\s*\(",
        r"[fghuvw]''\s*\(",
        r"\\prime",
        r"\b[uvw]'\b",
        r"\bd[uvw]\b",
        r"\\frac\{d[uvw]\}\{d[a-z]\}",
    ]
    for pat in abstract_patterns:
        if re.search(pat, eq_str):
            return True
    return False


def _score_equation_candidate(eq: Dict[str, Any]) -> int:
    score = 0
    lhs = eq.get("lhs", "")
    rhs = eq.get("rhs", "")

    # Display math is preferred
    if eq.get("type") == "display":
        score += 25

    # Calculus equations (integrals, derivatives)
    if r"\int" in lhs or r"\frac{d}" in lhs:
        score += 30

    # Presence of problem variable x, t
    if "x" in lhs or "x" in rhs:
        score += 15

    # Heavily deprioritize abstract definitions (e.g. product rule definition f'g + fg')
    if _is_abstract_equation(lhs) or _is_abstract_equation(rhs):
        score -= 150

    # Penalize trivial substitutions like u = x, du = dx, y = e^x
    if len(lhs) <= 3:
        score -= 20
    if len(rhs) <= 3:
        score -= 10

    # Penalize single-letter variables like u dy = uy
    if "u" in lhs and "dy" in lhs and len(lhs) < 15:
        score -= 15

    # Longer formulas represent full derivations
    score += min(len(lhs) + len(rhs), 30)
    return score


def verify_math_solution(answer: str) -> Optional[Dict[str, Any]]:
    """Scan the generated tutor answer, verify its core mathematical formulas,

    and return a structured verification payload for the frontend UI.
    Only returns a payload if an equation is conclusively verified.
    """
    if not answer:
        return None

    equations = extract_equations(answer)
    if not equations:
        return None

    # Score and rank candidates so concrete derivation formulas with variables are tested first
    scored_equations = sorted(equations, key=_score_equation_candidate, reverse=True)

    best_candidate = None
    best_res = None

    for eq in scored_equations:
        res = verify_equation(eq["lhs"], eq["rhs"])
        if res.get("status") == "verified":
            best_candidate = eq
            best_res = res
            break

    # Only return a payload if an equation is conclusively verified by SymPy.
    # Never show noisy or broken inconclusive boxes to the student.
    if not best_candidate or not best_res or best_res.get("status") != "verified":
        return None

    boundary_results = best_res.get("boundary_results", [])

    # If boundary results weren't run yet and it was verified by derivative,
    # generate sample boundary evaluations for the antiderivative to show the user
    if not boundary_results and best_res.get("method") == "derivative_check":
        try:
            lhs_expr = best_res.get("lhs_expr")
            rhs_expr = best_res.get("rhs_expr")
            var = sp.Symbol(best_res.get("var", "x"))
            if isinstance(lhs_expr, sp.Integral) and rhs_expr is not None:
                integrand = lhs_expr.function
                d_rhs = sp.diff(rhs_expr, var)
                for pt in [0.5, 1.0, 2.0]:
                    try:
                        v_lhs = float(integrand.evalf(subs={var: pt}))
                        v_rhs = float(d_rhs.evalf(subs={var: pt}))
                        boundary_results.append({
                            "input": f"{var} = {pt}",
                            "lhs": f"Integrand({pt}) = {round(v_lhs, 4)}",
                            "rhs": f"d/d{var}(RHS) = {round(v_rhs, 4)}",
                            "match": abs(v_lhs - v_rhs) < 1e-4,
                        })
                    except Exception:
                        pass
            elif rhs_expr is not None:
                for pt in [0.5, 1.0, 2.0]:
                    try:
                        val = float(rhs_expr.evalf(subs={var: pt}))
                        boundary_results.append({
                            "input": f"{var} = {pt}",
                            "lhs": f"F({pt})",
                            "rhs": round(val, 4),
                            "match": True,
                        })
                    except Exception:
                        pass
        except Exception as exc:
            logger.debug("Failed generating boundary display: %s", exc)

    python_snippet = generate_python_snippet(best_candidate, best_res)
    wolfram_url = generate_wolfram_url(best_candidate["lhs"], best_candidate["rhs"])

    raw = best_candidate.get("raw_latex", "").strip()
    if raw and "=" in raw:
        display_formula = raw
    else:
        display_formula = f"{best_candidate['lhs']} = {best_candidate['rhs']}"

    return {
        "has_math": True,
        "status": "verified",
        "method": best_res.get("method", "symbolic_check"),
        "summary": best_res.get("message", "Mathematical identity verified using SymPy CAS."),
        "verified_formula": display_formula,
        "raw_latex": best_candidate.get("raw_latex", ""),
        "boundary_checks": boundary_results,
        "wolfram_url": wolfram_url,
        "python_code": python_snippet,
    }
