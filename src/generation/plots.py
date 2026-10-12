"""Render a tutor plot spec into a PNG. Expressions are arithmetic only."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CACHE_DIR = Path("./data/plot_cache")
_FENCE = re.compile(r"```plot\s*\n([\s\S]*?)```", re.IGNORECASE)
_FUNCS: Dict[str, Callable[..., Any]] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "pi": np.pi,
    "e": np.e,
}
_BINOPS = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.divide,
    ast.Pow: np.power,
}


def _eval_node(node: ast.AST, variables: Dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        if node.id in _FUNCS:
            return _FUNCS[node.id]
        raise ValueError(f"Unknown name: {node.id}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand, variables)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left, variables), _eval_node(node.right, variables))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        args = [_eval_node(arg, variables) for arg in node.args]
        return _FUNCS[node.func.id](*args)
    raise ValueError("Expression is not a plain formula")


def evaluate_formula(expr: str, variables: Dict[str, Any]) -> Any:
    cleaned = expr.replace("^", "**")
    tree = ast.parse(cleaned, mode="eval")
    return _eval_node(tree.body, variables)


def _curves(spec: Dict[str, Any]) -> List[Dict[str, str]]:
    curves = spec.get("curves")
    if isinstance(curves, list) and curves:
        return [item for item in curves if isinstance(item, dict) and item.get("expr")]
    expr = spec.get("expr")
    if isinstance(expr, str) and expr.strip():
        return [{"expr": expr, "label": spec.get("label") or spec.get("title") or expr}]
    return []


def render_plot(spec: Dict[str, Any], cache_dir: Path = CACHE_DIR) -> Path:
    curves = _curves(spec)
    if not curves:
        raise ValueError("Plot needs an expr")
    dim = int(spec.get("dim") or (3 if any("z" == key for key in spec) else 2))
    xmin = float(spec.get("xmin", -5))
    xmax = float(spec.get("xmax", 5))
    if xmin >= xmax:
        xmin, xmax = -5.0, 5.0

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    path = cache_dir / f"{key}.png"
    if path.exists():
        return path

    figure = plt.figure(figsize=(6.4, 4.2), dpi=120)
    try:
        if dim == 3:
            ymin = float(spec.get("ymin", xmin))
            ymax = float(spec.get("ymax", xmax))
            axis = figure.add_subplot(111, projection="3d")
            grid = np.linspace(xmin, xmax, 60)
            y_grid = np.linspace(ymin, ymax, 60)
            x_mesh, y_mesh = np.meshgrid(grid, y_grid)
            z_mesh = evaluate_formula(curves[0]["expr"], {"x": x_mesh, "y": y_mesh})
            axis.plot_surface(x_mesh, y_mesh, z_mesh, cmap="viridis", linewidth=0, antialiased=True)
            axis.set_xlabel(str(spec.get("xlabel") or "x"))
            axis.set_ylabel(str(spec.get("ylabel") or "y"))
            axis.set_zlabel(str(spec.get("zlabel") or "z"))
        else:
            axis = figure.add_subplot(111)
            x_values = np.linspace(xmin, xmax, 400)
            for curve in curves[:4]:
                y_values = evaluate_formula(curve["expr"], {"x": x_values})
                axis.plot(x_values, y_values, label=str(curve.get("label") or curve["expr"]))
            axis.axhline(0, color="#bbbbbb", linewidth=0.8)
            axis.axvline(0, color="#bbbbbb", linewidth=0.8)
            shade = spec.get("shade")
            if isinstance(shade, dict) and shade.get("expr"):
                left = float(shade.get("a", xmin))
                right = float(shade.get("b", xmax))
                shade_x = np.linspace(left, right, 200)
                shade_y = evaluate_formula(str(shade["expr"]), {"x": shade_x})
                axis.fill_between(shade_x, shade_y, 0, alpha=0.25, label="accumulated area")
            axis.set_xlabel(str(spec.get("xlabel") or "x"))
            axis.set_ylabel(str(spec.get("ylabel") or "y"))
            axis.grid(True, alpha=0.3)
            if len(curves) > 1:
                axis.legend()
        title = spec.get("title")
        if title:
            axis.set_title(str(title))
        figure.tight_layout()
        figure.savefig(path, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)
    return path


def embed_plots(answer: str, cache_dir: Path = CACHE_DIR) -> str:
    """Replace ```plot blocks with rendered images. Invalid specs are dropped."""

    def replace(match: re.Match) -> str:
        raw = match.group(1).strip()
        try:
            spec = json.loads(raw)
            if not isinstance(spec, dict):
                raise ValueError("Plot spec must be an object")
            image = render_plot(spec, cache_dir=cache_dir)
        except Exception:
            return ""
        title = spec.get("title") or "Graph"
        return f"\n![{title}](/api/plots/{image.name})\n"

    return _FENCE.sub(replace, answer)


_PLOT_REQUEST = re.compile(r"\b(plot|plots|graph|graphs|curve|curves|sketch|surface)\b", re.IGNORECASE)


def teaching_plot_spec(query: str) -> Dict[str, Any]:
    """A concrete figure when the student asked for a graph and the model drew none."""
    text = (query or "").lower()
    if any(word in text for word in ("surface", "x, y, z", "x,y,z", "x, z, y", "x,z,y")):
        return {
            "dim": 3,
            "expr": "x**2 - y**2",
            "xmin": -2,
            "xmax": 2,
            "title": "z = x^2 - y^2",
        }
    if "integrat" in text and "deriv" in text:
        return {
            "dim": 2,
            "curves": [
                {"expr": "0.25*x**2", "label": "f(x) = x^2/4"},
                {"expr": "0.5*x", "label": "f'(x) = x/2"},
            ],
            "shade": {"expr": "0.25*x**2", "a": 0, "b": 3},
            "xmin": -1,
            "xmax": 4,
            "title": "Area under f, and the slope f'",
        }
    if "integrat" in text:
        return {
            "dim": 2,
            "expr": "0.25*x**2",
            "shade": {"expr": "0.25*x**2", "a": 0, "b": 3},
            "xmin": -1,
            "xmax": 4,
            "title": "Integral as accumulated area",
        }
    return {"dim": 2, "expr": "x**2", "xmin": -3, "xmax": 3, "title": "y = x^2"}


def ensure_requested_plot(query: str, answer: str, cache_dir: Path = CACHE_DIR) -> str:
    """Draw a graph when the student asked for one and the answer has no plot."""
    if "/api/plots/" in (answer or "") or not _PLOT_REQUEST.search(query or ""):
        return answer
    spec = teaching_plot_spec(query)
    image = render_plot(spec, cache_dir=cache_dir)
    title = spec.get("title") or "Graph"
    return answer.rstrip() + f"\n\n![{title}](/api/plots/{image.name})\n"
