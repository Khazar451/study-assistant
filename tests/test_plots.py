from src.generation.plots import embed_plots, ensure_requested_plot, evaluate_formula, render_plot


def test_evaluate_formula_rejects_calls_outside_math():
    try:
        evaluate_formula("__import__('os').system('echo no')", {"x": 1})
        raise AssertionError("unsafe expression was accepted")
    except ValueError:
        pass


def test_render_curve_and_surface(tmp_path):
    curve = render_plot(
        {"dim": 2, "expr": "x**2", "xmin": -2, "xmax": 2, "title": "parabola"},
        cache_dir=tmp_path,
    )
    surface = render_plot(
        {"dim": 3, "expr": "x**2 - y**2", "xmin": -2, "xmax": 2, "title": "saddle"},
        cache_dir=tmp_path,
    )
    assert curve.exists() and curve.stat().st_size > 1000
    assert surface.exists() and surface.stat().st_size > 1000


def test_requested_plot_is_added_when_the_model_skips_it(tmp_path):
    answer = ensure_requested_plot(
        "explain integration with plots and compare it with the derivative",
        "Integration measures area. The derivative measures slope.",
        cache_dir=tmp_path,
    )
    assert "/api/plots/" in answer
    assert answer.index("slope") < answer.index("/api/plots/")


def test_embed_plots_replaces_fence(tmp_path, monkeypatch):
    monkeypatch.setattr("src.generation.plots.CACHE_DIR", tmp_path)
    answer = (
        "A parabola.\n"
        "```plot\n"
        '{"dim": 2, "expr": "sin(x)", "xmin": -3, "xmax": 3, "title": "sine"}\n'
        "```\n"
    )
    rendered = embed_plots(answer, cache_dir=tmp_path)
    assert "```plot" not in rendered
    assert "/api/plots/" in rendered
    assert rendered.endswith(".png)\n") or ".png)" in rendered
