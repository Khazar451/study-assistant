from src import providers


def test_extract_citations_accepts_fullwidth_brackets_and_page_lists():
    answer = (
        "Batch prediction is asynchronous【Source: dmls.pdf, Page: 279】. "
        "Online prediction is synchronous [Source: dmls.pdf, Pages: 279, 281]. "
        "Repeated [Source: notes.md, Page: 2] and [Source: notes.md, Page: 2]."
    )
    assert providers.extract_citations(answer) == [
        "dmls.pdf (p. 279)",
        "dmls.pdf (p. 281)",
        "notes.md (p. 2)",
    ]


def test_normalize_citation_brackets():
    assert providers.normalize_citation_brackets("【Source: a.pdf, Page: 1】") == "[Source: a.pdf, Page: 1]"


def test_filter_reasoning_stream_passes_through_plain_angle_brackets():
    tokens = ["if a ", "<", " b then ", "<thin", "g>"]
    assert "".join(providers.filter_reasoning_stream(tokens)) == "if a < b then <thing>"


def test_fast_role_disables_reasoning_only_on_nebius(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "neb-key")
    assert providers.resolve_llm_provider("fast").extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert providers.resolve_llm_provider("answer").extra_body is None

    monkeypatch.delenv("NEBIUS_API_KEY")
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-key")
    assert providers.resolve_llm_provider("fast").extra_body is None
