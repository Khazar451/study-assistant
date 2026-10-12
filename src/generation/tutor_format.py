"""Tutor answer cleanup: headings, emoji, and citations that were not retrieved."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse

_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\u200d"
    "\ufe0f"
    "]+",
    flags=re.UNICODE,
)
_SETEXT_H1 = re.compile(r"^=+$")
_SETEXT_H2 = re.compile(r"^-+$")
_URL_RE = re.compile(r"https?://[^\s)>\]]+")
_CITE_RE = re.compile(
    r"\s*\[(?:Source|Web Source):\s*([^\]]+?)\]",
    re.IGNORECASE,
)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")


def infer_grounding(chunks: Sequence[Dict[str, Any]] | None) -> str:
    """Classify an answer as course notes, academic web, or ungrounded tutoring."""
    if not chunks:
        return "general"
    kinds = {
        (chunk.get("metadata") or {}).get("source_type", "document")
        for chunk in chunks
        if isinstance(chunk, dict)
    }
    if kinds and kinds <= {"web"}:
        return "web"
    return "course"


def allowed_source_keys(chunks: Sequence[Dict[str, Any]] | None, context: str = "") -> List[str]:
    keys: List[str] = []
    for chunk in chunks or []:
        if not isinstance(chunk, dict):
            continue
        meta = chunk.get("metadata") or {}
        for value in (meta.get("source"), meta.get("title"), meta.get("domain")):
            if isinstance(value, str) and value.strip():
                keys.append(value.strip().lower())
    for match in re.findall(r"Source:\s*([^|\n)]+)", context or ""):
        cleaned = match.strip()
        if cleaned and cleaned.lower() != "unknown source":
            keys.append(cleaned.lower())
    return keys


_SOURCE_CITE = re.compile(
    r"\[(?:Source|Web Source):\s*([^,\]]+?),\s*Pages?:\s*([^\]]+?)\]",
    re.IGNORECASE,
)
_NUM_CITE = re.compile(r"\[(\d{1,2})\]")


def build_references(chunks: Sequence[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    """One reference per retrieved chunk, in rank order."""
    references: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks or [], start=1):
        if not isinstance(chunk, dict):
            continue
        meta = chunk.get("metadata") or {}
        source = str(meta.get("source") or "Unknown")
        page = meta.get("page") or 1
        source_type = meta.get("source_type") or "document"
        url = source if source.startswith("http") else ""
        label = source if source_type == "web" or url else f"{source}, page {page}"
        preview = chunk.get("text") or chunk.get("page_content") or chunk.get("preview") or ""
        references.append({
            "n": index,
            "label": label,
            "source": source,
            "page": page,
            "source_type": source_type,
            "file_path": meta.get("file_path") or "",
            "url": url,
            "preview": str(preview)[:360],
        })
    return references


def bind_numbered_citations(
    answer: str,
    chunks: Sequence[Dict[str, Any]] | None,
) -> tuple[str, List[Dict[str, Any]]]:
    """Turn verified source tags into [1], [2], ... and drop numbers that were not retrieved."""
    references = build_references(chunks)
    if not references:
        return _drop_all_citations(answer or "").strip(), []

    def match_number(source: str, page: str) -> Optional[int]:
        source_l = source.strip().lower()
        page_numbers = re.findall(r"\d+", page or "")
        page_n = page_numbers[0] if page_numbers else ""
        for ref in references:
            ref_source = str(ref["source"]).lower()
            same_source = source_l in ref_source or ref_source in source_l
            if same_source and (not page_n or str(ref["page"]) == page_n):
                return int(ref["n"])
        for ref in references:
            ref_source = str(ref["source"]).lower()
            if source_l and (source_l in ref_source or ref_source in source_l):
                return int(ref["n"])
        return None

    def replace_source(match: re.Match) -> str:
        number = match_number(match.group(1), match.group(2))
        return f"[{number}]" if number else ""

    text = _SOURCE_CITE.sub(replace_source, answer or "")

    def keep_known(match: re.Match) -> str:
        number = int(match.group(1))
        return match.group(0) if 1 <= number <= len(references) else ""

    text = _NUM_CITE.sub(keep_known, text)

    def lift_bare_line(match: re.Match) -> str:
        number = int(match.group(1))
        if 1 <= number <= len(references):
            return f"[{number}]"
        return match.group(0)

    text = re.sub(r"(?m)^(\d{1,2})\s*$", lift_bare_line, text)
    text = re.sub(r"\n+## References\b[\s\S]*$", "", text).strip()
    return text, references


_MATH_DELIM = re.compile(r"\$\$[\s\S]*?\$\$|\$(?!\$)(?:\\.|[^$\n])+\$")
_MATH_COMMAND = re.compile(
    r"\\(?:displaystyle|int|iint|oint|frac|sum|prod|lim|sqrt|partial|Big|left|right|to|infty)(?![A-Za-z])"
)
_BRACKET_DISPLAY = re.compile(
    r"\[\s*(\\(?:displaystyle|frac|partial|int|iint|sum|lim|prod)[\s\S]*?)\s*\]"
)
_TEX_INLINE = re.compile(r"\\\(([\s\S]*?)\\\)")
_TEX_DISPLAY = re.compile(r"\\\[([\s\S]*?)\\\]")


def _repair_latex(formula: str) -> str:
    """Turn model shorthand into LaTeX KaTeX can render."""
    formula = formula.replace(r"\displaystyle", " ")
    formula = re.sub(r"(?<!\\),\s*(d[a-zA-Z])", r"\\,\1", formula)
    formula = re.sub(r"\s+", " ", formula).strip(" ,")
    return formula


def _matching_paren(text: str, open_at: int) -> int:
    depth = 0
    for index in range(open_at, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _is_display_formula(formula: str) -> bool:
    return bool(re.search(r"\\(?:frac|int|iint|sum|lim|prod)(?![A-Za-z])", formula)) or len(formula) > 80


def _math_span(formula: str) -> str:
    repaired = _repair_latex(formula)
    if _is_display_formula(repaired):
        return f"\n$$\n{repaired}\n$$\n"
    return f"${repaired}$"


def _looks_like_inline_math(body: str) -> bool:
    if re.search(r"[A-Za-z]{5,}", body) and not _MATH_COMMAND.search(body):
        return False
    if _MATH_COMMAND.search(body) or "_" in body or "\\partial" in body:
        return True
    compact = re.sub(r"\s+", "", body)
    if re.fullmatch(r"[A-Za-z](?:\([^()]{0,40}\))+(?:,[A-Za-z](?:\([^()]{0,40}\))?)*", compact):
        return True
    if re.fullmatch(r"[A-Za-z](?:=[A-Za-z0-9_()+\\-]{1,40})+", compact):
        return True
    return False


def _wrap_bare_math(segment: str) -> str:
    segment = _TEX_DISPLAY.sub(lambda match: _math_span(match.group(1)), segment)
    segment = _TEX_INLINE.sub(lambda match: _math_span(match.group(1)), segment)
    segment = _BRACKET_DISPLAY.sub(lambda match: _math_span(match.group(1)), segment)

    pieces: List[str] = []
    index = 0
    while index < len(segment):
        if segment[index] == "(":
            end = _matching_paren(segment, index)
            body = segment[index + 1:end] if end != -1 else ""
            if end != -1 and _looks_like_inline_math(body):
                pieces.append(_math_span(body))
                index = end + 1
                continue
        pieces.append(segment[index])
        index += 1
    wrapped = "".join(pieces)

    lines: List[str] = []
    for line in wrapped.splitlines():
        stripped = line.strip().strip("\\").strip()
        if "$" in line:
            lines.append(line)
        elif _MATH_COMMAND.search(stripped) and len(stripped) < 300 and not re.search(r"[A-Za-z]{6,}", stripped):
            lines.append(_math_span(stripped).strip("\n"))
        else:
            lines.append(line)
    return "\n".join(lines)


def normalize_math(text: str) -> str:
    """Put bare TeX into $$...$$ and fix differentials written with a comma."""
    pieces: List[str] = []
    cursor = 0
    for match in _MATH_DELIM.finditer(text):
        pieces.append(_wrap_bare_math(text[cursor:match.start()]))
        pieces.append(_repair_latex(match.group(0)) if match.group(0).startswith("$$") or match.group(0).startswith("$") else match.group(0))
        # Keep delimiters; only repair the formula body.
        block = match.group(0)
        if block.startswith("$$"):
            pieces[-1] = "$$" + _repair_latex(block[2:-2]) + "$$"
        elif block.startswith("$"):
            pieces[-1] = "$" + _repair_latex(block[1:-1]) + "$"
        cursor = match.end()
    pieces.append(_wrap_bare_math(text[cursor:]))
    return "".join(pieces)


def polish_tutor_answer(text: str, allowed_sources: Iterable[str] | None = None, grounding: str = "course") -> str:
    """Normalize tutor markdown and drop citations that are not in the retrieved set."""
    cleaned = strip_emoji(text or "")
    cleaned = setext_to_atx(cleaned)
    if grounding == "general":
        cleaned = _drop_all_citations(cleaned)
    elif allowed_sources is not None:
        cleaned = drop_unverified_citations(cleaned, list(allowed_sources))
    return normalize_math(cleaned).strip()


def strip_emoji(text: str) -> str:
    return _EMOJI_RE.sub("", text)


def setext_to_atx(text: str) -> str:
    """Turn `Title\\n===` / `Title\\n---` into `##` / `###` headings."""
    lines = text.splitlines()
    output: List[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        nxt = lines[index + 1] if index + 1 < len(lines) else ""
        title = line.strip()
        if title and not title.startswith(("#", "-", "*", ">")) and _SETEXT_H1.fullmatch(nxt.strip() or ""):
            output.append(f"## {title}")
            index += 2
            continue
        if (
            title
            and not title.startswith(("#", "-", "*", ">", "|"))
            and len(nxt.strip()) >= 3
            and _SETEXT_H2.fullmatch(nxt.strip())
        ):
            output.append(f"### {title}")
            index += 2
            continue
        output.append(line)
        index += 1
    return "\n".join(output)


def drop_unverified_citations(text: str, allowed_sources: Sequence[str]) -> str:
    allowed = [item.lower() for item in allowed_sources if item]

    def cite_ok(body: str) -> bool:
        lowered = body.strip().lower()
        if not lowered:
            return False
        if any(lowered in item or item in lowered for item in allowed):
            return True
        host = urlparse(lowered).netloc.removeprefix("www.")
        return bool(host) and any(host in item or item in host for item in allowed)

    def replace_cite(match: re.Match) -> str:
        return match.group(0) if cite_ok(match.group(1)) else ""

    cleaned = _CITE_RE.sub(replace_cite, text)

    def replace_link(match: re.Match) -> str:
        return match.group(0) if cite_ok(match.group(2)) else match.group(1)

    cleaned = _MD_LINK_RE.sub(replace_link, cleaned)

    def replace_url(match: re.Match) -> str:
        return match.group(0) if cite_ok(match.group(0)) else ""

    return _URL_RE.sub(replace_url, cleaned)


def _drop_all_citations(text: str) -> str:
    cleaned = _CITE_RE.sub("", text)
    cleaned = _MD_LINK_RE.sub(r"\1", cleaned)
    return _URL_RE.sub("", cleaned)
