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
    text = re.sub(r"\n+## References\b[\s\S]*$", "", text).strip()
    return text, references


def polish_tutor_answer(text: str, allowed_sources: Iterable[str] | None = None, grounding: str = "course") -> str:
    """Normalize tutor markdown and drop citations that are not in the retrieved set."""
    cleaned = strip_emoji(text or "")
    cleaned = setext_to_atx(cleaned)
    if grounding == "general":
        cleaned = _drop_all_citations(cleaned)
    elif allowed_sources is not None:
        cleaned = drop_unverified_citations(cleaned, list(allowed_sources))
    return cleaned.strip()


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
