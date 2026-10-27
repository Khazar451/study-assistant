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

    text = re.sub(r"(?m)^[^\S\n]*(\d{1,2})[^\S\n]*$", lift_bare_line, text)

    def attach_orphan_cite(match: re.Match) -> str:
        number = int(match.group(2))
        if 1 <= number <= len(references):
            return f"{match.group(1)} [{number}]"
        return match.group(0)

    text = re.sub(r"([^\n])\n+(\d{1,2})[^\S\n]*(?=\n|$)", attach_orphan_cite, text)
    text = re.sub(r"\n+## References\b[\s\S]*$", "", text).strip()
    text = _clean_math_citations(text)
    return text, references


_MATH_CITE_RE = re.compile(
    r"(?:\\[q]?quad|\\hspace\{[^}]*\}|\s*)*(?:\\text|\\tag)?\s*\{?\s*(\[(?:Source|Web Source):[^\]]+?\]|\[\d{1,2}\])\s*\}?\s*([.,;:!?])?",
    re.IGNORECASE,
)


def _clean_math_citations(text: str) -> str:
    """Extract any citation [1], [2] or [Source: ...] out of $$...$$ or $...$ math blocks."""
    if not text:
        return ""

    def fix_display_block(match: re.Match) -> str:
        body = match.group(1)
        cites: List[str] = []
        punct = ""

        def repl(m: re.Match) -> str:
            nonlocal punct
            cites.append(m.group(1).strip())
            if m.group(2):
                punct = m.group(2)
            return ""

        cleaned_body = _MATH_CITE_RE.sub(repl, body).rstrip()
        if not cites:
            return match.group(0)
        cite_str = " ".join(cites) + punct
        if not cleaned_body.strip():
            return f" {cite_str}"  # math that held only a citation; an empty $$ would pair with the next formula
        return f"\n$$\n{cleaned_body}\n$$\n{cite_str}\n"

    # 1. Process display math $$ ... $$
    text = re.sub(r"\$\$([\s\S]*?)\$\$", fix_display_block, text)

    def fix_inline_block(match: re.Match) -> str:
        body = match.group(1)
        cites: List[str] = []
        punct = ""

        def repl(m: re.Match) -> str:
            nonlocal punct
            cites.append(m.group(1).strip())
            if m.group(2):
                punct = m.group(2)
            return ""

        cleaned_body = _MATH_CITE_RE.sub(repl, body).rstrip()
        if not cites:
            return match.group(0)
        cite_str = " ".join(cites) + punct
        if not cleaned_body.strip():
            return cite_str  # "$[3]$" would otherwise become a stray "$$ [3]"
        return f"${cleaned_body}$ {cite_str}"

    # 2. Process inline math $ ... $
    text = re.sub(r"(?<!\$)\$(?!\$)([\s\S]*?)(?<!\$)\$(?!\$)", fix_inline_block, text)
    return text



_UNICODE_MATH_MAP = {
    "∫": r"\int ",
    "∬": r"\iint ",
    "∭": r"\iiint ",
    "∮": r"\oint ",
    "∑": r"\sum ",
    "∏": r"\prod ",
    "√": r"\sqrt",
    "∂": r"\partial ",
    "∞": r"\infty ",
    "≈": r"\approx ",
    "≠": r"\ne ",
    "≤": r"\le ",
    "≥": r"\ge ",
    "→": r"\to ",
    "±": r"\pm ",
    "·": r"\cdot ",
    "×": r"\times ",
    "÷": r"\div ",
}


def _convert_unicode_math(text: str) -> str:
    """Convert raw Unicode mathematical symbols into standard LaTeX equivalents."""
    if not text:
        return ""
    for uni, lat in _UNICODE_MATH_MAP.items():
        text = text.replace(uni, lat)
    return text


_MATH_DELIM = re.compile(r"\$\$[\s\S]*?\$\$|\$(?!\$)(?:\\.|[^$\n])+\$")
_MATH_COMMAND = re.compile(
    r"\\(?:displaystyle|int|iint|oint|frac|sum|prod|lim|sqrt|partial|Big|left|right|to|infty)(?![A-Za-z])"
)
_BRACKET_DISPLAY = re.compile(
    r"(?m)^\s*\[\s*(\\(?:displaystyle|frac|partial|int|iint|sum|lim|prod)[\s\S]*?)\s*\]\s*$"
)
_TEX_INLINE = re.compile(r"\\\(([\s\S]*?)\\\)")
_TEX_DISPLAY = re.compile(r"\\\[([\s\S]*?)\\\]")
# \( \) \[ \] but not the \\[4pt] line break inside aligned environments
_TEX_MARK = re.compile(r"(?<!\\)\\([()\[\]])")
_BLANK_LINE = re.compile(r"\n[^\S\n]*\n")
_HEADING_LINE = re.compile(r"\n[^\S\n]{0,3}#{1,6}\s")
_TABLE_LINE = re.compile(r"\n[^\S\n]*\|")
_CITATIONS_ONLY = re.compile(r"\s*(?:(?:\[\d{1,2}\]|\[(?:Web )?Source:[^\]]*\])\s*,?\s*)+[.,;:]?\s*", re.IGNORECASE)
# What a stray $$ is followed by on its line: only citations, punctuation or a table bar
_STRAY_TAIL = re.compile(r"(?:\s|\[\d{1,2}\]|\[(?:Web )?Source:[^\]]*\]|[.,;:!?|)])+", re.IGNORECASE)


def open_math_closer(text: str) -> str:
    """The mark that closes a formula `text` leaves open, or "" when none is open."""
    if text.count("$$") % 2:
        return "\n$$"
    marks = _TEX_MARK.findall(text)
    if marks and marks[-1] == "[":
        return "\n\\]"
    if marks and marks[-1] == "(":
        return "\\)"
    return ""


def _pair_tex_delimiters(text: str) -> str:
    """Turn \\( \\) into $...$ and \\[ \\] into $$...$$, dropping any mark without a partner.

    Opening and closing marks differ, so a stray one is recognised and removed instead of
    pairing with the next formula and turning the prose between them into math.
    """
    marks = list(_TEX_MARK.finditer(text))
    if not marks:
        return text
    pairs: List[tuple] = []
    opener: Optional[re.Match] = None
    for mark in marks:
        kind = mark.group(1)
        if kind in "([":
            opener = mark  # an earlier opener with no closer yet is stray
            continue
        if opener is None or (opener.group(1), kind) not in {("(", ")"), ("[", "]")}:
            continue
        body = text[opener.end():mark.start()]
        broken = _BLANK_LINE.search(body) if kind == ")" else _HEADING_LINE.search("\n" + body.strip())
        if not broken:
            pairs.append((opener, mark))
        opener = None

    pieces: List[str] = []
    cursor = 0
    for opener_mark, closer_mark in pairs:
        pieces.append(_TEX_MARK.sub("", text[cursor:opener_mark.start()]))
        body = text[opener_mark.end():closer_mark.start()]
        if _CITATIONS_ONLY.fullmatch(body):
            pieces.append(" ".join(body.split()))  # \([3][4]\) is a citation, not math
        elif opener_mark.group(1) == "(":
            body = " ".join(body.split())
            pieces.append(f"${body}$" if body else "")
        else:
            body = _BLANK_LINE.sub("\n", body.strip())
            pieces.append(f"\n$$\n{body}\n$$\n" if body else "")
        cursor = closer_mark.end()
    pieces.append(_TEX_MARK.sub("", text[cursor:]))
    return "".join(pieces)


def _balance_display_dollars(text: str) -> str:
    """Drop a $$ that opens nothing instead of letting it swallow the rest of the answer.

    An opening $$ is kept only when its closing $$ comes before any heading or paragraph break,
    and it is not followed on its line by just a citation or punctuation, as in the model's
    "exponent $n$ $$ [2]." A formula left open at the very end of the answer is closed.
    """
    marks = [m.start() for m in re.finditer(r"\$\$", text)]
    if not marks:
        return text
    drop = set()
    close_at_end = False
    index = 0
    while index < len(marks):
        start = marks[index] + 2
        line_end = text.find("\n", start)
        tail = text[start:] if line_end == -1 else text[start:line_end]
        if tail.strip() and _STRAY_TAIL.fullmatch(tail):
            drop.add(marks[index])
            index += 1
            continue
        end = marks[index + 1] if index + 1 < len(marks) else len(text)
        line_start = text.rfind("\n", 0, marks[index]) + 1
        lead = text[line_start:marks[index]].strip()
        in_table = lead.startswith("|")
        # Blank lines just inside a $$ that sits alone on its line are harmless
        body = text[start:end].rstrip() if lead else text[start:end].strip()
        if (
            _BLANK_LINE.search(body)
            or _HEADING_LINE.search("\n" + body)
            or _TABLE_LINE.search("\n" + body)
            or (in_table and "\n" in text[start:end])
        ):
            drop.add(marks[index])
            index += 1
            continue
        if index + 1 == len(marks):
            close_at_end = True
        index += 2
    if not drop and not close_at_end:
        return text
    pieces: List[str] = []
    cursor = 0
    for at in sorted(drop):
        before = text[cursor:at].rstrip(" \t")
        pieces.append(before + " " if text[at + 2:at + 3].isalnum() else before)
        cursor = at + 2
    pieces.append(text[cursor:])
    result = "".join(pieces)
    return result + "\n$$" if close_at_end else result


def _repair_latex(formula: str) -> str:
    """Turn model shorthand into LaTeX KaTeX can render and repair unclosed sizing tags."""
    formula = formula.replace(r"\displaystyle", " ")
    formula = re.sub(r"(?<!\\),\s*(d[a-zA-Z])", r"\\,\1", formula)
    # Strip trailing dangling sizing tags with no delimiter (e.g. \bigl with no bracket)
    formula = re.sub(r"\\(?:left|right|bigl|bigr|Bigl|Bigr|biggl|biggr|Biggl|Biggr)\s*$", "", formula)
    # Strip lone sizing command where no delimiter follows
    formula = re.sub(r"\\(?:left|right|bigl|bigr|Bigl|Bigr|biggl|biggr|Biggl|Biggr)(?=\s*($|[^()\[\]{}|.\\/]))", "", formula)
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


_COMMON_STOP_WORDS = {
    "into", "another", "that", "is", "are", "was", "were", "often", "simpler", "to",
    "evaluate", "and", "or", "for", "with", "from", "then", "gives", "where", "when",
    "using", "by", "the", "a", "an", "in", "on", "at", "as", "of", "we", "can",
    "see", "which", "this", "these", "let", "suppose", "assuming", "yields", "results",
    "such", "starting", "how", "why", "what", "because", "so", "if", "not", "but"
}


def _is_display_formula(formula: str) -> bool:
    if any(k in formula for k in (r"\begin", r"\displaystyle", r"\Big", r"\tag")):
        return True
    has_eq = bool(re.search(r"[=<>]|\\approx|\\le|\\ge", formula))
    has_complex = bool(re.search(r"\\(?:frac|partial|sum|lim|prod|int|iint)", formula))
    if has_eq and has_complex and len(formula) >= 28:
        return True
    return len(formula) > 75


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


def _formula_end(text: str, start: int) -> int:
    """End index of one TeX expression, including nested braces and parentheses."""
    brace = 0
    paren = 0
    index = start
    while index < len(text):
        char = text[index]
        if char == "{":
            brace += 1
        elif char == "}":
            if brace == 0:
                break
            brace -= 1
        elif char == "(":
            paren += 1
        elif char == ")":
            if paren == 0:
                break
            paren -= 1
        elif char in "\n$":
            break
        elif char == "." and brace == 0 and paren == 0:
            nxt = text[index + 1] if index + 1 < len(text) else ""
            if not nxt.isdigit():
                break
        elif char == " " and brace == 0 and paren == 0:
            rest = text[index:].lstrip()
            if rest.startswith("#") or re.match(r"^\[(?:\d{1,2}|Source|Web Source)\b", rest, re.IGNORECASE):
                break
            word_match = re.match(r"^([a-zA-Z]{2,})\b", rest)
            if word_match and word_match.group(1).lower() in _COMMON_STOP_WORDS:
                break
        index += 1
    return index


def _extract_formula_runs(segment: str) -> str:
    pieces: List[str] = []
    index = 0
    while index < len(segment):
        if segment[index] == "\\" and _MATH_COMMAND.match(segment, index):
            end = _formula_end(segment, index)
            pieces.append(_math_span(segment[index:end]))
            index = end
            continue
        pieces.append(segment[index])
        index += 1
    return "".join(pieces)


def _process_bare_text(text: str) -> str:
    text = _extract_formula_runs(text)
    pieces: List[str] = []
    index = 0
    while index < len(text):
        if text.startswith("$$", index):
            end = text.find("$$", index + 2)
            end = len(text) if end == -1 else end + 2
            pieces.append(text[index:end])
            index = end
            continue
        if text[index] == "$":
            end = text.find("$", index + 1)
            end = len(text) if end == -1 else end + 1
            pieces.append(text[index:end])
            index = end
            continue
        if text[index] == "(":
            end = _matching_paren(text, index)
            body = text[index + 1:end] if end != -1 else ""
            if end != -1 and "$" not in body and _looks_like_inline_math(body):
                pieces.append(_math_span(body))
                index = end + 1
                continue
        pieces.append(text[index])
        index += 1
    return "".join(pieces)


def _wrap_bare_math(segment: str) -> str:
    segment = _TEX_DISPLAY.sub(lambda match: _math_span(match.group(1)), segment)
    segment = _TEX_INLINE.sub(lambda match: _math_span(match.group(1)), segment)
    segment = _BRACKET_DISPLAY.sub(lambda match: _math_span(match.group(1)), segment)

    pieces: List[str] = []
    cursor = 0
    for match in _MATH_DELIM.finditer(segment):
        if match.start() > cursor:
            pieces.append(_process_bare_text(segment[cursor:match.start()]))
        pieces.append(match.group(0))
        cursor = match.end()
    if cursor < len(segment):
        pieces.append(_process_bare_text(segment[cursor:]))
    wrapped = "".join(pieces)
    return re.sub(r"\(\s*(\$\$(?:[\s\S]*?)\$\$|\$[^$]+\$)\s*\)", r"\1", wrapped)


def _ensure_math_spacing(s: str) -> str:
    """Ensure a single whitespace exists between inline math ($...$) and adjacent words."""
    if not s or "$" not in s:
        return s
    parts = re.split(r'(\$\$[\s\S]*?\$\$|\$(?!\$)[^\$\n]+?(?<!\$)\$)', s)
    out: List[str] = []
    for i, part in enumerate(parts):
        if part.startswith("$"):
            if not part.startswith("$$"):
                # Prepend space if previous text ended with an alphanumeric character
                if out and out[-1] and re.search(r"[a-zA-Z0-9]$", out[-1]):
                    out[-1] += " "
            out.append(part)
        else:
            # Append space if previous block was inline math and this text starts with an alphanumeric character
            if i > 0 and parts[i - 1].startswith("$") and not parts[i - 1].startswith("$$"):
                if part and re.match(r"^[a-zA-Z0-9]", part):
                    part = " " + part
            out.append(part)
    return "".join(out)


def normalize_math(text: str) -> str:
    """Put bare TeX into $$...$$ and fix differentials written with a comma."""
    if not text:
        return ""

    # Convert raw Unicode math symbols (like ∫, ∑, ∂, √, →) into standard LaTeX
    text = _convert_unicode_math(text)

    # The model writes \( \) and \[ \]; the page renders $ and $$. Stray marks are dropped here,
    # before anything pairs dollars, so one stray cannot turn the following prose into math.
    text = _balance_display_dollars(_pair_tex_delimiters(text))

    # Pre-clean: extract citations out of math and collapse duplicate/empty delimiters
    text = _clean_math_citations(text)
    text = re.sub(r"\$\$(?:\s*\$\$)+", "$$", text)
    text = re.sub(r"\$\$\s*\$\$", "", text)
    text = re.sub(r"(?<!\$)\$(?!\$)\s*(?<!\$)\$(?!\$)", "", text)

    pieces: List[str] = []
    cursor = 0
    for match in _MATH_DELIM.finditer(text):
        if match.start() > cursor:
            pieces.append(_wrap_bare_math(text[cursor:match.start()]))
        block = match.group(0)
        if block.startswith("$$"):
            body = _repair_latex(block[2:-2])
            pieces.append(f"\n$$\n{body}\n$$\n")
        elif block.startswith("$"):
            body = _repair_latex(block[1:-1])
            pieces.append(f"${body}$")
        else:
            pieces.append(block)
        cursor = match.end()
    if cursor < len(text):
        pieces.append(_wrap_bare_math(text[cursor:]))

    result = "".join(pieces)
    result = re.sub(r"\$\$\s*\$\$", "", result)
    result = re.sub(r"\$\$\s*([.,;:!?])\s*\$\$", r"\1", result)
    result = _clean_math_citations(result)
    if result.count("$$") % 2 != 0:
        result = _balance_display_dollars(result)
    return _ensure_math_spacing(result)



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
    """Turn `Title\\n===` / `Title\\n---` into `##` / `###` headings and normalize ATX headings."""
    # Unescape escaped hashes: '\#\#' -> '##'
    text = re.sub(r"\\+(#{1,6})", r"\1", text)
    # Normalize non-breaking spaces
    text = re.sub(r"[\u00A0\u2000-\u200B]", " ", text)
    # Collapse duplicate hashes: '## ## Title' -> '## Title'
    text = re.sub(r"(?m)^(#{1,6})\s*(#{1,6})\s*", r"\1 ", text)
    # Ensure space after hashes: '##Title' -> '## Title'
    text = re.sub(r"(?m)^(#{1,6})([^\s#\n])", r"\1 \2", text)
    # Ensure headings that are glued to the previous line get preceded by a newline
    text = re.sub(r"([^#\n])\s*(#{1,6}\s+[^\n]+)", r"\1\n\n\2", text)

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
