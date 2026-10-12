import os
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_DEFAULT_MODEL = "meta/llama-3.2-11b-vision-instruct"

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
NEBIUS_ANSWER_MODEL = "nvidia/nemotron-3-super-120b-a12b"
NEBIUS_FAST_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"

PROVIDER_LABELS = {
    "nebius": "Nebius Token Factory",
    "nvidia": "NVIDIA NIM",
}


@dataclass(frozen=True)
class LLMProvider:
    name: str
    api_key: str
    base_url: str
    model: str

    role: str = "answer"

    @property
    def label(self) -> str:
        return PROVIDER_LABELS.get(self.name, self.name)

    @property
    def extra_body(self) -> Optional[dict]:
        """Request extras for this provider and role.

        Nemotron reasoning tokens count against `max_tokens`; on short helper calls they can
        consume the whole budget and return no content, so reasoning is off for `role="fast"`.
        """
        if self.name == "nebius" and self.role == "fast":
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return None


def resolve_llm_provider(role: str = "answer") -> Optional[LLMProvider]:
    """Pick the chat-completion provider from the environment.

    Nebius Token Factory wins whenever NEBIUS_API_KEY is set. `role="answer"` selects the
    reasoning model used for tutor answers; `role="fast"` selects the low-latency model used
    for query augmentation and reranking.
    """
    nebius_key = os.getenv("NEBIUS_API_KEY")
    if nebius_key:
        if role == "answer":
            model = os.getenv("NEBIUS_MODEL", NEBIUS_ANSWER_MODEL)
        else:
            model = os.getenv("NEBIUS_FAST_MODEL", NEBIUS_FAST_MODEL)
        return LLMProvider(
            name="nebius",
            api_key=nebius_key,
            base_url=os.getenv("NEBIUS_BASE_URL", NEBIUS_BASE_URL),
            model=model,
            role=role,
        )

    nvidia_key = os.getenv("NVIDIA_API_KEY")
    if nvidia_key:
        return LLMProvider(
            name="nvidia",
            api_key=nvidia_key,
            base_url=os.getenv("NVIDIA_BASE_URL", NVIDIA_BASE_URL),
            model=os.getenv("LLM_MODEL", NVIDIA_DEFAULT_MODEL),
            role=role,
        )

    return None


def provider_name_for_url(base_url: str) -> str:
    return "nebius" if "nebius" in base_url else "nvidia"


_CITATION_RE = re.compile(r"\[Source:\s*([^,\]]+?),\s*Pages?:\s*([\d,\s\-–]+?)\s*\]")


def normalize_citation_brackets(text: str) -> str:
    """Nemotron often writes citations with full-width lenticular brackets (【...】)."""
    return text.replace("【", "[").replace("】", "]")


def extract_citations(answer: str) -> List[str]:
    """Return sorted unique `"<source> (p. <page>)"` labels cited in a generated answer."""
    labels = set()
    for source, pages in _CITATION_RE.findall(normalize_citation_brackets(answer)):
        for page in re.findall(r"\d+", pages):
            labels.add(f"{source.strip()} (p. {page})")
    return sorted(labels)


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_BLOCK_RE = re.compile(r"<think>.*?(</think>|$)\s*", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove `<think>...</think>` blocks that reasoning models may emit inline."""
    return _THINK_BLOCK_RE.sub("", text).strip()


def _partial_tag_suffix(text: str, tag: str) -> int:
    for size in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


def filter_reasoning_stream(tokens: Iterable[str]) -> Iterator[str]:
    """Streaming counterpart of `strip_reasoning`; tags may be split across tokens."""
    buffer = ""
    in_think = False
    for token in tokens:
        buffer += token
        while buffer:
            if in_think:
                end = buffer.find(_THINK_CLOSE)
                if end == -1:
                    buffer = buffer[-(len(_THINK_CLOSE) - 1):]
                    break
                buffer = buffer[end + len(_THINK_CLOSE):].lstrip()
                in_think = False
            else:
                start = buffer.find(_THINK_OPEN)
                if start != -1:
                    if start:
                        yield buffer[:start]
                    buffer = buffer[start + len(_THINK_OPEN):]
                    in_think = True
                    continue
                keep = _partial_tag_suffix(buffer, _THINK_OPEN)
                if len(buffer) > keep:
                    yield buffer[: len(buffer) - keep]
                    buffer = buffer[len(buffer) - keep:]
                break
    if buffer and not in_think:
        yield buffer
