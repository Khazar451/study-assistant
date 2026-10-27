import logging
import os
import re
from typing import Any, Dict, Iterator, List, Optional, Union
from dotenv import load_dotenv
from openai import OpenAI

from src import providers
from src.generation.tutor_format import polish_tutor_answer

# Automatically load environment variables from .env.local or .env
load_dotenv(".env.local")
load_dotenv()

logger = logging.getLogger(__name__)

CUT_SHORT_NOTE = "*(Answer cut short. Ask me to continue.)*"


def cut_short_ending(text: str) -> str:
    """What to append when the model hit its token limit mid-answer.

    An open $$ would otherwise be closed at the very end by the math cleanup, turning everything
    after it, prose and headings included, into one formula that cannot render.
    """
    close = "\n$$" if text.count("$$") % 2 else ""
    return f"{close}\n\n{CUT_SHORT_NOTE}"


class StudyGenerator:
    """Answers academic and study questions strictly grounded in retrieved document context.

    Supports:
    1. Primary provider: Nebius Token Factory running NVIDIA Nemotron (https://api.tokenfactory.nebius.com/v1/)
    2. Fallback provider: NVIDIA NIM (https://integrate.api.nvidia.com/v1)
    3. Grounded citation formatting with anti-hallucination guardrails.
    4. Non-streaming and streaming generation.
    """

    NVIDIA_BASE_URL = providers.NVIDIA_BASE_URL
    NVIDIA_DEFAULT_MODEL = providers.NVIDIA_DEFAULT_MODEL

    NEBIUS_BASE_URL = providers.NEBIUS_BASE_URL
    NEBIUS_DEFAULT_MODEL = providers.NEBIUS_ANSWER_MODEL

    DEFAULT_SYSTEM_PROMPT = (
        "You are a study tutor. Teach the student; do not dump an encyclopedia entry.\n\n"
        "Rules:\n"
        "1. Grounding: Use only facts in the context excerpts. Do not add outside facts, dates, or URLs.\n"
        "2. Citations: The excerpts are numbered Document 1, Document 2, and so on. After each key fact write only that number in brackets, like [1]. Never invent a number, filename, or URL.\n"
        "3. Teaching: Open with a one-sentence core idea. Then explain with short ## and ### headings, bold terms, and bullets. When two ideas in the excerpts are easy to confuse on an exam, contrast them in a ### Exam contrast section.\n"
        "4. Check-in: End with a ## Check-in heading and exactly one question that asks which course they are preparing for, or offers a quiz, flashcards, or a deeper breakdown.\n"
        "5. Format: Use ATX headings (##, ###). Never duplicate hashes like '## ##'. Do not use underlined setext headings. Do not use emoji.\n"
        "7. Mathematics: Write all formulas, expressions, variables, and substitutions using strict LaTeX. "
        "CRITICAL: Never use raw Unicode math symbols (like ∫, ∑, ∂, √, →, ≠, ≤, ≥); always use standard LaTeX commands (\\int, \\sum, \\partial, \\sqrt, \\to, \\ne, \\le, \\ge). "
        "Wrap all inline symbols, substitutions, and short expressions in single dollars, like $u = e^x$, $dv = \\cos x\\,dx$, $v = \\sin x$, $\\int u\\,dy$, and $\\partial f/\\partial x$. "
        "Every standalone derivation step or full equation goes on its own line between exactly one pair of double dollars (never output consecutive '$$ $$'):\n"
        "$$\n\\int u\\,dv = uv - \\int v\\,du\n$$\n"
        "Never leave formulas as plain text. Never write bare backslashes to break lines. "
        "Use \\\\, before differentials (like \\,dx). CRITICAL: Never put citations inside math mode. Never write \\text{[1]}, \\tag{[1]}, or [1] inside $$ or $. "
        "ALWAYS place citations strictly outside the dollar delimiters on the regular markdown text line, like: $$...$$\\n[1].\n"
        "8. Graphs: when a curve or surface would teach the idea, add one plot block and no other code:\n"
        "```plot\n"
        "{\"dim\": 2, \"expr\": \"x**2\", \"xmin\": -3, \"xmax\": 3, \"title\": \"y = x^2\"}\n"
        "```\n"
        "For a surface use \"dim\": 3 and an expr in x and y, which is drawn as z. "
        "To compare two curves, use \"curves\": [{\"expr\": \"sin(x)\", \"label\": \"sin\"}, {\"expr\": \"cos(x)\", \"label\": \"cos\"}].\n"
        "9. Comparisons: use a markdown table with a header row. Do not describe the comparison as loose columns.\n"
        "6. If the excerpts do not support an answer, say so in one sentence and ask what notes they can upload. Do not guess."
    )

    WEB_SYSTEM_PROMPT = (
        "You are a study tutor answering from academic web excerpts, not from the student's syllabus.\n\n"
        "Rules:\n"
        "1. Use only the provided excerpts. Do not add facts or links that are not in them.\n"
        "2. Cite a web excerpt only as its document number, like [1]. Never write a URL that is not in the excerpts.\n"
        "3. Teach: one-sentence core idea, then ## and ### headings. Contrast exam-confusable ideas only when both sides are in the excerpts.\n"
        "4. End with ## Check-in and one follow-up question.\n"
        "5. ATX headings only (##, ###, never duplicate hashes like '## ##'). No emoji. No underlined headings.\n"
        "6. Mathematics: Write all formulas and symbols in valid LaTeX. Never use raw Unicode math characters (like ∫, ∑, ∂, √, →); use \\int, \\sum, \\partial, \\sqrt, \\to. Wrap short inline formulas and variables in $...$, and standalone equations in $$...$$ on their own line with valid LaTeX and \\\\, before differentials. Never put citations inside math delimiters; place them after closing $$ or $."
    )

    GENERAL_SYSTEM_PROMPT = (
        "You are a study tutor. No course notes or academic pages were retrieved for this question, "
        "so teach from general knowledge and say what is commonly tested. Do not pretend this is cited from a textbook.\n\n"
        "Rules:\n"
        "1. Do not include URLs, filenames, or [Source: ...] citations.\n"
        "2. Open with a one-sentence core idea. Use ## and ### headings. Contrast ideas students mix up on exams.\n"
        "3. End with ## Check-in and exactly one question about their course, or an offer to quiz them.\n"
        "4. ATX headings only (##, ###, never duplicate hashes like '## ##'). No emoji. No underlined headings. No lifestyle or trivia unless it is a standard exam point.\n"
        "5. Mathematics: Write all formulas and symbols in valid LaTeX. Never use raw Unicode math characters (like ∫, ∑, ∂, √, →); use \\int, \\sum, \\partial, \\sqrt, \\to. Wrap short inline formulas and variables in $...$, and standalone equations in $$...$$ on their own line with valid LaTeX and \\\\, before differentials."
    )



    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
        system_prompt: Optional[str] = None,
        client: Optional[OpenAI] = None,
    ):
        """Initialize the StudyGenerator client.

        Priority order for credentials:
        1. Explicit api_key argument
        2. Primary: NEBIUS_API_KEY environment variable (Nebius Token Factory)
        3. Fallback: NVIDIA_API_KEY environment variable (NVIDIA NIM)
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        if api_key:
            self.api_key = api_key
            if base_url:
                self.base_url = base_url
                self.provider = providers.provider_name_for_url(base_url)
            else:
                self.base_url = self.NVIDIA_BASE_URL
                self.provider = "nvidia"
            self.model = model or (
                self.NEBIUS_DEFAULT_MODEL if self.provider == "nebius" else self.NVIDIA_DEFAULT_MODEL
            )
        else:
            resolved = providers.resolve_llm_provider("answer")
            if resolved is None:
                raise ValueError(
                    "No API key found. Please set NEBIUS_API_KEY (Nebius Token Factory, primary) or "
                    "NVIDIA_API_KEY (NVIDIA NIM, fallback) as an environment variable or pass api_key."
                )
            self.api_key = resolved.api_key
            self.base_url = base_url or resolved.base_url
            self.model = model or resolved.model
            self.provider = resolved.name

        self.provider_label = providers.PROVIDER_LABELS.get(self.provider, self.provider)

        # Nemotron 3 Ultra handles derivations/proofs on Token Factory; Super handles the rest.
        # Only used when the answer model was resolved from the environment (not pinned by the caller).
        self.reasoning_model: Optional[str] = None
        if not api_key and not model and self.provider == "nebius":
            reasoning = providers.resolve_llm_provider("reasoning")
            if reasoning and os.getenv("ENABLE_REASONING_ROUTER", "true").lower() != "false":
                self.reasoning_model = reasoning.model
        self.last_model = self.model

        # The SDK default (600 s, 2 retries) would hold a stalled request for up to half an hour;
        # 120 s still leaves room for a long non-streamed Ultra answer.
        self.client = client or OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=120.0,
            max_retries=1,
        )

    def prepare_context(self, context: Union[str, List[Dict[str, Any]]]) -> str:
        """Format raw context string or list of retrieved chunk dictionaries.

        Args:
            context: Either a pre-formatted context string or a list of chunk dicts
                     (each containing 'text'/'page_content' and optional 'metadata').

        Returns:
            Formatted context string with document tags and source/page metadata.
        """
        if isinstance(context, str):
            return context.strip()

        if not isinstance(context, list) or len(context) == 0:
            return ""

        context_blocks: List[str] = []
        for idx, chunk in enumerate(context, start=1):
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", "Unknown Source")
            page = metadata.get("page", 1)
            source_type = metadata.get("source_type", "document")
            text = chunk.get("text", chunk.get("page_content", "")).strip()

            if source_type == "web":
                block = f"[Document {idx}] (Web Source: {source})\n{text}"
            else:
                block = f"[Document {idx}] (Source: {source} | Page: {page})\n{text}"
            context_blocks.append(block)

        return "\n\n---\n\n".join(context_blocks)

    def _extract_sources(self, context_str: str) -> List[str]:
        """Extract unique source filenames from context string."""
        matches = re.findall(r"Source:\s*([^|\n)]+)", context_str)
        sources: List[str] = []
        seen = set()
        for m in matches:
            clean = m.strip()
            if clean and clean not in seen and clean != "Unknown Source":
                seen.add(clean)
                sources.append(clean)
        return sources

    def _prompt_for(self, grounding: str) -> str:
        if grounding == "web":
            return self.WEB_SYSTEM_PROMPT
        if grounding == "general":
            return self.GENERAL_SYSTEM_PROMPT
        return self.system_prompt

    # Derivations and proofs need room: the reasoning model gets this budget, and so does the answer
    # model when it stands in for it, so a fallback answer is not cut off mid-formula.
    REASONING_MAX_TOKENS = 4000
    # A 550B reasoning model can take minutes to start under load; other calls keep the client's 120 s.
    REASONING_TIMEOUT_S = 300.0

    def _create_completion(self, query: str, messages: List[Dict[str, str]], stream: bool):
        """Call the chat API, escalating hard questions to the reasoning model with a safe fallback."""
        kwargs: Dict[str, Any] = {"messages": messages, "temperature": self.temperature}
        if stream:
            kwargs["stream"] = True
        deep = bool(self.reasoning_model) and providers.needs_deep_reasoning(query)
        budget = max(self.max_tokens, self.REASONING_MAX_TOKENS) if deep else self.max_tokens
        if deep:
            # Logged at WARNING so the reason shows in the host's log (INFO is not printed by default).
            if not providers.reserve_reasoning_call():
                logger.warning("Hourly cap for %s reached; %s answers instead", self.reasoning_model, self.model)
            else:
                try:
                    response = self.client.chat.completions.create(
                        model=self.reasoning_model, max_tokens=budget, timeout=self.REASONING_TIMEOUT_S, **kwargs
                    )
                    self.last_model = self.reasoning_model
                    return response
                except Exception as exc:  # model unavailable / rate limited: fall back to the answer model
                    logger.warning(
                        "Reasoning model %s failed (%s: %s); %s answers instead",
                        self.reasoning_model, type(exc).__name__, exc, self.model,
                    )
        self.last_model = self.model
        return self.client.chat.completions.create(model=self.model, max_tokens=budget, **kwargs)

    def complete_text(self, system_prompt: str, user_prompt: str, max_tokens: int = 1500) -> str:
        """One short non-streaming completion for structured helper output (e.g. flashcards JSON).

        Uses the provider's fast model with thinking off when one is configured (Nemotron Nano on
        Token Factory), so reasoning cannot consume the token budget; any <think> block is stripped.
        """
        fast = providers.resolve_llm_provider("fast")
        same_provider = fast is not None and fast.name == self.provider
        model = fast.model if same_provider else self.model
        extra = {"extra_body": fast.extra_body} if same_provider and fast.extra_body else {}
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.2,
            max_tokens=max_tokens,
            **extra,
        )
        return providers.strip_reasoning(response.choices[0].message.content or "")

    def generate(
        self,
        query: str,
        context: Union[str, List[Dict[str, Any]]],
        grounding: str = "course",
        allowed_sources: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Generate a complete, grounded answer to the user query.

        Args:
            query: The student's question.
            context: Retrieved context (formatted string or chunk dict list).
            grounding: Grounding classification ("course", "web", or "general").
            allowed_sources: Optional set of allowed source names.
            history: Optional conversation history turns [{"role": "user"|"assistant", "content": ...}].

        Returns:
            Dictionary containing 'answer', 'model', 'provider', and 'sources'.
        """
        if not query or not query.strip():
            return {
                "answer": "Please provide a valid question.",
                "model": self.model,
                "provider": self.provider,
                "sources": [],
            }

        formatted_context = self.prepare_context(context)
        sources = self._extract_sources(formatted_context)

        if not formatted_context:
            user_content = (
                f"Student Question: {query.strip()}\n\n"
                f"Context Excerpts:\n[No study materials found for this topic.]"
            )
        else:
            user_content = (
                f"Student Question: {query.strip()}\n\n"
                f"Context Excerpts:\n{formatted_context}"
            )

        messages: List[Dict[str, str]] = [{"role": "system", "content": self._prompt_for(grounding)}]
        if history:
            for turn in history[-8:]:
                role = turn.get("role", "user")
                if role not in {"user", "assistant", "system"}:
                    role = "user" if role == "student" else "assistant"
                content = turn.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_content})

        response = self._create_completion(query, messages, stream=False)

        raw = providers.strip_reasoning(response.choices[0].message.content or "")
        if getattr(response.choices[0], "finish_reason", None) == "length":
            raw += cut_short_ending(raw)

        answer_text = polish_tutor_answer(
            providers.normalize_citation_brackets(raw),
            allowed_sources=allowed_sources if allowed_sources is not None else sources,
            grounding=grounding,
        )

        return {
            "answer": answer_text,
            "model": self.last_model,
            "provider": self.provider,
            "provider_label": self.provider_label,
            "sources": sources,
        }

    def generate_stream(
        self,
        query: str,
        context: Union[str, List[Dict[str, Any]]],
        grounding: str = "course",
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Iterator[str]:
        """Stream generated response tokens one by one for interactive UI typewriter effect.

        Args:
            query: The student's question.
            context: Retrieved context (formatted string or chunk dict list).
            grounding: Grounding classification ("course", "web", or "general").
            history: Optional conversation history turns [{"role": "user"|"assistant", "content": ...}].

        Yields:
            Token text chunks as they are generated by the model.
        """
        if not query or not query.strip():
            yield "Please provide a valid question."
            return

        formatted_context = self.prepare_context(context)

        if not formatted_context:
            user_content = (
                f"Student Question: {query.strip()}\n\n"
                f"Context Excerpts:\n[No study materials found for this topic.]"
            )
        else:
            user_content = (
                f"Student Question: {query.strip()}\n\n"
                f"Context Excerpts:\n{formatted_context}"
            )

        messages: List[Dict[str, str]] = [{"role": "system", "content": self._prompt_for(grounding)}]
        if history:
            for turn in history[-8:]:
                role = turn.get("role", "user")
                if role not in {"user", "assistant", "system"}:
                    role = "user" if role == "student" else "assistant"
                content = turn.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_content})

        stream = self._create_completion(query, messages, stream=True)
        finish = {"reason": None}

        def tokens() -> Iterator[str]:
            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if getattr(choice, "finish_reason", None):
                    finish["reason"] = choice.finish_reason
                if choice.delta and choice.delta.content:
                    yield choice.delta.content

        streamed: List[str] = []
        for token in providers.filter_reasoning_stream(tokens()):
            token = providers.normalize_citation_brackets(token)
            streamed.append(token)
            yield token
        if finish["reason"] == "length":
            yield cut_short_ending("".join(streamed))
