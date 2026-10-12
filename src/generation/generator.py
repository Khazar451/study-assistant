import re
from typing import Any, Dict, Iterator, List, Optional, Union
from dotenv import load_dotenv
from openai import OpenAI

from src import providers
from src.generation.tutor_format import polish_tutor_answer

# Automatically load environment variables from .env.local or .env
load_dotenv(".env.local")
load_dotenv()


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
        "5. Format: Use ATX headings (##, ###). Do not use underlined setext headings. Do not use emoji.\n"
        "7. Mathematics: Copy formulas from the excerpts. Inline symbols use single dollars, like $f(x,y)$ and $\\partial f/\\partial x$. "
        "A full equation goes on its own line between double dollars:\n"
        "$$\n\\frac{\\partial f}{\\partial x}(x_0,y_0)=\\lim_{h\\to 0}\\frac{f(x_0+h,y_0)-f(x_0,y_0)}{h}\n$$\n"
        "Never surround a formula with parentheses or square brackets. Never write a bare backslash to break a line. "
        "Use \\\\, before dx or dy. Keep citations like [1] outside the dollars.\n"
        "6. If the excerpts do not support an answer, say so in one sentence and ask what notes they can upload. Do not guess."
    )

    WEB_SYSTEM_PROMPT = (
        "You are a study tutor answering from academic web excerpts, not from the student's syllabus.\n\n"
        "Rules:\n"
        "1. Use only the provided excerpts. Do not add facts or links that are not in them.\n"
        "2. Cite a web excerpt only as its document number, like [1]. Never write a URL that is not in the excerpts.\n"
        "3. Teach: one-sentence core idea, then ## and ### headings. Contrast exam-confusable ideas only when both sides are in the excerpts.\n"
        "4. End with ## Check-in and one follow-up question.\n"
        "5. ATX headings only. No emoji. No underlined headings.\n"
        "6. Mathematics: put every formula in $$...$$ using valid LaTeX and \\\\, before differentials."
    )

    GENERAL_SYSTEM_PROMPT = (
        "You are a study tutor. No course notes or academic pages were retrieved for this question, "
        "so teach from general knowledge and say what is commonly tested. Do not pretend this is cited from a textbook.\n\n"
        "Rules:\n"
        "1. Do not include URLs, filenames, or [Source: ...] citations.\n"
        "2. Open with a one-sentence core idea. Use ## and ### headings. Contrast ideas students mix up on exams.\n"
        "3. End with ## Check-in and exactly one question about their course, or an offer to quiz them.\n"
        "4. ATX headings only. No emoji. No underlined headings. No lifestyle or trivia unless it is a standard exam point.\n"
        "5. Mathematics: put every formula in $$...$$ using valid LaTeX and \\\\, before differentials."
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

        self.client = client or OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
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

    def generate(
        self,
        query: str,
        context: Union[str, List[Dict[str, Any]]],
        grounding: str = "course",
        allowed_sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate a complete, grounded answer to the user query.

        Args:
            query: The student's question.
            context: Retrieved context (formatted string or chunk dict list).

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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._prompt_for(grounding)},
                {"role": "user", "content": user_content},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        answer_text = polish_tutor_answer(
            providers.normalize_citation_brackets(
                providers.strip_reasoning(response.choices[0].message.content or "")
            ),
            allowed_sources=allowed_sources if allowed_sources is not None else sources,
            grounding=grounding,
        )

        return {
            "answer": answer_text,
            "model": self.model,
            "provider": self.provider,
            "provider_label": self.provider_label,
            "sources": sources,
        }

    def generate_stream(
        self,
        query: str,
        context: Union[str, List[Dict[str, Any]]],
        grounding: str = "course",
    ) -> Iterator[str]:
        """Stream generated response tokens one by one for interactive UI typewriter effect.

        Args:
            query: The student's question.
            context: Retrieved context (formatted string or chunk dict list).

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

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._prompt_for(grounding)},
                {"role": "user", "content": user_content},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        tokens = (
            chunk.choices[0].delta.content
            for chunk in stream
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content
        )
        for token in providers.filter_reasoning_stream(tokens):
            yield providers.normalize_citation_brackets(token)
