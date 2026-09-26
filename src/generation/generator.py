import os
import re
from typing import Any, Dict, Iterator, List, Optional, Union
from dotenv import load_dotenv
from openai import OpenAI

# Automatically load environment variables from .env.local or .env
load_dotenv(".env.local")
load_dotenv()


class StudyGenerator:
    """Answers academic and study questions strictly grounded in retrieved document context.

    Supports:
    1. Primary provider: NVIDIA NIM (https://integrate.api.nvidia.com/v1)
    2. Fallback provider: Nebius Token Factory (https://api.studio.nebius.ai/v1)
    3. Grounded citation formatting with anti-hallucination guardrails.
    4. Non-streaming and streaming generation.
    """

    NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
    NVIDIA_DEFAULT_MODEL = "meta/llama-3.2-11b-vision-instruct"

    NEBIUS_BASE_URL = "https://api.studio.nebius.ai/v1"
    NEBIUS_DEFAULT_MODEL = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert AI Academic Study Assistant and Tutor. "
        "Your objective is to provide clear, thorough, and strictly grounded answers to student questions "
        "using exclusively the provided study material excerpts.\n\n"
        "Rules for Answering:\n"
        "1. Grounding: Rely ONLY on the facts directly stated in the context. Do NOT fabricate, assume, or extrapolate beyond what is given.\n"
        "2. Inline Citations: Whenever you state a key fact, definition, or theorem, include an inline citation with the source and page number in the format: [Source: <filename>, Page: <page>].\n"
        "3. Structure & Pedagogy: Format your answer for high readability using structured markdown: bold key concepts, use bullet points for lists, and explain complex concepts step-by-step.\n"
        "4. Insufficient Information: If the context does not contain enough information to answer the question, clearly state: "
        "'Based on the provided study materials, there is not enough information to answer this question.' "
        "Do NOT attempt to guess or answer from outside knowledge."
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
        2. Primary: NVIDIA_API_KEY environment variable
        3. Fallback: NEBIUS_API_KEY environment variable
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        if api_key:
            self.api_key = api_key
            if base_url:
                self.base_url = base_url
                self.provider = "nebius" if "nebius" in base_url else "nvidia"
            else:
                self.base_url = self.NVIDIA_BASE_URL
                self.provider = "nvidia"
            self.model = model or (
                self.NEBIUS_DEFAULT_MODEL if self.provider == "nebius" else self.NVIDIA_DEFAULT_MODEL
            )
        elif os.getenv("NVIDIA_API_KEY"):
            # 1. Primary: NVIDIA NIM
            self.api_key = os.getenv("NVIDIA_API_KEY")
            self.base_url = base_url or os.getenv("NVIDIA_BASE_URL", self.NVIDIA_BASE_URL)
            self.model = model or os.getenv("LLM_MODEL", self.NVIDIA_DEFAULT_MODEL)
            self.provider = "nvidia"
        elif os.getenv("NEBIUS_API_KEY"):
            # 2. Fallback: Nebius Token Factory
            self.api_key = os.getenv("NEBIUS_API_KEY")
            self.base_url = base_url or os.getenv("NEBIUS_BASE_URL", self.NEBIUS_BASE_URL)
            self.model = model or os.getenv("LLM_MODEL", self.NEBIUS_DEFAULT_MODEL)
            self.provider = "nebius"
        else:
            raise ValueError(
                "No API key found. Please set NVIDIA_API_KEY (primary) or "
                "NEBIUS_API_KEY (fallback) as an environment variable or pass api_key."
            )

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
            text = chunk.get("text", chunk.get("page_content", "")).strip()

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

    def generate(
        self,
        query: str,
        context: Union[str, List[Dict[str, Any]]],
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
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        answer_text = response.choices[0].message.content.strip()

        return {
            "answer": answer_text,
            "model": self.model,
            "provider": self.provider,
            "sources": sources,
        }

    def generate_stream(
        self,
        query: str,
        context: Union[str, List[Dict[str, Any]]],
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
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
