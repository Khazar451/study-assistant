import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Automatically load environment variables from .env.local or .env if present
load_dotenv(".env.local")
load_dotenv()


class QueryAugmenter:
    """Enhances and expands user queries for improved vector retrieval in RAG.

    Supports three primary augmentation strategies:
    1. Query Rewriting: Clarifies vague phrasing and optimizes academic terminology.
    2. Multi-Query Expansion: Generates diverse paraphrases and sub-queries to maximize retrieval recall.
    3. HyDE (Hypothetical Document Embeddings): Synthesizes a hypothetical answer passage for passage-to-passage search.
    """

    DEFAULT_MODEL = "meta/llama-3.2-11b-vision-instruct"
    DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ):
        """Initialize the QueryAugmenter client.

        Args:
            api_key: NVIDIA NIM or OpenAI-compatible API key.
            base_url: Base endpoint URL for the LLM API.
            model: Model identifier (e.g. meta/llama-3.1-8b-instruct).
            temperature: Sampling temperature for generation (lower = more deterministic).
        """
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "NVIDIA_API_KEY is not set. Please set the environment variable "
                "or pass api_key to QueryAugmenter."
            )

        self.base_url = base_url or os.getenv("NVIDIA_BASE_URL", self.DEFAULT_BASE_URL)
        self.model = model or os.getenv("LLM_MODEL", self.DEFAULT_MODEL)
        self.temperature = temperature

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def rewrite_query(self, query: str) -> str:
        """Rewrite an informal or ambiguous query into a search-optimized academic query.

        Args:
            query: The original user question or prompt.

        Returns:
            Rewritten, keyword-enriched search query string.
        """
        if not query or not query.strip():
            return ""

        clean_query = query.strip()
        system_prompt = (
            "You are an expert AI assistant specialized in information retrieval for academic study materials. "
            "Your task is to rewrite the user's input query into a clear, precise, and search-optimized query. "
            "Preserve the core intent, eliminate colloquialisms, resolve ambiguous terms, and include relevant "
            "academic or technical keywords. Output ONLY the rewritten query text, with no preamble, quotes, or explanation."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User query: {clean_query}"},
                ],
                temperature=self.temperature,
                max_tokens=150,
            )
            rewritten = response.choices[0].message.content.strip().strip('"\'')
            return rewritten if rewritten else clean_query
        except Exception:
            # Fall back to original query if LLM service encounters issues
            return clean_query

    def expand_query(self, query: str, num_queries: int = 3) -> List[str]:
        """Generate diverse variations and sub-queries to maximize retrieval recall.

        Args:
            query: The user query string.
            num_queries: Number of alternative query variations to generate.

        Returns:
            List of unique queries, starting with the original query followed by variations.
        """
        if not query or not query.strip():
            return []

        clean_query = query.strip()
        system_prompt = (
            f"You are an expert AI assistant specialized in academic search and information retrieval. "
            f"Your task is to generate {num_queries} diverse search query variations based on the user's question. "
            f"These variations should explore synonyms, related technical terms, sub-topics, and alternative phrasing "
            f"to maximize document retrieval recall across academic textbooks and lecture notes.\n"
            f"Rules:\n"
            f"1. Generate exactly {num_queries} alternative queries.\n"
            f"2. Output each query on a new line.\n"
            f"3. Do NOT include numbers, bullet points, introductory text, or quotes."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User query: {clean_query}"},
                ],
                temperature=max(0.5, self.temperature),
                max_tokens=250,
            )
            raw_text = response.choices[0].message.content.strip()

            # Clean each line: strip numbering, bullet markers, whitespace
            variations = []
            for line in raw_text.splitlines():
                cleaned = line.strip().lstrip("0123456789.-)•* ").strip('"\'')
                if cleaned:
                    variations.append(cleaned)

            # Combine original query with generated variations, deduplicating while preserving order
            all_queries: List[str] = []
            seen_lower = set()

            for q in [clean_query] + variations:
                normalized = q.lower().strip()
                if normalized and normalized not in seen_lower:
                    seen_lower.add(normalized)
                    all_queries.append(q)

            return all_queries
        except Exception:
            # Fall back to original query if LLM service encounters issues
            return [clean_query]

    def generate_hyde_query(self, query: str) -> str:
        """Generate a hypothetical textbook passage that directly answers the query (HyDE).

        Hypothetical Document Embeddings (HyDE) creates a simulated document chunk,
        which allows searching the vector space with passage-to-passage similarity
        rather than query-to-passage similarity.

        Args:
            query: The user's study question.

        Returns:
            A hypothetical passage string answering the question.
        """
        if not query or not query.strip():
            return ""

        clean_query = query.strip()
        system_prompt = (
            "You are a university professor and academic textbook author. "
            "Given a student's question, write a concise, authoritative hypothetical textbook passage "
            "or lecture note excerpt that answers the question directly and factually. "
            "Write in a formal academic tone with relevant domain terminology. "
            "Do NOT include introductory phrases like 'Here is...' or conversational filler. "
            "Output ONLY the textbook passage."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {clean_query}"},
                ],
                temperature=self.temperature,
                max_tokens=300,
            )
            passage = response.choices[0].message.content.strip()
            return passage if passage else clean_query
        except Exception:
            return clean_query

    def augment(
        self,
        query: str,
        mode: str = "expand",
        num_queries: int = 3,
    ) -> List[str]:
        """Unified entry point for query augmentation.

        Args:
            query: Original user query.
            mode: Augmentation strategy ('expand', 'rewrite', 'hyde', 'passthrough').
            num_queries: Number of variations for 'expand' mode.

        Returns:
            List of augmented query strings ready for vector embedding and retrieval.
        """
        if not query or not query.strip():
            return []

        clean_query = query.strip()

        if mode == "expand":
            return self.expand_query(clean_query, num_queries=num_queries)
        elif mode == "rewrite":
            rewritten = self.rewrite_query(clean_query)
            return [rewritten] if rewritten else [clean_query]
        elif mode == "hyde":
            hyde_passage = self.generate_hyde_query(clean_query)
            return [hyde_passage] if hyde_passage else [clean_query]
        elif mode == "passthrough":
            return [clean_query]
        else:
            raise ValueError(
                f"Unsupported augmentation mode: '{mode}'. "
                "Supported modes are: 'expand', 'rewrite', 'hyde', 'passthrough'."
            )
