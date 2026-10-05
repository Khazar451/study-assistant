"""Tavily Search API client for web retrieval and hybrid academic grounding."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None  # type: ignore


class TavilySearchClient:
    """Client for retrieving and formatting web search results via Tavily API.

    Supports:
    1. Real-time web search with clean content extraction.
    2. Conversion of search results to standard RAG chunk dictionaries.
    3. Seamless offline mock mode for continuous integration testing without API calls.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        mock_mode: bool = False,
    ):
        """Initialize the TavilySearchClient.

        Args:
            api_key: Optional Tavily API key. Defaults to TAVILY_API_KEY environment variable.
            mock_mode: If True, returns deterministic mock results for offline testing.
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.mock_mode = mock_mode
        self._client: Optional[Any] = None

        if not self.mock_mode and self.api_key and TavilyClient is not None:
            try:
                self._client = TavilyClient(api_key=self.api_key)
            except Exception:
                self._client = None

    @property
    def is_available(self) -> bool:
        """Check if the client has valid credentials or is in mock mode."""
        return self.mock_mode or (bool(self.api_key) and self._client is not None)

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute a web search and return results formatted as standardized chunks.

        Args:
            query: The academic or student search query.
            max_results: Maximum number of search results to return (default: 5).
            search_depth: Search depth ('basic' or 'advanced').
            include_answer: Whether to ask Tavily for an AI-synthesized answer.

        Returns:
            List of chunk dictionaries compatible with StudyRetriever and StudyReranker.
        """
        if not query or not query.strip():
            return []

        if self.mock_mode:
            return self._mock_search_results(query.strip(), max_results=max_results)

        if not self.is_available:
            return []

        try:
            response = self._client.search(
                query=query.strip(),
                search_depth=search_depth,
                max_results=max_results,
                include_answer=include_answer,
            )
            raw_results = response.get("results", [])
        except Exception:
            return []

        chunks: List[Dict[str, Any]] = []
        for idx, res in enumerate(raw_results):
            url = res.get("url", "")
            title = res.get("title", "Web Source")
            content = res.get("content", "").strip()
            score = float(res.get("score", 0.7))

            domain = urlparse(url).netloc or "web"

            chunk = {
                "id": f"web_{idx}_{abs(hash(url)) % 100000}",
                "text": f"{title}\n{content}",
                "metadata": {
                    "source": url,
                    "title": title,
                    "domain": domain,
                    "page": 1,
                    "source_type": "web",
                    "score": score,
                },
                "similarity_score": score,
            }
            chunks.append(chunk)

        return chunks

    def _mock_search_results(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Return deterministic mock chunks for offline unit testing."""
        mock_items = [
            {
                "url": "https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion",
                "title": "Kepler's laws of planetary motion",
                "content": (
                    "In astronomy, Kepler's laws of planetary motion, published by Johannes Kepler "
                    "between 1609 and 1619, describe the orbits of planets around the Sun. "
                    "The first law states that the orbit of a planet is an ellipse with the Sun at one of the two foci."
                ),
                "score": 0.95,
            },
            {
                "url": "https://nasa.gov/kepler-mission-overview",
                "title": "NASA Kepler Mission Overview",
                "content": (
                    "The Kepler second law states that a line segment joining a planet and the Sun sweeps out "
                    "equal areas during equal intervals of time. The planet moves faster when closer to perihelion."
                ),
                "score": 0.88,
            },
            {
                "url": "https://britannica.com/science/Keplers-laws-of-planetary-motion",
                "title": "Kepler's third law: Harmonic Law",
                "content": (
                    "Kepler's third law establishes that the square of the orbital period of a planet is directly "
                    "proportional to the cube of the semi-major axis of its orbit (P^2 = a^3)."
                ),
                "score": 0.82,
            },
        ]

        chunks: List[Dict[str, Any]] = []
        for idx, item in enumerate(mock_items[:max_results]):
            url = item["url"]
            domain = urlparse(url).netloc
            chunks.append({
                "id": f"mock_web_{idx}",
                "text": f"{item['title']}\n{item['content']}",
                "metadata": {
                    "source": url,
                    "title": item["title"],
                    "domain": domain,
                    "page": 1,
                    "source_type": "web",
                    "score": item["score"],
                },
                "similarity_score": item["score"],
            })

        return chunks
