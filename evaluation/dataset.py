"""Curated Academic Evaluation Dataset for Study Assistant.

Includes formal technical queries, colloquial student phrasing, ground-truth facts,
and target citations for quantitative RAG benchmarking.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BenchmarkItem:
    """A single academic question benchmark case with ground-truth facts and metadata."""

    id: str
    category: str  # 'colloquial_query', 'technical_definition', 'multi_hop_comparison'
    subject: str   # 'astronomy', 'physics', 'calculus'
    formal_query: str
    colloquial_query: str
    ground_truth_answer: str
    required_facts: List[str]
    target_source: str
    target_page: int
    distractor_keywords: List[str] = field(default_factory=list)


DEFAULT_BENCHMARK_DATASET: List[BenchmarkItem] = [
    BenchmarkItem(
        id="astro_kepler_1",
        category="technical_definition",
        subject="astronomy",
        formal_query="What does Kepler's first law state regarding the shape of planetary orbits and focal points?",
        colloquial_query="what shape does a planet travel around the sun and where is the sun?",
        ground_truth_answer="Kepler's first law states that planets orbit the Sun in elliptical orbits, with the Sun situated at one of the two focal points (foci). When eccentricity equals zero, the ellipse is a circle.",
        required_facts=["elliptical", "focal points", "focus", "eccentricity"],
        target_source="astronomy_kepler.txt",
        target_page=1,
        distractor_keywords=["orbital period", "harmonies", "equal areas"],
    ),
    BenchmarkItem(
        id="astro_kepler_2",
        category="multi_hop_comparison",
        subject="astronomy",
        formal_query="How does a planet's velocity vary between perihelion and aphelion under Kepler's second law?",
        colloquial_query="does a planet speed up when it gets closer to the sun or when its far away?",
        ground_truth_answer="Under Kepler's second law, a line joining a planet and the Sun sweeps out equal areas during equal intervals of time. Therefore, a planet moves faster when it is near the Sun (at perihelion) and slower when it is farthest (at aphelion).",
        required_facts=["equal areas", "perihelion", "aphelion", "faster", "slower"],
        target_source="astronomy_kepler.txt",
        target_page=1,
        distractor_keywords=["semi-major axis", "eccentricity"],
    ),
    BenchmarkItem(
        id="astro_kepler_3",
        category="colloquial_query",
        subject="astronomy",
        formal_query="What mathematical relationship connects orbital period and semi-major axis in Kepler's third law?",
        colloquial_query="why do far planets take longer to loop around the sun formula?",
        ground_truth_answer="Kepler's third law states that the square of the orbital period (P) is directly proportional to the cube of the semi-major axis (a) of its orbit: P^2 = a^3, where P is in Earth years and a in astronomical units (AU).",
        required_facts=["P^2 = a^3", "orbital period", "semi-major axis", "proportional"],
        target_source="astronomy_kepler.txt",
        target_page=1,
        distractor_keywords=["focal points", "equal areas"],
    ),
    BenchmarkItem(
        id="astro_kepler_4",
        category="technical_definition",
        subject="astronomy",
        formal_query="What geometric condition occurs when orbital eccentricity equals zero according to Keplerian mechanics?",
        colloquial_query="what if eccentricity is 0 what shape do you get?",
        ground_truth_answer="When orbital eccentricity (e) equals zero, the elliptical orbit becomes a perfect circle.",
        required_facts=["eccentricity equals zero", "circle", "ellipse"],
        target_source="astronomy_kepler.txt",
        target_page=1,
        distractor_keywords=["perihelion", "aphelion"],
    ),
    BenchmarkItem(
        id="astro_kepler_5",
        category="colloquial_query",
        subject="astronomy",
        formal_query="In which publication years and works did Johannes Kepler introduce his three planetary laws?",
        colloquial_query="when did kepler publish his laws and in what books?",
        ground_truth_answer="Kepler published his first two laws in 1609 in Astronomia Nova, and his third law in 1619 in Harmonices Mundi.",
        required_facts=["1609", "1619", "Astronomia Nova", "Harmonices Mundi"],
        target_source="astronomy_kepler.txt",
        target_page=1,
        distractor_keywords=["Newton", "Calculus"],
    ),
]


def load_default_dataset() -> List[BenchmarkItem]:
    """Return a copy of the default academic benchmark dataset."""
    return list(DEFAULT_BENCHMARK_DATASET)
