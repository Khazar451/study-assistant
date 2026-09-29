"""Academic Study Assistant Package.

Powered by NVIDIA NIM & Nebius AI Studio.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.assistant import StudyAssistant

def __getattr__(name: str):
    if name == "StudyAssistant":
        from src.assistant import StudyAssistant
        return StudyAssistant
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["StudyAssistant"]
