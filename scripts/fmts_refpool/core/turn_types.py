"""Dialogue turn representation shared by the FMTS selectors."""
from dataclasses import dataclass

@dataclass
class Turn:
    speaker: str
    text: str
