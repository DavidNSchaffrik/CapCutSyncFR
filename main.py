
from __future__ import annotations
from dataclasses import dataclass



@dataclass(frozen=True)
class ScriptLine:
    id: str
    text: str
    kind: str  # "question" | "answer" | "cta" | "narration" | etc.

@dataclass(frozen=True)
class Script:
    lines: list[ScriptLine]

@dataclass(frozen=True)
class AlignedSpan:
    line_id: str
    start_s: float
    end_s: float

@dataclass(frozen=True)
class Alignment:
    spans: dict[str, AlignedSpan]  # keyed by line_id

@dataclass(frozen=True)
class TextCue:
    text: str
    start_s: float
    duration_s: float
    track: str  # "auto_text" / "subtitle" etc.
