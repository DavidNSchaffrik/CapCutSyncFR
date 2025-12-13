
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Optional

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



@dataclass(frozen=True)
class TimelinePlan:
    cues: list[TextCue]
    total_duration_s: float


@dataclass(frozen=True)
class TranscriptWord:
    text: str
    start_s: float
    end_s: float



@dataclass(frozen=True)
class TranscriptSegement:
    id: int
    text: str
    start_s = float
    end_s = float
    words: Optional[list[TranscriptWord]] = None


@dataclass(frozen=True)
class Transcript:
    language: Optional[str]
    segements: list[TranscriptSegement]

    @property
    def duration_s(self) -> float:
        return max((s.end_s for s in self.segements), default=0.0)
    

    