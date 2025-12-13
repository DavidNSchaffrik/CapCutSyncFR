
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
    

    class FFmpegAudioExtractor:
        """
        Extract audio from Adobe Express video file. 
        """

        def __init__(self, sample_rate: int = 16000):
            self.sample_rate = sample_rate


        def extract_wav(self, videopath: str, output_wav_path: str) -> str:
            vp = Path(videopath)
            wp = Path(output_wav_path)
            wp.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                "ffmpeg",
                "-y",
                "-i", str(vp),
                "-vn",
                "-ac", "1",
                "-ar", str(self.sample_rate),
                "-c:a", "pcm_s16le"
                str(wp),
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed:\n{proc.stderr}")
            
            return str(wp)
        

        
