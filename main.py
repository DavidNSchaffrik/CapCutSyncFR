
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Optional
import whisperx

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
        

class WhisperTranscriber:
    """
    Transcribes the WAV file and returns timestamped segments (+ word timestamps when available).
    Requires: whisperx + torch.
    """
    def __init__(self, model_name: str = "small", device: str = "cpu"):
        self.model_name = model_name
        self.device = device

    def transcribe(self, wav_path: str) -> Transcript:
        audio = whisperx.load_audio(wav_path)

        model = whisperx.load_model(self.model_name, self.device)
        result = model.transcribe(audio)
        language = result.get("language") or "en"

        align_model, metadata = whisperx.load_align_model(language_code=language, device=self.device)
        aligned = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        segments: list[TranscriptSegment] = []
        for seg in aligned["segments"]:
            words: Optional[list[TranscriptWord]] = None

            if seg.get("words"):
                word_list: list[TranscriptWord] = []
                for w in seg["words"]:
                    start = w.get("start")
                    end = w.get("end")
                    if start is None or end is None:
                        continue

                    word_list.append(
                        TranscriptWord(
                            text=w.get("word", "").strip(),
                            start_s=float(start),
                            end_s=float(end),
                            confidence=w.get("score"),
                        )
                    )
                words = word_list

            segments.append(
                TranscriptSegment(
                    id=int(seg.get("id", len(segments))),
                    text=seg.get("text", "").strip(),
                    start_s=float(seg["start"]),
                    end_s=float(seg["end"]),
                    words=words,
                )
            )

        return Transcript(language=aligned.get("language") or language, segments=segments)



            


            

