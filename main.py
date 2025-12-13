from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Optional
import whisperx
import pycapcut as cc
from pycapcut import trange



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
    confidence: Optional[float] = None


@dataclass(frozen=True)
class TranscriptSegment:
    id: int
    text: str
    start_s: float
    end_s: float
    words: Optional[list[TranscriptWord]] = None


@dataclass(frozen=True)
class Transcript:
    language: Optional[str]
    segments: list[TranscriptSegment]

    @property
    def duration_s(self) -> float:
        return max((s.end_s for s in self.segments), default=0.0)
    

@dataclass(frozen=True)
class QuizItem:
    question_text: str
    answer_text: str
    q_start: float
    q_end: float
    a_start: float
    a_end: float



@dataclass(frozen=True)
class TextSlot:
    name: str
    track_name: str
    segment_index: int


@dataclass(frozen=True)
class PlannedCue:
    text: str
    start_s: float
    duration_s: float
    slot_name: str

class FFmpegAudioExtractor:
    """Extract mono WAV audio from a video file using ffmpeg."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

    def extract_wav(self, video_path: str, output_wav_path: str) -> str:
        vp = Path(video_path)
        wp = Path(output_wav_path)
        wp.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",                 # overwrite output
            "-i", str(vp),        # input video
            "-vn",                # no video stream
            "-ac", "1",           # mono
            "-ar", str(self.sample_rate),
            "-c:a", "pcm_s16le",  # 16-bit PCM WAV
            str(wp),              # output path
        ]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{proc.stderr}")

        return str(wp)


class WhisperTranscriber:
    def __init__(self, model_name: str = "small", device: str = "cpu", compute_type: str = "int8"):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type

    def transcribe(self, wav_path: str) -> Transcript:
        audio = whisperx.load_audio(wav_path)

        model = whisperx.load_model(
            self.model_name,
            self.device,
            compute_type=self.compute_type,
            vad_method="silero",
        )
        result = model.transcribe(audio, language="en")
        language = "en"

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
                wl: list[TranscriptWord] = []
                for w in seg["words"]:
                    start = w.get("start")
                    end = w.get("end")
                    if start is None or end is None:
                        continue
                    wl.append(
                        TranscriptWord(
                            text=w.get("word", "").strip(),
                            start_s=float(start),
                            end_s=float(end),
                            confidence=w.get("score"),
                        )
                    )
                words = wl

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


class VideoToTranscriptPipeline:
    def __init__(self, extractor: FFmpegAudioExtractor, transcriber: WhisperTranscriber):
        self.extractor = extractor
        self.transcriber = transcriber

    def run(self, video_path: str, work_dir: str = "./work") -> Transcript:
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)

        wav_path = work / (Path(video_path).stem + ".wav")
        wav = self.extractor.extract_wav(video_path, str(wav_path))
        return self.transcriber.transcribe(wav)


class QuizParser:
    """Transcript -> list[QuizItem] assuming Question? then Answer."""
    def parse(self, transcript: Transcript) -> list[QuizItem]:
        items: list[QuizItem] = []
        segs = transcript.segments

        i = 0
        while i < len(segs) - 1:
            q = segs[i]
            a = segs[i + 1]

            if not q.text.strip().endswith("?"):
                i += 1
                continue

            items.append(
                QuizItem(
                    question_text=q.text.strip(),
                    answer_text=a.text.strip(),
                    q_start=q.start_s,
                    q_end=q.end_s,
                    a_start=a.start_s,
                    a_end=a.end_s,
                )
            )
            i += 2

        return items


class TemplateLayout:
    def __init__(self, slots: list[TextSlot]):
        self._slots = {s.name: s for s in slots}
    
    def slot(self, name: str) -> TextSlot:
        return self._slots[name]
    
    def slots(self) -> list[TextSlot]:
        return list(self._slots.values())
    

def make_list_reveal_layout_multi_track() -> TemplateLayout:
    """
    Use this if you have:
      - EN_LIST_ANCHOR track with 1 segment (index 0)
      - 5 separate tracks: FR_SLOTS_ANCHOR_1 .. FR_SLOTS_ANCHOR_5
        each with 1 segment (index 0)
    """
    return TemplateLayout(slots=[
        TextSlot("en_list", "EN_LIST_ANCHOR", 0),
        TextSlot("fr_0", "FR_SLOTS_ANCHOR_1", 0),
        TextSlot("fr_1", "FR_SLOTS_ANCHOR_2", 0),
        TextSlot("fr_2", "FR_SLOTS_ANCHOR_3", 0),
        TextSlot("fr_3", "FR_SLOTS_ANCHOR_4", 0),
        TextSlot("fr_4", "FR_SLOTS_ANCHOR_5", 0),
    ])

    


class ListRevealModule:
    """
    English list stays up; French answers reveal using fr_0..fr_4 slots.
    """
    name = "list_reveal"

    def __init__(self, max_items: int = 5, lead_in_s: float = 0.05, tail_s: float = 0.10):
        self.max_items = max_items
        self.lead_in_s = lead_in_s
        self.tail_s = tail_s

    def build(self, transcript: Transcript, layout: TemplateLayout) -> list[PlannedCue]:
        items = QuizParser().parse(transcript)[: self.max_items]
        if not items:
            return []

        cues: list[PlannedCue] = []

        # English list block
        en_text = "\n".join(it.question_text for it in items)
        start = max(0.0, items[0].q_start - self.lead_in_s)
        end = min(transcript.duration_s, items[-1].a_end + self.tail_s)

        cues.append(PlannedCue(
            text=en_text,
            start_s=start,
            duration_s=max(0.1, end - start),
            slot_name="en_list",
        ))

        # French reveals
        for i, it in enumerate(items):
            slot_name = f"fr_{min(i, self.max_items - 1)}"
            a_start = max(0.0, it.a_start - self.lead_in_s)
            a_end = min(transcript.duration_s, it.a_end + self.tail_s)

            cues.append(PlannedCue(
                text=it.answer_text,
                start_s=a_start,
                duration_s=max(0.2, a_end - a_start),
                slot_name=slot_name,
            ))

        return cues

class ModuleRegistry:
    def __init__(self, modules: list[QuizModule]):
        self._modules = {m.name: m for m in modules}

    def get(self, name: str) -> QuizModule:
        return self._modules[name]

    def names(self) -> list[str]:
        return list(self._modules.keys())



class CapCutProject:
    def __init__(self, drafts_folder: str):
        self.cc = cc
        self.trange = trange
        self.draft_folder = cc.DraftFolder(drafts_folder)
        self.script_file = None


    def open_from_template(self, template_name: str, new_name: str) -> None:
        self.script_file = self.draft_folder.duplicate_as_template(template_name, new_name)

    def replace_video_by_name(self, placeholder_filename: str, new_video_path: str) -> None:
        self.script_file.replace_material_by_name(
            placeholder_filename,
            self.cc.VideoMaterial(new_video_path)
        )

    def ensure_text_track(self, track_name: str) -> None:
        try:
            self.script_file.add_track(self.cc.TrackType.text, track_name)
        except Exception:
            pass

    def add_text_cue(self, cue: TextCue) -> None:
        seg = self.cc.TextSegment(
            cue.text,
            self.trange(f"{cue.start_s}s", f"{cue.duration_s}s"),
        )
        self.script_file.add_segment(seg, cue.track)


    def save(self) -> None:
        self.script_file.save()


class SlotRenderer:
    """
    Renders PlannedCues by copying styling from template anchor slots.
    - Adds generated segments into a new track (RENDER_TEXT)
    - Wipes template anchor text so only generated text remains visible
    """

    def __init__(self, drafts_folder: str, render_track: str = "RENDER_TEXT"):
        self.drafts_folder = drafts_folder
        self.render_track = render_track

    def render(
        self,
        template_name: str,
        new_draft_name: str,
        placeholder_video_filename: str,
        final_video_path: str,
        layout: TemplateLayout,
        cues: list[PlannedCue],
    ) -> None:
        project = CapCutProject(self.drafts_folder)
        project.open_from_template(template_name, new_draft_name)
        project.replace_video_by_name(placeholder_video_filename, final_video_path)
        project.ensure_text_track(self.render_track)

        # Create new segments using anchor styling
        for cue in cues:
            slot = layout.slot(cue.slot_name)

            anchor_track = project.script_file.get_imported_track(
                cc.TrackType.text,
                name=slot.track_name,
            )

            # If this errors in your environment, paste the error; track/segment access differs by version.
            anchor_seg = anchor_track.segments[slot.segment_index]

            seg = cc.TextSegment(
                cue.text,
                trange(f"{cue.start_s}s", f"{cue.duration_s}s"),
                font=anchor_seg.font,
                style=anchor_seg.style,
                clip_settings=anchor_seg.clip_settings,
            )
            project.script_file.add_segment(seg, self.render_track)

        # Wipe the anchor text in template tracks (neutralise originals)
        for slot in layout.slots():
            t = project.script_file.get_imported_track(cc.TrackType.text, name=slot.track_name)
            for idx in range(len(t.segments)):
                try:
                    project.script_file.replace_text(t, idx, "")
                except Exception:
                    pass

        project.save()



if __name__ == "__main__":
    VIDEO_PATH = "final_video.mp4"   # change this
    WORK_DIR = "./work"
    MODEL = "small"
    DEVICE = "cpu"  # use "cuda" if you have an NVIDIA GPU

    extractor = FFmpegAudioExtractor(sample_rate=16000)
    transcriber = WhisperTranscriber(model_name=MODEL, device=DEVICE, compute_type="int8")
    pipeline = VideoToTranscriptPipeline(extractor, transcriber)

    transcript = pipeline.run(VIDEO_PATH, work_dir=WORK_DIR)

    print("Language:", transcript.language)
    print("Duration:", transcript.duration_s)
    for seg in transcript.segments[:10]:
        print(f"[{seg.start_s:.2f} - {seg.end_s:.2f}] {seg.text}")
