from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Optional
import whisperx
import pycapcut as cc
from pycapcut import trange
import json
import copy
from typing import Any



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
    track_type: cc.TrackType
    track_index: int
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

        TextSlot("en_list", cc.TrackType.text, track_index=0, segment_index=0),
        TextSlot("fr_0",    cc.TrackType.text, track_index=1, segment_index=0),
        TextSlot("fr_1",    cc.TrackType.text, track_index=2, segment_index=0),
        TextSlot("fr_2",    cc.TrackType.text, track_index=3, segment_index=0),
        TextSlot("fr_3",    cc.TrackType.text, track_index=4, segment_index=0),
        TextSlot("fr_4",    cc.TrackType.text, track_index=5, segment_index=0),
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
        debug_list_imported_tracks(project.script_file)
        project.replace_video_by_name(placeholder_video_filename, final_video_path)
        project.ensure_text_track(self.render_track)

        # Create new segments using anchor styling
        for cue in cues:
            slot = layout.slot(cue.slot_name)

            anchor_track = project.script_file.get_imported_track(
                slot.track_type,
                index=slot.track_index,
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
        seen: set[tuple[cc.TrackType, int]] = set()

        for slot in layout.slots():
            key = (slot.track_type, slot.track_index)
            if key in seen:
                continue
            seen.add(key)

            t = project.script_file.get_imported_track(slot.track_type, index=slot.track_index)

            for idx in range(len(t.segments)):
                try:
                    project.script_file.replace_text(t, idx, "")
                except Exception:
                    pass
       


        project.save()

def unique_draft_name(base: str) -> str:
    # tries base, then base_002, base_003, ...
    for i in range(1, 1000):
        name = base if i == 1 else f"{base}_{i:03d}"
        try:
            cc.DraftFolder(DRAFTS_FOLDER).load_template(name)
        except Exception:
            return name
    raise RuntimeError("Could not find a free draft name.")

def debug_list_imported_tracks(script_file) -> None:
    for ttype in (cc.TrackType.text, cc.TrackType.video, cc.TrackType.audio):
        print(f"\n--- Imported tracks for {ttype} ---")
        for i in range(20):
            try:
                t = script_file.get_imported_track(ttype, index=i)
            except Exception:
                break
            # Best-effort introspection (pycapcut versions differ)
            name = getattr(t, "name", None)
            segs = getattr(t, "segments", None)
            seg_count = len(segs) if isinstance(segs, list) else "?"
            print(f"index={i} name={name!r} segments={seg_count}")




def s_to_us(s: float) -> int:
    return int(round(s * 1_000_000))


class DraftJsonPatcher:
    """
    Fallback path: edit the duplicated draft's draft_content.json in-place.
    This preserves styling because we're modifying the existing template segments.
    """

    def __init__(self, draft_dir: Path):
        self.draft_dir = draft_dir
        self.path = self._find_draft_content_json(draft_dir)

    def _find_draft_content_json(self, d: Path) -> Path:
        # common file name in CapCut drafts
        p = d / "draft_content.json"
        if p.exists():
            return p
        # fallback: search one level deep
        for cand in d.rglob("draft_content.json"):
            return cand
        raise FileNotFoundError(f"Could not find draft_content.json under: {d}")

    def load(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, data: dict[str, Any]) -> None:
        # backup once
        backup = self.path.with_suffix(".json.bak")
        if not backup.exists():
            backup.write_text(self.path.read_text(encoding="utf-8"), encoding="utf-8")
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def retime_text_segment(
        self,
        track_index: int,
        segment_index: int,
        start_s: float,
        end_s: float,
    ) -> None:
        data = self.load()

        # The schema varies slightly by CapCut version; this is the most common shape:
        # data["tracks"] -> list of track dicts, each has ["type"] and ["segments"].
        tracks = data.get("tracks")
        if not isinstance(tracks, list):
            raise KeyError("draft_content.json: missing 'tracks' list")

        # Filter text tracks in the order they appear
        text_tracks = [t for t in tracks if t.get("type") in ("text", "Text", "TRACK_TYPE_TEXT")]
        if track_index >= len(text_tracks):
            raise IndexError(f"Text track index {track_index} out of range (found {len(text_tracks)} text tracks)")

        track = text_tracks[track_index]
        segs = track.get("segments")
        if not isinstance(segs, list) or segment_index >= len(segs):
            raise IndexError("Segment index out of range for that track")

        seg = segs[segment_index]

        start_us = s_to_us(start_s)
        dur_us = max(1, s_to_us(end_s - start_s))

        # Common field name:
        # seg["target_timerange"] = {"start": ..., "duration": ...}
        tr = seg.get("target_timerange")
        if isinstance(tr, dict):
            tr["start"] = start_us
            tr["duration"] = dur_us
        else:
            # Sometimes it's directly fields
            seg["target_timerange"] = {"start": start_us, "duration": dur_us}

        self.save(data)


class TemplateTextSlotWriter:
    """
    Writes into EXISTING styled template segments:
      - replace_text(): preserves styling
      - retime(): try via object, fallback to JSON patch
    """

    def __init__(self, project: "CapCutProject"):
        self.project = project

    def apply(self, slot: TextSlot, text: str, start_s: float, end_s: float) -> None:
        sf = self.project.script_file
        if sf is None:
            raise RuntimeError("CapCutProject has no open script_file")

        # 1) get the imported text track by index
        t = sf.get_imported_track(cc.TrackType.text, index=slot.track_index)

        # 2) replace the segment text (keeps styling)
        sf.replace_text(t, slot.segment_index, text)

        # 3) attempt to retime directly (often fails / no-op for imported segments)
        seg_obj = t.segments[slot.segment_index]
        if self._try_retime_imported_segment_object(seg_obj, start_s, end_s):
            return

        # 4) fallback: patch draft_content.json and resave
        draft_dir = self.project.get_draft_dir()
        DraftJsonPatcher(draft_dir).retime_text_segment(
            track_index=slot.track_index,
            segment_index=slot.segment_index,
            start_s=start_s,
            end_s=end_s,
        )

    def _try_retime_imported_segment_object(self, seg_obj: object, start_s: float, end_s: float) -> bool:
        """
        Best-effort. Returns True if it looks like it worked.
        Many pycapcut imported segment objects won't allow this.
        """
        start_us = s_to_us(start_s)
        dur_us = max(1, s_to_us(end_s - start_s))

        # Possible attribute names across versions:
        for attr in ("target_timerange", "target_range", "timerange"):
            if hasattr(seg_obj, attr):
                tr = getattr(seg_obj, attr)
                try:
                    # If it's a Timerange-like object
                    if hasattr(tr, "start") and hasattr(tr, "duration"):
                        tr.start = start_us
                        tr.duration = dur_us
                        return True
                    # If it's a dict
                    if isinstance(tr, dict):
                        tr["start"] = start_us
                        tr["duration"] = dur_us
                        return True
                except Exception:
                    pass

        # If no known timerange attribute exists or is writable
        return False

if __name__ == "__main__":
    # ---- INPUT VIDEO (the final mp4 you want to use) ----
    VIDEO_PATH = r"final_video.mp4"
    WORK_DIR = "./work"

    # ---- WHISPERX ----
    MODEL = "small"
    DEVICE = "cpu"           # "cuda" if you have NVIDIA GPU
    COMPUTE_TYPE = "int8"    # good for CPU

    # ---- CAPCUT TEMPLATE SETTINGS ----
    DRAFTS_FOLDER = r"C:\Users\david\AppData\Local\CapCut\User Data\Projects\com.lveditor.draft"
    TEMPLATE_NAME = "5_word_template"
    NEW_DRAFT_NAME = unique_draft_name("ListReveal_Output")


    # This is the filename of the placeholder video material INSIDE the template
    PLACEHOLDER_VIDEO_FILENAME = "ElevenLabs_2025-10-25T10_46_53_Guillaume-Narration_pvc_sp100_s52_sb47_t2-5.mp4"

    # ---- 1) Transcribe ----
    extractor = FFmpegAudioExtractor(sample_rate=16000)
    transcriber = WhisperTranscriber(model_name=MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    pipeline = VideoToTranscriptPipeline(extractor, transcriber)
    transcript = pipeline.run(VIDEO_PATH, work_dir=WORK_DIR)

    print("Language:", transcript.language)
    print("Duration:", transcript.duration_s)
    for seg in transcript.segments[:12]:
        print(f"[{seg.start_s:.2f} - {seg.end_s:.2f}] {seg.text}")

    # ---- 2) Choose module ----
    registry = ModuleRegistry([
        ListRevealModule(max_items=5),
    ])
    module = registry.get("list_reveal")

    # ---- 3) Choose layout (pick ONE) ----
    # Use ONE of these depending on how your anchors are structured in the template:
    # layout = make_list_reveal_layout_single_track()
    layout = make_list_reveal_layout_multi_track()

    # ---- 4) Build cues ----
    cues = module.build(transcript, layout)

    # ---- 5) Render into CapCut draft ----
    renderer = SlotRenderer(drafts_folder=DRAFTS_FOLDER, render_track="RENDER_TEXT")
    renderer.render(
        template_name=TEMPLATE_NAME,
        new_draft_name=NEW_DRAFT_NAME,
        placeholder_video_filename=PLACEHOLDER_VIDEO_FILENAME,
        final_video_path=VIDEO_PATH,
        layout=layout,
        cues=cues,
    )

    print("Done. Draft created:", NEW_DRAFT_NAME)
