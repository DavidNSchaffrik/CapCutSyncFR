# CapCut Quiz Video Generator

This project automates the creation of **quiz-style language learning videos** in **CapCut** by:

1. Transcribing spoken audio from a video
2. Extracting question/answer pairs (e.g. *“How do we say summer in French?” → “L’été.”*)
3. Injecting correctly-timed, styled text into a **CapCut template draft**
4. Producing a ready-to-edit CapCut project with preserved styling and timing

The system is designed for **high-volume, repeatable content generation** using CapCut templates.

---

## What This Produces

* A **new CapCut draft project** per input video
* English prompt words (e.g. `summer`, `winter`, …)
* French answer reveals (e.g. `L’été.`)
* Text timing aligned to the spoken audio
* All text inherits **exact styling from the template anchors**

You open the generated draft directly in CapCut and export as usual.

---

## High-Level Pipeline

```
Input Video (.mp4)
        ↓
FFmpeg (audio extraction)
        ↓
WhisperX (speech → text + timestamps)
        ↓
Quiz Parser (Q/A extraction)
        ↓
Module (layout + timing logic)
        ↓
CapCut Draft Duplication
        ↓
JSON Patch (text + timing only)
        ↓
Ready-to-use CapCut Draft
```

---

## Core Concepts

### 1. Templates (CapCut)

You create **CapCut templates manually** that contain:

* Pre-styled text tracks (anchors)
* Correct positioning, fonts, colors, shadows, alignment, etc.

This tool **never recreates styling**.
It only **replaces the text content and timing** inside those template anchors.

> Styling issues should always be fixed in the CapCut template, not in code.

---

### 2. Anchors & Slots

Each text element in the template is treated as a **slot**.

Example for a 6-question template:

```
Text tracks (in order):
0 → EN_0
1 → EN_1
2 → EN_2
3 → EN_3
4 → EN_4
5 → EN_5
6 → FR_0
7 → FR_1
8 → FR_2
9 → FR_3
10 → FR_4
11 → FR_5
```

The code maps transcript items to these slots using `TemplateLayout`.

---

### 3. Modules

A **module** defines:

* How transcript segments are interpreted
* How many items are used
* When text appears and disappears
* Which slot each piece of text goes into

Currently implemented:

* `list_reveal`

  * English words stay visible
  * French answers reveal one-by-one

Modules are swappable and configurable.

---

### 4. Quiz Item Extraction

The system expects a spoken pattern like:

```
How do we say summer in French ?
L'été.
```

From this it extracts:

* English prompt word → `summer`
* French answer → `L’été.`
* Precise timing from audio alignment

---

## Quick Start (Single Video)

### Requirements

* Python 3.10+
* CapCut installed (desktop)
* FFmpeg available on PATH
* WhisperX compatible environment

### Install

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

### Run on a Single Video

```bash
py main.py --n 6
```

This will:

* Transcribe `final_video.mp4`
* Extract up to 6 quiz items
* Duplicate the specified CapCut template
* Insert timed EN/FR text
* Produce a new CapCut draft project

---

## CLI Parameters

| Argument | Description                                 |
| -------- | ------------------------------------------- |
| `--n`    | Number of quiz items to render (default: 6) |

The following are currently configured **inside the script**:

* Template name
* Drafts folder
* Placeholder video name
* Output draft naming

(These will be externalised in batch mode.)

---

## Output

* A new CapCut draft appears under:

```
CapCut/User Data/Projects/com.lveditor.draft/
```

* Open CapCut → Projects → Your generated draft
* Review → Export

No rendering is done by this tool; CapCut handles final export.

---

## Known Constraints & Assumptions

* Audio must be **English questions followed immediately by answers**
* One question → one answer
* Text anchors must already exist in the template
* Track order must match the expected layout
* This tool **does not reposition or restyle text**

If alignment looks wrong:
→ Fix it in the CapCut template, not in Python.

---

## Intended Extensions (Next Steps)

* Batch processing (multiple videos in one run)
* Per-video template selection
* Multiple module types
* Config-driven runs (YAML/JSON)
* Manifest output for large batches

---

## Design Philosophy

* **CapCut is the design tool**
* **Python is the automation layer**
* Never fight CapCut’s styling system
* Always preserve template intent
