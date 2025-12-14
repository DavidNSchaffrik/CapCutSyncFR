@echo off
setlocal ENABLEDELAYEDEXPANSION

REM =========================
REM Activate venv
REM =========================
call "%~dp0venv\Scripts\activate.bat"
if errorlevel 1 (
  echo Failed to activate venv at: %~dp0venv\Scripts\activate.bat
  pause
  exit /b 1
)

REM =========================
REM DEFAULTS (edit once here)
REM =========================
set "DEFAULT_MANIFEST=%CD%\jobs.xlsx"
set "DEFAULT_BATCHNAME=batchvideo1"
set "DEFAULT_STARTINDEX=0"
set "DEFAULT_PAD=0"
set "DEFAULT_DRAFTS_FOLDER=C:\Users\david\AppData\Local\CapCut\User Data\Projects\com.lveditor.draft"
set "DEFAULT_PLACEHOLDER=ElevenLabs_2025-10-25T10_46_53_Guillaume-Narration_pvc_sp100_s52_sb47_t2-5.mp4"

set "DEFAULT_WORKDIR=./work"
set "DEFAULT_MODEL=small"
set "DEFAULT_DEVICE=cpu"
set "DEFAULT_COMPUTE=int8"

REM =========================
REM UI
REM =========================
echo ==========================================
echo CapCut Quiz Batch Runner
echo ==========================================
echo.

set /p MANIFEST=Path to jobs.xlsx [!DEFAULT_MANIFEST!]: 
if "!MANIFEST!"=="" set "MANIFEST=!DEFAULT_MANIFEST!"

set /p BATCHNAME=Batch name (e.g. batchvideoqqq) [!DEFAULT_BATCHNAME!]: 
if "!BATCHNAME!"=="" set "BATCHNAME=!DEFAULT_BATCHNAME!"

set /p STARTINDEX=Start index [!DEFAULT_STARTINDEX!]: 
if "!STARTINDEX!"=="" set "STARTINDEX=!DEFAULT_STARTINDEX!"

set /p PAD=Index padding (0=no pad, 3=001) [!DEFAULT_PAD!]: 
if "!PAD!"=="" set "PAD=!DEFAULT_PAD!"

set /p DRAFTS=CapCut drafts folder path [!DEFAULT_DRAFTS_FOLDER!]: 
if "!DRAFTS!"=="" set "DRAFTS=!DEFAULT_DRAFTS_FOLDER!"

set /p PLACEHOLDER=Placeholder video filename (inside template) [!DEFAULT_PLACEHOLDER!]: 
if "!PLACEHOLDER!"=="" set "PLACEHOLDER=!DEFAULT_PLACEHOLDER!"

echo.
echo Optional settings (press ENTER to accept defaults)
set /p WORKDIR=Work dir [!DEFAULT_WORKDIR!]: 
if "!WORKDIR!"=="" set "WORKDIR=!DEFAULT_WORKDIR!"

set /p MODEL=Whisper model [!DEFAULT_MODEL!]: 
if "!MODEL!"=="" set "MODEL=!DEFAULT_MODEL!"

set /p DEVICE=Device cpu/cuda [!DEFAULT_DEVICE!]: 
if "!DEVICE!"=="" set "DEVICE=!DEFAULT_DEVICE!"

set /p COMPUTE=int8/float16 [!DEFAULT_COMPUTE!]: 
if "!COMPUTE!"=="" set "COMPUTE=!DEFAULT_COMPUTE!"

echo.
echo ===== CONFIRM =====
echo Manifest: !MANIFEST!
echo Batch name: !BATCHNAME!
echo Start index: !STARTINDEX!
echo Pad: !PAD!
echo Drafts folder: !DRAFTS!
echo Placeholder: !PLACEHOLDER!
echo Work dir: !WORKDIR!
echo Model: !MODEL!
echo Device: !DEVICE!
echo Compute: !COMPUTE!
echo.
pause

python main.py batch ^
  --manifest "!MANIFEST!" ^
  --batch-name "!BATCHNAME!" ^
  --start-index !STARTINDEX! ^
  --pad !PAD! ^
  --drafts-folder "!DRAFTS!" ^
  --placeholder "!PLACEHOLDER!" ^
  --work-dir "!WORKDIR!" ^
  --model "!MODEL!" ^
  --device "!DEVICE!" ^
  --compute-type "!COMPUTE!" ^
  --continue-on-error

echo.
echo Batch run complete.
pause
