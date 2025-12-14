@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ======================================================
REM Resolve paths relative to THIS bat file
REM tools\add_jobs_from_folder.bat -> repo root is ..\
REM ======================================================
set "TOOLSDIR=%~dp0"
set "REPODIR=%TOOLSDIR%.."

pushd "%REPODIR%" >nul

REM ---- Activate venv if present ----
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else (
    echo WARNING: venv not found, using system Python
)

REM =========================
REM Defaults
REM =========================
set "DEFAULT_MANIFEST=%REPODIR%\jobs.xlsx"
set "DEFAULT_FOLDER=%REPODIR%\input_videos"
set "DEFAULT_RECURSIVE=0"
set "DEFAULT_ENABLED=true"
set "DEFAULT_N=6"
set "DEFAULT_MODULE=list_reveal"
set "DEFAULT_TEMPLATE=6_word_template"
set "DEFAULT_OUTPUT_PREFIX="
set "DEFAULT_PLACEHOLDER="
set "DEFAULT_SKIP_EXISTING=1"

echo ==========================================
echo CapCut Quiz - Add Jobs From Folder
echo Repo: %REPODIR%
echo ==========================================
echo.

set /p MANIFEST=Manifest path [!DEFAULT_MANIFEST!]:
if "!MANIFEST!"=="" set "MANIFEST=!DEFAULT_MANIFEST!"

set /p FOLDER=Folder with videos [!DEFAULT_FOLDER!]:
if "!FOLDER!"=="" set "FOLDER=!DEFAULT_FOLDER!"

set /p RECURSIVE=Recursive? 1=yes 0=no [!DEFAULT_RECURSIVE!]:
if "!RECURSIVE!"=="" set "RECURSIVE=!DEFAULT_RECURSIVE!"

set /p ENABLED=Enabled true/false [!DEFAULT_ENABLED!]:
if "!ENABLED!"=="" set "ENABLED=!DEFAULT_ENABLED!"

set /p N=Number of questions [!DEFAULT_N!]:
if "!N!"=="" set "N=!DEFAULT_N!"

set /p MODULE=Module [!DEFAULT_MODULE!]:
if "!MODULE!"=="" set "MODULE=!DEFAULT_MODULE!"

set /p TEMPLATE=Template [!DEFAULT_TEMPLATE!]:
if "!TEMPLATE!"=="" set "TEMPLATE=!DEFAULT_TEMPLATE!"

set /p OUTPUT_PREFIX=Output name prefix (optional):
if "!OUTPUT_PREFIX!"=="" set "OUTPUT_PREFIX="

set /p PLACEHOLDER=Placeholder video filename (optional):
if "!PLACEHOLDER!"=="" set "PLACEHOLDER="

set /p SKIP=Skip existing rows? 1=yes 0=no [!DEFAULT_SKIP_EXISTING!]:
if "!SKIP!"=="" set "SKIP=!DEFAULT_SKIP_EXISTING!"

echo.
echo ===== RUNNING =====
echo Manifest: !MANIFEST!
echo Folder:   !FOLDER!
echo.
pause

REM ---- Build args ----
set "ARGS=--manifest "!MANIFEST!" --folder "!FOLDER!" --enabled !ENABLED! --n !N! --module "!MODULE!" --template "!TEMPLATE!""

if "!RECURSIVE!"=="1" set "ARGS=!ARGS! --recursive"
if not "!OUTPUT_PREFIX!"=="" set "ARGS=!ARGS! --output-name-prefix "!OUTPUT_PREFIX!""
if not "!PLACEHOLDER!"=="" set "ARGS=!ARGS! --placeholder-video "!PLACEHOLDER!""
if "!SKIP!"=="1" set "ARGS=!ARGS! --skip-existing"

REM ---- Run python ----
if not exist "tools\add_jobs_from_folder.py" (
    echo ERROR: tools\add_jobs_from_folder.py not found
    echo Expected: %REPODIR%\tools\add_jobs_from_folder.py
    pause
    popd
    exit /b 1
)

py "tools\add_jobs_from_folder.py" !ARGS!

echo.
pause
popd
