# ============================================================
# BRAHMS — Windows installer
#
# What this does:
#   1. Creates a Python virtual environment and installs requirements.txt
#   2. Creates a Desktop shortcut (with the app icon) that launches the
#      GUI directly, with no console window
#
# What it does NOT do (documented, manual steps below):
#   - Build engine_omp (CPU/OpenMP engine): needs a C++ compiler with
#     OpenMP + FFTW3 + HDF5 + nlohmann/json, which Windows does not ship.
#     Easiest path: install MSYS2 (https://www.msys2.org/), then from the
#     MSYS2 MinGW64 shell:
#         pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-fftw \
#                   mingw-w64-x86_64-hdf5 mingw-w64-x86_64-nlohmann-json make
#         cd engine_omp && mingw32-make
#     Alternative: use WSL (Windows Subsystem for Linux) and run
#     install/setup_linux.sh inside it — the GUI can call into WSL.
#   - Build engine_gpu (CUDA engine): needs the NVIDIA CUDA Toolkit and
#     Visual Studio Build Tools (MSVC). Install both, open an
#     "x64 Native Tools Command Prompt", then:
#         cd engine_gpu && nmake /f Makefile.win     (or run nvcc by hand)
#     A ready-made Windows Makefile is not shipped yet — this is the one
#     manual step left for CUDA users on native Windows.
#
# Usage (from an ordinary PowerShell prompt, no admin rights needed):
#   cd BRAHMS\            (the cloned repo root)
#   powershell -ExecutionPolicy Bypass -File install\setup_windows.ps1
# ============================================================

$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $RootDir

# ── Find a Python 3 interpreter ─────────────────────────────────────────
$PythonCmd = $null
foreach ($cand in @("py -3", "python", "python3")) {
    $parts = $cand -split " "
    $exe = $parts[0]
    if (Get-Command $exe -ErrorAction SilentlyContinue) {
        $PythonCmd = $cand
        break
    }
}
if (-not $PythonCmd) {
    Write-Error "Python 3 was not found. Install it from https://www.python.org/downloads/ (check 'Add python.exe to PATH') and re-run this script."
    exit 1
}
Write-Host "==> Using Python: $PythonCmd"

# ── Virtual environment ──────────────────────────────────────────────────
Write-Host "==> Creating virtual environment: venv\"
Invoke-Expression "$PythonCmd -m venv venv"

$VenvPython  = Join-Path $RootDir "venv\Scripts\python.exe"
$VenvPythonw = Join-Path $RootDir "venv\Scripts\pythonw.exe"

Write-Host "==> Installing Python packages..."
& $VenvPython -m pip install --upgrade pip -q
& $VenvPython -m pip install -r requirements.txt -q

# ── Optional: try building the CPU engine if a MinGW toolchain is present ─
$make = Get-Command mingw32-make -ErrorAction SilentlyContinue
$gpp  = Get-Command g++ -ErrorAction SilentlyContinue
if ($make -and $gpp) {
    Write-Host "==> MinGW toolchain found — building CPU engine (engine_omp)..."
    Push-Location "$RootDir\engine_omp"
    try { & mingw32-make -B } catch { Write-Warning "CPU engine build failed — see manual steps in this script's header." }
    Pop-Location
} else {
    Write-Host "==> No MinGW toolchain (g++ / mingw32-make) found in PATH."
    Write-Host "    Skipping engine build — see this script's header for manual setup"
    Write-Host "    (MSYS2/MinGW-w64, or use WSL). The GUI itself still runs fine."
}

# ── Desktop shortcut ─────────────────────────────────────────────────────
Write-Host "==> Creating Desktop shortcut..."
$IconPath = Join-Path $RootDir "icon\icon.ico"
$Desktop  = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $Desktop "BRAHMS.lnk"

$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath       = $VenvPythonw
$Shortcut.Arguments        = '"' + (Join-Path $RootDir "main.py") + '"'
$Shortcut.WorkingDirectory = $RootDir
$Shortcut.IconLocation     = $IconPath
$Shortcut.Description      = "BRAHMS - Three-Wave Mixing Simulator"
$Shortcut.Save()

# Also drop a Start Menu entry
$StartMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
Copy-Item $ShortcutPath (Join-Path $StartMenu "BRAHMS.lnk") -Force

# ── Convenience launcher for the command line ────────────────────────────
$LauncherBat = Join-Path $RootDir "install\run_brahms.bat"
@"
@echo off
cd /d "$RootDir"
"$VenvPython" main.py
"@ | Set-Content -Path $LauncherBat -Encoding ASCII

Write-Host ""
Write-Host "==> Done!"
Write-Host "    A 'BRAHMS' shortcut was added to your Desktop and Start Menu."
Write-Host "    You can also launch it manually with:"
Write-Host "        install\run_brahms.bat"
