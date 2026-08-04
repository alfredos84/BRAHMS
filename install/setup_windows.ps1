# ============================================================
# BRAHMS - Windows installer
#
# What this does (fully automated, safe to re-run):
#   1. Creates a Python virtual environment and installs requirements.txt
#   2. Detects/installs the CPU engine toolchain (MSYS2 g++ + mingw32-make)
#      and builds engine_omp, if a MinGW toolchain is already on PATH.
#   3. Detects/installs everything needed for the GPU engine (engine_gpu):
#        - NVIDIA CUDA Toolkit                          (winget)
#        - Visual Studio Build Tools + C++ workload      (winget)
#        - mingw32-make (GNU Make, drives the Makefile)  (winget: MSYS2 +
#          pacman for just the make package - nvcc/cl.exe do the actual
#          compiling, this is only the build orchestrator)
#        - HDF5 (MSVC build, via the vcpkg bundled with Build Tools),
#          installed inside engine_gpu\vcpkg_installed\ (repo-local,
#          portable — NOT the HDF Group installer build, which links
#          against Intel's libmmd.dll and won't run on a plain Windows box)
#      Then compiles engine_gpu\twm.exe.
#   4. Creates a Desktop + Start Menu shortcut that launches the GUI
#      directly, with no console window.
#
# Every step first checks whether it's already satisfied and skips the
# (large) download/install if so — re-running this script after a partial
# install, or on a machine that already has some tools, only fills gaps.
#
# Usage (from an ordinary PowerShell prompt, no admin rights needed):
#   cd BRAHMS\            (the cloned repo root)
#   powershell -ExecutionPolicy Bypass -File install\setup_windows.ps1
#
# Flags:
#   -SkipGpu     Skip all CUDA/MSVC/HDF5 steps (Python + CPU engine only)
#   -SkipCpu     Skip the engine_omp (MinGW/OpenMP) build attempt
# ============================================================

param(
    [switch]$SkipGpu,
    [switch]$SkipCpu
)

$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $RootDir

function Write-Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Test-CommandExists($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

function Update-SessionPath {
    # winget/VS installers write to the machine/user registry PATH but don't
    # touch this already-running process's environment; re-derive it so
    # newly installed tools (nvcc, vswhere, ...) are usable without having
    # to close and reopen the shell.
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

function Find-VsWhere {
    # vswhere.exe ships inside the VS bootstrapper itself, so it exists the
    # moment any VS product/Build Tools install completes - but on a machine
    # with zero prior VS history, winget's installer process can return
    # slightly before the Installer\ folder is fully populated on disk.
    # Poll briefly instead of checking once.
    $path = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    for ($i = 0; $i -lt 6; $i++) {
        if (Test-Path $path) { return $path }
        Start-Sleep -Seconds 5
    }
    return $null
}

# ── 1. Python virtual environment ─────────────────────────────────────────

Write-Step "Looking for a Python 3 interpreter..."
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
Write-Host "    Using: $PythonCmd"

$VenvPython  = Join-Path $RootDir "venv\Scripts\python.exe"
$VenvPythonw = Join-Path $RootDir "venv\Scripts\pythonw.exe"

if (Test-Path $VenvPython) {
    Write-Step "Virtual environment already exists (venv\) - reusing it."
} else {
    Write-Step "Creating virtual environment: venv\"
    Invoke-Expression "$PythonCmd -m venv venv"
}

Write-Step "Installing/updating Python packages..."
& $VenvPython -m pip install --upgrade pip -q
& $VenvPython -m pip install -r requirements.txt -q

# ── 2. CPU engine (engine_omp) - best-effort, only if MinGW is present ────

if (-not $SkipCpu) {
    Write-Step "Checking for a MinGW toolchain (engine_omp / CPU backend)..."
    $make = Get-Command mingw32-make -ErrorAction SilentlyContinue
    $gpp  = Get-Command g++ -ErrorAction SilentlyContinue
    if ($make -and $gpp) {
        Write-Host "    Found - building engine_omp..."
        Push-Location "$RootDir\engine_omp"
        try { & mingw32-make -B } catch { Write-Warning "CPU engine build failed - it is optional; the GUI still runs fine without it." }
        Pop-Location
    } else {
        Write-Host "    Not found - skipping engine_omp (optional CPU backend)."
        Write-Host "    To enable it later: install MSYS2 (https://www.msys2.org/) and run:"
        Write-Host "        pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-make mingw-w64-x86_64-fftw mingw-w64-x86_64-hdf5 mingw-w64-x86_64-nlohmann-json"
        Write-Host "    then add C:\msys64\mingw64\bin to PATH and re-run this script."
    }
}

# ── 3. GPU engine (engine_gpu) - CUDA + MSVC + HDF5, then build twm.exe ───

if ($SkipGpu) {
    Write-Step "Skipping GPU engine setup (-SkipGpu)."
} else {
    Write-Step "Checking for winget (needed to auto-install CUDA / Build Tools)..."
    if (-not (Test-CommandExists "winget")) {
        Write-Warning "winget not found - cannot auto-install CUDA Toolkit / VS Build Tools."
        Write-Warning "Install App Installer from the Microsoft Store, or install CUDA and"
        Write-Warning "Visual Studio Build Tools (C++ workload) manually, then re-run this script."
    } else {

        # -- 3a. NVIDIA CUDA Toolkit --------------------------------------------
        Write-Step "Checking for NVIDIA CUDA Toolkit (nvcc)..."
        if (Test-CommandExists "nvcc") {
            $ver = (nvcc --version | Select-String "release").ToString()
            Write-Host "    Found: $ver - skipping install."
        } else {
            Write-Host "    Not found - installing via winget (Nvidia.CUDA, ~3-4 GB, this can take a while)..."
            winget install --id Nvidia.CUDA --silent --accept-package-agreements --accept-source-agreements
            Update-SessionPath
            if (-not (Test-CommandExists "nvcc")) {
                Write-Warning "CUDA installed but nvcc isn't on PATH in this session yet - close and reopen PowerShell, then re-run this script to finish the GPU engine build."
            }
        }

        # -- 3b. Visual Studio Build Tools (C++ workload = cl.exe for nvcc) -----
        Write-Step "Checking for MSVC (cl.exe) via Visual Studio Build Tools..."
        $vswhere = Find-VsWhere
        $vsInstallPath = $null
        if ($vswhere) {
            $vsInstallPath = & $vswhere -latest -products '*' `
                -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
                -property installationPath
        }
        if ($vsInstallPath) {
            Write-Host "    Found: $vsInstallPath - skipping install."
        } else {
            Write-Host "    Not found - installing Visual Studio 2022 Build Tools + C++ workload"
            Write-Host "    via winget (~2-4 GB, this can take a while)..."
            winget install --id Microsoft.VisualStudio.2022.BuildTools --silent `
                --accept-package-agreements --accept-source-agreements `
                --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
            Update-SessionPath
            # First-ever VS install on this machine: vswhere.exe and the
            # installationPath registration can lag a few seconds behind
            # winget's process exit, so poll instead of checking once.
            $vswhere = Find-VsWhere
            if ($vswhere) {
                for ($i = 0; $i -lt 6 -and -not $vsInstallPath; $i++) {
                    $vsInstallPath = & $vswhere -latest -products '*' `
                        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
                        -property installationPath
                    if (-not $vsInstallPath) { Start-Sleep -Seconds 5 }
                }
            }
            if (-not $vsInstallPath) {
                Write-Warning "Visual Studio Build Tools install did not complete as expected."
                Write-Warning "Re-run this script after it finishes, or install manually with the"
                Write-Warning "'Desktop development with C++' workload."
            }
        }

        # -- 3c. HDF5 (MSVC build via vcpkg, installed inside the repo) ---------
        Write-Step "Checking for HDF5 (MSVC/vcpkg build, engine_gpu\vcpkg_installed\)..."
        $EngineGpuDir = Join-Path $RootDir "engine_gpu"
        $Hdf5Lib = Join-Path $EngineGpuDir "vcpkg_installed\x64-windows\lib\hdf5.lib"
        if (Test-Path $Hdf5Lib) {
            Write-Host "    Found - skipping HDF5 build."
        } elseif ($vsInstallPath) {
            $vcpkgExe = Join-Path $vsInstallPath "VC\vcpkg\vcpkg.exe"
            if (-not (Test-Path $vcpkgExe)) {
                Write-Warning "vcpkg.exe not found under $vsInstallPath - skipping HDF5 build."
                Write-Warning "(It ships with the 'administrador de paquetes vcpkg' / 'vcpkg package manager' component of Build Tools.)"
            } else {
                Write-Host "    Building HDF5 via vcpkg (compiles from source, ~5-15 min)..."
                $VcpkgProjectDir = Join-Path $EngineGpuDir "vcpkg_project"
                New-Item -ItemType Directory -Force -Path $VcpkgProjectDir | Out-Null
                $ManifestPath = Join-Path $VcpkgProjectDir "vcpkg.json"
                # builtin-baseline pinned to the registry commit embedded in this
                # vcpkg distribution (see VC\vcpkg\vcpkg-bundle.json / embeddedsha)
                $BundleJson = Join-Path $vsInstallPath "VC\vcpkg\vcpkg-bundle.json"
                $Baseline = "e03dc9b29710050cd1018bc5674688108658d327"
                if (Test-Path $BundleJson) {
                    try { $Baseline = (Get-Content $BundleJson | ConvertFrom-Json).embeddedsha } catch {}
                }
                @"
{
  "name": "brahms-deps",
  "version": "1.0.0",
  "builtin-baseline": "$Baseline",
  "dependencies": [
    {
      "name": "hdf5",
      "default-features": false,
      "features": ["cpp", "zlib"]
    }
  ]
}
"@ | Set-Content -Path $ManifestPath -Encoding utf8

                Push-Location $VcpkgProjectDir
                try {
                    & $vcpkgExe install --triplet x64-windows "--x-install-root=$($EngineGpuDir -replace '\\','/')/vcpkg_installed"
                } finally {
                    Pop-Location
                }
                if (-not (Test-Path $Hdf5Lib)) {
                    Write-Warning "HDF5 build via vcpkg did not produce the expected library."
                    Write-Warning "Check the output above; re-run this script to retry."
                }
            }
        } else {
            Write-Warning "Skipping HDF5 build - Visual Studio Build Tools are required first."
        }

        # -- 3d. mingw32-make (build orchestrator for the Makefile; nvcc/cl.exe
        #        remain the actual compilers - this is NOT MinGW-GCC) ----------
        Write-Step "Checking for mingw32-make (GNU Make - drives the GPU Makefile)..."
        if (Test-CommandExists "mingw32-make") {
            Write-Host "    Found - skipping install."
        } else {
            $Msys2Bin = "C:\msys64\usr\bin\bash.exe"
            if (-not (Test-Path $Msys2Bin)) {
                Write-Host "    Not found - installing MSYS2 via winget (~500 MB)..."
                winget install --id MSYS2.MSYS2 --silent --accept-package-agreements --accept-source-agreements
            }
            if (Test-Path $Msys2Bin) {
                Write-Host "    Installing mingw-w64-x86_64-make via pacman..."
                # -Sy (not -Syu): a full system upgrade can require MSYS2 to
                # relaunch itself mid-update on a first run, which breaks a
                # non-interactive script. We only need the make package, not
                # a fully upgraded MSYS2 base.
                & $Msys2Bin -lc "pacman -Sy --noconfirm mingw-w64-x86_64-make" | Out-Host
                $Msys2MakeDir = "C:\msys64\mingw64\bin"
                if ((Test-Path (Join-Path $Msys2MakeDir "mingw32-make.exe")) -and ($env:Path -notlike "*$Msys2MakeDir*")) {
                    $env:Path = "$Msys2MakeDir;$env:Path"
                }
            }
            if (-not (Test-CommandExists "mingw32-make")) {
                Write-Warning "mingw32-make still not available - the GPU engine build (step below) will be skipped."
                Write-Warning "Install it manually (see this script's header) and re-run to build engine_gpu."
            }
        }

        # -- 3e. Build engine_gpu\twm.exe ---------------------------------------
        Write-Step "Building engine_gpu (GPU engine)..."
        $vcvars64 = if ($vsInstallPath) { Join-Path $vsInstallPath "VC\Auxiliary\Build\vcvars64.bat" } else { $null }
        if ((Test-CommandExists "nvcc") -and (Test-CommandExists "mingw32-make") -and $vcvars64 -and (Test-Path $vcvars64) -and (Test-Path $Hdf5Lib)) {
            $BuildBat = Join-Path $env:TEMP "brahms_build_gpu_$([guid]::NewGuid().ToString('N')).bat"
            @"
@echo off
call "$vcvars64" >NUL
mingw32-make -B -C "$EngineGpuDir"
"@ | Set-Content -Path $BuildBat -Encoding ASCII
            & cmd.exe /c "`"$BuildBat`""
            $buildExit = $LASTEXITCODE
            Remove-Item $BuildBat -Force -ErrorAction SilentlyContinue
            if ($buildExit -eq 0) {
                Write-Host "    engine_gpu\twm.exe built successfully."
            } else {
                Write-Warning "engine_gpu build failed (exit $buildExit) - see output above."
                Write-Warning "The GUI will rebuild it automatically on first GPU run once the issue is fixed."
            }
        } else {
            Write-Host "    Skipping build - one or more prerequisites (nvcc / MSVC / mingw32-make / HDF5) are still missing."
            Write-Host "    The GUI will attempt to build it on first GPU run; re-run this script to retry now."
        }
    }
}

# ── 4. Desktop / Start Menu shortcuts ──────────────────────────────────────

Write-Step "Creating Desktop shortcut..."
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

# -- Convenience launcher for the command line
$LauncherBat = Join-Path $RootDir "install\run_brahms.bat"
@"
@echo off
cd /d "$RootDir"
"$VenvPython" main.py
"@ | Set-Content -Path $LauncherBat -Encoding ASCII

Write-Host ""
Write-Host "==> Done!" -ForegroundColor Green
Write-Host "    A 'BRAHMS' shortcut was added to your Desktop and Start Menu."
Write-Host "    You can also launch it manually with:"
Write-Host "        install\run_brahms.bat"
Write-Host ""
Write-Host "    If CUDA/MSVC were just installed for the first time, close and reopen"
Write-Host "    PowerShell (or sign out/in) once so PATH changes take effect, then"
Write-Host "    re-run this script if the GPU engine build was skipped above."
