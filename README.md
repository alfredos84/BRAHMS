# BRAHMS — Three-Wave Mixing Simulator

A cross-platform GUI for simulating three-wave mixing processes (SHG, SFG,
OPG, DFG) in nonlinear crystals, with CPU (OpenMP) and GPU (CUDA) simulation
engines.

## Requirements

- Python 3.10+
- Linux: a C++ compiler (g++), FFTW3, HDF5, nlohmann/json — installed
  automatically by `install/setup_linux.sh`.
- Optional: an NVIDIA GPU + CUDA Toolkit, to also build the GPU engine.

## Install

### Linux (Ubuntu/Debian)

```bash
git clone https://github.com/alfredos84/BRAHMS.git
cd BRAHMS
bash install/setup_linux.sh
```

This installs system dependencies, creates a Python virtual environment,
builds the CPU simulation engine (and the GPU engine too, if an NVIDIA GPU
+ CUDA toolkit is detected), and adds a **BRAHMS** launcher to your
applications menu and Desktop — just double-click it to open the app.

To launch it manually instead:

```bash
bash install/run_brahms.sh
```

### Windows

```powershell
git clone https://github.com/alfredos84/BRAHMS.git
cd BRAHMS
powershell -ExecutionPolicy Bypass -File install\setup_windows.ps1
```

This creates a Python virtual environment and a **BRAHMS** shortcut on your
Desktop and Start Menu. Building the C++ simulation engines on native
Windows needs an extra manual step (MSYS2/MinGW for the CPU engine, CUDA
Toolkit + MSVC for the GPU engine) — see the comments at the top of
`install/setup_windows.ps1` for exact commands, or use WSL and run the
Linux installer instead.

## Manual run (any platform, if you prefer not to use the installers)

```bash
python3 -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
python main.py
```

## Repository layout

```
gui/            PyQt6 application (tabs, widgets, config builder)
engine_omp/     CPU simulation engine (C++/OpenMP)
engine_gpu/     GPU simulation engine (C++/CUDA)
crystals/       Nonlinear-crystal database (SQLite, created on first run)
icon/           App icon (icon.png / icon.ico)
install/        Installers and desktop launchers
main.py         Entry point
```
