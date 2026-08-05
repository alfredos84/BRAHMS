"""
EngineRunner
============
Launches the C++ engine (GPU/CUDA or CPU/OpenMP) via QProcess.

backend="cuda" → engine_gpu/twm     (compiled with nvcc)
backend="cpu"  → engine_omp/twm_cpu (compiled with g++ -fopenmp)

Flow:
  1. Write config.json to output directory
  2. If grid/arch changed or binary missing, rebuild with make -B
  3. Run the binary: twm config.json out_dir/
  4. Stream stdout/stderr line-by-line via log_line signal
  5. Emit finished(success, out_dir) when done
"""

import json
import os
import shutil
import subprocess
import sys

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, pyqtSignal

# MSYS2's mingw-w64 "make" package installs mingw32-make.exe, not make.exe
# (see install/setup_windows.ps1). Prefer it on Windows, but fall back to
# "make" in case the user has a different toolchain (e.g. WSL make on PATH).
if sys.platform == "win32" and shutil.which("mingw32-make"):
    MAKE_CMD = "mingw32-make"
else:
    MAKE_CMD = "make"


def _find_vcvars64() -> str | None:
    """Locate VC\\Auxiliary\\Build\\vcvars64.bat from any VS/Build Tools
    install (2022, 2026, ...) via vswhere. nvcc on Windows needs cl.exe on
    PATH, which only happens once this script has been sourced."""
    if sys.platform != "win32":
        return None
    vswhere = (r"C:\Program Files (x86)\Microsoft Visual Studio\Installer"
               r"\vswhere.exe")
    if not os.path.isfile(vswhere):
        return None
    try:
        r = subprocess.run(
            [vswhere, "-latest", "-products", "*",
             "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
             "-property", "installationPath"],
            capture_output=True, text=True, timeout=10)
        install_path = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
    except Exception:
        return None
    if not install_path:
        return None
    candidate = os.path.join(install_path, "VC", "Auxiliary", "Build", "vcvars64.bat")
    return candidate if os.path.isfile(candidate) else None


VCVARS64 = _find_vcvars64()


class EngineRunner(QObject):
    log_line = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, project_root: str, parent=None):
        super().__init__(parent)
        self._root           = project_root
        self._engine_dir     = os.path.join(project_root, "engine_gpu")
        self._engine_cpu_dir = os.path.join(project_root, "engine_omp")
        self._twm_bin        = os.path.join(self._engine_dir,     "twm")
        self._twm_cpu_bin    = os.path.join(self._engine_cpu_dir, "twm_cpu")
        self._backend        = "cuda"

        self._last_build: dict = {}   # tracks grid + arch of last successful build
        self._pending_grid: dict = {}
        self._config_path = ""
        self._out_dir     = ""
        self._pending_arch = ""

        self._proc_make: QProcess | None = None
        self._proc_twm:  QProcess | None = None
        self._omp_threads: int | None = None   # None → OpenMP runtime default (all logical cores)

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, config: dict, grid: dict, out_dir: str, backend: str = "cuda",
            omp_threads: int | None = None):
        """
        Start a simulation run.
        config      : dict from config_builder.build_config()
        grid        : {"NX": int, "NY": int, "NZ": int, "NT": int, ...}
        out_dir     : absolute path where HDF5 outputs will be written
        omp_threads : CPU backend only — sets OMP_NUM_THREADS for the run;
                      None leaves the OpenMP runtime default (all logical cores).
        """
        self._backend = backend
        self._omp_threads = omp_threads
        os.makedirs(out_dir, exist_ok=True)
        self._config_path = os.path.join(out_dir, "config.json")
        self._out_dir = out_dir
        self._pending_grid = dict(grid)

        with open(self._config_path, "w") as f:
            json.dump(config, f, indent=2)
        self.log_line.emit(f"[INFO] Config → {self._config_path}")

        if self._backend == "cpu":
            self._pending_arch = "cpu"
        else:
            self._pending_arch = self._detect_arch()
        pending_build = dict(grid) | {"ARCH": self._pending_arch}
        needs_build = (
            pending_build != self._last_build
            or not os.path.isfile(self._active_bin())
        )
        if needs_build:
            self._start_make()
        else:
            self._start_twm()

    def stop(self):
        for proc in (self._proc_make, self._proc_twm):
            if proc and proc.state() != QProcess.ProcessState.NotRunning:
                proc.kill()
                proc.waitForFinished(1000)
        self.log_line.emit("[INFO] Simulation stopped by user.")

    # ── Make step ──────────────────────────────────────────────────────────────

    def _active_bin(self) -> str:
        return self._twm_cpu_bin if self._backend == "cpu" else self._twm_bin

    def _active_engine_dir(self) -> str:
        return self._engine_cpu_dir if self._backend == "cpu" else self._engine_dir

    def _detect_arch(self) -> str:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15)
            if r.returncode == 0 and r.stdout.strip():
                cap = r.stdout.strip().split("\n")[0].strip().replace(".", "")
                return f"sm_{cap}"
            self.log_line.emit(
                f"[WARN] nvidia-smi returned code {r.returncode} "
                f"(stderr: {r.stderr.strip()!r}) - falling back to ARCH=sm_75")
        except Exception as e:
            self.log_line.emit(
                f"[WARN] Could not run nvidia-smi ({type(e).__name__}: {e}) "
                "- falling back to ARCH=sm_75")
        return "sm_75"

    def _start_make(self):
        g    = self._pending_grid
        arch = self._pending_arch
        eng  = self._active_engine_dir()
        # -B forces recompilation — needed because -D flags don't change timestamps
        if self._backend == "cpu":
            args = ["-B", "-C", eng,
                    f"NX={g['NX']}", f"NY={g['NY']}",
                    f"NZ={g['NZ']}", f"NT={g['NT']}"]
        else:
            args = ["-B", "-C", eng,
                    f"NX={g['NX']}", f"NY={g['NY']}",
                    f"NZ={g['NZ']}", f"NT={g['NT']}",
                    f"ARCH={arch}",
                    f"BLKX={g.get('BLKX', 16)}",
                    f"BLKY={g.get('BLKY', 16)}",
                    f"BLKT={g.get('BLKT', 16)}"]
        self.log_line.emit(f"[BUILD] {MAKE_CMD} {' '.join(args)}")

        self._proc_make = QProcess(self)
        self._proc_make.readyReadStandardOutput.connect(self._on_make_stdout)
        self._proc_make.readyReadStandardError.connect(self._on_make_stderr)
        self._proc_make.finished.connect(self._on_make_done)
        self._proc_make.errorOccurred.connect(self._on_make_error)

        if self._backend != "cpu" and VCVARS64:
            # nvcc needs cl.exe (MSVC) on PATH; only vcvars64.bat puts it there.
            # QProcess.start(program, args) quotes every arg itself, so handing
            # it a pre-quoted "/c <command line>" string double-quotes it and
            # cmd.exe fails to parse the result. setNativeArguments() bypasses
            # that: it's passed to CreateProcess verbatim, unquoted.
            quoted_args = " ".join(f'"{a}"' if " " in a else a for a in args)
            cmd_line = f'/c call "{VCVARS64}" >NUL && {MAKE_CMD} {quoted_args}'
            self._proc_make.setProgram("cmd.exe")
            self._proc_make.setNativeArguments(cmd_line)
            self._proc_make.start()
        else:
            self._proc_make.start(MAKE_CMD, args)

    def _on_make_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.log_line.emit(
                f"[ERROR] Could not run '{MAKE_CMD}' — no C++ build toolchain "
                "found in PATH. On Windows, install MSYS2/MinGW (or use WSL) — "
                "see install/setup_windows.ps1 for instructions — then try again.")
            self.finished.emit(False, "")

    def _on_make_stdout(self):
        data = self._proc_make.readAllStandardOutput().data().decode("utf-8", "replace")
        for line in data.splitlines():
            if line.strip():
                self.log_line.emit(f"[BUILD] {line}")

    def _on_make_stderr(self):
        data = self._proc_make.readAllStandardError().data().decode("utf-8", "replace")
        for line in data.splitlines():
            if line.strip():
                self.log_line.emit(f"[BUILD] {line}")

    def _on_make_done(self, exit_code: int, exit_status):
        if exit_code != 0:
            self.log_line.emit(f"[ERROR] Build failed (exit {exit_code})")
            self.finished.emit(False, "")
            return
        self.log_line.emit("[BUILD] Binary ready.")
        self._last_build = dict(self._pending_grid) | {"ARCH": self._pending_arch}
        self._start_twm()

    # ── TWM step ───────────────────────────────────────────────────────────────

    def _start_twm(self):
        binary = self._active_bin()
        self.log_line.emit(
            f"[RUN] {os.path.basename(binary)} {os.path.basename(self._config_path)} → {self._out_dir}/")
        self._proc_twm = QProcess(self)
        if self._backend != "cpu" and sys.platform == "win32":
            # twm.exe dynamically links hdf5.dll / hdf5_cpp.dll (vcpkg build,
            # installed in-repo by install/setup_windows.ps1); their
            # directory isn't on the default PATH.
            env = QProcessEnvironment.systemEnvironment()
            hdf5_bin = os.path.join(self._engine_dir, "vcpkg_installed", "x64-windows", "bin")
            if os.path.isdir(hdf5_bin):
                env.insert("PATH", hdf5_bin + os.pathsep + env.value("PATH"))
            self._proc_twm.setProcessEnvironment(env)
        elif self._backend == "cpu" and self._omp_threads:
            # OMP_NUM_THREADS overrides the OpenMP runtime default of "all
            # logical cores"; the CPU engine reads it implicitly via
            # omp_get_max_threads() — no code change needed on that side.
            env = QProcessEnvironment.systemEnvironment()
            env.insert("OMP_NUM_THREADS", str(self._omp_threads))
            self._proc_twm.setProcessEnvironment(env)
            self.log_line.emit(f"[INFO] OMP_NUM_THREADS={self._omp_threads}")
        self._proc_twm.readyReadStandardOutput.connect(self._on_twm_stdout)
        self._proc_twm.readyReadStandardError.connect(self._on_twm_stderr)
        self._proc_twm.finished.connect(self._on_twm_done)
        self._proc_twm.start(binary, [self._config_path, self._out_dir])

    def _on_twm_stdout(self):
        data = self._proc_twm.readAllStandardOutput().data().decode("utf-8", "replace")
        for line in data.splitlines():
            if line.strip():
                self.log_line.emit(line)

    def _on_twm_stderr(self):
        data = self._proc_twm.readAllStandardError().data().decode("utf-8", "replace")
        for line in data.splitlines():
            if line.strip():
                self.log_line.emit(f"[ERR] {line}")

    def _on_twm_done(self, exit_code: int, exit_status):
        if exit_code == 0:
            self.log_line.emit("[OK] Simulation complete.")
            self.finished.emit(True, self._out_dir)
        else:
            self.log_line.emit(f"[ERROR] Engine exited with code {exit_code}")
            self.finished.emit(False, "")
