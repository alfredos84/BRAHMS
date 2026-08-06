import os
import json
import subprocess
import tempfile
import time
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QGroupBox,
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QPushButton,
    QRadioButton, QButtonGroup, QProgressBar, QTextEdit,
    QGridLayout, QFrame, QScrollArea, QCheckBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer

from ..widgets.plot_canvas import PlotCanvas
from ..core_py.config_builder import build_config


# Parameters that can be swept, with their display name, units, and defaults
SWEEP_PARAMS = [
    ("Pump power",          "pump/power_W",      "W",    1e-3,  10.0,   20, "log"),
    ("Beam waist",          "pump/waist_um",     "μm",   5.0,   200.0,  20, "lin"),
    ("Crystal temperature", "temperature",       "°C",   20.0,  200.0,  40, "lin"),
    ("Crystal length",      "length_mm",         "mm",   1.0,   50.0,   20, "lin"),
    ("Grating period Λ",    "grating_um",        "μm",   5.0,   12.0,   20, "lin"),
    ("Phase mismatch Δk",   "dk",                "μm⁻¹", -1e-3,  1e-3,  80, "lin"),
    ("Pump wavelength λ",   "pump/lambda_um",    "μm",   0.9,   1.2,    20, "lin"),
    ("Pulse FWHM",          "pump/fwhm_ps",      "ps",   0.05,  2.0,    20, "lin"),
]

# Keys that each SWEEP_PARAMS entry makes variable (for Fixed params display filtering)
_SWEEP_KEY_EXCLUDES = {
    "pump/power_W":   {"Power"},
    "pump/waist_um":  {"Waist"},
    "temperature":    {"T"},
    "length_mm":      {"L"},
    "grating_um":     {"Λ"},
    "dk":             {"Δk"},
    "pump/lambda_um": {"λ pump"},
    "pump/fwhm_ps":   {"FWHM"},
}

# Module-level compilation cache: binary_path → (NX, NY, NZ, NT)
# Persists for the app session so we skip recompilation when grid is unchanged.
_compiled_grid_cache: dict[str, tuple] = {}


class SweepWorker(QObject):
    """Runs real simulations for each sweep point via EngineRunner."""
    progress   = pyqtSignal(int, float, object)  # index, param_value, dict of outputs
    finished   = pyqtSignal()
    log_line   = pyqtSignal(str)

    def __init__(self, values, base_params, param_key,
                 project_root: str = None, backend: str = "cuda"):
        super().__init__()
        self.values      = values
        self.base_params = base_params
        self.param_key   = param_key
        self.project_root = project_root or os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.backend  = backend
        self._running = True
        self._proc    = None

    def run(self):
        """Execute real simulations for each sweep point."""
        # Grid comes from base_params (from Simulation tab) — build_config does NOT
        # include "grid" in its output, so never rely on its fallback.
        gp = self.base_params.get("grid", {})
        self._grid = {
            "NX": int(gp.get("NX", 64)),
            "NY": int(gp.get("NY", 64)),
            "NZ": int(gp.get("NZ", 1000)),
            "NT": int(gp.get("NT", 1)),
        }
        self.log_line.emit(
            f"[SWEEP] Grid: NX={self._grid['NX']} NY={self._grid['NY']} "
            f"NZ={self._grid['NZ']} NT={self._grid['NT']}")
        self._binary = self._compile_engine(self._grid)
        if self._binary is None:
            self.finished.emit()
            return

        t_sweep_start = time.perf_counter()

        for i, v in enumerate(self.values):
            if not self._running:
                break
            self.log_line.emit(f"  [{i+1}/{len(self.values)}]  {self.param_key} = {v:.5g}")

            t_point_start = time.perf_counter()
            try:
                result = self._run_real_simulation(v)
                self.progress.emit(i, v, result)
            except Exception as e:
                self.log_line.emit(f"    [ERROR] {str(e)}")
                self.progress.emit(i, v, {"eta": 0.0, "P_signal": 0.0, "P_idler": 0.0, "P_pump_out": 0.0})
            t_point = time.perf_counter() - t_point_start
            self.log_line.emit(f"    [TIME]  point {i+1}: {t_point:.2f} s")

        t_total = time.perf_counter() - t_sweep_start
        n_done = len(self.values) if self._running else sum(1 for _ in range(len(self.values)))
        self.log_line.emit(
            f"\n[SWEEP] Total execution time: {t_total:.2f} s  "
            f"({len(self.values)} points,  avg {t_total/max(len(self.values),1):.2f} s/point)")
        self.finished.emit()

    def stop(self):
        self._running = False
        if self._proc:
            self._proc.terminate()

    def _detect_arch(self) -> str:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                cap = r.stdout.strip().split("\n")[0].strip().replace(".", "")
                return f"sm_{cap}"
        except Exception:
            pass
        return "sm_75"

    def _compile_engine(self, grid: dict) -> str | None:
        """Compile engine binary with correct grid macros. Returns binary path or None.
        Skips recompilation if the binary already exists with the same grid dimensions."""
        if self.backend == "cpu":
            eng_dir = os.path.join(self.project_root, "engine_omp")
            binary  = os.path.join(eng_dir, "twm_cpu")
            make_args = ["-B", "-C", eng_dir,
                         f"NX={grid['NX']}", f"NY={grid['NY']}",
                         f"NZ={grid['NZ']}", f"NT={grid['NT']}"]
        else:
            eng_dir = os.path.join(self.project_root, "engine_gpu")
            binary  = os.path.join(eng_dir, "twm")
            arch    = self._detect_arch()
            make_args = ["-B", "-C", eng_dir,
                         f"NX={grid['NX']}", f"NY={grid['NY']}",
                         f"NZ={grid['NZ']}", f"NT={grid['NT']}",
                         f"ARCH={arch}"]

        grid_key = (grid['NX'], grid['NY'], grid['NZ'], grid['NT'])
        if os.path.isfile(binary) and _compiled_grid_cache.get(binary) == grid_key:
            self.log_line.emit(
                f"[BUILD] Binary up-to-date "
                f"(NX={grid_key[0]} NY={grid_key[1]} NZ={grid_key[2]} NT={grid_key[3]}) "
                f"→ skipping recompilation")
            return binary

        self.log_line.emit(
            f"[BUILD] make NX={grid['NX']} NY={grid['NY']} "
            f"NZ={grid['NZ']} NT={grid['NT']} ...")

        try:
            self._proc = subprocess.Popen(
                ["make"] + make_args,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
        except OSError as e:
            self.log_line.emit(
                f"[ERROR] Could not run 'make': {e}\n"
                f"        No C++ build toolchain found in PATH. Install "
                f"MSYS2/MinGW (or use WSL) — see install/setup_windows.ps1 "
                f"for instructions — then try again.")
            return None
        for line in self._proc.stdout:
            line = line.rstrip()
            if line:
                self.log_line.emit(f"  {line}")
        self._proc.wait()

        if self._proc.returncode != 0:
            self.log_line.emit(f"[ERROR] Compilation failed (exit {self._proc.returncode})")
            return None

        _compiled_grid_cache[binary] = grid_key
        self.log_line.emit(f"[BUILD] Done → {binary}")
        return binary

    def _run_real_simulation(self, param_value: float) -> dict:
        """Run one simulation with modified parameter using the pre-compiled binary."""
        params = self._modify_params(param_value)
        config = build_config(params)

        with tempfile.TemporaryDirectory(prefix="twm_sweep_") as out_dir:
            config_path = os.path.join(out_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            self.log_line.emit(f"    [RUN] twm ...")
            self._proc = subprocess.Popen(
                [self._binary, config_path, out_dir + "/"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in self._proc.stdout:
                if not self._running:
                    self._proc.terminate()
                    raise RuntimeError("Sweep stopped by user")
                line = line.rstrip()
                if line:
                    self.log_line.emit(f"    {line}")

            self._proc.wait()
            if self._proc.returncode != 0:
                raise RuntimeError(f"Engine exited with code {self._proc.returncode}")

            return self._calculate_outputs(out_dir)

    def _modify_params(self, param_value: float) -> dict:
        """Create a copy of base_params with one parameter modified."""
        params = json.loads(json.dumps(self.base_params))  # deep copy

        key = self.param_key
        if key.startswith("pump/"):
            params["pump"][key[5:]] = param_value
        else:
            params[key] = param_value

        return params

    def _calculate_outputs(self, out_dir: str) -> dict:
        """
        Computes all output quantities from one simulation run.
        Returns dict: {"eta": %, "P_signal": W, "P_idler": W, "P_pump_out": W}

        η = P_signal / P_pump_in × 100
        P_pump_in is read from config.json["fields"]["pump"]["power_W"] — exact input
        power used to initialize the field; avoids reading pump_input_XY.h5 which
        the engine does NOT save.
        P = 0.5 * n * ε₀ * c * Σ|A|² * dx * dy  (CW, integrated over transverse grid)
        """
        import h5py
        zero = {"P_signal": 0.0, "P_idler": 0.0, "P_pump_out": 0.0,
                "eta_s": 0.0, "eta_i": 0.0, "pump_dep": 0.0}

        signal_file   = os.path.join(out_dir, "signal_output.h5")
        pump_out_file = os.path.join(out_dir, "pump_output.h5")
        idler_file    = os.path.join(out_dir, "idler_output.h5")

        try:
            if not os.path.isfile(signal_file):
                self.log_line.emit("    [ERROR] signal_output.h5 not found")
                return zero

            # Read optical constants and geometry from config
            n_p = 1.0; n_s = 1.0; n_i = 1.0
            LX_um = None; LY_um = None
            P_pump_in = None
            cfg_path = os.path.join(out_dir, "config.json")
            if os.path.isfile(cfg_path):
                with open(cfg_path) as f:
                    cfg = json.load(f)
                opt = cfg["crystal"]["optics"]
                geo = cfg["crystal"]["geometry"]
                n_p = opt["np"]; n_s = opt["ns"]; n_i = opt["ni"]
                LX_um = geo["LX_mm"] * 1000.0
                LY_um = geo["LY_mm"] * 1000.0
                # Use the declared input power — engine initialises the field with
                # exactly this value; no need to read pump_input_XY.h5 (not saved).
                P_pump_in = cfg["fields"]["pump"]["power_W"]

            if P_pump_in is None or P_pump_in <= 0:
                self.log_line.emit("    [ERROR] Cannot determine pump input power")
                return zero

            with h5py.File(signal_file, "r") as f:
                A_s = np.array(f["real"]) + 1j * np.array(f["imag"])

            _, Ny, Nx = A_s.shape
            dx = (LX_um or float(Nx)) / Nx
            dy = (LY_um or float(Ny)) / Ny
            C0   = 299792458e6 / 1e12    # μm/ps
            EPS0 = 8.8541878128e-12 * 1e12 / 1e6  # C²/(V·μm·ps)

            def power(A, n):
                return 0.5 * n * EPS0 * C0 * float(np.nansum(np.abs(A[-1])**2)) * dx * dy

            P_s_W    = power(A_s, n_s)
            eta      = P_s_W / P_pump_in * 100.0

            P_i_W = 0.0
            if os.path.isfile(idler_file):
                with h5py.File(idler_file, "r") as f:
                    A_i = np.array(f["real"]) + 1j * np.array(f["imag"])
                P_i_W = power(A_i, n_i)

            P_pout_W = 0.0
            if os.path.isfile(pump_out_file):
                with h5py.File(pump_out_file, "r") as f:
                    A_pout = np.array(f["real"]) + 1j * np.array(f["imag"])
                P_pout_W = power(A_pout, n_p)

            eta_s    = P_s_W   / P_pump_in * 100.0
            eta_i    = P_i_W   / P_pump_in * 100.0
            pump_dep = (1.0 - P_pout_W / P_pump_in) * 100.0
            self.log_line.emit(
                f"    η_s={eta_s:.3f}%  η_i={eta_i:.3f}%  dep={pump_dep:.2f}%"
                f"  P_s={P_s_W:.3e}W  P_i={P_i_W:.3e}W"
                f"  P_pout={P_pout_W:.3e}W  (P_pin={P_pump_in:.3e}W)")
            return {
                "P_signal":  P_s_W,
                "P_idler":   P_i_W,
                "P_pump_out": P_pout_W,
                "eta_s":     eta_s,
                "eta_i":     eta_i,
                "pump_dep":  pump_dep,
            }

        except Exception as e:
            import traceback
            self.log_line.emit(f"    [ERROR] {str(e)}")
            self.log_line.emit(f"    [TRACEBACK] {traceback.format_exc()}")
            return zero


class SweepTab(QWidget):

    def __init__(self, sim_tab_ref=None, crystal_tab_ref=None, parent=None):
        super().__init__(parent)
        self._sim_tab     = sim_tab_ref
        self._crystal_tab = crystal_tab_ref
        self._worker   = None
        self._thread   = None
        self._x_data   = []
        self._y_data   = {"P_signal": [], "P_idler": [], "P_pump_out": [],
                          "eta_s": [], "eta_i": [], "pump_dep": []}
        self._lines    = {}   # key → matplotlib Line2D
        self._axes_right = []  # twinx axes (one per panel)
        self._x_display_scale = 1.0      # linear multiplier for x-axis (e.g. Δk→Δk·L_cr)
        self._x_display_label = ""       # x-axis label for current sweep
        self._x_transform     = None     # optional callable(array) → array for non-linear display

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Radio buttons / checkboxes are stretched to the full column width by
        # the layout, but Qt's stylesheet-based hit-testing only registers
        # clicks within their natural (label-sized) rect. Pin them to their
        # natural width so the whole visible control is clickable.
        for w in self.findChildren(QRadioButton):
            w.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        for w in self.findChildren(QCheckBox):
            w.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)

    # ── Left panel ─────────────────────────────────────────────────────
    def _build_left(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left = QWidget()
        left.setMinimumWidth(255)
        left.setMaximumWidth(310)
        lv = QVBoxLayout(left)
        lv.setSpacing(10)
        lv.setContentsMargins(4, 4, 4, 4)

        # ── Crystal selector ───────────────────────────────────────────
        gb_crys = QGroupBox("Nonlinear Crystal")
        gc = QHBoxLayout(gb_crys)
        gc.setSpacing(6)
        self.cb_crystal = QComboBox()
        self.cb_crystal.addItems(["MgO:PPLN", "MgO:sPPLT", "ZGP"])
        gc.addWidget(self.cb_crystal)
        lv.addWidget(gb_crys)

        # ── Parameter to sweep ─────────────────────────────────────────
        gb_par = QGroupBox("Parameter to Sweep")
        gp = QGridLayout(gb_par)
        gp.setSpacing(6)

        gp.addWidget(QLabel("Parameter"), 0, 0)
        self.cb_param = QComboBox()
        for name, *_ in SWEEP_PARAMS:
            self.cb_param.addItem(name)
        gp.addWidget(self.cb_param, 0, 1)

        self.chk_xi_display = QCheckBox("Show as focusing parameter ξ")
        self.chk_xi_display.setVisible(False)   # only shown for Beam waist sweep
        gp.addWidget(self.chk_xi_display, 1, 0, 1, 2)
        lv.addWidget(gb_par)

        # ── Sweep range ────────────────────────────────────────────────
        gb_rng = QGroupBox("Sweep Range")
        gr = QGridLayout(gb_rng)
        gr.setSpacing(6)

        gr.addWidget(QLabel("Min"), 0, 0)
        self.sb_min = QDoubleSpinBox()
        self.sb_min.setRange(-1e9, 1e9); self.sb_min.setDecimals(5)
        gr.addWidget(self.sb_min, 0, 1)

        self.lbl_min_unit = QLabel("")
        self.lbl_min_unit.setObjectName("unitLabel")
        gr.addWidget(self.lbl_min_unit, 0, 2)

        gr.addWidget(QLabel("Max"), 1, 0)
        self.sb_max = QDoubleSpinBox()
        self.sb_max.setRange(-1e9, 1e9); self.sb_max.setDecimals(5)
        gr.addWidget(self.sb_max, 1, 1)

        self.lbl_max_unit = QLabel("")
        self.lbl_max_unit.setObjectName("unitLabel")
        gr.addWidget(self.lbl_max_unit, 1, 2)

        gr.addWidget(QLabel("N points"), 2, 0)
        self.sb_npts = QSpinBox()
        self.sb_npts.setRange(2, 500); self.sb_npts.setValue(20)
        gr.addWidget(self.sb_npts, 2, 1)

        gr.addWidget(QLabel("Spacing"), 3, 0)
        spacing_row = QHBoxLayout()
        self.rb_lin = QRadioButton("Linear")
        self.rb_log = QRadioButton("Log")
        self.rb_lin.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self.rb_lin); bg.addButton(self.rb_log)
        spacing_row.addWidget(self.rb_lin)
        spacing_row.addWidget(self.rb_log)
        spacing_row.addStretch()
        gr.addLayout(spacing_row, 3, 1, 1, 2)

        lv.addWidget(gb_rng)

        # ── Fixed parameters summary ───────────────────────────────────
        gb_fix = QGroupBox("Fixed parameters")
        gf = QVBoxLayout(gb_fix)
        self.lbl_fixed = QLabel("—")
        self.lbl_fixed.setObjectName("unitLabel")
        self.lbl_fixed.setWordWrap(True)
        self.btn_refresh_fixed = QPushButton("Load from Single Simulation")
        self.btn_refresh_fixed.clicked.connect(self._refresh_fixed)
        gf.addWidget(self.lbl_fixed)
        gf.addWidget(self.btn_refresh_fixed)
        lv.addWidget(gb_fix)

        # ── Force Δk override ─────────────────────────────────────────
        gb_dk = QGroupBox("Override Δk")
        gdk = QGridLayout(gb_dk)
        gdk.setSpacing(6)
        self.chk_force_dk = QCheckBox("Force Δk value")
        gdk.addWidget(self.chk_force_dk, 0, 0, 1, 2)
        gdk.addWidget(QLabel("Δk (μm⁻¹)"), 1, 0)
        self.sb_dk_override = QDoubleSpinBox()
        self.sb_dk_override.setLocale(self.sb_min.locale())
        self.sb_dk_override.setRange(-10.0, 10.0)
        self.sb_dk_override.setDecimals(6)
        self.sb_dk_override.setValue(0.0)
        self.sb_dk_override.setEnabled(False)
        gdk.addWidget(self.sb_dk_override, 1, 1)
        self.chk_force_dk.toggled.connect(self.sb_dk_override.setEnabled)
        lv.addWidget(gb_dk)

        # ── Backend ───────────────────────────────────────────────────
        gb_back = QGroupBox("Backend")
        gback = QHBoxLayout(gb_back)
        self.rb_gpu = QRadioButton("GPU (CUDA)")
        self.rb_mpi = QRadioButton("CPU (OMP)")
        self.rb_gpu.setChecked(True)
        bg2 = QButtonGroup(self)
        bg2.addButton(self.rb_gpu); bg2.addButton(self.rb_mpi)
        gback.addWidget(self.rb_gpu)
        gback.addWidget(self.rb_mpi)
        lv.addWidget(gb_back)

        # ── Run / Stop ─────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_run  = QPushButton("  Run Sweep  ")
        self.btn_run.setObjectName("runButton")
        self.btn_stop = QPushButton("  Stop  ")
        self.btn_stop.setObjectName("stopButton")
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_stop)
        lv.addLayout(btn_row)

        self.btn_export = QPushButton("Export results (.txt)")
        lv.addWidget(self.btn_export)
        lv.addStretch()

        scroll.setWidget(left)

        # connections
        self.cb_param.currentIndexChanged.connect(self._on_param_changed)
        self.chk_xi_display.toggled.connect(self._on_xi_toggle)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_export.clicked.connect(self._on_export)
        self._on_param_changed(0)

        return scroll

    # ── Right panel ────────────────────────────────────────────────────
    def _build_right(self):
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setSpacing(6)
        rv.setContentsMargins(4, 4, 4, 4)

        # 3-panel result plot: Signal | Idler | Pump  (dual Y: power left, η right)
        self.canvas = PlotCanvas(nrows=1, ncols=3, figsize=(12, 4), dpi=95)
        rv.addWidget(self.canvas, stretch=3)

        # Progress bar
        prog_row = QHBoxLayout()
        self.lbl_prog = QLabel("Ready")
        self.lbl_prog.setObjectName("unitLabel")
        prog_row.addWidget(self.lbl_prog)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximumHeight(14)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #252525;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                text-align: center;
                color: #888888;
                font-size: 9px;
            }
            QProgressBar::chunk {
                background-color: #00bcd4;
                border-radius: 2px;
            }
        """)
        prog_row.addWidget(self.progress_bar, stretch=1)
        rv.addLayout(prog_row)

        # Log
        self.log = QTextEdit()
        self.log.setObjectName("logOutput")
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(130)
        self.log.append("Parameter sweep ready.")
        rv.addWidget(self.log, stretch=1)

        self._init_plot()
        return right

    # 3-panel definitions: (left_key, right_key, title, color_left, color_right, ylabel_left, ylabel_right)
    _PANELS = [
        ("P_signal",   "eta_s",    "Signal",       "#4caf50", "#80cbc4", "P signal (W)",  "η signal (%)"),
        ("P_idler",    "eta_i",    "Idler",        "#ff9800", "#ffcc80", "P idler (W)",   "η idler (%)"),
        ("P_pump_out", "pump_dep", "Pump",         "#ab47bc", "#ce93d8", "P pump out (W)", "Depletion (%)"),
    ]

    # ── Plot init ──────────────────────────────────────────────────────
    def _init_plot(self):
        for i, (_, _, title, c_l, c_r, yl, yr) in enumerate(self._PANELS):
            ax = self.canvas.get_ax(i)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("—", fontsize=8)
            ax.set_ylabel(yl, fontsize=8, color=c_l)
        self.canvas.refresh()

    # ── Slots ──────────────────────────────────────────────────────────
    def _on_param_changed(self, index):
        _, key, unit, vmin, vmax, npts, spacing = SWEEP_PARAMS[index]
        self.sb_min.setValue(vmin)
        self.sb_max.setValue(vmax)
        self.sb_npts.setValue(npts)
        self.rb_log.setChecked(spacing == "log")
        self.rb_lin.setChecked(spacing == "lin")
        self.lbl_min_unit.setText(unit)
        self.lbl_max_unit.setText(unit)
        # Show ξ-display toggle only when sweeping beam waist
        self.chk_xi_display.setVisible(key == "pump/waist_um")
        # Refresh fixed-params display so swept param disappears immediately
        if self._sim_tab is not None and self.lbl_fixed.text() != "—":
            self._refresh_fixed()

    def _refresh_fixed(self):
        """Load parameters from Simulation tab."""
        self.btn_refresh_fixed.setText("Loading...")
        self.btn_refresh_fixed.setEnabled(False)

        if self._sim_tab is None:
            self.lbl_fixed.setText("Simulation tab not connected.")
            self.btn_refresh_fixed.setText("Load from Single Simulation")
            self.btn_refresh_fixed.setEnabled(True)
            return

        try:
            p = self._sim_tab._collect_params()
            pump = p.get("pump", {})
            grid = p.get("grid", {})

            # Sync crystal combobox from sim tab
            crystal_name = p.get("crystal", "")
            idx = self.cb_crystal.findText(crystal_name)
            if idx >= 0:
                self.cb_crystal.setCurrentIndex(idx)

            # Get the key currently being swept so we can hide it
            swept_key = SWEEP_PARAMS[self.cb_param.currentIndex()][1]
            excluded  = _SWEEP_KEY_EXCLUDES.get(swept_key, set())

            # Each entry: (tag, text) — tag=None means always shown
            all_lines = [
                (None,     f"Process: {p.get('process','—')}"),
                ("L",      f"L = {p.get('length_mm','—')} mm"),
                ("T",      f"T = {p.get('temperature','—')} °C"),
                ("Λ",      f"Λ = {p.get('grating_um','—')} μm"),
                ("λ pump", f"λ pump = {pump.get('lambda_um','—')} μm"),
                ("Power",  f"Power = {pump.get('power_W','—')} W"),
                ("Waist",  f"Waist = {pump.get('waist_um','—')} μm"),
                ("FWHM",   f"FWHM = {pump.get('fwhm_ps','—')} ps"),
                ("Δk",     f"Δk = {p.get('dk', 0.0):.4g} μm⁻¹"),
                (None,     f"Grid: NX={grid.get('NX','—')} NY={grid.get('NY','—')} NZ={grid.get('NZ','—')} NT={grid.get('NT','—')}"),
            ]
            lines = [txt for tag, txt in all_lines if tag not in excluded]
            self.lbl_fixed.setText("\n".join(lines))
        except Exception as e:
            self.lbl_fixed.setText(f"Error loading params: {str(e)}")
        finally:
            self.btn_refresh_fixed.setText("✓ Load from Single Simulation")
            QTimer.singleShot(1500, lambda: self.btn_refresh_fixed.setText("Load from Single Simulation"))
            self.btn_refresh_fixed.setEnabled(True)

    def _build_sweep_values(self):
        vmin = self.sb_min.value()
        vmax = self.sb_max.value()
        n    = self.sb_npts.value()
        if self.rb_log.isChecked() and vmin > 0 and vmax > 0:
            return np.logspace(np.log10(vmin), np.log10(vmax), int(n))
        return np.linspace(vmin, vmax, int(n))

    def _on_run(self):
        self._x_data = []
        self._y_data = {"P_signal": [], "P_idler": [], "P_pump_out": [],
                        "eta_s": [], "eta_i": [], "pump_dep": []}
        self._lines       = {}
        self._axes_right  = []
        self._x_display_scale = 1.0
        self._x_display_label = ""
        self._x_transform     = None

        idx       = self.cb_param.currentIndex()
        name, key, unit, *_ = SWEEP_PARAMS[idx]
        values    = self._build_sweep_values()

        # Always pull base params and refresh the fixed-params display
        base_params = {}
        if self._sim_tab:
            base_params = self._sim_tab._collect_params()
            self._refresh_fixed()

        # Crystal selected directly in the sweep tab takes precedence
        base_params["crystal"] = self.cb_crystal.currentText()

        # Merge alpha/beta on-off flags from Crystals tab
        if self._crystal_tab is not None:
            base_params.update(self._crystal_tab.alpha_flags())
            base_params.update(self._crystal_tab.beta_flags())
            base_params.update(self._crystal_tab.rho_flags())

        # Inject forced Δk into base params (only when not sweeping dk itself)
        if key != "dk" and self.chk_force_dk.isChecked():
            base_params["dk"] = self.sb_dk_override.value()

        # ── X-axis label / transform ───────────────────────────────────
        if key == "dk":
            L_cr_mm = float(base_params.get("length_mm", 1.0))
            self._x_display_scale = L_cr_mm * 1000.0   # Δk[μm⁻¹] × L[μm] = rad
            self._x_display_label = "Δk · L_cr  (rad)"
        elif key == "pump/waist_um":
            # Pre-compute ξ↔waist transform so the checkbox can toggle at any time
            try:
                cfg0  = build_config(base_params)
                n_p   = cfg0["crystal"]["optics"]["np"]
                L_um  = float(base_params.get("length_mm", 10.0)) * 1000.0
                lp_um = float(base_params["pump"]["lambda_um"])
                _A    = L_um * lp_um / (2 * np.pi * n_p)
                self._x_transform = lambda w, A=_A: A / np.asarray(w, dtype=float)**2
                self.log.append(
                    f"[SWEEP] ξ toggle available: n_p={n_p:.4f}  "
                    f"L={L_um:.0f} μm  λ_p={lp_um:.4f} μm")
            except Exception as e:
                self.log.append(f"[SWEEP][WARN] ξ toggle not available: {e}")
            # Initial xlabel from current checkbox state
            if self._x_transform is not None and self.chk_xi_display.isChecked():
                self._x_display_label = "Focusing parameter ξ"
            else:
                self._x_display_label = f"{name}  ({unit})"
        else:
            self._x_display_scale = 1.0
            self._x_display_label = f"{name}  ({unit})"

        xlabel = self._x_display_label
        log_x  = self.rb_log.isChecked() and self._x_transform is None

        # ── Choose active panels ───────────────────────────────────────
        undepleted = bool(base_params.get("undepleted_pump", False))
        degenerate = bool(base_params.get("pump", {}).get("degenerate", False))
        # _PANELS order: 0=Signal, 1=Idler, 2=Pump
        active_panels = [p for i, p in enumerate(self._PANELS)
                         if not (i == 1 and degenerate)      # hide Idler if degenerate
                         and not (i == 2 and undepleted)]     # hide Pump if undepleted

        self.canvas.clear()   # re-creates plain axes from scratch
        for ai, (lk, rk, title, c_l, c_r, yl, yr) in enumerate(active_panels):
            ax = self.canvas.get_ax(ai)
            self.canvas._style_ax(ax)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(yl, fontsize=8, color=c_l)
            ax.tick_params(axis="y", labelcolor=c_l, labelsize=7)
            if log_x:
                ax.set_xscale("log")

            ax2 = ax.twinx()
            self.canvas._style_ax(ax2)
            ax2.set_ylabel(yr, fontsize=8, color=c_r)
            ax2.tick_params(axis="y", labelcolor=c_r, labelsize=7)
            self._axes_right.append(ax2)

            line_l, = ax.plot([], [], "o-",  color=c_l, linewidth=1.5, markersize=4)
            line_r, = ax2.plot([], [], "s--", color=c_r, linewidth=1.2, markersize=3)
            self._lines[lk] = line_l
            self._lines[rk] = line_r

        # Hide unused axes
        for ai in range(len(active_panels), len(self._PANELS)):
            self.canvas.get_ax(ai).set_visible(False)

        self.canvas.refresh()

        self.progress_bar.setMaximum(len(values))
        self.progress_bar.setValue(0)
        self.log.append(f"\n[SWEEP] {name}  from {values[0]:.4g} to "
                        f"{values[-1]:.4g}  ({len(values)} points)")
        if degenerate:
            self.log.append("[SWEEP] Degenerate — idler panel hidden (signal = idler).")
        if undepleted:
            self.log.append("[SWEEP] Undepleted pump — pump panel hidden.")

        project_root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        backend = "cuda" if self.rb_gpu.isChecked() else "cpu"

        self._thread = QThread()
        self._worker = SweepWorker(values, base_params, key,
                                   project_root=project_root, backend=backend)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_point_received)
        self._worker.log_line.connect(self.log.append)
        self._worker.finished.connect(self._on_sweep_finished)
        self._worker.finished.connect(self._thread.quit)

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._thread.start()

    def _on_xi_toggle(self, checked: bool):
        """Switch x-axis between beam waist (μm) and focusing parameter ξ on existing data."""
        if not self._x_data or self._x_transform is None:
            return
        raw = np.array(self._x_data)
        x      = self._x_transform(raw) if checked else raw
        xlabel = "Focusing parameter ξ" if checked else "Beam waist  (μm)"
        for k, line in self._lines.items():
            line.set_xdata(x)
            ax = line.axes
            if ax not in self._axes_right:   # update xlabel only on primary axes
                ax.set_xlabel(xlabel, fontsize=8)
            ax.relim()
            ax.autoscale_view()
        self.canvas.refresh()

    def _on_stop(self):
        if self._worker:
            self._worker.stop()
        self.log.append("[SWEEP] Stopped by user.")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_point_received(self, index, x_val, result):
        self._x_data.append(x_val)
        for k in self._y_data:
            self._y_data[k].append(result.get(k, 0.0))

        raw = np.array(self._x_data)
        if self._x_transform is not None and self.chk_xi_display.isChecked():
            x = self._x_transform(raw)
        else:
            x = raw * self._x_display_scale
        for k, line in self._lines.items():
            y = np.array(self._y_data[k])
            line.set_data(x, y)
            ax = line.axes
            ax.relim(); ax.autoscale_view()
        self.canvas.refresh()

        n = self.progress_bar.maximum()
        self.progress_bar.setValue(index + 1)
        self.lbl_prog.setText(f"Point {index+1} / {n}")

    def _on_sweep_finished(self):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        n = len(self._x_data)
        self.lbl_prog.setText(f"Done — {n} points")
        self.log.append(f"[SWEEP] Finished. {n} points computed.")

    def _on_export(self):
        if not self._x_data:
            self.log.append("[SWEEP] No data to export.")
            return
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Export sweep data", "sweep_results.txt",
            "Text files (*.txt);;CSV (*.csv);;All files (*)")
        if not path:
            return
        idx  = self.cb_param.currentIndex()
        name, key, unit, *_ = SWEEP_PARAMS[idx]

        # Build key→label map from panel definitions
        _key_label = {}
        for lk, rk, _, _, _, yl, yr in self._PANELS:
            _key_label[lk] = yl
            _key_label[rk] = yr

        # Only export the y-quantities currently being plotted
        active_keys = list(self._lines.keys())
        col_labels  = "  ".join(f"{_key_label.get(k, k):>20}" for k in active_keys)

        # First column: raw swept-parameter values (never transformed)
        x_export       = np.array(self._x_data)
        x_header_label = f"{name} ({unit})"

        header = (f"# TWM parameter sweep\n"
                  f"# X: {x_header_label}\n"
                  f"{'X':>20}  {col_labels}")
        cols = [x_export] + [self._y_data[k] for k in active_keys]
        data = np.column_stack(cols)
        np.savetxt(path, data, header=header, fmt="%20.8e", comments="")
        self.log.append(f"[SWEEP] Exported to {path}")
