"""
Full Simulation tab — (3+1)D pulsed + focused Gaussian mode.

Always pulsed (temporal envelope) + focused Gaussian beam.
Temporal window auto-computed as 6×FWHM (overridable).
"""

import os

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QGroupBox, QLabel, QComboBox, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPushButton, QScrollArea, QTextEdit,
    QFrame, QGridLayout, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal, QLocale

from ..core_py.crystal_db import get_db

_LOCALE_C = QLocale(QLocale.Language.C)

# GPU memory budget constants (bytes)
_BYTES_PER_COMPLEX = 8    # 2 × float32
_N_COMPLEX_ARRAYS  = 22   # solver stages + aux + field buffers
_N_REAL_ARRAYS     = 3    # vol_p, vol_s, vol_i (NZ×NY×NX)
_BYTES_PER_REAL    = 4


class FullSimulationTab(QWidget):
    run_requested  = pyqtSignal(dict)
    stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        main = QHBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(splitter)

        # ── Column 1: process + pump + signal/idler ────────────────────
        scroll1 = QScrollArea()
        scroll1.setWidgetResizable(True)
        scroll1.setFrameShape(QFrame.Shape.NoFrame)
        scroll1.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left = QWidget()
        left.setMinimumWidth(280); left.setMaximumWidth(350)
        lv = QVBoxLayout(left)
        lv.setSpacing(10); lv.setContentsMargins(6, 6, 6, 6)
        lv.addWidget(self._build_process_group())
        lv.addWidget(self._build_pump_group())
        lv.addWidget(self._build_signal_group())
        lv.addStretch()
        scroll1.setWidget(left)
        splitter.addWidget(scroll1)

        # ── Column 2: crystal + grid + temporal window ─────────────────
        scroll2 = QScrollArea()
        scroll2.setWidgetResizable(True)
        scroll2.setFrameShape(QFrame.Shape.NoFrame)
        scroll2.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        mid = QWidget()
        mid.setMinimumWidth(260); mid.setMaximumWidth(350)
        mv = QVBoxLayout(mid)
        mv.setSpacing(10); mv.setContentsMargins(6, 6, 6, 6)
        mv.addWidget(self._build_crystal_group())
        mv.addWidget(self._build_grid_group())
        mv.addWidget(self._build_temporal_group())
        mv.addStretch()
        scroll2.setWidget(mid)
        splitter.addWidget(scroll2)

        # ── Column 3: log + run buttons ────────────────────────────────
        right = QWidget()
        right.setMinimumWidth(360); right.setMaximumWidth(620)
        rv = QVBoxLayout(right)
        rv.setSpacing(6); rv.setContentsMargins(4, 4, 4, 4)
        self.log = QTextEdit()
        self.log.setObjectName("logOutput")
        self.log.setReadOnly(True)
        self.log.append("Full (3+1)D simulation ready.")
        rv.addWidget(self.log, stretch=1)
        btn_row = QVBoxLayout()
        self.btn_run  = QPushButton("  Run (GPU / CUDA)  ")
        self.btn_run.setObjectName("runButton")
        self.btn_mpi  = QPushButton("  Run (CPU / OpenMP)  ")
        self.btn_mpi.setObjectName("runButton")
        self.btn_stop = QPushButton("  Stop  ")
        self.btn_stop.setObjectName("stopButton")
        self.btn_stop.setEnabled(False)
        for b in (self.btn_run, self.btn_mpi, self.btn_stop):
            btn_row.addWidget(b)
        rv.addLayout(btn_row)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 2)

        for sb in self.findChildren(QDoubleSpinBox):
            sb.setLocale(_LOCALE_C)

        # Radio buttons / checkboxes are stretched to the full column width by
        # the layout, but Qt's stylesheet-based hit-testing only registers
        # clicks within their natural (label-sized) rect. Pin them to their
        # natural width so the whole visible control is clickable.
        for w in self.findChildren(QCheckBox):
            w.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)

        # ── Connections ────────────────────────────────────────────────
        self.btn_run.clicked.connect(self._on_run_gpu)
        self.btn_mpi.clicked.connect(self._on_run_cpu)
        self.btn_stop.clicked.connect(self.stop_requested)
        self.sb_waist.valueChanged.connect(self._auto_lxly)
        self.sb_fwhm.valueChanged.connect(self._on_fwhm_changed)
        self.chk_twindow_auto.toggled.connect(self._on_twindow_auto_toggle)
        self.cb_process.currentTextChanged.connect(self._on_process_changed)

        self._auto_lxly()
        # example_6 default: LX=LY=500 μm = 0.5 mm
        self.sb_lx.setValue(0.500)
        self.sb_ly.setValue(0.500)
        self._update_twindow_auto()
        self._update_memory_estimate()

    # ── Group builders ─────────────────────────────────────────────────

    def _build_process_group(self):
        gb = QGroupBox("Nonlinear Process")
        g  = QGridLayout(gb); g.setSpacing(6)
        g.addWidget(QLabel("Process"), 0, 0)
        self.cb_process = QComboBox()
        self.cb_process.addItems(["OPG", "SHG", "SFG", "DFG"])
        g.addWidget(self.cb_process, 0, 1)
        self.chk_undepleted = QCheckBox("Undepleted pump")
        self.chk_undepleted.setToolTip("Fix pump amplitude — undepleted pump approximation")
        g.addWidget(self.chk_undepleted, 1, 0, 1, 2)
        return gb

    def _build_pump_group(self):
        gb = QGroupBox("Pump — Focused Gaussian Pulse")
        g  = QGridLayout(gb); g.setSpacing(6)

        g.addWidget(QLabel("λ pump (μm)"), 0, 0)
        self.sb_lp = QDoubleSpinBox()
        self.sb_lp.setRange(0.1, 20); self.sb_lp.setDecimals(4); self.sb_lp.setValue(1.064)
        g.addWidget(self.sb_lp, 0, 1)

        g.addWidget(QLabel("Power (W)"), 1, 0)
        self.sb_power = QDoubleSpinBox()
        self.sb_power.setRange(1e-9, 1e9); self.sb_power.setDecimals(4); self.sb_power.setValue(1.0)
        g.addWidget(self.sb_power, 1, 1)

        g.addWidget(QLabel("Waist w₀ (μm)"), 2, 0)
        self.sb_waist = QDoubleSpinBox()
        self.sb_waist.setRange(0.1, 10000); self.sb_waist.setValue(60.0)
        g.addWidget(self.sb_waist, 2, 1)

        g.addWidget(QLabel("FWHM (ps)"), 3, 0)
        self.sb_fwhm = QDoubleSpinBox()
        self.sb_fwhm.setRange(0.001, 1e5); self.sb_fwhm.setValue(0.2)
        g.addWidget(self.sb_fwhm, 3, 1)

        g.addWidget(QLabel("Focal point"), 4, 0)
        self.cb_focus = QComboBox()
        self.cb_focus.addItems(["Crystal centre (0.5·L)", "Crystal start (z=0)"])
        g.addWidget(self.cb_focus, 4, 1)

        return gb

    def _build_signal_group(self):
        gb = QGroupBox("Signal / Idler Wavelengths")
        g  = QGridLayout(gb); g.setSpacing(6)

        g.addWidget(QLabel("λ signal (μm)"), 0, 0)
        self.sb_ls = QDoubleSpinBox()
        self.sb_ls.setRange(0.1, 20); self.sb_ls.setDecimals(4); self.sb_ls.setValue(1.4500)
        g.addWidget(self.sb_ls, 0, 1)

        g.addWidget(QLabel("λ idler (μm)"), 1, 0)
        self.sb_li = QDoubleSpinBox()
        self.sb_li.setRange(0.1, 20); self.sb_li.setDecimals(4); self.sb_li.setValue(3.9969)
        g.addWidget(self.sb_li, 1, 1)

        self.chk_degen = QCheckBox("Degenerate  (signal = idler)")
        g.addWidget(self.chk_degen, 2, 0, 1, 2)

        note = QLabel("Signal/idler start as vacuum noise.")
        note.setObjectName("unitLabel"); note.setWordWrap(True)
        g.addWidget(note, 3, 0, 1, 2)

        return gb

    def _build_crystal_group(self):
        gb = QGroupBox("Crystal")
        g  = QGridLayout(gb); g.setSpacing(6)

        g.addWidget(QLabel("Material"), 0, 0)
        self.cb_crystal = QComboBox()
        self.cb_crystal.addItems(["MgO:PPLN", "MgO:sPPLT", "BBO"])
        g.addWidget(self.cb_crystal, 0, 1)

        g.addWidget(QLabel("Length L (mm)"), 1, 0)
        self.sb_length = QDoubleSpinBox()
        self.sb_length.setRange(0.1, 200); self.sb_length.setValue(10.0)
        g.addWidget(self.sb_length, 1, 1)

        g.addWidget(QLabel("Temperature (°C)"), 2, 0)
        self.sb_temp = QDoubleSpinBox()
        self.sb_temp.setRange(-100, 500); self.sb_temp.setValue(27)
        g.addWidget(self.sb_temp, 2, 1)

        g.addWidget(QLabel("Λ grating (μm)"), 3, 0)
        self.sb_lambda = QDoubleSpinBox()
        self.sb_lambda.setRange(0.1, 100); self.sb_lambda.setDecimals(3)
        self.sb_lambda.setValue(6.920)
        g.addWidget(self.sb_lambda, 3, 1)

        # Birefringent-uniaxial crystals (e.g. BBO) have no grating — instead
        # every field's axis (o, or e mixed at this angle) is set by the PM
        # type + angle found in Phase Matching → Birefringent and sent here
        # via "Load into Simulation". Both are read-only summaries of that
        # choice; re-run "Load into Simulation" to change them.
        g.addWidget(QLabel("PM type (birefringent)"), 4, 0)
        self.lbl_pm_type = QLabel("—")
        self.lbl_pm_type.setObjectName("valueLabel")
        g.addWidget(self.lbl_pm_type, 4, 1)

        g.addWidget(QLabel("θ_pm (deg, birefringent)"), 5, 0)
        self.lbl_theta = QLabel("—")
        self.lbl_theta.setObjectName("valueLabel")
        g.addWidget(self.lbl_theta, 5, 1)

        self.chk_force_dk = QCheckBox("Force Δk (μm⁻¹)")
        g.addWidget(self.chk_force_dk, 6, 0)
        self.sb_dk_forced = QDoubleSpinBox()
        self.sb_dk_forced.setRange(-10.0, 10.0); self.sb_dk_forced.setDecimals(6)
        self.sb_dk_forced.setValue(0.0)
        self.sb_dk_forced.setEnabled(False)
        g.addWidget(self.sb_dk_forced, 6, 1)
        self.chk_force_dk.toggled.connect(self.sb_dk_forced.setEnabled)

        g.addWidget(QLabel("LX (mm)"), 7, 0)
        self.sb_lx = QDoubleSpinBox()
        self.sb_lx.setRange(0.01, 100); self.sb_lx.setDecimals(3); self.sb_lx.setValue(0.5)
        g.addWidget(self.sb_lx, 7, 1)

        g.addWidget(QLabel("LY (mm)"), 8, 0)
        self.sb_ly = QDoubleSpinBox()
        self.sb_ly.setRange(0.01, 100); self.sb_ly.setDecimals(3); self.sb_ly.setValue(0.5)
        g.addWidget(self.sb_ly, 8, 1)

        note = QLabel("LX, LY auto-set from waist (6·w₀ each side)")
        note.setObjectName("unitLabel"); note.setWordWrap(True)
        g.addWidget(note, 9, 0, 1, 2)

        self._pm_type_value: str | None = None
        self._theta_deg_value: float | None = None
        self.cb_crystal.currentIndexChanged.connect(self._on_crystal_changed)

        return gb

    def _on_crystal_changed(self, _idx=None):
        db = get_db()
        cr = db.get(self.cb_crystal.currentText()) or {}
        is_birefringent = cr.get("type", "") == "Birefringent-uniaxial"
        self.sb_lambda.setEnabled(not is_birefringent)
        self.lbl_pm_type.setEnabled(is_birefringent)
        self.lbl_theta.setEnabled(is_birefringent)
        if not is_birefringent:
            self._pm_type_value = None
            self._theta_deg_value = None
            self.lbl_pm_type.setText("—")
            self.lbl_theta.setText("—")

    def _build_grid_group(self):
        gb = QGroupBox("Numerical Grid")
        g  = QGridLayout(gb); g.setSpacing(5)

        for col, lbl in enumerate(["NX", "NY", "NZ", "NT"]):
            g.addWidget(QLabel(lbl), 0, col)

        self.sb_nx = QSpinBox(); self.sb_nx.setRange(4, 1024);    self.sb_nx.setValue(64)
        self.sb_ny = QSpinBox(); self.sb_ny.setRange(4, 1024);    self.sb_ny.setValue(64)
        self.sb_nz = QSpinBox(); self.sb_nz.setRange(10, 1000000); self.sb_nz.setValue(1000)
        self.sb_nt = QSpinBox(); self.sb_nt.setRange(2, 4096);    self.sb_nt.setValue(256)

        for col, sb in enumerate([self.sb_nx, self.sb_ny, self.sb_nz, self.sb_nt]):
            g.addWidget(sb, 1, col)

        self.lbl_mem = QLabel("Memory: —")
        self.lbl_mem.setObjectName("unitLabel")
        self.lbl_mem.setWordWrap(True)
        g.addWidget(self.lbl_mem, 2, 0, 1, 4)

        for sb in [self.sb_nx, self.sb_ny, self.sb_nz, self.sb_nt]:
            sb.valueChanged.connect(self._update_memory_estimate)

        lbl_gpu = QLabel("Only for GPU kernels:")
        lbl_gpu.setObjectName("unitLabel")
        g.addWidget(lbl_gpu, 3, 0, 1, 4)
        for col, lbl in enumerate(["BLKX", "BLKY", "BLKT"]):
            g.addWidget(QLabel(lbl), 4, col)
        self.sb_blkx = QSpinBox(); self.sb_blkx.setRange(1, 1024); self.sb_blkx.setValue(16)
        self.sb_blky = QSpinBox(); self.sb_blky.setRange(1, 1024); self.sb_blky.setValue(16)
        self.sb_blkt = QSpinBox(); self.sb_blkt.setRange(1, 1024); self.sb_blkt.setValue(16)
        for tip in (self.sb_blkx, self.sb_blky, self.sb_blkt):
            tip.setToolTip("CUDA thread-block dimensions (-DBLKX/-DBLKY/-DBLKT).\n"
                           "Compile-time only; ignored by the CPU (OpenMP) backend.")
        for col, sb in enumerate([self.sb_blkx, self.sb_blky, self.sb_blkt]):
            g.addWidget(sb, 5, col)

        lbl_cpu = QLabel("Only for CPU (OpenMP) backend:")
        lbl_cpu.setObjectName("unitLabel")
        g.addWidget(lbl_cpu, 6, 0, 1, 4)
        g.addWidget(QLabel("Threads"), 7, 0)
        self.sb_omp_threads = QSpinBox()
        _ncpu = os.cpu_count() or 1
        self.sb_omp_threads.setRange(1, _ncpu)
        self.sb_omp_threads.setValue(_ncpu)
        self.sb_omp_threads.setToolTip(
            "OMP_NUM_THREADS for the CPU run. Defaults to all logical cores "
            f"detected on this machine ({_ncpu}). Ignored by the GPU backend.")
        g.addWidget(self.sb_omp_threads, 7, 1)

        return gb

    def _build_temporal_group(self):
        gb = QGroupBox("Temporal Window")
        g  = QGridLayout(gb); g.setSpacing(5)

        g.addWidget(QLabel("T_window (ps)"), 0, 0)
        self.sb_twindow = QDoubleSpinBox()
        self.sb_twindow.setRange(0.001, 1e6); self.sb_twindow.setDecimals(3)
        self.sb_twindow.setValue(3.0)
        self.sb_twindow.setEnabled(True)
        g.addWidget(self.sb_twindow, 0, 1)

        self.chk_twindow_auto = QCheckBox("Auto  (6 × FWHM)")
        self.chk_twindow_auto.setChecked(False)
        g.addWidget(self.chk_twindow_auto, 1, 0, 1, 2)

        note = QLabel("dt = T_window / NT")
        note.setObjectName("unitLabel")
        g.addWidget(note, 2, 0, 1, 2)

        return gb

    # ── Logic ──────────────────────────────────────────────────────────

    def _auto_lxly(self):
        waist_um = self.sb_waist.value()
        half_mm  = max(6.0 * waist_um * 1e-3, 0.25)
        val      = round(2.0 * half_mm, 3)
        for sb in (self.sb_lx, self.sb_ly):
            sb.blockSignals(True); sb.setValue(val); sb.blockSignals(False)

    def _update_twindow_auto(self):
        if self.chk_twindow_auto.isChecked():
            self.sb_twindow.blockSignals(True)
            self.sb_twindow.setValue(round(6.0 * self.sb_fwhm.value(), 4))
            self.sb_twindow.blockSignals(False)

    def _on_fwhm_changed(self):
        self._update_twindow_auto()

    def _on_twindow_auto_toggle(self, checked):
        self.sb_twindow.setEnabled(not checked)
        if checked:
            self._update_twindow_auto()

    def _on_process_changed(self, proc):
        shg = (proc == "SHG")
        self.sb_ls.setEnabled(not shg)
        self.sb_li.setEnabled(not shg)

    def _update_memory_estimate(self):
        nx = self.sb_nx.value()
        ny = self.sb_ny.value()
        nz = self.sb_nz.value()
        nt = self.sb_nt.value()

        # ~22 complex NX×NY×NT arrays (solver stages + field buffers)
        # + 3 real NX×NY×NZ volume arrays
        field_b = _N_COMPLEX_ARRAYS * nx * ny * nt * _BYTES_PER_COMPLEX
        vol_b   = _N_REAL_ARRAYS    * nx * ny * nz * _BYTES_PER_REAL
        total_gb = (field_b + vol_b) / 1024**3

        if total_gb > 10:
            color = "color: #f44336;"
            warn  = " ⚠ EXCEEDS 10 GB"
        elif total_gb > 6:
            color = "color: #ff9800;"
            warn  = " ⚠ Large — verify GPU RAM"
        else:
            color = "color: #00e676;"
            warn  = ""

        self.lbl_mem.setText(
            f"GPU ≈ {total_gb:.2f} GB  ({nx}×{ny}×{nz}×{nt}){warn}")
        self.lbl_mem.setStyleSheet(color)

    # ── Button slots ───────────────────────────────────────────────────

    def _on_run_gpu(self):
        self.log.append("[GPU] Building configuration...")
        self.run_requested.emit(self._collect_params(backend="cuda"))

    def _on_run_cpu(self):
        self.log.append("[CPU] Building configuration...")
        self.run_requested.emit(self._collect_params(backend="cpu"))

    def _collect_params(self, backend="cuda") -> dict:
        focus_text = self.cb_focus.currentText()
        focal_preset = "Crystal centre (0.5·L)" if "centre" in focus_text else "Crystal start"

        return {
            "backend":     backend,
            "omp_threads": self.sb_omp_threads.value(),
            "process":  self.cb_process.currentText(),
            "crystal":  self.cb_crystal.currentText(),
            "length_mm":       self.sb_length.value(),
            "temperature":     self.sb_temp.value(),
            "grating_um":      self.sb_lambda.value(),
            **({"pm_type": self._pm_type_value} if self._pm_type_value else {}),
            **({"theta_deg": self._theta_deg_value} if self._theta_deg_value is not None else {}),
            **({"dk": self.sb_dk_forced.value()} if self.chk_force_dk.isChecked() else {}),
            "lx_mm":           self.sb_lx.value(),
            "ly_mm":           self.sb_ly.value(),
            "undepleted_pump": self.chk_undepleted.isChecked(),
            "pump": {
                "lambda_um":   self.sb_lp.value(),
                "regime":      "Pulsed",
                "profile":     "Focused Gaussian",
                "power_W":     self.sb_power.value(),
                "waist_um":    self.sb_waist.value(),
                "fwhm_ps":     self.sb_fwhm.value(),
                "focal_preset": focal_preset,
            },
            "signal": {
                "lambda_um": self.sb_ls.value(),
                "mode":      "noise",
            },
            "idler": {
                "lambda_um":  self.sb_li.value(),
                "degenerate": self.chk_degen.isChecked(),
                "mode":       "noise",
            },
            "grid": {
                "NX": self.sb_nx.value(),
                "NY": self.sb_ny.value(),
                "NZ": self.sb_nz.value(),
                "NT": self.sb_nt.value(),
                "BLKX": self.sb_blkx.value(),
                "BLKY": self.sb_blky.value(),
                "BLKT": self.sb_blkt.value(),
            },
            "thermal":      {"enabled": False},
            "t_window_ps":  self.sb_twindow.value(),
        }

    # ── Public helpers ─────────────────────────────────────────────────

    def append_log(self, text: str):
        self.log.append(text)

    def load_pm_wavelengths(self, lp: float, ls: float, li: float,
                            T: float, Lambda: float):
        self.sb_lp.setValue(lp)
        self.sb_ls.setValue(ls)
        self.sb_li.setValue(li)
        self.sb_temp.setValue(T)
        self.sb_lambda.setValue(Lambda)

    def load_pm_birefringent(self, crystal: str, process: str, lp: float, ls: float,
                             li: float, T: float, pm_type: str, theta_deg: float):
        idx = self.cb_crystal.findText(crystal)
        if idx >= 0:
            self.cb_crystal.setCurrentIndex(idx)
        idx_p = self.cb_process.findText(process)
        if idx_p >= 0:
            self.cb_process.setCurrentIndex(idx_p)
        self.sb_lp.setValue(lp)
        self.sb_ls.setValue(ls)
        self.sb_li.setValue(li)
        self.sb_temp.setValue(T)
        if abs(ls - li) > 1e-6:
            self.chk_degen.setChecked(False)
        self._pm_type_value   = pm_type
        self._theta_deg_value = theta_deg
        self.lbl_pm_type.setText(pm_type)
        self.lbl_theta.setText(f"{theta_deg:.3f}°")
