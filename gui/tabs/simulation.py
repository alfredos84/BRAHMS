from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QGroupBox, QLabel, QComboBox, QDoubleSpinBox, QSpinBox,
    QRadioButton, QCheckBox, QPushButton, QScrollArea,
    QButtonGroup, QTextEdit, QFrame, QSizePolicy, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, QLocale
from PyQt6.QtGui import QFont

_LOCALE_C = QLocale(QLocale.Language.C)   # always uses '.' as decimal separator


class FieldGroupBox(QGroupBox):
    """Reusable group for configuring signal or idler field."""

    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        self.rb_noise  = QRadioButton("Vacuum noise")
        self.rb_inject = QRadioButton("Inject field")
        self.rb_noise.setChecked(True)

        bg = QButtonGroup(self)
        bg.addButton(self.rb_noise)
        bg.addButton(self.rb_inject)

        layout.addWidget(self.rb_noise)
        layout.addWidget(self.rb_inject)

        self.inject_widget = QWidget()
        ig = QGridLayout(self.inject_widget)
        ig.setContentsMargins(0, 4, 0, 0)
        ig.setSpacing(4)
        ig.addWidget(QLabel("Power (W)"), 0, 0)
        self.sb_power = QDoubleSpinBox()
        self.sb_power.setLocale(_LOCALE_C)
        self.sb_power.setRange(1e-9, 1e6); self.sb_power.setDecimals(4)
        self.sb_power.setValue(0.001)
        ig.addWidget(self.sb_power, 0, 1)
        ig.addWidget(QLabel("Waist (μm)"), 1, 0)
        self.sb_waist = QDoubleSpinBox()
        self.sb_waist.setLocale(_LOCALE_C)
        self.sb_waist.setRange(0.1, 10000); self.sb_waist.setValue(30)
        ig.addWidget(self.sb_waist, 1, 1)
        ig.addWidget(QLabel("FWHM (ps)"), 2, 0)
        self.sb_fwhm = QDoubleSpinBox()
        self.sb_fwhm.setLocale(_LOCALE_C)
        self.sb_fwhm.setRange(0.001, 1e5); self.sb_fwhm.setValue(1.0)
        ig.addWidget(self.sb_fwhm, 2, 1)
        self.inject_widget.hide()

        layout.addWidget(self.inject_widget)

        self.rb_inject.toggled.connect(self.inject_widget.setVisible)
        self.rb_noise.toggled.connect(
            lambda checked: self.inject_widget.setVisible(not checked))


class SimulationTab(QWidget):
    run_requested  = pyqtSignal(dict)
    stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)
        main_layout.addWidget(splitter)

        # ── COLUMN 1: scrollable main parameter panel ──────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left = QWidget()
        left.setMinimumWidth(290)
        left.setMaximumWidth(360)
        lv = QVBoxLayout(left)
        lv.setSpacing(10)
        lv.setContentsMargins(6, 6, 6, 6)

        lv.addWidget(self._build_process_group())
        lv.addWidget(self._build_pump_group())
        lv.addWidget(self._build_signal_group())
        lv.addWidget(self._build_idler_group())
        lv.addStretch()

        scroll.setWidget(left)
        splitter.addWidget(scroll)

        # ── COLUMN 2: crystal + grid + output + thermal ───────────────
        thermal_scroll = QScrollArea()
        thermal_scroll.setWidgetResizable(True)
        thermal_scroll.setFrameShape(QFrame.Shape.NoFrame)
        thermal_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        thermal_col = QWidget()
        thermal_col.setMinimumWidth(260)
        thermal_col.setMaximumWidth(360)
        tv = QVBoxLayout(thermal_col)
        tv.setSpacing(10)
        tv.setContentsMargins(6, 6, 6, 6)
        tv.addWidget(self._build_crystal_group())
        tv.addWidget(self._build_grid_group())
        tv.addWidget(self._build_output_group())
        tv.addWidget(self._build_thermal_group())
        tv.addStretch()

        thermal_scroll.setWidget(thermal_col)
        splitter.addWidget(thermal_scroll)

        # ── COLUMN 3: log + run buttons ────────────────────────────────
        right = QWidget()
        right.setMinimumWidth(360)
        right.setMaximumWidth(620)
        rv = QVBoxLayout(right)
        rv.setSpacing(6)
        rv.setContentsMargins(4, 4, 4, 4)

        self.log = QTextEdit()
        self.log.setObjectName("logOutput")
        self.log.setReadOnly(True)
        self.log.append("BRAHMS ready.")
        rv.addWidget(self.log, stretch=1)

        btn_row = QVBoxLayout()
        self.btn_run  = QPushButton("  Run (GPU / CUDA)  ")
        self.btn_run.setObjectName("runButton")
        self.btn_mpi  = QPushButton("  Run (CPU / OpenMP)  ")
        self.btn_mpi.setObjectName("runButton")
        self.btn_stop = QPushButton("  Stop  ")
        self.btn_stop.setObjectName("stopButton")
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_mpi)
        btn_row.addWidget(self.btn_stop)
        rv.addLayout(btn_row)

        splitter.addWidget(right)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 2)

        # Apply C locale (period decimal) to all remaining QDoubleSpinBox children
        for sb in self.findChildren(QDoubleSpinBox):
            sb.setLocale(_LOCALE_C)

        # ── Signal connections ─────────────────────────────────────────
        self.btn_run.clicked.connect(self._on_run_gpu)
        self.btn_mpi.clicked.connect(self._on_run_mpi)
        self.btn_stop.clicked.connect(self.stop_requested)
        self.cb_regime.currentTextChanged.connect(self._update_visibility)
        self.cb_profile.currentTextChanged.connect(self._update_visibility)
        self.cb_process.currentTextChanged.connect(self._update_visibility)
        self.chk_degen.toggled.connect(self._on_degenerate)
        self.sb_waist.valueChanged.connect(self._auto_lxly)
        self._update_visibility()
        self._auto_lxly()

    # ── Group builders ─────────────────────────────────────────────────
    def _build_process_group(self):
        gb = QGroupBox("Nonlinear Process")
        g  = QGridLayout(gb)
        g.setSpacing(6)
        g.addWidget(QLabel("Process"), 0, 0)
        self.cb_process = QComboBox()
        self.cb_process.addItems(["SHG", "SFG", "OPG", "DFG"])
        g.addWidget(self.cb_process, 0, 1)
        return gb

    def _build_crystal_group(self):
        gb = QGroupBox("Crystal")
        g  = QGridLayout(gb)
        g.setSpacing(6)

        g.addWidget(QLabel("Material"), 0, 0)
        self.cb_crystal = QComboBox()
        self.cb_crystal.addItems(["MgO:PPLN", "PPLN", "MgO:sPPLT", "ZGP"])
        g.addWidget(self.cb_crystal, 0, 1)

        g.addWidget(QLabel("Length L (mm)"), 1, 0)
        self.sb_length = QDoubleSpinBox()
        self.sb_length.setLocale(_LOCALE_C)
        self.sb_length.setRange(0.1, 200); self.sb_length.setValue(30)
        g.addWidget(self.sb_length, 1, 1)

        g.addWidget(QLabel("Temperature (°C)"), 2, 0)
        self.sb_temp = QDoubleSpinBox()
        self.sb_temp.setLocale(_LOCALE_C)
        self.sb_temp.setRange(-100, 500); self.sb_temp.setValue(27)
        g.addWidget(self.sb_temp, 2, 1)

        g.addWidget(QLabel("Λ grating (μm)"), 3, 0)
        self.sb_lambda = QDoubleSpinBox()
        self.sb_lambda.setLocale(_LOCALE_C)
        self.sb_lambda.setRange(0.1, 100); self.sb_lambda.setDecimals(3)
        self.sb_lambda.setValue(6.99)
        g.addWidget(self.sb_lambda, 3, 1)

        self.chk_force_dk = QCheckBox("Force Δk (μm⁻¹)")
        g.addWidget(self.chk_force_dk, 4, 0)
        self.sb_dk_forced = QDoubleSpinBox()
        self.sb_dk_forced.setLocale(_LOCALE_C)
        self.sb_dk_forced.setRange(-10.0, 10.0); self.sb_dk_forced.setDecimals(6)
        self.sb_dk_forced.setValue(0.0)
        self.sb_dk_forced.setEnabled(False)
        g.addWidget(self.sb_dk_forced, 4, 1)
        self.chk_force_dk.toggled.connect(self.sb_dk_forced.setEnabled)

        g.addWidget(QLabel("LX (mm)"), 5, 0)
        self.sb_lx = QDoubleSpinBox()
        self.sb_lx.setLocale(_LOCALE_C)
        self.sb_lx.setRange(0.01, 100); self.sb_lx.setDecimals(3)
        self.sb_lx.setValue(0.5)
        g.addWidget(self.sb_lx, 5, 1)

        g.addWidget(QLabel("LY (mm)"), 6, 0)
        self.sb_ly = QDoubleSpinBox()
        self.sb_ly.setLocale(_LOCALE_C)
        self.sb_ly.setRange(0.01, 100); self.sb_ly.setDecimals(3)
        self.sb_ly.setValue(0.5)
        g.addWidget(self.sb_ly, 6, 1)

        note = QLabel("LX, LY auto-set from waist (6·w₀ each side)")
        note.setObjectName("unitLabel")
        note.setWordWrap(True)
        g.addWidget(note, 7, 0, 1, 2)

        return gb

    def _build_pump_group(self):
        gb = QGroupBox("Pump Field")
        g  = QGridLayout(gb)
        g.setSpacing(6)
        g.addWidget(QLabel("λ pump (μm)"), 0, 0)
        self.sb_lp = QDoubleSpinBox()
        self.sb_lp.setLocale(_LOCALE_C)
        self.sb_lp.setRange(0.1, 20); self.sb_lp.setDecimals(4)
        self.sb_lp.setValue(1.064)
        g.addWidget(self.sb_lp, 0, 1)
        g.addWidget(QLabel("Regime"), 1, 0)
        self.cb_regime = QComboBox()
        self.cb_regime.addItems(["CW", "Pulsed"])
        g.addWidget(self.cb_regime, 1, 1)
        g.addWidget(QLabel("Profile"), 2, 0)
        self.cb_profile = QComboBox()
        self.cb_profile.addItems(["Focused Gaussian", "Plane wave"])
        g.addWidget(self.cb_profile, 2, 1)
        g.addWidget(QLabel("Power (W)"), 3, 0)
        self.sb_power = QDoubleSpinBox()
        self.sb_power.setLocale(_LOCALE_C)
        self.sb_power.setRange(1e-9, 1e6); self.sb_power.setDecimals(4)
        self.sb_power.setValue(10.0)
        g.addWidget(self.sb_power, 3, 1)
        g.addWidget(QLabel("Waist (μm)"), 4, 0)
        self.sb_waist = QDoubleSpinBox()
        self.sb_waist.setLocale(_LOCALE_C)
        self.sb_waist.setRange(0.1, 10000); self.sb_waist.setValue(30)
        g.addWidget(self.sb_waist, 4, 1)
        self.lbl_fwhm = QLabel("FWHM (ps)")
        self.sb_fwhm  = QDoubleSpinBox()
        self.sb_fwhm.setLocale(_LOCALE_C)
        self.sb_fwhm.setRange(0.001, 1e5); self.sb_fwhm.setValue(0.2)
        g.addWidget(self.lbl_fwhm, 5, 0)
        g.addWidget(self.sb_fwhm, 5, 1)
        self.lbl_focus = QLabel("Focal position")
        self.cb_focus  = QComboBox()
        self.cb_focus.addItems(["Crystal centre (0.5·L)", "Crystal start", "Custom"])
        g.addWidget(self.lbl_focus, 6, 0)
        g.addWidget(self.cb_focus, 6, 1)
        self.chk_undepleted = QCheckBox("Undepleted pump")
        self.chk_undepleted.setToolTip("Fix pump amplitude (dAp/dz = 0) — undepleted pump approximation")
        g.addWidget(self.chk_undepleted, 7, 0, 1, 2)
        return gb

    def _build_signal_group(self):
        self.signal_gb = FieldGroupBox("Signal Field")
        g = self.signal_gb.layout()
        self.lbl_ls = QLabel("λ signal (μm)")
        self.sb_ls  = QDoubleSpinBox()
        self.sb_ls.setLocale(_LOCALE_C)
        self.sb_ls.setRange(0.1, 20); self.sb_ls.setDecimals(4)
        self.sb_ls.setValue(0.532)
        row = QHBoxLayout()
        row.addWidget(self.lbl_ls)
        row.addWidget(self.sb_ls)
        g.insertLayout(0, row)
        return self.signal_gb

    def _build_idler_group(self):
        # Wrapper keeps chk_degen OUTSIDE idler_gb so disabling the group
        # does not propagate to the checkbox (Qt child-disable propagation).
        wrapper = QWidget()
        wl = QVBoxLayout(wrapper)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(4)

        self.chk_degen = QCheckBox("Degenerate  (signal = idler)")
        wl.addWidget(self.chk_degen)

        self.idler_gb = FieldGroupBox("Idler Field")
        g = self.idler_gb.layout()
        self.lbl_li = QLabel("λ idler (μm)")
        self.sb_li  = QDoubleSpinBox()
        self.sb_li.setLocale(_LOCALE_C)
        self.sb_li.setRange(0.1, 20); self.sb_li.setDecimals(4)
        self.sb_li.setValue(0.532)
        row = QHBoxLayout()
        row.addWidget(self.lbl_li)
        row.addWidget(self.sb_li)
        g.insertLayout(0, row)
        wl.addWidget(self.idler_gb)
        return wrapper

    def _build_grid_group(self):
        gb = QGroupBox("Numerical Grid")
        g  = QGridLayout(gb)
        g.setSpacing(5)
        for col, label in enumerate(["NX", "NY", "NZ", "NT"]):
            g.addWidget(QLabel(label), 0, col)
        self.sb_nx = QSpinBox(); self.sb_nx.setRange(4, 1024); self.sb_nx.setValue(64)
        self.sb_ny = QSpinBox(); self.sb_ny.setRange(4, 1024); self.sb_ny.setValue(64)
        self.sb_nz = QSpinBox(); self.sb_nz.setRange(10, 1000000); self.sb_nz.setValue(200)
        self.sb_nt = QSpinBox(); self.sb_nt.setRange(1, 1024); self.sb_nt.setValue(1)
        for col, sb in enumerate([self.sb_nx, self.sb_ny, self.sb_nz, self.sb_nt]):
            g.addWidget(sb, 1, col)
        self.lbl_mem = QLabel("Memory: ~— MB")
        self.lbl_mem.setObjectName("unitLabel")
        g.addWidget(self.lbl_mem, 2, 0, 1, 4)
        for sb in [self.sb_nx, self.sb_ny, self.sb_nz, self.sb_nt]:
            sb.valueChanged.connect(self._update_memory_estimate)
        return gb

    def _build_output_group(self):
        gb = QGroupBox("Output")
        g  = QVBoxLayout(gb)
        g.setSpacing(5)
        self.chk_save_pump   = QCheckBox("Save pump");   self.chk_save_pump.setChecked(True)
        self.chk_save_signal = QCheckBox("Save signal"); self.chk_save_signal.setChecked(True)
        self.chk_save_idler  = QCheckBox("Save idler")
        self.chk_save_inside = QCheckBox("Save field inside crystal")
        for w in [self.chk_save_pump, self.chk_save_signal,
                  self.chk_save_idler, self.chk_save_inside]:
            g.addWidget(w)
        return gb

    def _build_thermal_group(self):
        gb = QGroupBox("Thermal Simulation")
        gv = QVBoxLayout(gb)
        gv.setSpacing(8)

        self.chk_thermal = QCheckBox("Enable thermal simulation")
        gv.addWidget(self.chk_thermal)

        self._thermal_params_widget = QWidget()
        tp = QVBoxLayout(self._thermal_params_widget)
        tp.setContentsMargins(0, 4, 0, 0)
        tp.setSpacing(6)

        # Thermal boundary conditions
        gb_bc = QGroupBox("Thermal Boundary Conditions")
        gb2 = QGridLayout(gb_bc)
        gb2.setSpacing(5)
        gb2.addWidget(QLabel("Oven T (°C)"), 0, 0)
        self.sb_T_oven = QDoubleSpinBox()
        self.sb_T_oven.setRange(-100, 500); self.sb_T_oven.setValue(25.0)
        gb2.addWidget(self.sb_T_oven, 0, 1)
        gb2.addWidget(QLabel("Env T (°C)"), 1, 0)
        self.sb_T_env = QDoubleSpinBox()
        self.sb_T_env.setRange(-100, 500); self.sb_T_env.setValue(25.0)
        gb2.addWidget(self.sb_T_env, 1, 1)
        note = QLabel("Oven heats bottom face (y=0).\nAll other faces → T_env.")
        note.setObjectName("unitLabel")
        note.setWordWrap(True)
        gb2.addWidget(note, 2, 0, 1, 2)
        tp.addWidget(gb_bc)

        # Solver settings
        gb_sol = QGroupBox("Thermal Solver")
        gs = QGridLayout(gb_sol)
        gs.setSpacing(5)
        gs.addWidget(QLabel("Max iterations"), 0, 0)
        self.sb_th_maxiter = QDoubleSpinBox()
        self.sb_th_maxiter.setRange(100, 1e6); self.sb_th_maxiter.setDecimals(0)
        self.sb_th_maxiter.setValue(100000)
        gs.addWidget(self.sb_th_maxiter, 0, 1)
        gs.addWidget(QLabel("Tolerance"), 1, 0)
        self.sb_th_tol = QDoubleSpinBox()
        self.sb_th_tol.setRange(1e-12, 1); self.sb_th_tol.setDecimals(8)
        self.sb_th_tol.setValue(5e-4)
        gs.addWidget(self.sb_th_tol, 1, 1)
        gs.addWidget(QLabel("h conv (W/m²K)"), 2, 0)
        self.sb_h_conv = QDoubleSpinBox()
        self.sb_h_conv.setRange(0.1, 10000.0); self.sb_h_conv.setDecimals(1)
        self.sb_h_conv.setValue(10.0)
        self.sb_h_conv.setToolTip(
            "Convective heat transfer coefficient at crystal surfaces.\n"
            "Natural convection (air): 5–25 W/m²K\n"
            "Forced convection (air):  25–250 W/m²K")
        gs.addWidget(self.sb_h_conv, 2, 1)
        tp.addWidget(gb_sol)

        self._thermal_params_widget.setVisible(False)
        self.chk_thermal.toggled.connect(self._thermal_params_widget.setVisible)

        gv.addWidget(self._thermal_params_widget)
        return gb

    # ── Visibility logic ───────────────────────────────────────────────
    def _update_visibility(self):
        pulsed  = self.cb_regime.currentText() == "Pulsed"
        focused = self.cb_profile.currentText() == "Focused Gaussian"
        proc    = self.cb_process.currentText()

        self.lbl_fwhm.setVisible(pulsed)
        self.sb_fwhm.setVisible(pulsed)
        self.lbl_focus.setVisible(focused)
        self.cb_focus.setVisible(focused)
        self.sb_nt.setEnabled(pulsed)

        shg = proc == "SHG"
        self.sb_ls.setEnabled(not shg)
        self.lbl_ls.setEnabled(not shg)
        if shg:
            self.sb_ls.setValue(self.sb_lp.value() / 2)

        # SHG is always degenerate — show checkbox but disabled+checked
        if proc == "SHG":
            self.chk_degen.setVisible(True)
            self.chk_degen.setChecked(True)
            self.chk_degen.setEnabled(False)
        else:
            if not self.chk_degen.isEnabled():
                # Was locked by SHG — uncheck first so idler group re-enables
                self.chk_degen.setChecked(False)
            self.chk_degen.setEnabled(True)
            self.chk_degen.setVisible(proc == "OPG")
        self._update_memory_estimate()

    def _on_degenerate(self, checked):
        self.idler_gb.setEnabled(not checked)

    def _auto_lxly(self):
        """Auto-update LX/LY from waist (6·w₀ each side = 12·w₀ total), in mm."""
        waist_um = self.sb_waist.value()
        half_mm  = max(6.0 * waist_um * 1e-3, 0.25)   # min 0.25 mm/side → 0.5 mm total
        val      = round(2.0 * half_mm, 3)
        for sb in (self.sb_lx, self.sb_ly):
            sb.blockSignals(True)
            sb.setValue(val)
            sb.blockSignals(False)

    def _update_memory_estimate(self):
        nx = self.sb_nx.value(); ny = self.sb_ny.value()
        nz = self.sb_nz.value(); nt = self.sb_nt.value()
        mb = 3 * nx * ny * nz * nt * 8 / 1024**2
        self.lbl_mem.setText(f"Memory: ~{mb:.0f} MB  ({nx}×{ny}×{nz}×{nt})")

    # ── Button slots ───────────────────────────────────────────────────
    def _on_run_gpu(self):
        self.log.append("[GPU] Building configuration...")
        self.run_requested.emit(self._collect_params(backend="cuda"))

    def _on_run_mpi(self):
        self.log.append("[CPU] Building configuration...")
        self.run_requested.emit(self._collect_params(backend="cpu"))

    def _collect_params(self, backend="cuda"):
        params = {
            "backend":  backend,
            "process":  self.cb_process.currentText(),
            "crystal":  self.cb_crystal.currentText(),
            "length_mm":    self.sb_length.value(),
            "temperature":  self.sb_temp.value(),
            "grating_um":   self.sb_lambda.value(),
            "lx_mm":        self.sb_lx.value(),
            **({"dk": self.sb_dk_forced.value()} if self.chk_force_dk.isChecked() else {}),
            "ly_mm":        self.sb_ly.value(),
            "undepleted_pump": self.chk_undepleted.isChecked(),
            "pump": {
                "lambda_um": self.sb_lp.value(),
                "regime":    self.cb_regime.currentText(),
                "profile":   self.cb_profile.currentText(),
                "power_W":   self.sb_power.value(),
                "waist_um":  self.sb_waist.value(),
                "fwhm_ps":   self.sb_fwhm.value(),
            },
            "signal": {
                "lambda_um": self.sb_ls.value(),
                "mode": "noise" if self.signal_gb.rb_noise.isChecked() else "inject",
            },
            "idler": {
                "lambda_um": self.sb_li.value(),
                "degenerate": self.chk_degen.isChecked(),
                "mode": "noise" if self.idler_gb.rb_noise.isChecked() else "inject",
            },
            "grid": {
                "NX": self.sb_nx.value(), "NY": self.sb_ny.value(),
                "NZ": self.sb_nz.value(), "NT": self.sb_nt.value(),
            },
        }
        if self.chk_thermal.isChecked():
            params["thermal"] = {
                "enabled":        True,
                "T_oven_C":       self.sb_T_oven.value(),
                "T_ambient_C":    self.sb_T_env.value(),
                "h_conv_W_m2K":   self.sb_h_conv.value(),
                "max_iter":       int(self.sb_th_maxiter.value()),
                "tol":            self.sb_th_tol.value(),
                "max_outer_iter": 10,
                "outer_tol":      0.001,
            }
        else:
            params["thermal"] = {"enabled": False}
        return params

    def load_pm_wavelengths(self, lp: float, ls: float, li: float,
                            T: float, Lambda: float):
        """Load PM-computed wavelengths and operating conditions into the form."""
        self.sb_lp.setValue(lp)
        self.sb_ls.setValue(ls)
        self.sb_li.setValue(li)
        self.sb_temp.setValue(T)
        self.sb_lambda.setValue(Lambda)
        # Enable idler box if non-degenerate
        if abs(ls - li) > 1e-6:
            self.chk_degen.setChecked(False)

    def append_log(self, text: str):
        self.log.append(text)
