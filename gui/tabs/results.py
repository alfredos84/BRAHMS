import os
import json
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QGroupBox,
    QLabel, QPushButton, QFileDialog,
    QFrame, QLineEdit, QSlider, QTabWidget, QSpinBox, QCheckBox
)
from PyQt6.QtCore import Qt


def _hsep():
    """Thin horizontal separator line."""
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFrameShadow(QFrame.Shadow.Sunken)
    return f

from ..widgets.plot_canvas import PlotCanvas


def _load_h5_complex(path: str) -> np.ndarray | None:
    try:
        import h5py
        with h5py.File(path, "r") as f:
            r = f["real"][()]
            im = f["imag"][()]
        return r.astype(np.float32) + 1j * im.astype(np.float32)
    except Exception:
        return None


def _load_h5_real(path: str, dataset: str = "intensity") -> np.ndarray | None:
    """Load a real-valued HDF5 dataset (e.g., propagation profile)."""
    try:
        import h5py
        with h5py.File(path, "r") as f:
            return f[dataset][()].astype(np.float32)
    except Exception:
        return None


_FIELD_COLORS = {"Pump": "#ff6d00", "Signal": "#00e676", "Idler": "#40c4ff"}
_FIELDS       = ["Pump", "Signal", "Idler"]


class ResultsTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter)

        # ── LEFT: controls ──────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(210)
        left.setMaximumWidth(260)
        lv = QVBoxLayout(left)
        lv.setSpacing(10)
        lv.setContentsMargins(4, 4, 4, 4)

        gb_file = QGroupBox("Output")
        gf = QVBoxLayout(gb_file)
        self.le_file = QLineEdit()
        self.le_file.setPlaceholderText("Output directory or HDF5 file…")
        self.le_file.setReadOnly(True)
        btn_open = QPushButton("Browse…")
        btn_open.clicked.connect(self._open_file)
        gf.addWidget(self.le_file)
        gf.addWidget(btn_open)
        lv.addWidget(gb_file)

        gb_eff = QGroupBox("Conversion Efficiency")
        ge = QVBoxLayout(gb_eff)
        ge.setSpacing(2)
        def _eff_label(text, attr):
            lbl = QLabel(text)
            lbl.setObjectName("valueLabel")
            setattr(self, attr, lbl)
            ge.addWidget(lbl)
        _eff_label("η = —",              "lbl_eta")
        ge.addWidget(_hsep())
        _eff_label("P_pump   = —",       "lbl_p_pump")
        _eff_label("P_signal = —",       "lbl_p_signal")
        _eff_label("P_idler  = —",       "lbl_p_idler")
        ge.addWidget(_hsep())
        _eff_label("I_pump   = —",       "lbl_i_pump")
        _eff_label("I_signal = —",       "lbl_i_signal")
        _eff_label("I_idler  = —",       "lbl_i_idler")
        lv.addWidget(gb_eff)

        # Hidden sliders — still used by intensity/phase plots
        self.sl_z = QSlider(Qt.Orientation.Horizontal)
        self.sl_z.setRange(0, 99);  self.sl_z.setValue(99)
        self.lbl_z = QLabel()
        self.sl_x = QSlider(Qt.Orientation.Horizontal)
        self.sl_x.setRange(0, 63);  self.sl_x.setValue(32)
        self.lbl_x = QLabel()

        self.btn_load   = QPushButton("Load & Plot")
        self.btn_load.setObjectName("runButton")
        self.btn_export = QPushButton("Export Figures")
        lv.addWidget(self.btn_load)
        lv.addWidget(self.btn_export)
        self.chk_unwrap_phase = QCheckBox("Unwrap phase (bottom panels)")
        self.chk_unwrap_phase.setToolTip(
            "Bottom-row phase line-outs: wrapped (-π, π] when unchecked, "
            "unwrapped (continuous) when checked. Re-plots instantly from "
            "the already-loaded data — no need to re-run the simulation.")
        lv.addWidget(self.chk_unwrap_phase)
        lv.addStretch()

        splitter.addWidget(left)

        # ── RIGHT: sub-tabs ─────────────────────────────────────────────
        self.sub_tabs = QTabWidget()
        self.sub_tabs.setDocumentMode(True)
        self._build_intensity_tab()
        self._build_phase_tab()
        self._build_thermal_tab()
        splitter.addWidget(self.sub_tabs)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.btn_load.clicked.connect(self._load_and_plot)
        self.btn_export.clicked.connect(self._export)
        self.sl_z.valueChanged.connect(self._on_slice_changed)
        self.sl_x.valueChanged.connect(self._on_slice_changed)
        self.chk_unwrap_phase.toggled.connect(self._on_unwrap_toggled)

        self._fields_data:  dict[str, np.ndarray] = {}
        self._fields_input: dict[str, np.ndarray] = {}
        self._vol_data:     dict[str, np.ndarray] = {}  # (NZ, NY, NX) intensity volumes
        self._T3D = None
        self._n   = {"Pump": 1.0, "Signal": 1.0, "Idler": 1.0}
        self._LX_um  = None
        self._LY_um  = None
        self._Lcr_mm = None
        self._update_slice_labels()

    # ── Tab builders ────────────────────────────────────────────────────
    def _build_intensity_tab(self):
        w = QWidget()
        QVBoxLayout(w).setContentsMargins(2, 2, 2, 2)
        w.setLayout(QVBoxLayout())
        w.layout().setContentsMargins(2, 2, 2, 2)
        self.canvas_int = PlotCanvas(nrows=2, ncols=3, figsize=(14, 7), dpi=95)
        w.layout().addWidget(self.canvas_int)
        self._init_intensity_axes()
        self.sub_tabs.addTab(w, "  Beam Profiles  ")

    def _build_phase_tab(self):
        w = QWidget()
        w.setLayout(QVBoxLayout())
        w.layout().setContentsMargins(2, 2, 2, 2)
        self.canvas_ph = PlotCanvas(nrows=2, ncols=3, figsize=(14, 7), dpi=95)
        w.layout().addWidget(self.canvas_ph)
        self._init_phase_axes()
        self.sub_tabs.addTab(w, "  Phases  ")

    def _build_thermal_tab(self):
        w = QWidget()
        outer = QHBoxLayout(w)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(6)

        # ── Panel 1: T(x,y) at z-slice ──────────────────────────────────
        col0 = QVBoxLayout()
        col0.setSpacing(3)
        self.canvas_th0 = PlotCanvas(nrows=1, ncols=1, figsize=(5, 5), dpi=95)
        col0.addWidget(self.canvas_th0, stretch=1)

        row_z = QHBoxLayout()
        row_z.addWidget(QLabel("z-slice:"))
        self.slider_th_z = QSlider(Qt.Orientation.Horizontal)
        self.slider_th_z.setRange(0, 100); self.slider_th_z.setValue(99)
        self.lbl_th_z = QLabel("100 %")
        row_z.addWidget(self.slider_th_z, stretch=1)
        row_z.addWidget(self.lbl_th_z)
        col0.addLayout(row_z)
        outer.addLayout(col0, stretch=1)

        # ── Panel 2: T(y,z) at x-slice ──────────────────────────────────
        col1 = QVBoxLayout()
        col1.setSpacing(3)
        self.canvas_th1 = PlotCanvas(nrows=1, ncols=1, figsize=(5, 5), dpi=95)
        col1.addWidget(self.canvas_th1, stretch=1)

        row_x = QHBoxLayout()
        row_x.addWidget(QLabel("x-slice:"))
        self.slider_th_x = QSlider(Qt.Orientation.Horizontal)
        self.slider_th_x.setRange(0, 100); self.slider_th_x.setValue(50)
        self.lbl_th_x = QLabel("50 %")
        row_x.addWidget(self.slider_th_x, stretch=1)
        row_x.addWidget(self.lbl_th_x)
        col1.addLayout(row_x)
        outer.addLayout(col1, stretch=1)

        self._init_thermal_axes()
        self.sub_tabs.addTab(w, "  Thermal Profile  ")
        self.slider_th_z.valueChanged.connect(self._on_th_z_slider)
        self.slider_th_x.valueChanged.connect(self._on_th_x_slider)

    # ── Axis init placeholders ──────────────────────────────────────────
    def _init_intensity_axes(self):
        for col, name in enumerate(_FIELDS):
            for row, title in enumerate([
                    f"{name}   |A(x,y, z₀)|²",
                    f"{name}   |A(x₀, y, z)|²"]):
                ax = self.canvas_int.get_ax(col + row * 3)
                ax.set_title(title, fontsize=9)
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", color="#555555", fontsize=10)
        self.canvas_int.refresh()

    def _init_phase_axes(self):
        for col, name in enumerate(_FIELDS):
            for row, title in enumerate([
                    f"{name}   phase(x,y, z=L)",
                    f"{name}   phase(x=0, y, z=L)"]):
                ax = self.canvas_ph.get_ax(col + row * 3)
                ax.set_title(title, fontsize=9)
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", color="#555555", fontsize=10)
        self.canvas_ph.refresh()

    def _init_thermal_axes(self):
        for canvas, title, xl, yl in [
                (self.canvas_th0, "T(x, y)  z = z_slice", "x  (μm)", "y  (μm)"),
                (self.canvas_th1, "T(y, z)  x = x_slice", "z  (mm)",  "y  (μm)")]:
            ax = canvas.get_ax(0)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(xl, fontsize=9); ax.set_ylabel(yl, fontsize=9)
            ax.text(0.5, 0.5, "No thermal data", transform=ax.transAxes,
                    ha="center", va="center", color="#555555", fontsize=10)
            canvas.refresh()

    # ── File & load ─────────────────────────────────────────────────────
    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open HDF5 file", "", "HDF5 files (*.h5 *.hdf5);;All files (*)")
        if path:
            self.le_file.setText(path)

    def _update_slice_labels(self):
        Lcr   = self._Lcr_mm or 10.0
        LX    = self._LX_um  or 1.0
        iz    = self.sl_z.value()
        ix    = self.sl_x.value()
        NZ_   = max(self.sl_z.maximum() + 1, 1)
        NX_   = max(self.sl_x.maximum() + 1, 1)
        z_mm  = iz * Lcr / max(NZ_ - 1, 1)
        x_um  = (ix - NX_ / 2.0) * LX / NX_
        self.lbl_z.setText(f"{z_mm:.1f} mm")
        self.lbl_x.setText(f"{x_um:.0f} μm")

    def _on_slice_changed(self):
        self._update_slice_labels()
        if self._vol_data:
            self._plot_intensities()
            self._update_efficiency()

    def _on_unwrap_toggled(self, _checked):
        if self._fields_data:
            self._plot_phases()

    def _load_and_plot(self):
        path = self.le_file.text().strip()
        if not path:
            return
        if os.path.isdir(path):
            self.load_output_dir(path)
        elif os.path.isfile(path):
            for name in _FIELDS:
                if name.lower() in os.path.basename(path).lower():
                    A = _load_h5_complex(path)
                    if A is not None:
                        self._fields_data[name] = A
            self._plot_all()

    def load_output_dir(self, out_dir: str):
        """Called automatically after a successful engine run."""
        self._out_dir = out_dir
        self.le_file.setText(out_dir)

        # Read crystal geometry and n from config.json
        self._n      = {"Pump": 1.0, "Signal": 1.0, "Idler": 1.0}
        self._LX_um  = None
        self._LY_um  = None
        self._Lcr_mm = None
        degenerate   = False
        cfg_path = os.path.join(out_dir, "config.json")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
                opt = cfg["crystal"]["optics"]
                geo = cfg["crystal"]["geometry"]
                self._n      = {"Pump": opt["np"], "Signal": opt["ns"], "Idler": opt["ni"]}
                self._LX_um  = geo["LX_mm"] * 1000.0
                self._LY_um  = geo["LY_mm"] * 1000.0
                self._Lcr_mm = geo["Lcr_mm"]
                degenerate   = cfg["crystal"].get("degenerate", False)
            except Exception:
                pass

        # Load output fields
        self._fields_data  = {}
        self._fields_input = {}
        for name, fname_out, fname_in in [
                ("Pump",   "pump_output.h5",  "pump_input_XY.h5"),
                ("Signal", "signal_output.h5", None),
                ("Idler",  "idler_output.h5",  None)]:
            p = os.path.join(out_dir, fname_out)
            if os.path.isfile(p):
                A = _load_h5_complex(p)
                if A is not None:
                    self._fields_data[name] = A
            if fname_in:
                pi = os.path.join(out_dir, fname_in)
                if os.path.isfile(pi):
                    Ain = _load_h5_complex(pi)
                    if Ain is not None:
                        self._fields_input[name] = Ain

        # For degenerate processes (SHG): idler == signal
        if degenerate and "Idler" not in self._fields_data and "Signal" in self._fields_data:
            self._fields_data["Idler"] = self._fields_data["Signal"]

        # Load intensity volumes  |A(x, y, z)|²  — shape (NZ, NY, NX)
        self._vol_data = {}
        for name, fname in [("Pump",   "pump_volume.h5"),
                             ("Signal", "signal_volume.h5"),
                             ("Idler",  "idler_volume.h5")]:
            p = os.path.join(out_dir, fname)
            if os.path.isfile(p):
                arr = _load_h5_real(p, "intensity")
                if arr is not None:
                    self._vol_data[name] = arr   # shape (NZ, NY, NX)
        if degenerate and "Idler" not in self._vol_data and "Signal" in self._vol_data:
            self._vol_data["Idler"] = self._vol_data["Signal"]

        # Update slice slider ranges from actual volume dimensions
        if self._vol_data:
            vol0 = next(iter(self._vol_data.values()))
            NZ_, NY_, NX_ = vol0.shape
            self.sl_z.setMaximum(NZ_ - 1); self.sl_z.setValue(NZ_ - 1)
            self.sl_x.setMaximum(NX_ - 1); self.sl_x.setValue(NX_ // 2)
        self._update_slice_labels()

        if self._fields_data:
            self._plot_all()

        # Thermal (optional)
        th_path = os.path.join(out_dir, "thermal_output.h5")
        if os.path.isfile(th_path):
            self._load_thermal(th_path)

    # ── Helpers ─────────────────────────────────────────────────────────
    def _extent(self, NY, NX):
        """Physical extent [μm] for imshow."""
        LX = self._LX_um if self._LX_um else float(NX)
        LY = self._LY_um if self._LY_um else float(NY)
        return [-LX / 2, LX / 2, -LY / 2, LY / 2]

    def _y_axis(self, NY):
        LY = self._LY_um if self._LY_um else float(NY)
        return np.linspace(-LY / 2, LY / 2, NY)

    # ── Plotting ────────────────────────────────────────────────────────
    def _plot_all(self):
        self._plot_intensities()
        self._plot_phases()
        self._update_efficiency()

    def _plot_intensities(self):
        self.canvas_int.fig.clear()
        self.canvas_int._init_axes()

        iz  = self.sl_z.value()
        ix  = self.sl_x.value()
        LY  = self._LY_um  or 1.0
        LX  = self._LX_um  or 1.0
        Lcr = self._Lcr_mm or 10.0

        for col, name in enumerate(_FIELDS):
            ax_top = self.canvas_int.get_ax(col)
            ax_bot = self.canvas_int.get_ax(col + 3)

            if name not in self._vol_data:
                for ax, t in [(ax_top, f"{name}   |A(x,y, z₀)|²"),
                              (ax_bot, f"{name}   |A(x₀, y, z)|²")]:
                    ax.set_title(t, fontsize=9)
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                            ha="center", va="center", color="#555555", fontsize=10)
                continue

            vol = self._vol_data[name]          # (NZ, NY, NX)
            NZ_, NY_, NX_ = vol.shape

            # ── top: transverse slice at z = iz*dz ───────────────────
            z_mm  = iz * Lcr / max(NZ_ - 1, 1)
            I_xy  = vol[iz, :, :]               # (NY, NX)
            ext   = [-LX/2, LX/2, -LY/2, LY/2]
            ax_top.set_title(f"{name}   |A(x,y, z={z_mm:.1f} mm)|²", fontsize=12)
            ax_top.set_xlabel("x  (μm)", fontsize=12)
            ax_top.set_ylabel("y  (μm)", fontsize=12)
            im = ax_top.imshow(I_xy, cmap="RdBu_r", origin="lower",
                               aspect="auto", extent=ext, interpolation="bilinear")
            cb = self.canvas_int.fig.colorbar(im, ax=ax_top, pad=0.02)
            cb.ax.tick_params(colors="white", labelsize=10)

            # ── bottom: longitudinal slice at x = ix ─────────────────
            x_um  = (ix - NX_/2) * LX / NX_
            I_yz  = vol[:, :, ix]               # (NZ, NY)
            ext_p = [0, Lcr, -LY/2, LY/2]
            ax_bot.set_title(f"{name}   |A(x={x_um:.0f} μm, y, z)|²", fontsize=12)
            ax_bot.set_xlabel("z  (mm)", fontsize=12)
            ax_bot.set_ylabel("y  (μm)", fontsize=12)
            im_b = ax_bot.imshow(I_yz.T, cmap="RdBu_r", origin="lower",
                                 aspect="auto", extent=ext_p, interpolation="bilinear")
            cb_b = self.canvas_int.fig.colorbar(im_b, ax=ax_bot, pad=0.02)
            cb_b.ax.tick_params(colors="white", labelsize=10)

        self.canvas_int.fig.tight_layout()
        self.canvas_int.refresh()

    def _plot_phases(self):
        self.canvas_ph.fig.clear()
        self.canvas_ph._init_axes()

        for col, name in enumerate(_FIELDS):
            ax_top = self.canvas_ph.get_ax(col)
            ax_bot = self.canvas_ph.get_ax(col + 3)

            if name not in self._fields_data:
                for ax, t in [(ax_top, f"{name}   phase(x,y, z=L)"),
                              (ax_bot, f"{name}   phase(x=0, y, z=L)")]:
                    ax.set_title(t, fontsize=9)
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                            ha="center", va="center", color="#555555", fontsize=10)
                continue

            A   = self._fields_data[name][-1]
            NY, NX = A.shape
            ext = self._extent(NY, NX)
            y   = self._y_axis(NY)
            ph  = np.angle(A)
            c   = _FIELD_COLORS[name]

            # ── top: 2-D phase map (same colormap/formatting as intensities) ─
            ax_top.set_title(f"{name}   phase(x,y, z=L)", fontsize=12)
            ax_top.set_xlabel("x  (μm)", fontsize=12)
            ax_top.set_ylabel("y  (μm)", fontsize=12)
            im = ax_top.imshow(ph, cmap="RdBu_r", origin="lower",
                               aspect="auto", extent=ext, vmin=-np.pi, vmax=np.pi,
                               interpolation="bilinear")
            cb = self.canvas_ph.fig.colorbar(im, ax=ax_top, pad=0.02)
            cb.set_label("rad", fontsize=12, color="white")
            cb.ax.tick_params(colors="white", labelsize=10)

            # ── bottom: 1-D phase profile at x = 0 (wrap/unwrap toggle) ──
            ix0      = NX // 2
            unwrap   = self.chk_unwrap_phase.isChecked()
            ph_prof  = np.unwrap(ph[:, ix0]) if unwrap else ph[:, ix0]
            title_suffix = "  (unwrapped)" if unwrap else ""
            ax_bot.set_title(f"{name}   phase(x=0, y, z=L){title_suffix}", fontsize=12)
            ax_bot.set_xlabel("y  (μm)", fontsize=12)
            ax_bot.set_ylabel("phase  (rad)", fontsize=12)
            ax_bot.plot(y, ph_prof, color=c, lw=1.5)
            ax_bot.set_xlim(ext[2], ext[3])
            ax_bot.axhline(0, color="#555", lw=0.7, ls="--")
            if unwrap:
                # Unwrapped phase can range well beyond ±π — let it auto-scale.
                ax_bot.margins(y=0.08)
            else:
                ax_bot.set_ylim(-np.pi - 0.1, np.pi + 0.1)
                ax_bot.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
                ax_bot.set_yticklabels(["-π", "-π/2", "0", "π/2", "π"], fontsize=11)

        self.canvas_ph.fig.tight_layout()
        self.canvas_ph.refresh()

    def _update_efficiency(self):
        # Physical constants — same units as engine (μm, ps)
        C0   = 299.792458        # μm/ps
        EPS0 = 8.8541878128e-12 * 1e12 / 1e6  # W·ps/(V²·μm)
        UM2_TO_CM2 = 1e-8        # 1 μm² = 1e-8 cm²

        iz  = self.sl_z.value()
        lbl_p = {"Pump": self.lbl_p_pump,
                 "Signal": self.lbl_p_signal,
                 "Idler":  self.lbl_p_idler}
        lbl_i = {"Pump": self.lbl_i_pump,
                 "Signal": self.lbl_i_signal,
                 "Idler":  self.lbl_i_idler}

        powers = {}

        for name in _FIELDS:
            n_val = self._n.get(name, 1.0)
            P_W   = None
            I_W   = None

            if name in self._vol_data:
                # vol[iz, NY, NX] = ½·n·EPS0·C · Σ_t |A|²   [W/μm² × NT samples]
                vol       = self._vol_data[name]
                NZ_, NY_, NX_ = vol.shape
                NT_ = (self._fields_data[name].shape[0]
                       if name in self._fields_data else 1)
                dx_ = (self._LX_um or float(NX_)) / NX_
                dy_ = (self._LY_um or float(NY_)) / NY_
                iz_ = min(iz, NZ_ - 1)
                sl  = vol[iz_, :, :]          # shape (NY, NX)
                # divide by NT to get time-averaged intensity [W/μm²]
                P_W = float(np.sum(sl)) * dx_ * dy_ / NT_
                I_W = float(np.max(sl)) / NT_ / UM2_TO_CM2  # W/cm²
            elif name in self._fields_data:
                A = self._fields_data[name][-1]  # last t-slice, (NY, NX)
                NY_, NX_ = A.shape
                dx_ = (self._LX_um or float(NX_)) / NX_
                dy_ = (self._LY_um or float(NY_)) / NY_
                P_W = (0.5 * n_val * EPS0 * C0
                       * float(np.sum(np.abs(A)**2)) * dx_ * dy_)
                I_W = (0.5 * n_val * EPS0 * C0
                       * float(np.max(np.abs(A)**2)) / UM2_TO_CM2)

            if P_W is not None:
                powers[name] = P_W
                lbl_p[name].setText(f"P_{name.lower():<6s} = {P_W:.3e} W")
                lbl_i[name].setText(f"I_{name.lower():<6s} = {I_W:.3e} W/cm²")
            else:
                lbl_p[name].setText(f"P_{name.lower():<6s} = —")
                lbl_i[name].setText(f"I_{name.lower():<6s} = —")

        # Efficiency: P_signal at current z / P_pump at z=0
        P_p0 = None
        if "Pump" in self._vol_data:
            vol = self._vol_data["Pump"]
            NZ_, NY_, NX_ = vol.shape
            NT_ = (self._fields_data["Pump"].shape[0]
                   if "Pump" in self._fields_data else 1)
            dx_ = (self._LX_um or float(NX_)) / NX_
            dy_ = (self._LY_um or float(NY_)) / NY_
            P_p0 = float(np.sum(vol[0, :, :])) * dx_ * dy_ / NT_
        elif "Pump" in powers:
            P_p0 = powers["Pump"]

        if "Signal" in powers and P_p0 and P_p0 > 0:
            eta = powers["Signal"] / P_p0 * 100.0
            self.lbl_eta.setText(f"η = {eta:.4f} %")
        else:
            self.lbl_eta.setText("η = —")

    # ── Thermal ──────────────────────────────────────────────────────────
    def _load_thermal(self, path: str):
        try:
            import h5py
            with h5py.File(path, "r") as f:
                self._T3D = f["T"][()].astype(np.float32)  # shape (NZ, NY, NX)
            self._draw_thermal()
        except Exception as e:
            ax0 = self.canvas_th0.get_ax(0)
            ax0.clear()
            self.canvas_th0._style_ax(ax0)
            ax0.text(0.5, 0.5, f"Error loading thermal data:\n{e}",
                     transform=ax0.transAxes, ha="center", va="center",
                     color="#ff6d00", fontsize=9)
            self.canvas_th0.refresh()

    def _draw_thermal_xy(self):
        """Panel 0: T(x, y) cross-section at z = z_slice."""
        if self._T3D is None:
            return
        T = self._T3D          # shape (NZ, NY, NX)
        NZ_, NY_, NX_ = T.shape
        Lcr = self._Lcr_mm or 1.0
        LX  = self._LX_um  or float(NX_)
        LY  = self._LY_um  or float(NY_)

        z_idx = int(self.slider_th_z.value() * (NZ_ - 1) / 100)
        z_mm  = z_idx * Lcr / max(NZ_ - 1, 1)
        self.lbl_th_z.setText(f"{z_mm:.1f} mm")

        self.canvas_th0.fig.clear()
        self.canvas_th0._init_axes()
        ax = self.canvas_th0.get_ax(0)
        FS = 10

        # T[z_idx, ::-1, :] → rows=NY (flipped so oven face at bottom), cols=NX
        ext = [-LX / 2, LX / 2, 0, LY]
        im = ax.imshow(T[z_idx, ::-1, :], origin="upper", cmap="afmhot",
                       aspect="auto", extent=ext, interpolation="bilinear")
        ax.set_title(f"T(x, y)   z = {z_mm:.1f} mm", fontsize=FS)
        ax.set_xlabel("x  (μm)", fontsize=FS)
        ax.set_ylabel("y  (μm)  [0 = oven]", fontsize=FS)
        cb = self.canvas_th0.fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label("T  (°C)", color="white", fontsize=FS)
        cb.ax.tick_params(colors="white", labelsize=FS - 1)
        self.canvas_th0.fig.tight_layout()
        self.canvas_th0.refresh()

    def _draw_thermal_yz(self):
        """Panel 1: T(y, z) longitudinal cut at x = x_slice."""
        if self._T3D is None:
            return
        T = self._T3D          # shape (NZ, NY, NX)
        NZ_, NY_, NX_ = T.shape
        Lcr = self._Lcr_mm or 1.0
        LX  = self._LX_um  or float(NX_)
        LY  = self._LY_um  or float(NY_)

        x_idx = int(self.slider_th_x.value() * (NX_ - 1) / 100)
        x_um  = (x_idx - NX_ / 2.0) * LX / NX_
        self.lbl_th_x.setText(f"{x_um:.1f} μm")

        self.canvas_th1.fig.clear()
        self.canvas_th1._init_axes()
        ax = self.canvas_th1.get_ax(0)
        FS = 10

        T_max = T.max(); T_min = T.min()
        # T[:, ::-1, x_idx] → shape (NZ, NY_flipped) → .T → (NY_flipped, NZ)
        # origin="upper": flipped row-0 (cool top face) at top, oven at bottom
        ext = [0, Lcr, 0, LY]
        im = ax.imshow(T[:, ::-1, x_idx].T, origin="upper", cmap="afmhot",
                       aspect="auto", extent=ext, interpolation="bilinear")
        ax.set_title(
            f"T(y, z)   x = {x_um:.1f} μm   ΔT = {T_max - T_min:.1f} K",
            fontsize=FS)
        ax.set_xlabel("z  (mm)", fontsize=FS)
        ax.set_ylabel("y  (μm)  [0 = oven]", fontsize=FS)
        cb = self.canvas_th1.fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label("T  (°C)", color="white", fontsize=FS)
        cb.ax.tick_params(colors="white", labelsize=FS - 1)
        self.canvas_th1.fig.tight_layout()
        self.canvas_th1.refresh()

    def _draw_thermal(self):
        self._draw_thermal_xy()
        self._draw_thermal_yz()

    def _on_th_z_slider(self, value):
        self._draw_thermal_xy()

    def _on_th_x_slider(self, value):
        self._draw_thermal_yz()

    # ── Export ───────────────────────────────────────────────────────────
    def _export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export figures", "results.pdf",
            "PDF (*.pdf);;PNG (*.png);;SVG (*.svg)")
        if not path:
            return
        idx = self.sub_tabs.currentIndex()
        canvases = [self.canvas_int, self.canvas_ph, self.canvas_th0, self.canvas_th1]
        fig = canvases[min(idx, len(canvases)-1)].fig
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
