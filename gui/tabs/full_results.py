"""
Full Results tab — visualizes (3+1)D pulsed+focused simulation output.

Layout:
  Left  — NX×NY spatial beam profile at selected time slice (t-slider)
  Right — P(t) power curves for pump, signal, idler (stacked, energy in title)

Input files read from output directory:
  pump_output.h5, signal_output.h5, idler_output.h5  → [NT, NY, NX] complex
  time.h5                                              → [NT] real (ps)
  config.json                                          → crystal/grid metadata
"""

import os
import json
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QLabel, QPushButton, QFileDialog, QFrame,
    QLineEdit, QSlider, QComboBox,
)
from PyQt6.QtCore import Qt

from ..widgets.plot_canvas import PlotCanvas


# BRAHMS internal unit constants (see DataTypes_cpu.hpp / DataTypes.cuh)
_EPS0 = 8.8541878128e-12 * 1e12 / 1e6   # F·ps²/μm³  (= ε₀ × 1e6)
_C    = 299.792458                         # μm/ps


def _format_energy(E_J: float) -> str:
    """Format energy in the most appropriate SI sub-unit (2 decimal places)."""
    if E_J >= 1e-3:
        return f"{E_J * 1e3:.2f} mJ"
    if E_J >= 1e-6:
        return f"{E_J * 1e6:.2f} μJ"
    if E_J >= 1e-9:
        return f"{E_J * 1e9:.2f} nJ"
    if E_J >= 1e-12:
        return f"{E_J * 1e12:.2f} pJ"
    return f"{E_J * 1e12:.2e} pJ"


def _load_complex(path: str) -> "np.ndarray | None":
    """Load [NT, NY, NX] complex array from HDF5 (datasets 'real' and 'imag')."""
    try:
        import h5py
        with h5py.File(path, "r") as f:
            r  = f["real"][()].astype(np.float32)
            im = f["imag"][()].astype(np.float32)
        return r + 1j * im
    except Exception:
        return None


def _load_vector(path: str) -> "np.ndarray | None":
    """Load 1D real array from HDF5 (dataset 'data')."""
    try:
        import h5py
        with h5py.File(path, "r") as f:
            return f["data"][()].astype(np.float32)
    except Exception:
        return None


_WAVE_COLORS = {
    "Input Pump": "#ffd54f",
    "Pump":       "#ff6d00",
    "Signal":     "#00e676",
    "Idler":      "#40c4ff",
}
_WAVES     = ["Pump", "Signal", "Idler"]          # output fields
_ALL_WAVES = ["Input Pump", "Pump", "Signal", "Idler"]  # including input


class FullResultsTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter)

        # ── LEFT: spatial NX×NY panel ──────────────────────────────────
        left = QWidget()
        lv   = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0); lv.setSpacing(4)

        # Browse bar
        browse_row = QHBoxLayout()
        self.le_dir  = QLineEdit()
        self.le_dir.setPlaceholderText("Output directory…")
        self.le_dir.setReadOnly(True)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse)
        browse_row.addWidget(self.le_dir)
        browse_row.addWidget(btn_browse)
        lv.addLayout(browse_row)

        # Wave selector (includes Input Pump for density view)
        wave_row = QHBoxLayout()
        wave_row.addWidget(QLabel("Display:"))
        self.cb_wave = QComboBox()
        self.cb_wave.addItems(_ALL_WAVES)
        self.cb_wave.setCurrentIndex(2)   # default: Signal
        wave_row.addWidget(self.cb_wave)
        wave_row.addStretch()
        lv.addLayout(wave_row)

        # Spatial canvas
        self.canvas_xy = PlotCanvas(nrows=1, ncols=1, figsize=(5, 5), dpi=95)
        lv.addWidget(self.canvas_xy, stretch=1)
        self._cbar_xy = None  # single colorbar reference — removed before re-draw

        # t-slider
        t_row = QHBoxLayout()
        t_row.addWidget(QLabel("t-slice:"))
        self.sl_t   = QSlider(Qt.Orientation.Horizontal)
        self.sl_t.setRange(0, 0); self.sl_t.setValue(0)
        self.lbl_t  = QLabel("—")
        t_row.addWidget(self.sl_t, stretch=1)
        t_row.addWidget(self.lbl_t)
        lv.addLayout(t_row)

        splitter.addWidget(left)

        # ── RIGHT: 4 stacked P(t) panels (Input Pump + 3 outputs) ───
        right = QWidget()
        rv    = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0); rv.setSpacing(4)

        self.canvas_pt = {
            "Input Pump": PlotCanvas(nrows=1, ncols=1, figsize=(6, 2.5), dpi=90),
            "Pump":       PlotCanvas(nrows=1, ncols=1, figsize=(6, 2.5), dpi=90),
            "Signal":     PlotCanvas(nrows=1, ncols=1, figsize=(6, 2.5), dpi=90),
            "Idler":      PlotCanvas(nrows=1, ncols=1, figsize=(6, 2.5), dpi=90),
        }
        for w in _ALL_WAVES:
            rv.addWidget(self.canvas_pt[w], stretch=1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        # Internal data
        self._fields: dict[str, "np.ndarray | None"] = {w: None for w in _ALL_WAVES}
        self._t_ps:   "np.ndarray | None" = None
        self._cfg:    dict | None = None

        # Connections
        self.sl_t.valueChanged.connect(self._on_t_slider)
        self.cb_wave.currentTextChanged.connect(self._draw_xy)

        self._init_placeholders()

    # ── Init / placeholders ────────────────────────────────────────────

    def _init_placeholders(self):
        ax = self.canvas_xy.get_ax(0)
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                color="#555555", transform=ax.transAxes, fontsize=11)
        ax.set_title("NX×NY  |A|²  at  t-slice", fontsize=9)
        self.canvas_xy.refresh()

        for w in _ALL_WAVES:
            ax = self.canvas_pt[w].get_ax(0)
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    color="#555555", transform=ax.transAxes, fontsize=10)
            label = "Input Pump" if w == "Input Pump" else f"Output {w}"
            ax.set_title(f"{label} — P(t)", fontsize=9)
            self.canvas_pt[w].refresh()

    # ── Public API ──────────────────────────────────────────────────────

    def load_output_dir(self, out_dir: str):
        """Load results from an engine output directory."""
        self.le_dir.setText(out_dir)

        # Config
        cfg_path = os.path.join(out_dir, "config.json")
        self._cfg = json.load(open(cfg_path)) if os.path.isfile(cfg_path) else None

        # Fields  [NT, NY, NX]
        for name, fname in [("Input Pump", "pump_input_XY.h5"),
                             ("Pump",       "pump_output.h5"),
                             ("Signal",     "signal_output.h5"),
                             ("Idler",      "idler_output.h5")]:
            self._fields[name] = _load_complex(os.path.join(out_dir, fname))

        # Time vector  [NT] ps
        self._t_ps = _load_vector(os.path.join(out_dir, "time.h5"))

        # Determine NT from whichever field loaded
        ref = next((v for v in self._fields.values() if v is not None), None)
        if ref is None:
            return

        nt = ref.shape[0]
        self.sl_t.setRange(0, nt - 1)
        self.sl_t.setValue(nt // 2)

        self._draw_xy()
        self._draw_all_pt()

    # ── Browse ─────────────────────────────────────────────────────────

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select output directory", "")
        if d:
            self.load_output_dir(d)

    # ── Helpers ────────────────────────────────────────────────────────

    def _t_label(self, it: int) -> str:
        if self._t_ps is not None and it < len(self._t_ps):
            return f"{self._t_ps[it]:.3f} ps"
        if self._cfg:
            tw = self._cfg["fields"].get("t_window_ps", 1.0)
            ref = next((v for v in self._fields.values() if v is not None), None)
            nt  = ref.shape[0] if ref is not None else 1
            return f"{-tw/2 + it * tw / max(nt - 1, 1):.3f} ps"
        return f"it={it}"

    def _grid_params(self, nt: int, ny: int, nx: int):
        """Return (dx, dy, dt, n_p, n_s, n_i, lx_um, ly_um) from config or fallback."""
        dx = dy = dt = 1.0
        n_p = n_s = n_i = 2.0
        lx_um = ly_um = float(nx)

        if self._cfg:
            geom  = self._cfg["crystal"]["geometry"]
            opt   = self._cfg["crystal"]["optics"]
            lx_um = geom.get("LX_mm", 0.5) * 1e3
            ly_um = geom.get("LY_mm", 0.5) * 1e3
            dx    = lx_um / nx
            dy    = ly_um / ny
            tw    = self._cfg["fields"].get("t_window_ps", 1.0)
            dt    = tw / nt
            n_p   = opt.get("np", 2.0)
            n_s   = opt.get("ns", 2.0)
            n_i   = opt.get("ni", 2.0)

        return dx, dy, dt, n_p, n_s, n_i, lx_um, ly_um

    def _power_curve(self, A: np.ndarray, n: float,
                     dx: float, dy: float) -> np.ndarray:
        """P(t) [W] = 0.5 · n · ε₀ · c · dx · dy · Σ_{x,y}|A(t,y,x)|²"""
        return 0.5 * n * _EPS0 * _C * dx * dy * np.sum(np.abs(A) ** 2, axis=(1, 2))

    @staticmethod
    def _power_scale(p_max: float):
        """Return (scale_factor, unit_string) for the y-axis."""
        for threshold, scale, unit in [
            (1e9,  1e-9,  "GW"),
            (1e6,  1e-6,  "MW"),
            (1e3,  1e-3,  "kW"),
            (1.0,  1.0,   "W"),
            (1e-3, 1e3,   "mW"),
        ]:
            if p_max >= threshold:
                return scale, unit
        return 1e6, "μW"

    # ── Draw: spatial panel ────────────────────────────────────────────

    def _draw_xy(self):
        wave = self.cb_wave.currentText()
        A    = self._fields.get(wave)

        ax = self.canvas_xy.get_ax(0)

        # Remove previous colorbar before clearing axes to avoid accumulation
        if self._cbar_xy is not None:
            self._cbar_xy.remove()
            self._cbar_xy = None
        ax.cla()

        if A is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    color="#555555", transform=ax.transAxes, fontsize=11)
            ax.set_title(f"{wave}  |A|²  at  t-slice", fontsize=9)
            self.canvas_xy.refresh()
            return

        nt, ny, nx = A.shape
        it = min(int(self.sl_t.value()), nt - 1)
        dx, dy, dt, *_, lx_um, ly_um = self._grid_params(nt, ny, nx)

        I_slice = np.abs(A[it]) ** 2           # [NY, NX]
        ext     = [-lx_um/2, lx_um/2, -ly_um/2, ly_um/2]

        im = ax.imshow(I_slice, origin="lower", cmap="inferno",
                       aspect="equal", extent=ext, interpolation="bilinear")
        self._cbar_xy = self.canvas_xy.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self._cbar_xy.ax.tick_params(colors="white", labelsize=7)
        self._cbar_xy.ax.yaxis.label.set_color("white")

        t_str = self._t_label(it)
        ax.set_title(f"{wave}  |A|²  at  t = {t_str}", fontsize=9)
        ax.set_xlabel("x (μm)", fontsize=8)
        ax.set_ylabel("y (μm)", fontsize=8)
        self.lbl_t.setText(t_str)

        self.canvas_xy.fig.tight_layout(pad=0.4)
        self.canvas_xy.refresh()

    # ── Draw: P(t) panels ──────────────────────────────────────────────

    def _draw_all_pt(self):
        ref = next((v for v in self._fields.values() if v is not None), None)
        if ref is None:
            return
        nt, ny, nx = ref.shape
        dx, dy, dt, n_p, n_s, n_i = self._grid_params(nt, ny, nx)[:6]

        # Build time axis
        if self._t_ps is not None and len(self._t_ps) == nt:
            t_ax = self._t_ps
        else:
            t_ax = np.linspace(-dt * nt / 2, dt * nt / 2, nt, dtype=np.float32)

        n_map = {"Input Pump": n_p, "Pump": n_p, "Signal": n_s, "Idler": n_i}

        for wave in _ALL_WAVES:
            A      = self._fields[wave]
            canvas = self.canvas_pt[wave]
            color  = _WAVE_COLORS[wave]
            ax     = canvas.get_ax(0)
            ax.cla()

            label = "Input Pump" if wave == "Input Pump" else f"Output {wave}"

            if A is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        color="#555555", transform=ax.transAxes, fontsize=10)
                ax.set_title(f"{label} — P(t)", fontsize=9)
                canvas.refresh()
                continue

            Pt    = self._power_curve(A, n_map[wave], dx, dy)   # [W]
            E_J   = float(np.sum(Pt)) * float(dt) * 1e-12        # J
            e_str = _format_energy(E_J)

            p_max         = float(np.max(Pt)) if Pt.size else 0.0
            scale, p_unit = self._power_scale(p_max)

            ax.plot(t_ax, Pt * scale, color=color, linewidth=1.2)
            ax.fill_between(t_ax, Pt * scale, alpha=0.15, color=color)
            ax.set_title(f"{label} — Energy = {e_str}", fontsize=9)
            ax.set_xlabel("t (ps)", fontsize=8)
            ax.set_ylabel(f"P ({p_unit})", fontsize=8)
            ax.set_xlim(float(t_ax[0]), float(t_ax[-1]))

            canvas.fig.tight_layout(pad=0.4)
            canvas.refresh()

    # ── Slots ──────────────────────────────────────────────────────────

    def _on_t_slider(self):
        self._draw_xy()
