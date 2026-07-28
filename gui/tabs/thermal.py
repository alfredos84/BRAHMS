import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QGroupBox,
    QLabel, QDoubleSpinBox, QPushButton, QGridLayout,
    QFrame, QScrollArea, QSlider, QSizePolicy
)
from PyQt6.QtCore import Qt

from ..widgets.plot_canvas import PlotCanvas


class ThermalTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # ── LEFT: thermal parameters ───────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        left = QWidget()
        left.setMinimumWidth(230)
        left.setMaximumWidth(290)
        lv = QVBoxLayout(left)
        lv.setSpacing(10)
        lv.setContentsMargins(4, 4, 4, 4)

        # Crystal geometry
        gb_geom = QGroupBox("Crystal Geometry")
        g = QGridLayout(gb_geom)
        g.setSpacing(6)
        for row, (lbl, attr, val) in enumerate([
            ("LX (μm)",      "sb_lx", 500.0),
            ("LY (μm)",      "sb_ly", 500.0),
            ("Length (μm)",  "sb_lz", 20000.0),
            ("NX grid",      "sb_nx", 32),
            ("NY grid",      "sb_ny", 32),
            ("NZ grid",      "sb_nz", 100),
        ]):
            g.addWidget(QLabel(lbl), row, 0)
            sb = QDoubleSpinBox()
            sb.setRange(1, 1e7); sb.setDecimals(1); sb.setValue(val)
            setattr(self, attr, sb)
            g.addWidget(sb, row, 1)
        lv.addWidget(gb_geom)

        # Oven & environment
        gb_oven = QGroupBox("Thermal Boundary Conditions")
        g2 = QGridLayout(gb_oven)
        g2.setSpacing(6)

        g2.addWidget(QLabel("Oven T (°C)"), 0, 0)
        self.sb_T_oven = QDoubleSpinBox()
        self.sb_T_oven.setRange(-100, 500); self.sb_T_oven.setValue(27.0)
        g2.addWidget(self.sb_T_oven, 0, 1)

        g2.addWidget(QLabel("Env T (°C)"), 1, 0)
        self.sb_T_env = QDoubleSpinBox()
        self.sb_T_env.setRange(-100, 500); self.sb_T_env.setValue(25.0)
        g2.addWidget(self.sb_T_env, 1, 1)

        oven_note = QLabel("Oven heats the bottom face (y = 0).\nAll other faces → T_env.")
        oven_note.setObjectName("unitLabel")
        oven_note.setWordWrap(True)
        g2.addWidget(oven_note, 2, 0, 1, 2)
        lv.addWidget(gb_oven)

        # Material thermal props
        gb_mat = QGroupBox("Material (from crystal DB)")
        g3 = QGridLayout(gb_mat)
        g3.setSpacing(6)
        for row, (lbl, attr, val) in enumerate([
            ("κ (W/m·K)",       "sb_kappa", 8.0),
            ("α (cm⁻¹) pump",   "sb_ap",    0.002e-4 * 1e4),
            ("α (cm⁻¹) signal", "sb_as",    0.025e-4 * 1e4),
        ]):
            g3.addWidget(QLabel(lbl), row, 0)
            sb = QDoubleSpinBox()
            sb.setRange(0, 1e6); sb.setDecimals(6); sb.setValue(val)
            setattr(self, attr, sb)
            g3.addWidget(sb, row, 1)
        lv.addWidget(gb_mat)

        # Solver settings
        gb_sol = QGroupBox("Solver")
        g4 = QGridLayout(gb_sol)
        g4.setSpacing(6)
        g4.addWidget(QLabel("Max iterations"), 0, 0)
        self.sb_maxiter = QDoubleSpinBox()
        self.sb_maxiter.setRange(100, 1e6); self.sb_maxiter.setDecimals(0)
        self.sb_maxiter.setValue(100000)
        g4.addWidget(self.sb_maxiter, 0, 1)
        g4.addWidget(QLabel("Tolerance"), 1, 0)
        self.sb_tol = QDoubleSpinBox()
        self.sb_tol.setRange(1e-12, 1); self.sb_tol.setDecimals(8)
        self.sb_tol.setValue(5e-4)
        g4.addWidget(self.sb_tol, 1, 1)
        lv.addWidget(gb_sol)

        # Status display
        gb_stat = QGroupBox("Status")
        gs = QGridLayout(gb_stat)
        self.lbl_tmax = QLabel("T_max: —")
        self.lbl_dt   = QLabel("ΔT: —")
        self.lbl_conv = QLabel("Convergence: —")
        for i, lbl in enumerate([self.lbl_tmax, self.lbl_dt, self.lbl_conv]):
            gs.addWidget(lbl, i, 0)
        lv.addWidget(gb_stat)

        self.btn_run_thermal = QPushButton("Compute Thermal Profile")
        self.btn_run_thermal.setObjectName("runButton")
        lv.addWidget(self.btn_run_thermal)
        lv.addStretch()

        scroll.setWidget(left)
        splitter.addWidget(scroll)

        # ── RIGHT: plots ───────────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setSpacing(6)
        rv.setContentsMargins(4, 4, 4, 4)

        self.canvas = PlotCanvas(nrows=2, ncols=2, figsize=(10, 7), dpi=95)
        rv.addWidget(self.canvas, stretch=1)

        # z-slider
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("z slice:"))
        self.slider_z = QSlider(Qt.Orientation.Horizontal)
        self.slider_z.setRange(0, 99)
        self.slider_z.setValue(50)
        slider_row.addWidget(self.slider_z)
        self.lbl_z_val = QLabel("z = 50%")
        slider_row.addWidget(self.lbl_z_val)
        rv.addLayout(slider_row)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._init_axes()
        self.btn_run_thermal.clicked.connect(self._demo_thermal)
        self.slider_z.valueChanged.connect(self._on_slider)

    def _init_axes(self):
        labels = [
            ("x  (μm)", "y  (μm)",  "T(x,y)  cross-section"),
            ("z  (mm)", "T  (°C)",  "Axial profile  T(0,0,z)"),
            ("z  (mm)", "x  (μm)",  "T(x,z)  y = y_mid  [panel e]"),
            ("z  (mm)", "y  (μm)",  "T(y,z)  x = 0  [panel f]"),
        ]
        for i, (xl, yl, tl) in enumerate(labels):
            ax = self.canvas.get_ax(i)
            ax.set_xlabel(xl, fontsize=9)
            ax.set_ylabel(yl, fontsize=9)
            ax.set_title(tl, fontsize=9)
        self.canvas.refresh()

    def _demo_thermal(self):
        """Analytic Gaussian heating demo — replaced by real solver output."""
        NX, NY, NZ = 32, 32, 100
        LX, LY, LZ = (self.sb_lx.value(), self.sb_ly.value(),
                       self.sb_lz.value() * 1e-3)   # mm
        T0    = self.sb_T_oven.value()
        T_env = self.sb_T_env.value()
        waist = min(LX, LY) * 0.1

        x = np.linspace(-LX/2, LX/2, NX)
        y = np.linspace(0, self.sb_ly.value(), NY)
        z = np.linspace(0, LZ, NZ)
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Analytic steady-state approximation: Gaussian heating + oven BC
        Q_peak = 0.5
        T3D = T_env + Q_peak * np.exp(
            -(X[:, :, np.newaxis]**2 + (Y[:, :, np.newaxis] - LX/2)**2) /
            waist**2) * np.sin(np.pi * z[np.newaxis, np.newaxis, :] / LZ)
        # Apply oven BC (bottom face y=0 → T = T_oven)
        T3D[:, 0, :] = T0

        T_max = T3D.max()
        self.lbl_tmax.setText(f"T_max = {T_max:.2f} °C")
        self.lbl_dt.setText(f"ΔT = {T_max - T0:.2f} °C")
        self.lbl_conv.setText("Convergence: ✓ (demo)")

        z_idx = self.slider_z.value()
        self._draw(T3D, x, y, z, z_idx)

    def _draw(self, T3D, x, y, z, z_idx):
        # T3D shape in demo: (NX, NY, NZ) — axes T3D[ix, iy, iz]
        ax0, ax1, ax2, ax3 = [self.canvas.get_ax(i) for i in range(4)]

        for ax in [ax0, ax1, ax2, ax3]:
            ax.clear()
            self.canvas._style_ax(ax)

        FS = 12
        mid_x = T3D.shape[0] // 2
        mid_y = T3D.shape[1] // 2
        LY = y[-1] - y[0]

        # Panel 0: T(x,y) at z_idx — flip y so oven face (iy=0) is at bottom
        im = ax0.imshow(T3D[:, ::-1, z_idx].T, origin="upper", cmap="afmhot",
                        extent=[x[0], x[-1], y[0], y[-1]], aspect="auto")
        ax0.set_xlabel("x  (μm)", fontsize=FS)
        ax0.set_ylabel("y  (μm)", fontsize=FS)
        ax0.set_title(f"T(x,y) at z = {z[z_idx]:.1f} mm", fontsize=FS)
        cb0 = self.canvas.fig.colorbar(im, ax=ax0, pad=0.02)
        cb0.set_label("T  (°C)", color="white", fontsize=FS)
        cb0.ax.tick_params(colors="white", labelsize=FS - 1)

        # Panel 1: T(0,0,z) — axial profile through beam centre
        ax1.plot(z, T3D[mid_x, mid_y, :], color="#ff6d00", linewidth=1.5)
        ax1.set_xlabel("z  (mm)", fontsize=FS)
        ax1.set_ylabel("T  (°C)", fontsize=FS)
        ax1.set_title("Axial profile  T(0,0,z)", fontsize=FS)

        # Panel 2 [paper panel e]: T(x, z) at y = y_mid (beam-centre height)
        # T3D[:, mid_y, :] → shape (NX, NZ); rows=x, cols=z
        im2 = ax2.imshow(T3D[:, mid_y, :], origin="lower", cmap="afmhot",
                         extent=[z[0], z[-1], x[0], x[-1]], aspect="auto")
        ax2.set_xlabel("z  (mm)", fontsize=FS)
        ax2.set_ylabel("x  (μm)", fontsize=FS)
        ax2.set_title(f"T(x, z)  y = {y[mid_y]:.0f} μm  [panel e]", fontsize=FS)
        cb2 = self.canvas.fig.colorbar(im2, ax=ax2, pad=0.02)
        cb2.set_label("T  (°C)", color="white", fontsize=FS)
        cb2.ax.tick_params(colors="white", labelsize=FS - 1)

        # Panel 3 [paper panel f]: T(y, z) at x = 0 (beam centre)
        # T3D[mid_x, ::-1, :] → shape (NY_flipped, NZ): rows=y (flipped), cols=z
        # origin="upper" → flipped row-0 (cool top face) at top, oven face at bottom
        T_max = T3D.max(); T_min = T3D.min()
        im3 = ax3.imshow(T3D[mid_x, ::-1, :], origin="upper", cmap="afmhot",
                         extent=[z[0], z[-1], y[0], y[-1]], aspect="auto")
        ax3.set_xlabel("z  (mm)", fontsize=FS)
        ax3.set_ylabel("y  (μm)  [0 = oven face]", fontsize=FS)
        ax3.set_title(
            f"T(y, z)  x = {x[mid_x]:.0f} μm  ΔT = {T_max - T_min:.1f} K  [panel f]",
            fontsize=FS)
        cb3 = self.canvas.fig.colorbar(im3, ax=ax3, pad=0.02)
        cb3.set_label("T  (°C)", color="white", fontsize=FS)
        cb3.ax.tick_params(colors="white", labelsize=FS - 1)

        self.canvas.refresh()
        self._T3D = T3D
        self._axes_data = (x, y, z)

    def _on_slider(self, value):
        self.lbl_z_val.setText(f"z = {value}%")
        if hasattr(self, "_T3D"):
            x, y, z = self._axes_data
            self._draw(self._T3D, x, y, z, int(value * (len(z)-1) / 100))
