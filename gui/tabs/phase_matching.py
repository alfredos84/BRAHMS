import numpy as np
from scipy.optimize import brentq

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QGroupBox,
    QLabel, QComboBox, QDoubleSpinBox, QPushButton,
    QGridLayout, QFrame, QScrollArea, QSpinBox, QTabWidget
)
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal

from ..widgets.plot_canvas import PlotCanvas
from ..core_py.crystal_db  import get_db
from ..core_py.phase_matching import (
    PhaseMatchingCalc, UniaxialPMCalc, PM_TYPE_LABELS, wavelengths_for_scan,
)


# ── QPM phase-matching helpers ──────────────────────────────────────────────────
#
# Sign convention (same as Boyd / Fejer, verified from literature):
#
#   SHG  : Δk = k(λ_SH)  − 2·k(λ_fund) − 2π/Λ(T)
#           where λ_SH = λ_fund/2  (energy conservation: 2ω → ω + ω)
#
#   OPG/DFG/SFG : Δk = k(λ_pump) − k(λ_signal) − k(λ_idler) − 2π/Λ(T)
#           energy conservation: ω_p = ω_s + ω_i → 1/λ_p = 1/λ_s + 1/λ_i
#
#   K_QPM = 2π/Λ(T)  is SUBTRACTED (compensates the positive bare Δk
#           that arises from normal material dispersion)
#
# QPM crystals (PPLN, PPLT, …) are periodically-poled ferroelectrics: every
# field always propagates on the extraordinary axis, so PhaseMatchingCalc.dk()
# is used here with its "e","e","e" defaults throughout (no o/e choice).
# ──────────────────────────────────────────────────────────────────────────────

def _shg_find_pm_fund(calc: PhaseMatchingCalc,
                      T: float, Lambda: float, lp_hint: float,
                      lp_min: float = 0.3, lp_max: float = 6.0,
                      n_pts: int = 3000) -> tuple[float, float]:
    """
    SHG: find fundamental wavelength λ_fund where Δk = 0 at given T, Λ.

    Δk = k(λ_fund/2) − 2·k(λ_fund) − 2π/Λ(T) = 0

    Scans λ_fund ∈ [lp_min, lp_max] μm (widened around lp_hint).
    Returns (λ_SH, λ_SH) = (λ_fund_PM/2, λ_fund_PM/2), or (nan, nan).
    """
    # Widen search range symmetrically around lp_hint while staying in bounds
    lo = max(lp_min, lp_hint * 0.40)
    hi = min(lp_max, lp_hint * 2.50)
    lp_scan = np.linspace(lo, hi, n_pts)

    dk_scan = np.array([
        calc.dk(lf / 2.0, lf, lf, T, Lambda)
        for lf in lp_scan
    ])
    finite = np.isfinite(dk_scan)
    if not np.any(finite):
        return np.nan, np.nan

    lp_f = lp_scan[finite]
    dk_f = dk_scan[finite]
    signs = np.where(np.diff(np.sign(dk_f)))[0]
    if len(signs) == 0:
        return np.nan, np.nan

    j = signs[0]
    try:
        lf_pm = brentq(
            lambda lf: calc.dk(lf / 2.0, lf, lf, T, Lambda),
            float(lp_f[j]), float(lp_f[j + 1]), xtol=1e-8
        )
        lsh = lf_pm / 2.0
        return lsh, lsh
    except Exception:
        return np.nan, np.nan


def _pm_find_pair_opg(calc: PhaseMatchingCalc,
                      lp: float, T: float, Lambda: float,
                      lam_min: float = 0.3, lam_max: float = 5.0
                      ) -> tuple[float, float]:
    """
    OPG/DFG/SFG: find (λ_signal, λ_idler) satisfying simultaneously:
      • Energy conservation : 1/λ_p = 1/λ_s + 1/λ_i    (λ_p shortest)
      • Phase matching      : Δk = k(λ_p) − k(λ_s) − k(λ_i) − 2π/Λ = 0

    Only pairs where BOTH λ_s and λ_i are within [lam_min, lam_max] are
    considered — this prevents returning unphysical solutions where one
    wavelength lies outside the Sellmeier validity range of the crystal
    (e.g., λ_i = 18 μm for PPLN whose transparency ends at ~5 μm).
    """
    # λ_s > λ_p (from energy conservation); λ_i = λ_p·λ_s/(λ_s−λ_p) also > λ_p
    #
    # Use a composite scan: coarse coverage + dense region near the degenerate
    # point (ls = li = 2*lp).  Near degeneracy dk is very flat so the sign
    # change can be extremely narrow; without extra density it gets missed and
    # the tuning curves appear to terminate before reaching 2*lp.
    ls_coarse = np.linspace(lp * 1.001, lp * 10.0, 4000)
    ls_fine   = np.linspace(lp * 1.97,  lp * 2.03, 2000)  # dense near 2*lp
    ls_scan   = np.unique(np.concatenate([ls_coarse, ls_fine]))
    li_scan   = lp * ls_scan / (ls_scan - lp)

    # Keep only pairs where BOTH wavelengths are inside the crystal's
    # transparency / Sellmeier-validity window
    valid = (
        (ls_scan >= lam_min) & (ls_scan <= lam_max) &
        (li_scan >= lam_min) & (li_scan <= lam_max)
    )
    ls_scan = ls_scan[valid]
    li_scan = li_scan[valid]
    if len(ls_scan) < 2:
        return np.nan, np.nan

    dk_scan = np.array([
        calc.dk(lp, float(ls), float(li), T, Lambda)
        for ls, li in zip(ls_scan, li_scan)
    ])
    finite = np.isfinite(dk_scan)
    if not np.any(finite):
        return np.nan, np.nan

    ls_f = ls_scan[finite]
    li_f = li_scan[finite]
    dk_f = dk_scan[finite]
    signs = np.where(np.diff(np.sign(dk_f)))[0]

    if len(signs) == 0:
        # No sign change — happens at the exact degenerate temperature where dk
        # has a double root (tangent to zero) instead of a simple crossing.
        # Fall back to the minimum of |dk|: if it's small relative to the
        # overall dk scale, the operating point is very close to PM.
        abs_dk   = np.abs(dk_f)
        i_min    = int(np.argmin(abs_dk))
        dk_scale = float(np.max(abs_dk))
        if dk_scale > 0 and abs_dk[i_min] / dk_scale < 0.02:
            # Refine the minimum with a tighter Brent bracket around it
            i_lo = max(i_min - 1, 0)
            i_hi = min(i_min + 1, len(ls_f) - 1)
            try:
                from scipy.optimize import minimize_scalar
                def _obj(ls_val):
                    li_val = lp * ls_val / (ls_val - lp)
                    return abs(calc.dk(lp, ls_val, li_val, T, Lambda))
                res = minimize_scalar(
                    _obj,
                    bounds=(float(ls_f[i_lo]), float(ls_f[i_hi])),
                    method="bounded",
                )
                if res.success:
                    ls_opt = float(res.x)
                    li_opt = lp * ls_opt / (ls_opt - lp)
                    if lam_min <= ls_opt <= lam_max and lam_min <= li_opt <= lam_max:
                        if li_opt > ls_opt:
                            ls_opt, li_opt = li_opt, ls_opt
                        return ls_opt, li_opt
            except Exception:
                pass
            ls_ret, li_ret = float(ls_f[i_min]), float(li_f[i_min])
            if li_ret > ls_ret:
                ls_ret, li_ret = li_ret, ls_ret
            return ls_ret, li_ret
        return np.nan, np.nan

    # Among all sign changes, pick the one whose midpoint ls is closest to
    # the degenerate point (ls = 2*lp).  This selects the near-degenerate
    # OPG branch (e.g. 532 nm → 1064 + 1064 nm) rather than the first
    # (often unphysical or inverted) crossing that happens to appear first
    # in a left-to-right scan.
    ls_deg_target = 2.0 * lp
    j = int(min(signs,
                key=lambda jj: abs(0.5 * (ls_f[jj] + ls_f[jj + 1]) - ls_deg_target)))
    try:
        ls_pm = brentq(
            lambda ls_val: calc.dk(
                lp, ls_val, lp * ls_val / (ls_val - lp), T, Lambda),
            float(ls_f[j]), float(ls_f[j + 1]), xtol=1e-8
        )
        li_pm = lp * ls_pm / (ls_pm - lp)
        # Enforce convention: signal = longer wavelength
        if li_pm > ls_pm:
            ls_pm, li_pm = li_pm, ls_pm
        return ls_pm, li_pm
    except Exception:
        return np.nan, np.nan


# ── QPM background worker ───────────────────────────────────────────────────────

class _PMWorker(QObject):
    done  = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, calc: PhaseMatchingCalc, process: str,
                 lp_fix: float, T_fix: float, Lambda_fix: float,
                 T_arr: np.ndarray,
                 Lambda_arr: np.ndarray,
                 lam_min: float = 0.3, lam_max: float = 5.0):
        super().__init__()
        self._calc       = calc
        self._process    = process
        self._lp_fix     = lp_fix
        self._T_fix      = T_fix
        self._Lambda_fix = Lambda_fix
        self._T_arr      = T_arr
        self._Lambda_arr = Lambda_arr
        self._lam_min    = lam_min
        self._lam_max    = lam_max

    def run(self):
        try:
            calc    = self._calc
            process = self._process
            lp_fix  = self._lp_fix
            T_fix   = self._T_fix
            L_fix   = self._Lambda_fix

            ls_T  = np.full(len(self._T_arr),      np.nan)
            li_T  = np.full(len(self._T_arr),      np.nan)
            ls_L  = np.full(len(self._Lambda_arr), np.nan)
            li_L  = np.full(len(self._Lambda_arr), np.nan)

            if process == "SHG":
                # ── SHG sweeps ────────────────────────────────────────────────
                # Convention: Δk = k(λ_fund/2) − 2·k(λ_fund) − 2π/Λ(T) = 0
                # lp_fix is the reference fundamental wavelength (search hint).
                #
                # Sweep 1 (vs T): at each T find λ_fund_PM(T) then λ_SH = λ_fund_PM/2
                for i, T in enumerate(self._T_arr):
                    ls_T[i], li_T[i] = _shg_find_pm_fund(calc, T, L_fix, lp_fix)

                # Sweep 2 (vs Λ): at each Λ find λ_fund_PM(Λ) then λ_SH = λ_fund_PM/2
                for i, Lam in enumerate(self._Lambda_arr):
                    ls_L[i], li_L[i] = _shg_find_pm_fund(calc, T_fix, Lam, lp_fix)

            else:
                # ── OPG / DFG / SFG sweeps ────────────────────────────────────
                # Convention: Δk = k(λ_p) − k(λ_s) − k(λ_i) − 2π/Λ = 0
                # Energy conservation: 1/λ_p = 1/λ_s + 1/λ_i  (λ_p shortest)
                mn, mx = self._lam_min, self._lam_max
                for i, T in enumerate(self._T_arr):
                    ls_T[i], li_T[i] = _pm_find_pair_opg(calc, lp_fix, T, L_fix, mn, mx)
                for i, Lam in enumerate(self._Lambda_arr):
                    ls_L[i], li_L[i] = _pm_find_pair_opg(calc, lp_fix, T_fix, Lam, mn, mx)

            self.done.emit({
                "T_arr":    self._T_arr,
                "ls_vs_T":  ls_T,  "li_vs_T":  li_T,
                "L_arr":    self._Lambda_arr,
                "ls_vs_L":  ls_L,  "li_vs_L":  li_L,
                "lp_fix":   lp_fix, "T_fix": T_fix, "L_fix": L_fix,
                "process":  process,
            })
        except Exception as exc:
            self.error.emit(str(exc))


# ── QPM sub-tab ──────────────────────────────────────────────────────────────────

class _QPMPanel(QWidget):

    # lp_um, ls_um, li_um, T_C, Lambda_um
    pm_wavelengths_found = pyqtSignal(float, float, float, float, float)

    def __init__(self, db, parent=None):
        super().__init__(parent)
        self._db        = db
        self._thread    = None
        self._pm_result = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # ── LEFT: scroll controls ───────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left = QWidget()
        left.setMinimumWidth(260)
        left.setMaximumWidth(360)
        lv = QVBoxLayout(left)
        lv.setSpacing(10)
        lv.setContentsMargins(4, 4, 4, 4)

        # ── Crystal & Process ─────────────────────────────────────────
        gb_cr = QGroupBox("Crystal & Process")
        g = QGridLayout(gb_cr)
        g.setSpacing(6)
        g.addWidget(QLabel("Crystal"), 0, 0)
        self.cb_crystal = QComboBox()
        g.addWidget(self.cb_crystal, 0, 1)
        g.addWidget(QLabel("Process"), 1, 0)
        self.cb_process = QComboBox()
        self.cb_process.addItems(["SHG", "OPG", "SFG", "DFG"])
        g.addWidget(self.cb_process, 1, 1)
        g.addWidget(QLabel("Λ grating (μm)"), 2, 0)
        self.sb_Lambda = QDoubleSpinBox()
        self.sb_Lambda.setRange(0.1, 1000)
        self.sb_Lambda.setDecimals(3)
        self.sb_Lambda.setValue(6.99)
        g.addWidget(self.sb_Lambda, 2, 1)
        g.addWidget(QLabel("Crystal length (mm)"), 3, 0)
        self.sb_L = QDoubleSpinBox()
        self.sb_L.setRange(0.1, 200)
        self.sb_L.setDecimals(2)
        self.sb_L.setValue(20.0)
        g.addWidget(self.sb_L, 3, 1)
        lv.addWidget(gb_cr)

        # ── Operating Point ───────────────────────────────────────────
        gb_op = QGroupBox("Operating Point  →  find λ_s, λ_i")
        go = QGridLayout(gb_op)
        go.setSpacing(6)
        go.addWidget(QLabel("λ_p (μm)"), 0, 0)
        self.sb_op_lp = QDoubleSpinBox()
        self.sb_op_lp.setRange(0.1, 20)
        self.sb_op_lp.setDecimals(4)
        self.sb_op_lp.setValue(1.064)
        go.addWidget(self.sb_op_lp, 0, 1)
        go.addWidget(QLabel("T (°C)"), 1, 0)
        self.sb_op_T = QDoubleSpinBox()
        self.sb_op_T.setRange(-100, 1000)
        self.sb_op_T.setDecimals(1)
        self.sb_op_T.setValue(27.0)
        go.addWidget(self.sb_op_T, 1, 1)
        go.addWidget(QLabel("Λ (μm)"), 2, 0)
        self.sb_op_Lambda = QDoubleSpinBox()
        self.sb_op_Lambda.setRange(0.1, 1000)
        self.sb_op_Lambda.setDecimals(3)
        self.sb_op_Lambda.setValue(6.99)
        go.addWidget(self.sb_op_Lambda, 2, 1)

        self.btn_find = QPushButton("Find λ_s, λ_i")
        self.btn_find.setObjectName("runButton")
        go.addWidget(self.btn_find, 3, 0, 1, 2)

        self.lbl_op_ls  = QLabel("λ_s = —")
        self.lbl_op_li  = QLabel("λ_i = —")
        self.lbl_op_tpm = QLabel("T_PM = —")
        for lbl in (self.lbl_op_ls, self.lbl_op_li, self.lbl_op_tpm):
            lbl.setObjectName("valueLabel")
            lbl.setWordWrap(True)
        go.addWidget(self.lbl_op_ls,  4, 0, 1, 2)
        go.addWidget(self.lbl_op_li,  5, 0, 1, 2)
        go.addWidget(self.lbl_op_tpm, 6, 0, 1, 2)

        self.btn_load_sim = QPushButton("→ Load into Simulation")
        self.btn_load_sim.setEnabled(False)
        go.addWidget(self.btn_load_sim, 7, 0, 1, 2)
        lv.addWidget(gb_op)

        # ── Scan Ranges ───────────────────────────────────────────────
        gb_sc = QGroupBox("Scan Ranges")
        gs = QGridLayout(gb_sc)
        gs.setSpacing(5)

        def _add_range(row, label, attr_min, attr_max, attr_n,
                       vmin, vmax, vn, dec=2, rmin=-273, rmax=10000):
            gs.addWidget(QLabel(label), row, 0, 1, 3)
            sb_min = QDoubleSpinBox()
            sb_min.setRange(rmin, rmax); sb_min.setDecimals(dec); sb_min.setValue(vmin)
            sb_max = QDoubleSpinBox()
            sb_max.setRange(rmin, rmax); sb_max.setDecimals(dec); sb_max.setValue(vmax)
            sb_n   = QSpinBox()
            sb_n.setRange(10, 2000); sb_n.setValue(vn)
            gs.addWidget(QLabel("min"), row+1, 0); gs.addWidget(sb_min, row+1, 1, 1, 2)
            gs.addWidget(QLabel("max"), row+2, 0); gs.addWidget(sb_max, row+2, 1, 1, 2)
            gs.addWidget(QLabel("N"),   row+3, 0); gs.addWidget(sb_n,   row+3, 1, 1, 2)
            setattr(self, attr_min, sb_min)
            setattr(self, attr_max, sb_max)
            setattr(self, attr_n,   sb_n)

        _add_range(0,  "T scan (°C)",    "sb_Tmin",  "sb_Tmax",  "sb_TN",   10,   200, 100, 1,
                   rmin=-273, rmax=1000)
        _add_range(4,  "Λ scan (μm)",    "sb_Lmin",  "sb_Lmax",  "sb_LN",    5.0,  12.0, 100, 3,
                   rmin=0.001, rmax=1000)
        lv.addWidget(gb_sc)

        self.lbl_status = QLabel("")
        self.lbl_status.setObjectName("unitLabel")
        self.lbl_status.setWordWrap(True)
        lv.addWidget(self.lbl_status)

        self.btn_calc = QPushButton("Calculate")
        self.btn_calc.setObjectName("runButton")
        lv.addWidget(self.btn_calc)
        lv.addStretch()

        scroll.setWidget(left)
        splitter.addWidget(scroll)

        # ── RIGHT: 2 plots  (1 row × 2 cols) ───────────────────────────
        self.canvas = PlotCanvas(nrows=1, ncols=2, figsize=(14, 5), dpi=95)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._init_axes()
        self._populate_crystals()

        self.cb_crystal.currentIndexChanged.connect(self._on_crystal_changed)
        self.sb_Lambda.valueChanged.connect(
            lambda v: self.sb_op_Lambda.setValue(v))
        self.btn_calc.clicked.connect(self._calculate)
        self.btn_find.clicked.connect(self._find_pm_point)
        self.btn_load_sim.clicked.connect(self._emit_load_to_sim)

    # ── Setup ───────────────────────────────────────────────────────────
    def _populate_crystals(self):
        self.cb_crystal.clear()
        for name in self._db.all_names():
            cr = self._db.get(name)
            if cr and cr.get("type", "").startswith("QPM"):
                self.cb_crystal.addItem(name)
        self._on_crystal_changed(0)

    def _on_crystal_changed(self, _idx):
        name = self.cb_crystal.currentText()
        cr   = self._db.get(name)
        if cr:
            lam0 = cr.get("lambda0", 0.0)
            if lam0 > 0:
                self.sb_Lambda.setValue(lam0)
                self.sb_op_Lambda.setValue(lam0)
            self.lbl_status.setText(cr.get("reference", ""))

    def _init_axes(self):
        titles = [
            ("T  (°C)",  "λ  (μm)", "λ_s , λ_i  vs  Temperature"),
            ("Λ  (μm)",  "λ  (μm)", "λ_s , λ_i  vs  Grating period"),
        ]
        for i, (xl, yl, tl) in enumerate(titles):
            ax = self.canvas.get_ax(i)
            ax.set_xlabel(xl, fontsize=9)
            ax.set_ylabel(yl, fontsize=9)
            ax.set_title(tl, fontsize=9)
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="#555555", fontsize=10)
        self.canvas.fig.tight_layout()
        self.canvas.refresh()

    # ── Calculation ─────────────────────────────────────────────────────
    def _make_calc(self):
        """Build a PhaseMatchingCalc for the currently selected (QPM) crystal."""
        name = self.cb_crystal.currentText()
        cr   = self._db.get(name)
        if cr is None:
            return None, None
        sf_e = self._db.sellmeier(cr, axis="e")
        if not sf_e.is_ready:
            self.lbl_status.setText(f"Sellmeier error: {sf_e.error}")
            return None, None
        calc = PhaseMatchingCalc(
            sf_e, None,
            alpha_th=cr.get("alpha_th", 14.8e-6),
            T0=cr.get("T0", 27.0),
        )
        return calc, cr

    def _calculate(self):
        if self._thread and self._thread.isRunning():
            return

        calc, cr = self._make_calc()
        if calc is None:
            return

        process  = self.cb_process.currentText()
        lp_fix   = self.sb_op_lp.value()
        T_fix    = self.sb_op_T.value()
        L_fix    = self.sb_op_Lambda.value()
        T_arr    = np.linspace(self.sb_Tmin.value(),  self.sb_Tmax.value(),  int(self.sb_TN.value()))
        L_arr    = np.linspace(self.sb_Lmin.value(),  self.sb_Lmax.value(),  int(self.sb_LN.value()))

        self.btn_calc.setEnabled(False)
        self.lbl_status.setText("Computing…")

        lam_min = (cr or {}).get("lambda_min", 0.3)
        lam_max = (cr or {}).get("lambda_max", 5.0)
        self._worker = _PMWorker(calc, process, lp_fix, T_fix, L_fix,
                                 T_arr, L_arr, lam_min, lam_max)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.done.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_result(self, res: dict):
        self.btn_calc.setEnabled(True)

        ls_c = "#00e676"   # signal colour
        li_c = "#40c4ff"   # idler colour
        proc = res["process"]
        degenerate = (proc == "SHG")

        def _plot_pair(ax, x, ls, li, xlabel, title):
            ax.clear()
            self.canvas._style_ax(ax)
            mask_s = np.isfinite(ls)
            mask_i = np.isfinite(li)
            if mask_s.any():
                ax.plot(x[mask_s], ls[mask_s], color=ls_c, lw=1.8, label="λ_s")
            if not degenerate and mask_i.any():
                ax.plot(x[mask_i], li[mask_i], color=li_c, lw=1.8, label="λ_i")
            if mask_s.any() or mask_i.any():
                ax.legend(fontsize=8)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel("λ  (μm)", fontsize=9)
            ax.set_title(title, fontsize=9)

        # Fixed-point annotations in titles
        lp  = res["lp_fix"]; T = res["T_fix"]; L = res["L_fix"]

        if proc == "SHG":
            # Sweeps 1&2: y-axis = λ_SH (the phase-matched SH wavelength)
            _plot_pair(self.canvas.get_ax(0),
                       res["T_arr"],  res["ls_vs_T"],  res["li_vs_T"],
                       "T  (°C)",
                       f"λ_SH(T)  [Λ = {L:.3f} μm,  λ_fund hint = {lp:.4f} μm]")
            _plot_pair(self.canvas.get_ax(1),
                       res["L_arr"],  res["ls_vs_L"],  res["li_vs_L"],
                       "Λ  (μm)",
                       f"λ_SH(Λ)  [T = {T:.1f} °C,  λ_fund hint = {lp:.4f} μm]")
        else:
            _plot_pair(self.canvas.get_ax(0),
                       res["T_arr"],  res["ls_vs_T"],  res["li_vs_T"],
                       "T  (°C)",
                       f"λ_s , λ_i  vs  T    [λ_p = {lp:.4f} μm,  Λ = {L:.3f} μm]")
            _plot_pair(self.canvas.get_ax(1),
                       res["L_arr"],  res["ls_vs_L"],  res["li_vs_L"],
                       "Λ  (μm)",
                       f"λ_s , λ_i  vs  Λ    [λ_p = {lp:.4f} μm,  T = {T:.1f} °C]")

        self.canvas.fig.tight_layout()
        self.canvas.refresh()

        # Status summary
        valid_T = res["ls_vs_T"][np.isfinite(res["ls_vs_T"])]
        if valid_T.size:
            label_s = "λ_SH" if proc == "SHG" else "λ_s"
            self.lbl_status.setText(
                f"{label_s} range (vs T): {valid_T.min():.4f} – {valid_T.max():.4f} μm")
        else:
            self.lbl_status.setText(
                "No PM solution found. Check scan ranges and Λ.")

    def _on_error(self, msg: str):
        self.btn_calc.setEnabled(True)
        self.lbl_status.setText(f"Error: {msg}")

    # ── Operating-point finder ──────────────────────────────────────────
    def _find_pm_point(self):
        calc, cr = self._make_calc()
        if calc is None:
            return

        lp_fund = self.sb_op_lp.value()
        T       = self.sb_op_T.value()
        Lambda  = self.sb_op_Lambda.value()
        process = self.cb_process.currentText()

        if process == "SHG":
            lp_shg = lp_fund / 2.0
            # Δk at the current operating point (always computable)
            dk_val = calc.dk(lp_shg, lp_fund, lp_fund, T, Lambda)
            # Search for T_PM in the scan range
            T_arr = np.linspace(self.sb_Tmin.value(), self.sb_Tmax.value(), 600)
            T_pm  = calc.pm_temperature(lp_shg, lp_fund, lp_fund, Lambda, T_arr)
            self._pm_result = {"lp": lp_fund, "ls": lp_shg, "li": lp_shg,
                               "T": T, "Lambda": Lambda}
            self.lbl_op_ls.setText(f"λ_SH   = {lp_shg:.4f} μm")
            self.lbl_op_li.setText(f"λ_fund = {lp_fund:.4f} μm")
            if not np.isnan(T_pm):
                self.lbl_op_tpm.setText(
                    f"Δk = {dk_val:.5f} μm⁻¹   |   T_PM = {T_pm:.2f} °C")
            else:
                self.lbl_op_tpm.setText(
                    f"Δk = {dk_val:.5f} μm⁻¹   (no T_PM in scan range)")
            self.btn_load_sim.setEnabled(True)

        else:
            cr_now = self._db.get(self.cb_crystal.currentText()) or {}
            mn = cr_now.get("lambda_min", 0.3)
            mx = cr_now.get("lambda_max", 5.0)
            ls_pm, li_pm = _pm_find_pair_opg(calc, lp_fund, T, Lambda, mn, mx)
            if np.isnan(ls_pm):
                # No PM solution — compute Δk at the degenerate point as reference
                ls_ref = min(2.0 * lp_fund, mx * 0.98)
                li_ref = (lp_fund * ls_ref / (ls_ref - lp_fund)
                          if (ls_ref - lp_fund) > 1e-6 else mx)
                try:
                    dk_ref = calc.dk(lp_fund, ls_ref, li_ref, T, Lambda)
                    dk_str = f"Δk = {dk_ref:.5f} μm⁻¹  (at degenerate point)"
                except Exception:
                    dk_str = "Δk = —"
                self.lbl_op_ls.setText("λ_s = —  (no PM solution)")
                self.lbl_op_li.setText("λ_i = —")
                self.lbl_op_tpm.setText(dk_str + f"   T = {T:.1f} °C")
                self.btn_load_sim.setEnabled(False)
                return

            # PM found — Δk should be ≈ 0 at (ls_pm, li_pm); show it as verification
            dk_val = calc.dk(lp_fund, ls_pm, li_pm, T, Lambda)
            self._pm_result = {"lp": lp_fund, "ls": ls_pm, "li": li_pm,
                               "T": T, "Lambda": Lambda}
            self.lbl_op_ls.setText(f"λ_s = {ls_pm:.4f} μm")
            self.lbl_op_li.setText(f"λ_i = {li_pm:.4f} μm")
            self.lbl_op_tpm.setText(
                f"Δk = {dk_val:.5f} μm⁻¹   @  T = {T:.1f} °C")
            self.btn_load_sim.setEnabled(True)

    def _emit_load_to_sim(self):
        if self._pm_result is None:
            return
        r = self._pm_result
        self.pm_wavelengths_found.emit(
            r["lp"], r["ls"], r["li"], r["T"], r["Lambda"])

    # ── Public API ──────────────────────────────────────────────────────
    def refresh_crystals(self):
        current = self.cb_crystal.currentText()
        self._populate_crystals()
        idx = self.cb_crystal.findText(current)
        if idx >= 0:
            self.cb_crystal.setCurrentIndex(idx)


# ── Birefringent (angle-tuned) sub-tab ──────────────────────────────────────────
#
# Uses the closed-form Table 2.1 formulas (Dmitriev, Gurzadyan & Nikogosyan,
# "Handbook of Nonlinear Optical Crystals") via UniaxialPMCalc — no root
# search: theta_pm(lam1, lam2, lam3, T, pm_type) is evaluated directly and
# vectorises over numpy arrays, so both scan plots are computed synchronously
# without a background thread.
#
# Wavelength convention here (see wavelengths_for_scan in core_py):
#   SHG              : lam1 = lam2 = lam_scan (fundamental), lam3 = lam_scan/2.
#   SFG / DFG / OPG  : lam3 = lam_fixed (pump, or sum-frequency output — same
#                       role, always the shortest wavelength); lam1 = lam_scan;
#                       lam2 from energy conservation. All three processes
#                       share the same non-degenerate 3-wave math.
# ──────────────────────────────────────────────────────────────────────────────

class _BirefringentPanel(QWidget):

    # crystal, process, lp_um, ls_um, li_um, T_C, pm_type, theta_deg
    pm_birefringent_found = pyqtSignal(str, str, float, float, float, float, str, float)

    def __init__(self, db, parent=None):
        super().__init__(parent)
        self._db = db
        self._pm_result = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # ── LEFT: scroll controls ───────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left = QWidget()
        left.setMinimumWidth(280)
        left.setMaximumWidth(380)
        lv = QVBoxLayout(left)
        lv.setSpacing(10)
        lv.setContentsMargins(4, 4, 4, 4)

        # ── Crystal & Process ─────────────────────────────────────────
        gb_cr = QGroupBox("Crystal & Process")
        g = QGridLayout(gb_cr)
        g.setSpacing(6)
        g.addWidget(QLabel("Crystal"), 0, 0)
        self.cb_crystal = QComboBox()
        g.addWidget(self.cb_crystal, 0, 1)
        g.addWidget(QLabel("Uniaxial sign"), 1, 0)
        self.lbl_sign = QLabel("")
        self.lbl_sign.setObjectName("unitLabel")
        g.addWidget(self.lbl_sign, 1, 1)
        g.addWidget(QLabel("Process"), 2, 0)
        self.cb_process = QComboBox()
        self.cb_process.addItems(["SHG", "SFG", "DFG", "OPG"])
        g.addWidget(self.cb_process, 2, 1)
        g.addWidget(QLabel("PM type"), 3, 0)
        self.cb_pm_type = QComboBox()
        g.addWidget(self.cb_pm_type, 3, 1)
        lv.addWidget(gb_cr)

        # ── Operating Point ───────────────────────────────────────────
        # Which of (lam1, lam3) is "p" vs "s" flips with the process (see
        # _on_process_changed): for SHG lam1 is the fundamental (pump) and
        # lam3=lam1/2 is signal=idler; for SFG/DFG/OPG lam3 is the pump
        # (fixed) and lam1 is the signal being scanned. The row labels are
        # updated to always read lambda_p / lambda_s / lambda_i, matching
        # the naming used everywhere else in the app instead of the
        # Table-2.1 "wave 1/2/3" convention.
        gb_op = QGroupBox("Operating Point  →  compute θ_pm")
        go = QGridLayout(gb_op)
        go.setSpacing(6)
        self.lbl_op_l1_name = QLabel("λ_p (μm)")
        go.addWidget(self.lbl_op_l1_name, 0, 0)
        self.sb_op_l1 = QDoubleSpinBox()
        self.sb_op_l1.setRange(0.1, 20); self.sb_op_l1.setDecimals(4)
        self.sb_op_l1.setValue(0.6)
        go.addWidget(self.sb_op_l1, 0, 1)
        self.lbl_op_l3_name = QLabel("λ_p (μm)  [fixed]")
        go.addWidget(self.lbl_op_l3_name, 1, 0)
        self.sb_op_l3 = QDoubleSpinBox()
        self.sb_op_l3.setRange(0.1, 20); self.sb_op_l3.setDecimals(4)
        self.sb_op_l3.setValue(0.4)
        go.addWidget(self.sb_op_l3, 1, 1)
        go.addWidget(QLabel("T (°C)"), 2, 0)
        self.sb_op_T = QDoubleSpinBox()
        self.sb_op_T.setRange(-100, 1000); self.sb_op_T.setDecimals(1)
        self.sb_op_T.setValue(20.0)
        go.addWidget(self.sb_op_T, 2, 1)

        self.btn_find = QPushButton("Compute θ_pm")
        self.btn_find.setObjectName("runButton")
        go.addWidget(self.btn_find, 3, 0, 1, 2)

        self.lbl_op_wavelengths = QLabel("λ_p = —   λ_s = —   λ_i = —")
        self.lbl_op_pm  = QLabel("θ_pm = —")
        for lbl in (self.lbl_op_wavelengths, self.lbl_op_pm):
            lbl.setObjectName("valueLabel")
            lbl.setWordWrap(True)
        go.addWidget(self.lbl_op_wavelengths, 4, 0, 1, 2)
        go.addWidget(self.lbl_op_pm, 5, 0, 1, 2)

        self.btn_load_sim = QPushButton("→ Load into Simulation")
        self.btn_load_sim.setEnabled(False)
        go.addWidget(self.btn_load_sim, 6, 0, 1, 2)
        lv.addWidget(gb_op)

        # ── Scan Ranges ───────────────────────────────────────────────
        gb_sc = QGroupBox("Scan Ranges")
        gs = QGridLayout(gb_sc)
        gs.setSpacing(5)

        def _add_range(row, label, attr_min, attr_max, attr_n,
                       vmin, vmax, vn, dec=2, rmin=-273, rmax=10000):
            lbl = QLabel(label)
            gs.addWidget(lbl, row, 0, 1, 3)
            sb_min = QDoubleSpinBox()
            sb_min.setRange(rmin, rmax); sb_min.setDecimals(dec); sb_min.setValue(vmin)
            sb_max = QDoubleSpinBox()
            sb_max.setRange(rmin, rmax); sb_max.setDecimals(dec); sb_max.setValue(vmax)
            sb_n   = QSpinBox()
            sb_n.setRange(10, 2000); sb_n.setValue(vn)
            gs.addWidget(QLabel("min"), row+1, 0); gs.addWidget(sb_min, row+1, 1, 1, 2)
            gs.addWidget(QLabel("max"), row+2, 0); gs.addWidget(sb_max, row+2, 1, 1, 2)
            gs.addWidget(QLabel("N"),   row+3, 0); gs.addWidget(sb_n,   row+3, 1, 1, 2)
            setattr(self, attr_min, sb_min)
            setattr(self, attr_max, sb_max)
            setattr(self, attr_n,   sb_n)
            return lbl

        self.lbl_l1_scan_name = _add_range(
            0, "λ_p scan (μm)", "sb_l1min", "sb_l1max", "sb_l1N",
            0.48, 0.75, 300, 4, rmin=0.05, rmax=100)
        lv.addWidget(gb_sc)

        self.lbl_status = QLabel("")
        self.lbl_status.setObjectName("unitLabel")
        self.lbl_status.setWordWrap(True)
        lv.addWidget(self.lbl_status)

        self.btn_calc = QPushButton("Calculate")
        self.btn_calc.setObjectName("runButton")
        lv.addWidget(self.btn_calc)
        lv.addStretch()

        scroll.setWidget(left)
        splitter.addWidget(scroll)

        # ── RIGHT: 1 plot ───────────────────────────────────────────────
        self.canvas = PlotCanvas(nrows=1, ncols=1, figsize=(9, 5.5), dpi=95)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._init_axes()
        self._populate_crystals()

        self.cb_crystal.currentIndexChanged.connect(self._on_crystal_changed)
        self.cb_process.currentIndexChanged.connect(self._on_process_changed)
        self.btn_calc.clicked.connect(self._calculate)
        self.btn_find.clicked.connect(self._find_pm_point)
        self.btn_load_sim.clicked.connect(self._emit_load_to_sim)
        self._on_process_changed(0)

    # ── Setup ───────────────────────────────────────────────────────────
    def _populate_crystals(self):
        self.cb_crystal.clear()
        for name in self._db.all_names():
            cr = self._db.get(name)
            if cr and cr.get("type", "") == "Birefringent-uniaxial":
                self.cb_crystal.addItem(name)
        self._on_crystal_changed(0)

    def _on_process_changed(self, _idx):
        is_shg = (self.cb_process.currentText() == "SHG")
        self.sb_op_l3.setEnabled(not is_shg)
        if is_shg:
            self.lbl_op_l1_name.setText("λ_p (μm)  [fundamental]")
            self.lbl_op_l3_name.setText("λ_p / 2  (auto: λ_s = λ_i)")
            self.lbl_l1_scan_name.setText("λ_p scan (μm)")
        else:
            self.lbl_op_l1_name.setText("λ_s (μm)  [scanned]")
            self.lbl_op_l3_name.setText("λ_p (μm)  [pump, fixed]")
            self.lbl_l1_scan_name.setText("λ_s scan (μm)")

    def _on_crystal_changed(self, _idx):
        calc = self._make_calc()
        self.cb_pm_type.clear()
        if calc is None:
            self.lbl_sign.setText("")
            return
        self.lbl_sign.setText(calc.sign.capitalize())
        for t in calc.valid_pm_types():
            self.cb_pm_type.addItem(PM_TYPE_LABELS[t], t)
        name = self.cb_crystal.currentText()
        cr = self._db.get(name) or {}
        self.lbl_status.setText(cr.get("reference", ""))

    def _init_axes(self):
        ax = self.canvas.get_ax(0)
        ax.set_xlabel("λ₁  (μm)", fontsize=9)
        ax.set_ylabel("θ_pm  (deg)", fontsize=9)
        ax.set_title("θ_pm  vs  Wavelength", fontsize=9)
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color="#555555", fontsize=10)
        self.canvas.fig.tight_layout()
        self.canvas.refresh()

    # ── Calculation ─────────────────────────────────────────────────────
    def _make_calc(self) -> UniaxialPMCalc | None:
        """Build a UniaxialPMCalc for the currently selected crystal."""
        name = self.cb_crystal.currentText()
        cr   = self._db.get(name)
        if cr is None:
            return None
        sf_o = self._db.sellmeier(cr, axis="o")
        sf_e = self._db.sellmeier(cr, axis="e")
        if not sf_o.is_ready or not sf_e.is_ready:
            self.lbl_status.setText(
                "This crystal needs both o- and e-axis Sellmeier formulas.")
            return None
        lam_ref = 0.5 * (cr.get("lambda_min", 0.4) + cr.get("lambda_max", 5.0))
        T_ref   = cr.get("T0", 25.0)
        return UniaxialPMCalc(sf_o, sf_e, lam_ref=lam_ref, T_ref=T_ref)

    def _pm_type(self) -> str | None:
        idx = self.cb_pm_type.currentIndex()
        if idx < 0:
            return None
        return self.cb_pm_type.itemData(idx)

    def _calculate(self):
        calc = self._make_calc()
        pm_type = self._pm_type()
        if calc is None or pm_type is None:
            return

        process   = self.cb_process.currentText()
        is_shg    = (process == "SHG")
        lam_fixed = self.sb_op_l3.value()
        T_fixed   = self.sb_op_T.value()
        x_label   = "λ_p  (μm)" if is_shg else "λ_s  (μm)"

        # θ_pm vs the scanned wavelength (T fixed) — λ_p itself for SHG
        # (fundamental), or λ_s for SFG/DFG/OPG (pump λ_p held fixed).
        lam_scan = np.linspace(self.sb_l1min.value(), self.sb_l1max.value(),
                               int(self.sb_l1N.value()))
        lam1, lam2, lam3 = wavelengths_for_scan(lam_scan, lam_fixed, process)
        theta_vs_lam = calc.theta_pm(lam1, lam2, lam3, T_fixed, pm_type)

        pm_label = self.cb_pm_type.currentText()

        ax0 = self.canvas.get_ax(0)
        ax0.clear(); self.canvas._style_ax(ax0)
        mask = np.isfinite(theta_vs_lam)
        if mask.any():
            ax0.plot(lam_scan[mask], np.asarray(theta_vs_lam)[mask],
                    color="#00e676", lw=1.8)
        ax0.set_xlabel(x_label, fontsize=9)
        ax0.set_ylabel("θ_pm  (deg)", fontsize=9)
        ax0.set_title(f"θ_pm({x_label.split()[0]})  [{pm_label},  T = {T_fixed:.1f} °C]", fontsize=9)

        self.canvas.fig.tight_layout()
        self.canvas.refresh()

        x_name = x_label.split()[0]
        if mask.any():
            valid = np.asarray(theta_vs_lam)[mask]
            self.lbl_status.setText(
                f"θ_pm range (vs {x_name}): {valid.min():.2f}° – {valid.max():.2f}°")
        else:
            self.lbl_status.setText(
                f"No real θ_pm solution in this {x_name} range for the chosen PM type.")

    def _find_pm_point(self):
        calc = self._make_calc()
        pm_type = self._pm_type()
        if calc is None or pm_type is None:
            return

        process   = self.cb_process.currentText()
        lam_fixed = self.sb_op_l3.value()
        T         = self.sb_op_T.value()
        lam1      = self.sb_op_l1.value()

        lam1v, lam2v, lam3v = wavelengths_for_scan(lam1, lam_fixed, process)
        theta = calc.theta_pm(lam1v, lam2v, lam3v, T, pm_type)
        lam2_scalar = float(np.asarray(lam2v))
        lam3_scalar = float(np.asarray(lam3v))

        self._pm_result = None
        self.btn_load_sim.setEnabled(False)

        if np.isnan(lam2_scalar) or (np.isnan(theta) if np.isscalar(theta) else not np.isfinite(theta)):
            self.lbl_op_wavelengths.setText(
                "λ_p = —   λ_s = —   λ_i = —   (check energy conservation / λ range)")
            self.lbl_op_pm.setText("θ_pm = —  (no real solution for this PM type)")
            return

        # Engine convention (same as core_py/config_builder.py):
        #   SHG            : pump = fundamental (lam1), signal = idler = SH (lam3)
        #   SFG / DFG / OPG: pump = shortest (lam3), signal = lam1, idler = lam2
        if process == "SHG":
            lp_eng, ls_eng, li_eng = lam1, lam3_scalar, lam3_scalar
        else:
            lp_eng, ls_eng, li_eng = lam3_scalar, lam1, lam2_scalar

        self.lbl_op_wavelengths.setText(
            f"λ_p = {lp_eng:.4f} μm   λ_s = {ls_eng:.4f} μm   λ_i = {li_eng:.4f} μm")

        if process == "SHG" and pm_type[0] != pm_type[1]:
            self.lbl_op_pm.setText(
                f"θ_pm = {float(theta):.2f}°   [{pm_type} is Type II — cannot be "
                f"loaded into a simulation: the pump field would need mixed "
                f"o+e polarization on a single input]")
            return

        self.lbl_op_pm.setText(
            f"θ_pm = {float(theta):.2f}°   [{calc.sign} uniaxial, {pm_type}]")

        self._pm_result = {
            "crystal": self.cb_crystal.currentText(), "process": process,
            "lp": lp_eng, "ls": ls_eng, "li": li_eng, "T": T,
            "pm_type": pm_type, "theta_deg": float(theta),
        }
        self.btn_load_sim.setEnabled(True)

    def _emit_load_to_sim(self):
        if self._pm_result is None:
            return
        r = self._pm_result
        self.pm_birefringent_found.emit(
            r["crystal"], r["process"], r["lp"], r["ls"], r["li"], r["T"],
            r["pm_type"], r["theta_deg"])

    # ── Public API ──────────────────────────────────────────────────────
    def refresh_crystals(self):
        current = self.cb_crystal.currentText()
        self._populate_crystals()
        idx = self.cb_crystal.findText(current)
        if idx >= 0:
            self.cb_crystal.setCurrentIndex(idx)


# ── Top-level tab: QPM + Birefringent ───────────────────────────────────────────

class PhaseMatchingTab(QWidget):

    # lp_um, ls_um, li_um, T_C, Lambda_um
    pm_wavelengths_found = pyqtSignal(float, float, float, float, float)
    # crystal, process, lp_um, ls_um, li_um, T_C, pm_type, theta_deg
    pm_birefringent_found = pyqtSignal(str, str, float, float, float, float, str, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        db = get_db()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        inner_tabs = QTabWidget()
        self._qpm_panel = _QPMPanel(db)
        self._bi_panel  = _BirefringentPanel(db)
        inner_tabs.addTab(self._qpm_panel, "QPM")
        inner_tabs.addTab(self._bi_panel,  "Birefringent")
        layout.addWidget(inner_tabs)

        self._qpm_panel.pm_wavelengths_found.connect(self.pm_wavelengths_found.emit)
        self._bi_panel.pm_birefringent_found.connect(self.pm_birefringent_found.emit)

    # ── Public API ──────────────────────────────────────────────────────
    def refresh_crystals(self):
        self._qpm_panel.refresh_crystals()
        self._bi_panel.refresh_crystals()
