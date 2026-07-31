from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QGroupBox,
    QLabel, QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QPushButton,
    QListWidget, QListWidgetItem, QTabWidget, QGridLayout,
    QPlainTextEdit, QMessageBox, QScrollArea, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont

from ..widgets.plot_canvas import PlotCanvas
from ..core_py.crystal_db import get_db
from ..core_py.sellmeier  import SellmeierFormula
from ..core_py.bibtex_utils import bibtex_to_citation

import numpy as np


# ── Background worker: parses formula + computes curves ───────────────
_C_UM_PS = 299.792458   # speed of light in μm/ps


class _PlotWorker(QObject):
    done    = pyqtSignal(object, object, object, object, object)   # lam, n, vg, dn, gvd
    error   = pyqtSignal(str)

    def __init__(self, sf: SellmeierFormula, lam_min: float,
                 lam_max: float, T: float):
        super().__init__()
        self._sf  = sf
        self._min = lam_min
        self._max = lam_max
        self._T   = T

    def run(self):
        try:
            lam = np.linspace(self._min, self._max, 500)
            n   = self._sf.n(lam, self._T)
            dn  = self._sf.dn_dL(lam, self._T)      # dn/dλ  [μm⁻¹]
            gvd = self._sf.GVD_fs2_mm(lam, self._T)
            # Group velocity: v_g = c / n_g,  n_g = n - λ·(dn/dλ)
            n_g = n - lam * dn
            vg  = (_C_UM_PS / n_g) * 1e-2           # convert μm/ps → 10⁸ m/s
            self.done.emit(lam, n, vg, dn, gvd)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Background worker: phase-matching curves ──────────────────────────
class _PMCurveWorker(QObject):
    done     = pyqtSignal(dict)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, sf_e: SellmeierFormula,
                 sf_o: "SellmeierFormula | None",
                 params: dict):
        super().__init__()
        self._sf_e = sf_e
        self._sf_o = sf_o
        self._p    = params

    # ------------------------------------------------------------------
    def run(self):
        try:
            self.done.emit(self._compute())
        except Exception as exc:
            self.error.emit(str(exc))

    # ------------------------------------------------------------------
    def _k(self, lam: float, T: float, pol: str) -> float:
        sf = self._sf_o if (pol == "o" and self._sf_o) else self._sf_e
        return float(sf.k(lam, T))

    def _Lambda_T(self, Lambda0: float, T: float) -> float:
        return Lambda0 * (1.0 + self._p["alpha_th"] * (T - self._p["T0"]))

    def _dk(self, lam_p: float, lam_s: float, lam_i: float,
             T: float, Lambda: float) -> float:
        kp = self._k(lam_p, T, self._p["pol_p"])
        ks = self._k(lam_s, T, self._p["pol_s"])
        ki = self._k(lam_i, T, self._p["pol_i"])
        dk = kp - ks - ki
        if Lambda > 0:
            dk -= 2.0 * np.pi / self._Lambda_T(Lambda, T)
        return dk

    def _find_signal(self, lam_p: float, T: float, Lambda: float,
                     lam_range: tuple) -> tuple:
        """
        Find λ_s ∈ (λ_p, 2λ_p] such that Δk = 0 for OPG/SFG/DFG.
        Returns (λ_s, λ_i) or (nan, nan) if no PM condition found.
        Energy conservation: λ_i = λ_p·λ_s / (λ_s − λ_p)
        """
        from scipy.optimize import brentq
        ls_lo = lam_p + 0.005
        ls_hi = min(2.0 * lam_p - 0.001, lam_range[1])
        if ls_hi <= ls_lo:
            return np.nan, np.nan

        def _li(ls):
            denom = ls - lam_p
            return lam_p * ls / denom if denom > 1e-9 else None

        def func(ls):
            li = _li(ls)
            if li is None or li > lam_range[1] * 2.0:
                return np.nan
            try:
                return self._dk(lam_p, ls, li, T, Lambda)
            except Exception:
                return np.nan

        ls_arr = np.linspace(ls_lo, ls_hi, 400)
        dk_arr = np.array([func(ls) for ls in ls_arr])

        finite = np.isfinite(dk_arr)
        if not np.any(finite):
            return np.nan, np.nan

        ls_f  = ls_arr[finite]
        dk_f  = dk_arr[finite]
        signs = np.where(np.diff(np.sign(dk_f)))[0]
        if len(signs) == 0:
            return np.nan, np.nan

        j = signs[0]
        try:
            ls_root = brentq(func, ls_f[j], ls_f[j + 1],
                             xtol=1e-6, maxiter=200)
            li_root = _li(ls_root)
            if li_root is None or li_root > lam_range[1] * 5.0:
                return np.nan, np.nan
            return ls_root, li_root
        except Exception:
            return np.nan, np.nan

    def _find_sfg_input2(self, lam_1: float, T: float, Lambda: float,
                         lam_range: tuple) -> tuple:
        """
        For SFG: given λ_1 (first input), find λ_2 (second input) s.t. Δk=0.
        λ_SFG = λ_1·λ_2 / (λ_1 + λ_2)   [output, always shorter than both inputs]
        Δk = k(λ_SFG) − k(λ_1) − k(λ_2) − 2π/Λ = 0
        pol convention: pol_p=pol(λ_SFG), pol_s=pol(λ_1), pol_i=pol(λ_2)
        Returns (λ_2, λ_SFG) or (nan, nan).
        """
        from scipy.optimize import brentq

        def _l_sfg(l2):
            return lam_1 * l2 / (lam_1 + l2)

        def func(l2):
            lsfg = _l_sfg(l2)
            if lsfg < lam_range[0]:
                return np.nan
            try:
                return self._dk(lsfg, lam_1, l2, T, Lambda)
            except Exception:
                return np.nan

        l2_arr = np.linspace(lam_range[0] + 0.001, lam_range[1], 500)
        dk_arr = np.array([func(l2) for l2 in l2_arr])

        finite = np.isfinite(dk_arr)
        if not np.any(finite):
            return np.nan, np.nan

        l2_f  = l2_arr[finite]
        dk_f  = dk_arr[finite]
        signs = np.where(np.diff(np.sign(dk_f)))[0]
        if len(signs) == 0:
            return np.nan, np.nan

        j = signs[0]
        try:
            l2_root   = brentq(func, l2_f[j], l2_f[j + 1], xtol=1e-6, maxiter=200)
            lsfg_root = _l_sfg(l2_root)
            if lsfg_root < lam_range[0]:
                return np.nan, np.nan
            return l2_root, lsfg_root
        except Exception:
            return np.nan, np.nan

    # ------------------------------------------------------------------
    def _compute(self) -> dict:
        p = self._p
        if p["process"] == "SHG":
            return self._compute_shg()
        if p["process"] == "SFG":
            return self._compute_sfg()
        return self._compute_opg()

    # ------------------------------------------------------------------
    def _compute_sfg(self) -> dict:
        """
        SFG: λ_pump in GUI = λ_1 (first SFG input).
        Finds λ_2 (second input) and λ_SFG = λ_1·λ_2/(λ_1+λ_2) (output).
        """
        p          = self._p
        T_arr      = p["T_arr"]
        Lambda_arr = p["Lambda_arr"]
        l1_arr     = p["lp_arr"]
        Lambda_fix = p["Lambda_fixed"]
        T_fix      = p["T_fixed"]
        l1_fix     = p["lam_p"]
        lam_range  = p["lam_range"]

        nT  = len(T_arr)
        nL  = len(Lambda_arr)
        nl1 = len(l1_arr)
        done = 0

        # Panel 1: λ_2(T), λ_SFG(T) at fixed λ_1 and Λ
        l2_T   = np.full(nT, np.nan)
        lsfg_T = np.full(nT, np.nan)
        for i, T in enumerate(T_arr):
            l2_T[i], lsfg_T[i] = self._find_sfg_input2(l1_fix, T, Lambda_fix, lam_range)
            done += 1
            self.progress.emit(int(100 * done / (nT + nL + nl1)))

        # Panel 2: λ_2(Λ), λ_SFG(Λ) at fixed λ_1 and T  — QPM only
        l2_L   = np.full(nL, np.nan)
        lsfg_L = np.full(nL, np.nan)
        if p["is_qpm"] and nL > 1:
            for i, L in enumerate(Lambda_arr):
                l2_L[i], lsfg_L[i] = self._find_sfg_input2(l1_fix, T_fix, L, lam_range)
                done += 1
                self.progress.emit(int(100 * done / (nT + nL + nl1)))
        else:
            done += nL

        # Panel 3: λ_2(λ_1), λ_SFG(λ_1) at fixed T and Λ
        l2_l1   = np.full(nl1, np.nan)
        lsfg_l1 = np.full(nl1, np.nan)
        for i, l1 in enumerate(l1_arr):
            l2_l1[i], lsfg_l1[i] = self._find_sfg_input2(l1, T_fix, Lambda_fix, lam_range)
            done += 1
            self.progress.emit(int(100 * done / (nT + nL + nl1)))

        self.progress.emit(100)
        return {
            "process":      "SFG",
            "T_arr":        T_arr,
            "l2_vs_T":      l2_T,
            "lsfg_vs_T":    lsfg_T,
            "Lambda_arr":   Lambda_arr,
            "l2_vs_L":      l2_L,
            "lsfg_vs_L":    lsfg_L,
            "l1_arr":       l1_arr,
            "l2_vs_l1":     l2_l1,
            "lsfg_vs_l1":   lsfg_l1,
            "l1_fixed":     l1_fix,
            "lam_p_fixed":  l1_fix,   # alias for _plot_pm_curves compatibility
            "Lambda_fixed": Lambda_fix,
            "T_fixed":      T_fix,
            "is_qpm":       p["is_qpm"],
        }

    # ------------------------------------------------------------------
    def _compute_shg(self) -> dict:
        from scipy.optimize import brentq
        p          = self._p
        T_arr      = p["T_arr"]
        Lambda_arr = p["Lambda_arr"]
        lf_arr     = p["lp_arr"]          # sweep axis = λ_fundamental (panel 3)
        Lambda_fix = p["Lambda_fixed"]
        T_fix      = p["T_fixed"]
        lf_fix     = p["lam_p"]           # fixed λ_fundamental for panels 1 and 2
        lam_range  = p["lam_range"]
        nf         = len(lf_arr)
        nL         = len(Lambda_arr)

        # SHG Δk: k(λ_SH) − 2k(λ_fund) − 2π/Λ(T)
        def shg_dk(lf, T, L):
            lsh = lf / 2.0
            try:
                return self._dk(lsh, lf, lf, T, L)
            except Exception:
                return np.nan

        def find_T_pm(lf, L):
            """Return T_PM for given λ_fund and Λ, or nan."""
            if lf / 2.0 < lam_range[0] or lf > lam_range[1]:
                return np.nan
            dk_T = np.array([shg_dk(lf, T, L) for T in T_arr])
            fin  = np.isfinite(dk_T)
            if not np.any(fin):
                return np.nan
            chg = np.where(np.diff(np.sign(dk_T[fin])))[0]
            if not len(chg):
                return np.nan
            T_f = T_arr[fin]; j = chg[0]
            try:
                return brentq(lambda T: shg_dk(lf, T, L), T_f[j], T_f[j + 1], xtol=1e-4)
            except Exception:
                return np.nan

        # Panel 1: Δk(T) at fixed λ_fund = lf_fix and Λ = Lambda_fix
        dk_vs_T = np.array([shg_dk(lf_fix, T, Lambda_fix) for T in T_arr])
        self.progress.emit(10)

        # Panel 2: T_PM(Λ) at fixed λ_fund = lf_fix  — QPM only
        T_pm_vs_L = np.full(nL, np.nan)
        if p["is_qpm"] and nL > 1:
            for i, L in enumerate(Lambda_arr):
                T_pm_vs_L[i] = find_T_pm(lf_fix, L)
                self.progress.emit(10 + int(45 * (i + 1) / nL))

        # Panel 3: T_PM(λ_fund) at fixed Λ = Lambda_fix
        T_pm_vs_lf = np.full(nf, np.nan)
        for i, lf in enumerate(lf_arr):
            T_pm_vs_lf[i] = find_T_pm(lf, Lambda_fix)
            self.progress.emit(55 + int(45 * (i + 1) / nf))
        self.progress.emit(100)

        return {
            "process":      "SHG",
            "T_arr":        T_arr,
            "dk_vs_T":      dk_vs_T,
            "Lambda_arr":   Lambda_arr,
            "T_pm_vs_L":    T_pm_vs_L,
            "lf_arr":       lf_arr,
            "T_pm_vs_lf":   T_pm_vs_lf,
            "lam_p_fixed":  lf_fix,
            "Lambda_fixed": Lambda_fix,
            "T_fixed":      T_fix,
            "is_qpm":       p["is_qpm"],
        }

    # ------------------------------------------------------------------
    def _compute_opg(self) -> dict:
        p          = self._p
        T_arr      = p["T_arr"]
        Lambda_arr = p["Lambda_arr"]
        lp_arr     = p["lp_arr"]
        Lambda_fix = p["Lambda_fixed"]
        T_fix      = p["T_fixed"]
        lam_p_fix  = p["lam_p"]
        lam_range  = p["lam_range"]

        nT   = len(T_arr)
        nL   = len(Lambda_arr)
        nlp  = len(lp_arr)
        done = 0

        # Panel 1: λ_s(T), λ_i(T) at fixed λ_p and Λ
        ls_T = np.full(nT, np.nan)
        li_T = np.full(nT, np.nan)
        for i, T in enumerate(T_arr):
            ls_T[i], li_T[i] = self._find_signal(lam_p_fix, T, Lambda_fix, lam_range)
            done += 1
            self.progress.emit(int(100 * done / (nT + nL + nlp)))

        # Panel 2: λ_s(Λ), λ_i(Λ) at fixed λ_p and T  — QPM only
        ls_L = np.full(nL, np.nan)
        li_L = np.full(nL, np.nan)
        if p["is_qpm"] and nL > 1:
            for i, L in enumerate(Lambda_arr):
                ls_L[i], li_L[i] = self._find_signal(lam_p_fix, T_fix, L, lam_range)
                done += 1
                self.progress.emit(int(100 * done / (nT + nL + nlp)))
        else:
            done += nL

        # Panel 3: λ_s(λ_p), λ_i(λ_p) at fixed T and Λ
        ls_lp = np.full(nlp, np.nan)
        li_lp = np.full(nlp, np.nan)
        for i, lp in enumerate(lp_arr):
            ls_lp[i], li_lp[i] = self._find_signal(lp, T_fix, Lambda_fix, lam_range)
            done += 1
            self.progress.emit(int(100 * done / (nT + nL + nlp)))

        self.progress.emit(100)
        return {
            "process":      p["process"],
            "T_arr":        T_arr,
            "ls_vs_T":      ls_T,
            "li_vs_T":      li_T,
            "Lambda_arr":   Lambda_arr,
            "ls_vs_L":      ls_L,
            "li_vs_L":      li_L,
            "lp_arr":       lp_arr,
            "ls_vs_lp":     ls_lp,
            "li_vs_lp":     li_lp,
            "lam_p_fixed":  lam_p_fix,
            "Lambda_fixed": Lambda_fix,
            "T_fixed":      T_fix,
            "is_qpm":       p["is_qpm"],
        }


class CrystalsTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._db      = get_db()
        self._current = None   # dict of currently selected crystal
        self._worker_thread    = None
        self._pm_worker_thread = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # ── LEFT: crystal list ─────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(200)
        left.setMaximumWidth(260)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(6)

        lv.addWidget(QLabel("Crystal Library"))

        self.crystal_list = QListWidget()
        self.crystal_list.setAlternatingRowColors(True)
        lv.addWidget(self.crystal_list)

        btn_row = QHBoxLayout()
        self.btn_add    = QPushButton("Add")
        self.btn_add.setObjectName("addButton")
        self.btn_edit   = QPushButton("Save")
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setObjectName("deleteButton")
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_edit)
        btn_row.addWidget(self.btn_delete)
        lv.addLayout(btn_row)
        splitter.addWidget(left)

        # ── RIGHT: crystal editor + plots ──────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 0, 0, 0)
        rv.setSpacing(6)

        inner_tabs = QTabWidget()
        inner_tabs.addTab(self._build_properties_tab(),  "Properties")
        inner_tabs.addTab(self._build_sellmeier_tab(),   "Refractive Index")
        rv.addWidget(inner_tabs)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # ── Connections ────────────────────────────────────────────────
        self._defaults: dict | None = None   # DB snapshot at crystal-selection time

        self.crystal_list.currentItemChanged.connect(self._on_crystal_selected)
        self.btn_add.clicked.connect(self._on_add)
        self.btn_edit.clicked.connect(self._on_save)
        self.btn_delete.clicked.connect(self._on_delete)
        self.btn_plot_n.clicked.connect(self._plot_refractive_index)
        self.btn_set_changes.clicked.connect(self._on_set_changes)
        self.btn_set_defaults.clicked.connect(self._on_set_default_values)

        self._reload_list()

    # ── Sub-tabs ───────────────────────────────────────────────────────
    def _build_properties_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        w = QWidget()
        g = QGridLayout(w)
        g.setSpacing(4)
        g.setContentsMargins(12, 8, 12, 8)

        def _sb(lo=0, hi=1e6, dec=5, val=0.0):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi); sb.setDecimals(dec); sb.setValue(val)
            sb.setFixedHeight(24)
            sb.setMaximumWidth(120)
            return sb

        # ── Optical & identification ───────────────────────────────────
        optical_fields = [
            ("Name",                 "le_name",    None),
            ("Type",                 "cb_type",    "combo"),
            ("λ min (μm)",           "sb_lmin",    _sb(0.01, 100, 3, 0.4)),
            ("λ max (μm)",           "sb_lmax",    _sb(0.01, 100, 3, 5.0)),
            ("deff (pm/V)",          "sb_deff",    _sb(0, 1e4, 3, 25.0)),
            ("α pump (cm⁻¹)",        "sb_alpha_p", _sb(0, 1e6, 6, 0.002)),
            ("α signal (cm⁻¹)",      "sb_alpha_s", _sb(0, 1e6, 6, 0.025)),
            ("α idler (cm⁻¹)",       "sb_alpha_i", _sb(0, 1e6, 6, 0.025)),
            ("β pump (μm/W)",        "sb_beta_p",  _sb(0, 1, 8, 0.0)),
            ("β signal (μm/W)",      "sb_beta_s",  _sb(0, 1, 8, 5e-5)),
            ("β idler (μm/W)",       "sb_beta_i",  _sb(0, 1, 8, 0.0)),
            ("ρ_p (rad)",            "sb_rho_p",   _sb(-1.5708, 1.5708, 6, 0.0)),
            ("ρ_s (rad)",            "sb_rho_s",   _sb(-1.5708, 1.5708, 6, 0.0)),
            ("ρ_i (rad)",            "sb_rho_i",   _sb(-1.5708, 1.5708, 6, 0.0)),
        ]

        # ── Thermal & QPM ──────────────────────────────────────────────
        thermal_fields = [
            ("Thermal conductivity κ (W/m·K)",     "sb_kappa",    _sb(0, 1e4, 3, 4.6)),
            ("Thermal expansion α_th (10⁻⁶ K⁻¹)", "sb_alpha_th", _sb(0, 1e4, 4, 14.8)),
            ("Heat capacity Cₚ (J/kg·K)",           "sb_cp",       _sb(0, 1e5, 2, 628.0)),
            ("Density ρ (kg/m³)",                   "sb_rho",      _sb(0, 1e5, 1, 4640.0)),
            ("Grating period Λ₀ at T₀ (μm)",        "sb_lambda0",  _sb(0, 1e4, 4, 6.99)),
            ("Reference temperature T₀ (°C)",       "sb_T0",       _sb(-100, 1000, 2, 27.0)),
        ]

        _ALPHA_ATTRS = ("sb_alpha_p", "sb_alpha_s", "sb_alpha_i")
        _ALPHA_CHK   = ("chk_alpha_p", "chk_alpha_s", "chk_alpha_i")
        _BETA_ATTRS  = ("sb_beta_p",   "sb_beta_s",   "sb_beta_i")
        _BETA_CHK    = ("chk_beta_p",  "chk_beta_s",  "chk_beta_i")
        _RHO_ATTRS   = ("sb_rho_p",    "sb_rho_s",    "sb_rho_i")
        _RHO_CHK     = ("chk_rho_p",   "chk_rho_s",   "chk_rho_i")

        all_fields = optical_fields + thermal_fields
        for row, (label, attr, widget) in enumerate(all_fields):
            g.addWidget(QLabel(label), row, 0)
            if widget == "combo":
                cb = QComboBox()
                cb.addItems(["QPM-PPLN", "QPM-PPLT", "Birefringent-uniaxial",
                             "Birefringent-biaxial"])
                cb.setFixedHeight(24)
                cb.setMaximumWidth(160)
                setattr(self, attr, cb)
                g.addWidget(cb, row, 1)
            elif widget is None:
                le = QLineEdit()
                le.setFixedHeight(24)
                le.setMaximumWidth(160)
                setattr(self, attr, le)
                g.addWidget(le, row, 1)
            else:
                setattr(self, attr, widget)
                g.addWidget(widget, row, 1)

            if attr in _ALPHA_ATTRS:
                chk_name = _ALPHA_CHK[_ALPHA_ATTRS.index(attr)]
                chk = QCheckBox("on")
                chk.setChecked(True)
                chk.setToolTip("Uncheck to zero absorption in simulation (stored value is preserved)")
                setattr(self, chk_name, chk)
                g.addWidget(chk, row, 2)

            if attr in _BETA_ATTRS:
                chk_name = _BETA_CHK[_BETA_ATTRS.index(attr)]
                chk = QCheckBox("on")
                chk.setChecked(False)
                chk.setToolTip("Enable two-photon absorption for this wave in the simulation")
                setattr(self, chk_name, chk)
                g.addWidget(chk, row, 2)

            if attr in _RHO_ATTRS:
                chk_name = _RHO_CHK[_RHO_ATTRS.index(attr)]
                chk = QCheckBox("on")
                chk.setChecked(False)
                chk.setToolTip("Enable spatial walk-off for this wave in the simulation")
                setattr(self, chk_name, chk)
                g.addWidget(chk, row, 2)

        # ── Reference (BibTeX input → auto-generated short citation) ────
        bib_row = len(all_fields)
        g.addWidget(QLabel("Reference (BibTeX)"), bib_row, 0,
                    Qt.AlignmentFlag.AlignTop)
        self.te_bibtex = QPlainTextEdit()
        self.te_bibtex.setMaximumHeight(70)
        self.te_bibtex.setPlaceholderText(
            "@article{Gayer2008, author={Gayer, O. and Sacks, Z. and "
            "Galun, E. and Arie, A.}, journal={Applied Physics B}, "
            "volume={91}, pages={343--348}, year={2008}}")
        self.te_bibtex.textChanged.connect(self._on_bibtex_changed)
        g.addWidget(self.te_bibtex, bib_row, 1, 1, 2)

        # ── Action buttons ─────────────────────────────────────────────
        btn_row = bib_row + 1
        bh = QHBoxLayout()
        bh.setContentsMargins(0, 10, 0, 0)
        bh.setSpacing(8)
        self.btn_set_changes  = QPushButton("Set changes")
        self.btn_set_defaults = QPushButton("Set default values")
        self.btn_set_changes.setObjectName("addButton")
        bh.addWidget(self.btn_set_changes)
        bh.addWidget(self.btn_set_defaults)
        bh.addStretch()
        g.addLayout(bh, btn_row, 0, 1, 3)

        # ── Citation preview (auto-derived from the BibTeX above) ───────
        ref_row = btn_row + 1
        g.addWidget(QLabel("Citation:"), ref_row, 0, Qt.AlignmentFlag.AlignTop)
        self.lbl_ref = QLabel("")
        self.lbl_ref.setObjectName("unitLabel")
        self.lbl_ref.setWordWrap(True)
        g.addWidget(self.lbl_ref, ref_row, 1, 1, 2)

        g.setRowStretch(ref_row + 1, 1)
        scroll.setWidget(w)
        return scroll

    def _on_bibtex_changed(self):
        self.lbl_ref.setText(bibtex_to_citation(self.te_bibtex.toPlainText()))

    def alpha_flags(self) -> dict:
        """Return absorption on/off state for the three waves."""
        return {
            "alpha_p_active": self.chk_alpha_p.isChecked(),
            "alpha_s_active": self.chk_alpha_s.isChecked(),
            "alpha_i_active": self.chk_alpha_i.isChecked(),
        }

    def beta_flags(self) -> dict:
        """Return TPA on/off state for the three waves."""
        return {
            "beta_p_active": self.chk_beta_p.isChecked(),
            "beta_s_active": self.chk_beta_s.isChecked(),
            "beta_i_active": self.chk_beta_i.isChecked(),
        }

    def rho_flags(self) -> dict:
        """Return spatial walk-off on/off state for the three waves."""
        return {
            "rho_p_active": self.chk_rho_p.isChecked(),
            "rho_s_active": self.chk_rho_s.isChecked(),
            "rho_i_active": self.chk_rho_i.isChecked(),
        }

    def _build_sellmeier_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        info = QLabel(
            "Variables: L = wavelength [μm],  T = temperature [°C].\n"
            "Use Python/SymPy syntax: **, sqrt(), exp(). "
            "Named coefficients must be defined below."
        )
        info.setWordWrap(True)
        info.setObjectName("unitLabel")
        v.addWidget(info)

        axis_row = QHBoxLayout()
        axis_row.addWidget(QLabel("Optical axis:"))
        self.cb_axis = QComboBox()
        self.cb_axis.addItems(["extraordinary (e)", "ordinary (o)"])
        self.cb_axis.currentIndexChanged.connect(self._on_axis_changed)
        axis_row.addWidget(self.cb_axis)
        axis_row.addStretch()
        v.addLayout(axis_row)

        v.addWidget(QLabel("Formula  n(L, T) ="))
        self.te_formula = QPlainTextEdit()
        self.te_formula.setMaximumHeight(80)
        self.te_formula.setPlaceholderText(
            "sqrt(A1 + A2/(L**2 - A3**2) + A4/(L**2 - A5**2) - A6*L**2)")
        v.addWidget(self.te_formula)

        coeff_gb = QGroupBox("Coefficients  (name = value)")
        cg = QGridLayout(coeff_gb)
        self._coeff_widgets: list[tuple[QLabel, QLineEdit, QLineEdit]] = []
        # Up to 10 named coefficient rows (label | name field | value field)
        for i in range(10):
            row, col_offset = divmod(i, 2)
            lbl = QLabel(f"c{i+1}")
            lbl.setMinimumWidth(20)
            lbl.setObjectName("unitLabel")
            name_le  = QLineEdit()
            name_le.setMaximumWidth(60)
            name_le.setPlaceholderText("name")
            val_le   = QLineEdit("0.0")
            val_le.setMaximumWidth(110)
            cg.addWidget(lbl,     row, col_offset * 3)
            cg.addWidget(name_le, row, col_offset * 3 + 1)
            cg.addWidget(val_le,  row, col_offset * 3 + 2)
            self._coeff_widgets.append((lbl, name_le, val_le))
        v.addWidget(coeff_gb)

        T_row = QHBoxLayout()
        T_row.addWidget(QLabel("Preview at T (°C):"))
        self.sb_T_preview = QDoubleSpinBox()
        self.sb_T_preview.setRange(-50, 500); self.sb_T_preview.setValue(27.0)
        T_row.addWidget(self.sb_T_preview)
        self.btn_plot_n = QPushButton("Plot n(λ),  v_g(λ)  &  GVD(λ)")
        T_row.addWidget(self.btn_plot_n)
        T_row.addStretch()
        v.addLayout(T_row)

        self.lbl_n_status = QLabel("")
        self.lbl_n_status.setObjectName("unitLabel")
        v.addWidget(self.lbl_n_status)

        self.n_canvas = PlotCanvas(nrows=1, ncols=3, figsize=(11, 3), dpi=90)
        self.n_canvas.setMinimumHeight(200)
        v.addWidget(self.n_canvas)

        return w

    def _build_pm_curve_tab(self):
        w  = QWidget()
        mv = QVBoxLayout(w)
        mv.setContentsMargins(8, 8, 8, 4)
        mv.setSpacing(6)

        # ── Row 1: process + PM type + fixed values ───────────────────
        r1 = QHBoxLayout()
        r1.setSpacing(8)

        r1.addWidget(QLabel("Process:"))
        self.pm_cb_process = QComboBox()
        self.pm_cb_process.addItems(["OPG", "SFG", "DFG", "SHG"])
        r1.addWidget(self.pm_cb_process)

        r1.addWidget(QLabel("  PM type:"))
        self.pm_cb_type = QComboBox()
        self.pm_cb_type.addItems([
            "e → e e  (QPM)",
            "e → o o  (Type I)",
            "o → e e  (Type I)",
            "e → e o  (Type II)",
            "e → o e  (Type II)",
            "o → e o  (Type II)",
            "o → o e  (Type II)",
        ])
        r1.addWidget(self.pm_cb_type)

        r1.addWidget(QLabel("  λ_pump (μm):"))
        self.pm_sb_lp = QDoubleSpinBox()
        self.pm_sb_lp.setRange(0.1, 10.0); self.pm_sb_lp.setDecimals(4)
        self.pm_sb_lp.setValue(1.0642); self.pm_sb_lp.setMaximumWidth(90)
        r1.addWidget(self.pm_sb_lp)

        r1.addWidget(QLabel("  T fixed (°C):"))
        self.pm_sb_T_fixed = QDoubleSpinBox()
        self.pm_sb_T_fixed.setRange(-50, 500); self.pm_sb_T_fixed.setDecimals(1)
        self.pm_sb_T_fixed.setValue(27.0); self.pm_sb_T_fixed.setMaximumWidth(80)
        r1.addWidget(self.pm_sb_T_fixed)

        r1.addWidget(QLabel("  Λ fixed (μm):"))
        self.pm_sb_L_fixed = QDoubleSpinBox()
        self.pm_sb_L_fixed.setRange(0.1, 500.0); self.pm_sb_L_fixed.setDecimals(3)
        self.pm_sb_L_fixed.setValue(6.99); self.pm_sb_L_fixed.setMaximumWidth(80)
        r1.addWidget(self.pm_sb_L_fixed)

        r1.addStretch()
        mv.addLayout(r1)

        # ── Row 2: sweep ranges + N points + Compute ─────────────────
        r2 = QHBoxLayout()
        r2.setSpacing(8)

        def _dsb(lo, hi, dec, val, w=72):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi); sb.setDecimals(dec)
            sb.setValue(val); sb.setMaximumWidth(w)
            return sb

        r2.addWidget(QLabel("T:"))
        self.pm_sb_T_min = _dsb(-50, 500, 0, 20.0)
        r2.addWidget(self.pm_sb_T_min)
        r2.addWidget(QLabel("—"))
        self.pm_sb_T_max = _dsb(-50, 500, 0, 200.0)
        r2.addWidget(self.pm_sb_T_max)
        r2.addWidget(QLabel("°C"))

        r2.addWidget(QLabel("  Λ:"))
        self.pm_sb_L_min = _dsb(0.1, 500, 2, 5.0)
        r2.addWidget(self.pm_sb_L_min)
        r2.addWidget(QLabel("—"))
        self.pm_sb_L_max = _dsb(0.1, 500, 2, 25.0)
        r2.addWidget(self.pm_sb_L_max)
        r2.addWidget(QLabel("μm"))

        r2.addWidget(QLabel("  λ_p:"))
        self.pm_sb_lp_min = _dsb(0.1, 10, 3, 0.90)
        r2.addWidget(self.pm_sb_lp_min)
        r2.addWidget(QLabel("—"))
        self.pm_sb_lp_max = _dsb(0.1, 10, 3, 1.20)
        r2.addWidget(self.pm_sb_lp_max)
        r2.addWidget(QLabel("μm"))

        r2.addWidget(QLabel("  N:"))
        self.pm_sb_N = QSpinBox()
        self.pm_sb_N.setRange(20, 500); self.pm_sb_N.setValue(100)
        self.pm_sb_N.setMaximumWidth(60)
        r2.addWidget(self.pm_sb_N)

        self.pm_btn_compute = QPushButton("Compute")
        self.pm_btn_compute.setObjectName("addButton")
        self.pm_btn_compute.clicked.connect(self._run_pm_curves)
        r2.addWidget(self.pm_btn_compute)

        self.pm_lbl_status = QLabel("")
        self.pm_lbl_status.setObjectName("unitLabel")
        r2.addWidget(self.pm_lbl_status)

        r2.addStretch()
        mv.addLayout(r2)

        # ── 3 plots (1×3 horizontal) ──────────────────────────────────
        self.pm_canvas = PlotCanvas(nrows=1, ncols=3, figsize=(13, 4), dpi=88)
        self.pm_canvas.setMinimumHeight(270)
        mv.addWidget(self.pm_canvas, stretch=1)

        return w

    # ── PM-curve helpers ───────────────────────────────────────────────
    def _parse_pm_type(self, text: str) -> tuple:
        """'e → o o  (Type I)' → ('e', 'o', 'o')"""
        left, right = text.split("→")
        pol_p = left.strip()
        parts = right.strip().split()
        pol_s = parts[0] if len(parts) > 0 else "e"
        pol_i = parts[1] if len(parts) > 1 else pol_s
        return pol_p, pol_s, pol_i

    def _run_pm_curves(self):
        cr = self._current
        if cr is None:
            self.pm_lbl_status.setText("No crystal selected.")
            return

        # Build Sellmeier objects from stored data
        import json as _json
        coeffs_e = cr.get("coeffs_e", {})
        if isinstance(coeffs_e, str):
            coeffs_e = _json.loads(coeffs_e)
        sf_e = SellmeierFormula(cr.get("formula_e", ""), coeffs_e)
        if not sf_e.is_ready:
            self.pm_lbl_status.setText(f"e-axis formula error: {sf_e.error}")
            return

        sf_o = None
        coeffs_o = cr.get("coeffs_o", {})
        if isinstance(coeffs_o, str):
            coeffs_o = _json.loads(coeffs_o)
        if cr.get("formula_o"):
            sf_o = SellmeierFormula(cr.get("formula_o"), coeffs_o)
            if not sf_o.is_ready:
                sf_o = None

        pol_p, pol_s, pol_i = self._parse_pm_type(self.pm_cb_type.currentText())
        is_qpm   = "QPM" in cr.get("type", "")
        lam_p    = self.pm_sb_lp.value()
        T_fix    = self.pm_sb_T_fixed.value()
        L_fix    = self.pm_sb_L_fixed.value() if is_qpm else 0.0
        N        = self.pm_sb_N.value()

        T_arr    = np.linspace(self.pm_sb_T_min.value(),  self.pm_sb_T_max.value(),  N)
        L_arr    = np.linspace(self.pm_sb_L_min.value(),  self.pm_sb_L_max.value(),  N) \
                   if is_qpm else np.array([L_fix])
        lp_arr   = np.linspace(self.pm_sb_lp_min.value(), self.pm_sb_lp_max.value(), N)
        lam_range = (cr.get("lambda_min", 0.4), cr.get("lambda_max", 5.0))

        params = {
            "process":       self.pm_cb_process.currentText(),
            "pol_p": pol_p, "pol_s": pol_s, "pol_i": pol_i,
            "lam_p":         lam_p,
            "T_fixed":       T_fix,
            "Lambda_fixed":  L_fix,
            "T_arr":         T_arr,
            "Lambda_arr":    L_arr,
            "lp_arr":        lp_arr,
            "lam_range":     lam_range,
            "is_qpm":        is_qpm,
            "alpha_th":      cr.get("alpha_th", 14.8e-6),
            "T0":            cr.get("T0", 27.0),
        }

        if self._pm_worker_thread and self._pm_worker_thread.isRunning():
            return

        self.pm_btn_compute.setEnabled(False)
        self.pm_lbl_status.setText("Computing…")

        worker = _PMCurveWorker(sf_e, sf_o, params)
        thread = QThread(self)          # parent=self keeps thread alive
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.done.connect(self._on_pm_curves_done)
        worker.error.connect(self._on_pm_curves_error)
        worker.progress.connect(
            lambda p: self.pm_lbl_status.setText(f"Computing… {p}%"))
        worker.done.connect(thread.quit)
        worker.error.connect(thread.quit)
        # Keep strong Python refs until thread finishes
        self._pm_worker        = worker
        self._pm_worker_thread = thread
        thread.start()

    def _on_pm_curves_done(self, results: dict):
        self.pm_btn_compute.setEnabled(True)
        try:
            self._plot_pm_curves(results)
        except Exception as exc:
            self.pm_lbl_status.setText(f"Plot error: {exc}")
            import traceback; traceback.print_exc()

    def _on_pm_curves_error(self, msg: str):
        self.pm_btn_compute.setEnabled(True)
        self.pm_lbl_status.setText(f"Error: {msg}")

    def _plot_pm_curves(self, r: dict):
        ax0, ax1, ax2 = [self.pm_canvas.get_ax(i) for i in range(3)]
        for ax in (ax0, ax1, ax2):
            ax.clear()
            self.pm_canvas._style_ax(ax)

        c_s = "#00bcd4"   # cyan  → signal
        c_i = "#ff6d00"   # orange → idler
        c_d = "#00e676"   # green  → Δk / T_PM

        def _nm(arr):            # μm → nm
            return arr * 1000.0

        process    = r["process"]
        lam_p_fix  = r["lam_p_fixed"]
        L_fix      = r["Lambda_fixed"]
        T_fix      = r["T_fixed"]

        if process == "SHG":
            lf_nm = _nm(np.array([lam_p_fix]))[0]

            # Panel 1: Δk(T) at fixed λ_fund and Λ  — shows T_PM as zero crossing
            T_arr   = r["T_arr"]
            dk_vs_T = r["dk_vs_T"]
            finite  = np.isfinite(dk_vs_T)
            if np.any(finite):
                ax0.plot(T_arr[finite], dk_vs_T[finite], color=c_d, lw=1.5)
            ax0.axhline(0, color="#555", lw=0.8, ls="--")
            ax0.set_xlabel("T  (°C)"); ax0.set_ylabel("Δk  (μm⁻¹)")
            ax0.set_title(f"SHG: Δk(T)  (λ_fund={lf_nm:.0f} nm, Λ={L_fix:.2f} μm)")

            # Panel 2: T_PM(Λ) at fixed λ_fund  — QPM only
            if r["is_qpm"]:
                L_arr      = r["Lambda_arr"]
                T_pm_vs_L  = r["T_pm_vs_L"]
                valid = ~np.isnan(T_pm_vs_L)
                if np.any(valid):
                    ax1.plot(L_arr[valid], T_pm_vs_L[valid], color=c_s, lw=1.5)
                ax1.set_xlabel("Λ  (μm)"); ax1.set_ylabel("T_PM  (°C)")
                ax1.set_title(f"SHG: T_PM(Λ)  (λ_fund={lf_nm:.0f} nm)")
            else:
                ax1.text(0.5, 0.5, "QPM only", transform=ax1.transAxes,
                         ha="center", va="center", color="#777", fontsize=10)
                ax1.set_xlabel("Λ  (μm)"); ax1.set_ylabel("T_PM  (°C)")
                ax1.set_title("SHG: T_PM(Λ)  (not applicable)")

            # Panel 3: T_PM(λ_fund) at fixed Λ  — spectral tuning curve
            lf     = r["lf_arr"]
            T_pm   = r["T_pm_vs_lf"]
            valid  = ~np.isnan(T_pm)
            if np.any(valid):
                ax2.plot(_nm(lf[valid]), T_pm[valid], color=c_s, lw=1.5)
            ax2.set_xlabel("λ_fund  (nm)"); ax2.set_ylabel("T_PM  (°C)")
            ax2.set_title(f"SHG: T_PM(λ)  (Λ={L_fix:.2f} μm)")

            n_valid = int(np.sum(~np.isnan(T_pm)))

        elif process == "SFG":
            # SFG: λ_pump in GUI = λ_p (first / pump input, shortest wavelength).
            # Δk = k(λ_SFG) − k(λ_p) − k(λ_2) − 2π/Λ(T) = 0
            # Curves show λ_2 (second input) and λ_SFG (output) vs T, Λ, λ_p.
            lp_label = f"{r['l1_fixed']*1000:.0f} nm"

            def _sfg_panels(ax, x_arr, l2_arr, lsfg_arr, xlabel, title):
                v2 = ~np.isnan(l2_arr)
                vs = ~np.isnan(lsfg_arr)
                if np.any(v2):
                    ax.plot(x_arr[v2], _nm(l2_arr[v2]), color=c_i, lw=1.5,
                            label="λ₂ (input 2)", ls="--")
                if np.any(vs):
                    ax.plot(x_arr[vs], _nm(lsfg_arr[vs]), color=c_s, lw=1.5,
                            label="λ_SFG (output)")
                if np.any(v2) or np.any(vs):
                    ax.legend(fontsize=8)
                ax.set_xlabel(xlabel); ax.set_ylabel("λ  (nm)")
                ax.set_title(title)

            # Panel 1: vs T
            T_arr = r["T_arr"]
            _sfg_panels(ax0, T_arr, r["l2_vs_T"], r["lsfg_vs_T"],
                        "T  (°C)",
                        f"SFG: PM vs T  (λ_p={lp_label}, Λ={L_fix:.2f} μm)")

            # Panel 2: vs Λ
            if r["is_qpm"] and len(r["Lambda_arr"]) > 1:
                _sfg_panels(ax1, r["Lambda_arr"], r["l2_vs_L"], r["lsfg_vs_L"],
                            "Λ  (μm)",
                            f"SFG: PM vs Λ  (λ_p={lp_label}, T={T_fix:.0f} °C)")
            else:
                ax1.text(0.5, 0.5, "QPM only", transform=ax1.transAxes,
                         ha="center", va="center", color="#777", fontsize=10)
                ax1.set_xlabel("Λ  (μm)"); ax1.set_ylabel("λ  (nm)")
                ax1.set_title("SFG: PM vs Λ  (not applicable)")

            # Panel 3: vs λ_p
            l1_arr = r["l1_arr"]
            _sfg_panels(ax2, _nm(l1_arr), r["l2_vs_l1"], r["lsfg_vs_l1"],
                        "λ_p  (nm)",
                        f"SFG: PM vs λ_p  (T={T_fix:.0f} °C, Λ={L_fix:.2f} μm)")

            n_valid = int(np.sum(~np.isnan(r["l2_vs_T"])))

        else:
            # OPG / DFG: λ_pump = pump (highest freq, shortest λ)
            # Generates λ_s, λ_i both LONGER than λ_pump
            lp_label = f"{_nm(np.array([lam_p_fix]))[0]:.0f} nm"

            def _opg_panels(ax, x_arr, ls_arr, li_arr, xlabel, title):
                vs = ~np.isnan(ls_arr)
                vi = ~np.isnan(li_arr)
                if np.any(vs):
                    ax.plot(x_arr[vs], _nm(ls_arr[vs]), color=c_s, lw=1.5,
                            label="signal")
                if np.any(vi):
                    ax.plot(x_arr[vi], _nm(li_arr[vi]), color=c_i, lw=1.5,
                            label="idler", ls="--")
                if np.any(vs) or np.any(vi):
                    ax.legend(fontsize=8)
                ax.set_xlabel(xlabel); ax.set_ylabel("λ  (nm)")
                ax.set_title(title)

            # Panel 1: λ_s(T), λ_i(T)
            T_arr = r["T_arr"]
            _opg_panels(ax0, T_arr, r["ls_vs_T"], r["li_vs_T"],
                        "T  (°C)",
                        f"{process}: PM vs T  (λ_p={lp_label}, Λ={L_fix:.2f} μm)")

            # Panel 2: λ_s(Λ), λ_i(Λ)
            if r["is_qpm"] and len(r["Lambda_arr"]) > 1:
                _opg_panels(ax1, r["Lambda_arr"], r["ls_vs_L"], r["li_vs_L"],
                            "Λ  (μm)",
                            f"{process}: PM vs Λ  (λ_p={lp_label}, T={T_fix:.0f} °C)")
            else:
                ax1.text(0.5, 0.5, "QPM only", transform=ax1.transAxes,
                         ha="center", va="center", color="#777", fontsize=10)
                ax1.set_xlabel("Λ  (μm)"); ax1.set_ylabel("λ  (nm)")
                ax1.set_title(f"{process}: PM vs Λ  (not applicable)")

            # Panel 3: λ_s(λ_p), λ_i(λ_p)
            lp_arr = r["lp_arr"]
            _opg_panels(ax2, _nm(lp_arr), r["ls_vs_lp"], r["li_vs_lp"],
                        "λ_pump  (nm)",
                        f"{process}: PM vs λ_pump  (T={T_fix:.0f} °C,"
                        f" Λ={L_fix:.2f} μm)")

            n_valid = int(np.sum(~np.isnan(r["ls_vs_T"])))

        self.pm_lbl_status.setText(
            f"Done — {n_valid} PM solutions in panel 1.")
        self.pm_canvas.refresh()

    # ── Crystal list management ────────────────────────────────────────
    def _reload_list(self):
        self.crystal_list.blockSignals(True)
        self.crystal_list.clear()
        for name in self._db.all_names():
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            self.crystal_list.addItem(item)
        self.crystal_list.blockSignals(False)
        if self.crystal_list.count():
            self.crystal_list.setCurrentRow(0)

    # ── Populate UI from DB crystal dict ──────────────────────────────
    def _load_crystal(self, cr: dict, store_defaults: bool = True):
        self._current = cr
        if store_defaults:
            self._defaults = dict(cr)   # snapshot at selection time
        self.le_name.setText(cr.get("name", ""))
        self.te_bibtex.blockSignals(True)
        self.te_bibtex.setPlainText(cr.get("reference_bibtex", ""))
        self.te_bibtex.blockSignals(False)
        self.lbl_ref.setText(cr.get("reference", ""))

        idx = self.cb_type.findText(cr.get("type", ""), Qt.MatchFlag.MatchContains)
        if idx < 0: idx = 0
        self.cb_type.setCurrentIndex(idx)

        self.sb_lmin.setValue(cr.get("lambda_min", 0.4))
        self.sb_lmax.setValue(cr.get("lambda_max", 5.0))
        self.sb_deff.setValue(cr.get("deff", 25.0))
        self.sb_alpha_p.setValue(cr.get("alpha_p", 0.0))
        self.sb_alpha_s.setValue(cr.get("alpha_s", 0.0))
        self.sb_alpha_i.setValue(cr.get("alpha_i", 0.0))
        self.sb_beta_p.setValue(cr.get("beta_p", 0.0))
        self.sb_beta_s.setValue(cr.get("beta_s", 0.0))
        self.sb_beta_i.setValue(cr.get("beta_i", 0.0))
        self.sb_rho_p.setValue(cr.get("rho_p", 0.0))
        self.sb_rho_s.setValue(cr.get("rho_s", 0.0))
        self.sb_rho_i.setValue(cr.get("rho_i", 0.0))

        self.sb_kappa.setValue(cr.get("kappa", 4.6))
        self.sb_alpha_th.setValue(cr.get("alpha_th", 14.8e-6) * 1e6)
        self.sb_cp.setValue(cr.get("cp", 628.0))
        self.sb_rho.setValue(cr.get("rho", 4640.0))
        self.sb_lambda0.setValue(cr.get("lambda0", 0.0))
        self.sb_T0.setValue(cr.get("T0", 27.0))

        self._load_axis("e")

    def _load_axis(self, axis: str):
        cr = self._current
        if cr is None:
            return
        key_formula = "formula_e" if axis == "e" else "formula_o"
        key_coeffs  = "coeffs_e"  if axis == "e" else "coeffs_o"
        formula = cr.get(key_formula, "")
        coeffs  = cr.get(key_coeffs, {})
        if isinstance(coeffs, str):
            import json; coeffs = json.loads(coeffs)

        self.te_formula.setPlainText(formula)
        self._fill_coefficients(coeffs)

    def _fill_coefficients(self, coeffs: dict):
        items = list(coeffs.items())
        for i, (lbl_w, name_w, val_w) in enumerate(self._coeff_widgets):
            if i < len(items):
                k, v = items[i]
                name_w.setText(str(k))
                val_w.setText(f"{v:.8g}")
                lbl_w.show(); name_w.show(); val_w.show()
            else:
                name_w.setText("")
                val_w.setText("0.0")
                lbl_w.show(); name_w.show(); val_w.show()

    def _collect_coefficients(self) -> dict:
        coeffs = {}
        for _, name_w, val_w in self._coeff_widgets:
            k = name_w.text().strip()
            v = val_w.text().strip()
            if k:
                try:
                    coeffs[k] = float(v)
                except ValueError:
                    pass
        return coeffs

    # ── Slots ──────────────────────────────────────────────────────────
    def _on_crystal_selected(self, current, _previous):
        if current is None:
            return
        name = current.data(Qt.ItemDataRole.UserRole)
        cr   = self._db.get(name)
        if cr:
            self._load_crystal(cr)

    def _on_axis_changed(self, _idx):
        axis = "e" if self.cb_axis.currentIndex() == 0 else "o"
        self._load_axis(axis)

    def _on_add(self):
        name = "New Crystal"
        cr = {
            "name": name, "type": "QPM-PPLN",
            "lambda_min": 0.4, "lambda_max": 5.0,
            "deff": 25.0,
            "alpha_p": 0.0, "alpha_s": 0.0, "alpha_i": 0.0,
            "beta_p":  0.0, "beta_s":  0.0, "beta_i":  0.0,
            "rho_p":   0.0, "rho_s":   0.0, "rho_i":   0.0,
            "formula_e": "", "formula_o": "",
            "coeffs_e": {}, "coeffs_o": {},
            "kappa": 4.6, "alpha_th": 14.8e-6, "cp": 628.0, "rho": 4640.0,
            "lambda0": 0.0, "T0": 27.0, "reference": "",
            "reference_bibtex": "", "preloaded": 0,
        }
        self._db.add(cr)
        self._reload_list()
        for i in range(self.crystal_list.count()):
            if self.crystal_list.item(i).text() == name:
                self.crystal_list.setCurrentRow(i)
                break
        self.le_name.setFocus()

    def _on_save(self):
        cr = self._current
        if cr is None:
            return
        if cr.get("preloaded"):
            QMessageBox.warning(self, "TWM",
                "Pre-loaded crystals are read-only.\n"
                "Use 'Add' to create a copy with modifications.")
            return

        axis = "e" if self.cb_axis.currentIndex() == 0 else "o"
        key_f = "formula_e" if axis == "e" else "formula_o"
        key_c = "coeffs_e"  if axis == "e" else "coeffs_o"

        fields = {
            "name":       self.le_name.text().strip(),
            "type":       self.cb_type.currentText(),
            "lambda_min": self.sb_lmin.value(),
            "lambda_max": self.sb_lmax.value(),
            "deff":       self.sb_deff.value(),
            "alpha_p":    self.sb_alpha_p.value(),
            "alpha_s":    self.sb_alpha_s.value(),
            "alpha_i":    self.sb_alpha_i.value(),
            "beta_p":     self.sb_beta_p.value(),
            "beta_s":     self.sb_beta_s.value(),
            "beta_i":     self.sb_beta_i.value(),
            "rho_p":      self.sb_rho_p.value(),
            "rho_s":      self.sb_rho_s.value(),
            "rho_i":      self.sb_rho_i.value(),
            "kappa":      self.sb_kappa.value(),
            "alpha_th":   self.sb_alpha_th.value() * 1e-6,
            "cp":         self.sb_cp.value(),
            "rho":        self.sb_rho.value(),
            "lambda0":    self.sb_lambda0.value(),
            "T0":         self.sb_T0.value(),
            "reference":  self.lbl_ref.text().strip(),
            "reference_bibtex": self.te_bibtex.toPlainText().strip(),
            key_f:        self.te_formula.toPlainText().strip(),
            key_c:        self._collect_coefficients(),
        }
        old_name = cr["name"]
        self._db.update(old_name, fields)
        self._reload_list()

    def _on_set_changes(self):
        """Apply current UI values to DB (all crystals, including pre-loaded)."""
        cr = self._current
        if cr is None:
            return
        axis  = "e" if self.cb_axis.currentIndex() == 0 else "o"
        key_f = "formula_e" if axis == "e" else "formula_o"
        key_c = "coeffs_e"  if axis == "e" else "coeffs_o"
        fields = {
            "name":       self.le_name.text().strip(),
            "type":       self.cb_type.currentText(),
            "lambda_min": self.sb_lmin.value(),
            "lambda_max": self.sb_lmax.value(),
            "deff":       self.sb_deff.value(),
            "alpha_p":    self.sb_alpha_p.value(),
            "alpha_s":    self.sb_alpha_s.value(),
            "alpha_i":    self.sb_alpha_i.value(),
            "beta_p":     self.sb_beta_p.value(),
            "beta_s":     self.sb_beta_s.value(),
            "beta_i":     self.sb_beta_i.value(),
            "rho_p":      self.sb_rho_p.value(),
            "rho_s":      self.sb_rho_s.value(),
            "rho_i":      self.sb_rho_i.value(),
            "kappa":      self.sb_kappa.value(),
            "alpha_th":   self.sb_alpha_th.value() * 1e-6,
            "cp":         self.sb_cp.value(),
            "rho":        self.sb_rho.value(),
            "lambda0":    self.sb_lambda0.value(),
            "T0":         self.sb_T0.value(),
            "reference":  self.lbl_ref.text().strip(),
            "reference_bibtex": self.te_bibtex.toPlainText().strip(),
            key_f:        self.te_formula.toPlainText().strip(),
            key_c:        self._collect_coefficients(),
        }
        old_name = cr["name"]
        self._db.update(old_name, fields)
        self._current = self._db.get(fields["name"]) or self._current
        self._reload_list()

    def _on_set_default_values(self):
        """Restore UI and DB to the values stored at crystal-selection time."""
        if self._defaults is None or self._current is None:
            return
        self._db.update(self._current["name"], self._defaults)
        self._load_crystal(self._defaults, store_defaults=False)
        self._reload_list()

    def _on_delete(self):
        cr = self._current
        if cr is None:
            return
        if cr.get("preloaded"):
            QMessageBox.warning(self, "TWM",
                "Pre-loaded crystals cannot be deleted.")
            return
        ans = QMessageBox.question(
            self, "TWM", f"Delete crystal '{cr['name']}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ans == QMessageBox.StandardButton.Yes:
            self._db.delete(cr["name"])
            self._reload_list()

    # ── Plot ───────────────────────────────────────────────────────────
    def _plot_refractive_index(self):
        formula = self.te_formula.toPlainText().strip()
        if not formula:
            self.lbl_n_status.setText("No formula entered.")
            return

        coeffs = self._collect_coefficients()
        axis   = "e" if self.cb_axis.currentIndex() == 0 else "o"
        label  = f"{self._current['name'] if self._current else '?'} ({axis})"

        self.lbl_n_status.setText("Computing symbolic derivatives…")
        self.btn_plot_n.setEnabled(False)

        sf = SellmeierFormula(formula, coeffs, label=label)
        if not sf.is_ready:
            self.lbl_n_status.setText(f"Formula error: {sf.error}")
            self.btn_plot_n.setEnabled(True)
            return

        lam_min = self._current["lambda_min"] if self._current else 0.4
        lam_max = self._current["lambda_max"] if self._current else 5.0
        T_prev  = self.sb_T_preview.value()

        # Run in background thread to keep UI responsive
        if self._worker_thread and self._worker_thread.isRunning():
            return

        self._worker = _PlotWorker(sf, lam_min, lam_max, T_prev)
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_plot_done)
        self._worker.error.connect(self._on_plot_error)
        self._worker.done.connect(self._worker_thread.quit)
        self._worker.error.connect(self._worker_thread.quit)
        self._worker_thread.start()

    def _on_plot_done(self, lam, n, vg, dn, gvd):
        mid = len(lam) // 2
        self.btn_plot_n.setEnabled(True)
        self.lbl_n_status.setText(
            f"n({lam[mid]:.3f} μm, {self.sb_T_preview.value():.0f}°C) "
            f"= {n[mid]:.5f}   "
            f"v_g = {vg[mid]:.4f} × 10⁸ m/s   "
            f"GVD = {gvd[mid]:.1f} fs²/mm"
        )

        ax0, ax1, ax2 = [self.n_canvas.get_ax(i) for i in range(3)]
        for ax in (ax0, ax1, ax2):
            ax.clear()
            self.n_canvas._style_ax(ax)

        lbl = (self._current["name"] if self._current else "") + \
              ("  (e)" if self.cb_axis.currentIndex() == 0 else "  (o)")

        ax0.plot(lam, n, color="#00bcd4", lw=1.5, label=lbl)
        ax0.set_xlabel("λ  (μm)"); ax0.set_ylabel("n(λ)"); ax0.set_title("Refractive index")
        ax0.legend(fontsize=8)

        ax1.plot(lam, vg, color="#ff6d00", lw=1.5)
        ax1.set_xlabel("λ  (μm)"); ax1.set_ylabel("v_g  (10⁸ m/s)")
        ax1.set_title("Group Velocity")

        ax2.plot(lam, gvd, color="#00e676", lw=1.5)
        ax2.axhline(0, color="#555", lw=0.8, ls="--")
        ax2.set_xlabel("λ  (μm)"); ax2.set_ylabel("GVD  (fs²/mm)")
        ax2.set_title("Group Velocity Dispersion")

        self.n_canvas.refresh()

    def _on_plot_error(self, msg: str):
        self.btn_plot_n.setEnabled(True)
        self.lbl_n_status.setText(f"Error: {msg}")

    # ── Public API for other tabs ──────────────────────────────────────
    def current_crystal_name(self) -> str | None:
        item = self.crystal_list.currentItem()
        return item.text() if item else None
