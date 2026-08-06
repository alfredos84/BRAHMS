"""
PhaseMatchingCalc
=================
Computes QPM and birefringent phase-matching curves for TWM processes
(SHG, SFG, OPG, DFG) using SellmeierFormula objects.

All wavelengths in μm, temperatures in °C, crystal lengths in mm.

Conventions
-----------
    Δk = k_p − k_s − k_i − 2π / Λ(T)    (QPM)
    Δk = k_p − k_s − k_i                 (birefringent)

Acceptance bandwidths follow the sinc² criterion:
    ΔX · L  = 0.886 · 2π / |∂Δk/∂X|
"""

import numpy as np
from scipy.optimize import brentq

from .sellmeier import SellmeierFormula


class PhaseMatchingCalc:
    """
    Parameters
    ----------
    sf_e : SellmeierFormula
        Extraordinary (or only) axis formula.
    sf_o : SellmeierFormula, optional
        Ordinary axis formula (birefringent crystals only).
    alpha_th : float
        Thermal expansion coefficient [K⁻¹] — for Λ(T) correction.
    T0 : float
        Reference temperature for grating period [°C].
    """

    def __init__(self,
                 sf_e: SellmeierFormula,
                 sf_o: SellmeierFormula | None = None,
                 alpha_th: float = 14.8e-6,
                 T0: float = 27.0):
        self.sf_e     = sf_e
        self.sf_o     = sf_o
        self.alpha_th = alpha_th
        self.T0       = T0

    # ------------------------------------------------------------------
    # Wavenumbers
    # ------------------------------------------------------------------
    def _k(self, lam: float, T: float, pol: str = "e") -> float:
        sf = self.sf_o if (pol == "o" and self.sf_o) else self.sf_e
        return float(sf.k(lam, T))

    def _Lambda_T(self, Lambda0: float, T: float) -> float:
        """Grating period corrected for thermal expansion."""
        return Lambda0 * (1.0 + self.alpha_th * (T - self.T0))

    # ------------------------------------------------------------------
    # Phase mismatch Δk
    # ------------------------------------------------------------------
    def dk(self,
           lam_p: float, lam_s: float, lam_i: float,
           T: float,
           Lambda: float = 0.0,
           pol_p: str = "e", pol_s: str = "e", pol_i: str = "e") -> float:
        """
        Phase mismatch  Δk [μm⁻¹].
        Set Lambda=0 for birefringent PM (no QPM term).
        """
        kp = self._k(lam_p, T, pol_p)
        ks = self._k(lam_s, T, pol_s)
        ki = self._k(lam_i, T, pol_i)
        dk_val = kp - ks - ki
        if Lambda > 0:
            dk_val -= 2 * np.pi / self._Lambda_T(Lambda, T)
        return dk_val

    def dk_array(self,
                 lam_p_arr, lam_s_arr, lam_i_arr,
                 T: float,
                 Lambda: float = 0.0,
                 pol_p="e", pol_s="e", pol_i="e") -> np.ndarray:
        """Vectorised Δk over wavelength arrays at fixed T."""
        kp = self.sf_e.k(np.asarray(lam_p_arr), T)
        ks = self.sf_e.k(np.asarray(lam_s_arr), T)
        ki = self.sf_e.k(np.asarray(lam_i_arr), T)
        res = kp - ks - ki
        if Lambda > 0:
            res -= 2 * np.pi / self._Lambda_T(Lambda, T)
        return res

    # ------------------------------------------------------------------
    # Phase-matching temperature
    # ------------------------------------------------------------------
    def pm_temperature(self,
                       lam_p: float, lam_s: float, lam_i: float,
                       Lambda: float,
                       T_arr: np.ndarray,
                       pol_p="e", pol_s="e", pol_i="e") -> float:
        """
        Find the PM temperature (root of Δk = 0) via Brent's method.
        Returns NaN if no sign change found in T_arr.
        """
        dk_arr = np.array([
            self.dk(lam_p, lam_s, lam_i, T, Lambda, pol_p, pol_s, pol_i)
            for T in T_arr
        ])
        sign_idx = np.where(np.diff(np.sign(dk_arr)))[0]
        if len(sign_idx) == 0:
            return np.nan
        j = sign_idx[0]
        try:
            return brentq(
                lambda T: self.dk(lam_p, lam_s, lam_i, T, Lambda, pol_p, pol_s, pol_i),
                float(T_arr[j]), float(T_arr[j + 1]),
                xtol=1e-4, maxiter=100)
        except Exception:
            return np.nan

    def pm_curve(self,
                 lam_scan_arr: np.ndarray,
                 Lambda: float,
                 T_arr: np.ndarray,
                 process: str = "SHG",
                 pol_p="e", pol_s="e", pol_i="e") -> np.ndarray:
        """
        PM temperature vs scan wavelength (= fundamental for SHG, pump for OPG/DFG).
        """
        T_pm = np.full(len(lam_scan_arr), np.nan)
        for i, lscan in enumerate(lam_scan_arr):
            lp, ls, li = _three_wavelengths(lscan, process)
            T_pm[i] = self.pm_temperature(lp, ls, li, Lambda, T_arr,
                                          pol_p, pol_s, pol_i)
        return T_pm

    # ------------------------------------------------------------------
    # Acceptance bandwidths
    # ------------------------------------------------------------------
    def bw_temperature(self,
                       lam_p: float, lam_s: float, lam_i: float,
                       T_pm: float, L_mm: float,
                       Lambda: float,
                       dT: float = 0.05) -> float:
        """
        Temperature acceptance bandwidth-length product  ΔT·L  [°C·mm].

        Criterion: sinc²(ΔkL/2) = 0.5 → ΔkL = 0.886π
        ΔT·L [°C·mm] = 0.886π / |∂Δk/∂T| [μm⁻¹/°C] × 10⁻³ [mm/μm]
        """
        if np.isnan(T_pm):
            return np.nan
        dk1 = self.dk(lam_p, lam_s, lam_i, T_pm + dT, Lambda)
        dk2 = self.dk(lam_p, lam_s, lam_i, T_pm - dT, Lambda)
        d_dk_dT = (dk1 - dk2) / (2 * dT)
        if abs(d_dk_dT) < 1e-30:
            return np.nan
        return 0.886 * np.pi / abs(d_dk_dT) * 1e-3  # [°C·mm]

    def bw_wavelength(self,
                      lam_p: float, lam_s: float, lam_i: float,
                      T_pm: float, L_mm: float,
                      Lambda: float,
                      dlam: float = 1e-4) -> float:
        """
        Spectral acceptance bandwidth-length product  Δλ·L  [nm·mm].

        ΔkL = 0.886π → Δλ·L [nm·mm] = 0.886π / |∂Δk/∂λ| [μm⁻²]
        (1 μm² = 1 nm × 1 mm → no extra factor needed)
        """
        if np.isnan(T_pm):
            return np.nan
        # Δk depends on lam_p (high-freq field); shift by dlam
        dk1 = self.dk(lam_p + dlam, lam_s + 2*dlam, lam_i + 2*dlam, T_pm, Lambda)
        dk2 = self.dk(lam_p - dlam, lam_s - 2*dlam, lam_i - 2*dlam, T_pm, Lambda)
        d_dk_dl = (dk1 - dk2) / (4 * dlam)   # derivative w.r.t. λ_fundamental
        if abs(d_dk_dl) < 1e-30:
            return np.nan
        return 0.886 * np.pi / abs(d_dk_dl)  # [nm·mm]

    # ------------------------------------------------------------------
    # Full sweep
    # ------------------------------------------------------------------
    def full_analysis(self,
                      lam_scan_arr: np.ndarray,
                      Lambda: float,
                      T_arr: np.ndarray,
                      L_mm: float,
                      process: str = "SHG") -> dict:
        """
        Compute all PM curves while sweeping the scan wavelength.

        For SHG  : lam_scan = λ_fundamental (input pump).
        For OPG  : lam_scan = λ_pump (the driving field).

        Returns dict with keys:
            lam_scan, lam_p, lam_s, lam_i,
            dk_at_Tfix, T_fix, T_pm, bw_T, bw_lam
        """
        T_fix = T_arr[len(T_arr) // 2]

        trips  = [_three_wavelengths(ls, process) for ls in lam_scan_arr]
        lp_arr = np.array([t[0] for t in trips])
        ls_arr = np.array([t[1] for t in trips])
        li_arr = np.array([t[2] for t in trips])

        dk_fix = self.dk_array(lp_arr, ls_arr, li_arr, T_fix, Lambda)

        T_pm   = np.full(len(lam_scan_arr), np.nan)
        bw_T   = np.full(len(lam_scan_arr), np.nan)
        bw_lam = np.full(len(lam_scan_arr), np.nan)

        for i, (lp, ls, li) in enumerate(zip(lp_arr, ls_arr, li_arr)):
            Tpm = self.pm_temperature(lp, ls, li, Lambda, T_arr)
            T_pm[i]   = Tpm
            bw_T[i]   = self.bw_temperature(lp, ls, li, Tpm, L_mm, Lambda)
            bw_lam[i] = self.bw_wavelength(lp, ls, li, Tpm, L_mm, Lambda)

        return {
            "lam_scan": lam_scan_arr,
            "lam_p":    lp_arr,
            "lam_s":    ls_arr,
            "lam_i":    li_arr,
            "dk_at_Tfix": dk_fix,
            "T_fix":    T_fix,
            "T_pm":     T_pm,
            "bw_T":     bw_T,
            "bw_lam":   bw_lam,
        }


# ------------------------------------------------------------------
# Helper: derive CWE wavelengths (lam_p_high, lam_s, lam_i) from scan λ
# ------------------------------------------------------------------
def _three_wavelengths(lam_scan: float,
                       process: str) -> tuple[float, float, float]:
    """
    Return (lam_p, lam_s, lam_i) for the CWE convention:
        Δk = k(lam_p) − k(lam_s) − k(lam_i)
    where lam_p is always the SHORTEST wavelength (highest frequency).

    lam_scan convention
    -------------------
    SHG  : lam_scan = λ_fundamental  → lam_p = λ_fund/2,  lam_s = lam_i = λ_fund
    OPG  : lam_scan = λ_pump         → lam_p = λ_pump,    lam_s = lam_i = 2·λ_pump
    SFG  : lam_scan = λ_pump         → same as OPG (degenerate default)
    DFG  : lam_scan = λ_pump         → same as OPG (degenerate default)
    """
    if process == "SHG":
        lam_p = lam_scan / 2.0      # SH  (high freq, generated)
        lam_s = lam_i = lam_scan    # fundamental (low freq, input)
    else:
        # OPG / DFG / SFG: pump is the high-freq driver
        lam_p = lam_scan
        lam_s = lam_i = 2.0 * lam_scan   # degenerate signal/idler
    return lam_p, lam_s, lam_i


# ══════════════════════════════════════════════════════════════════════════
# Birefringent (angle-tuned) uniaxial phase matching
#
# Analytic closed-form phase-matching angle from Dmitriev, Gurzadyan &
# Nikogosyan, "Handbook of Nonlinear Optical Crystals", Table 2.1 — exact
# for ooe / oeo / eoo, accurate to ~0.1-0.2 deg for oee / eoe / eeo.
#
# Convention (same as the handbook): waves 1 and 2 are the two
# lower-frequency (longer-wavelength) fields, wave 3 is the highest-
# frequency (shortest-wavelength) field — always lam3 < lam1, lam3 < lam2.
# The letters in a PM-type string ("ooe", "eoe", ...) give the
# polarisation (o/e) of waves 1, 2, 3 in that order.
# ══════════════════════════════════════════════════════════════════════════

PM_TYPES = {
    "negative": ["ooe", "oee", "eoe"],   # k_o1 + k_o2 = k_e3(theta), etc.
    "positive": ["eeo", "oeo", "eoo"],
}

PM_TYPE_LABELS = {
    "ooe": "ooe  (Type I)",
    "oee": "oee  (Type II)",
    "eoe": "eoe  (Type II)",
    "eeo": "eeo  (Type I)",
    "oeo": "oeo  (Type II)",
    "eoo": "eoo  (Type II)",
}


def crystal_sign(sf_o: SellmeierFormula, sf_e: SellmeierFormula,
                  lam_ref: float, T_ref: float = 25.0) -> str:
    """Negative (n_o > n_e) or positive (n_e > n_o) uniaxial, from the
    crystal's own o/e Sellmeier formulas at a reference wavelength/T —
    not a stored property, since it follows directly from the formulas."""
    no = float(sf_o.n(lam_ref, T_ref))
    ne = float(sf_e.n(lam_ref, T_ref))
    return "negative" if no > ne else "positive"


def wavelengths_for_scan(lam_scan, lam_fixed: float, process: str):
    """
    Return (lam1, lam2, lam3) — lam3 always the shortest wavelength.

    SHG              : lam1 = lam2 = lam_scan (fundamental),  lam3 = lam_scan/2.
                        lam_fixed is unused (fully degenerate, no free 3rd λ).
    SFG / DFG / OPG  : lam3 = lam_fixed (pump, or sum-frequency output —
                        mathematically identical, always the shortest λ);
                        lam1 = lam_scan; lam2 from energy conservation
                        1/lam3 = 1/lam1 + 1/lam2. NaN where lam_scan <= lam_fixed
                        (energy conservation would require lam2 <= 0).

    lam_scan may be a scalar or a numpy array; lam_fixed is always scalar.
    """
    lam_scan = np.asarray(lam_scan, dtype=float)
    if process == "SHG":
        lam1 = lam2 = lam_scan
        lam3 = lam_scan / 2.0
    else:
        lam1 = lam_scan
        lam3 = np.full_like(lam_scan, float(lam_fixed))
        with np.errstate(divide="ignore", invalid="ignore"):
            inv2 = 1.0 / lam3 - 1.0 / lam1
            lam2 = np.where(inv2 > 0, 1.0 / np.where(inv2 > 0, inv2, np.nan), np.nan)
    return lam1, lam2, lam3


class UniaxialPMCalc:
    """
    Angle-tuned phase matching for birefringent uniaxial crystals, via the
    closed-form expressions of Table 2.1 (no root-finding needed).

    Parameters
    ----------
    sf_o, sf_e : SellmeierFormula
        Pure ordinary / extraordinary axis formulas (NOT angle-mixed).
    lam_ref, T_ref : reference point used only to determine the crystal's
        uniaxial sign (n_o vs n_e); does not affect theta_pm() itself.
    """

    def __init__(self, sf_o: SellmeierFormula, sf_e: SellmeierFormula,
                 lam_ref: float = 1.0, T_ref: float = 25.0):
        self.sf_o = sf_o
        self.sf_e = sf_e
        self._sign = crystal_sign(sf_o, sf_e, lam_ref, T_ref)

    @property
    def sign(self) -> str:
        return self._sign

    def valid_pm_types(self) -> list[str]:
        return list(PM_TYPES[self._sign])

    def theta_pm(self, lam1, lam2, lam3, T, pm_type: str) -> np.ndarray:
        """Phase-matching angle theta_pm [deg] (NaN where no real solution
        exists, i.e. tan^2(theta) < 0). lam1/lam2/lam3/T may be scalars or
        broadcastable numpy arrays."""
        sign = self._sign
        if pm_type not in PM_TYPES[sign]:
            raise ValueError(
                f"PM type '{pm_type}' is not valid for a {sign} uniaxial crystal "
                f"(valid: {PM_TYPES[sign]})")

        no1 = np.asarray(self.sf_o.n(lam1, T), dtype=float)
        no2 = np.asarray(self.sf_o.n(lam2, T), dtype=float)
        no3 = np.asarray(self.sf_o.n(lam3, T), dtype=float)
        ne1 = np.asarray(self.sf_e.n(lam1, T), dtype=float)
        ne2 = np.asarray(self.sf_e.n(lam2, T), dtype=float)
        ne3 = np.asarray(self.sf_e.n(lam3, T), dtype=float)

        A, B, C = no1 / lam1, no2 / lam2, no3 / lam3
        D, E, F = ne1 / lam1, ne2 / lam2, ne3 / lam3

        with np.errstate(divide="ignore", invalid="ignore"):
            if sign == "negative":
                U = (A + B) ** 2 / C ** 2
                W = (A + B) ** 2 / F ** 2
                if pm_type == "ooe":
                    tan2 = (1 - U) / (W - 1)
                elif pm_type == "oee":
                    R = (A + B) ** 2 / (D + B) ** 2
                    tan2 = (1 - U) / (W - R)
                else:  # eoe
                    Q = (A + B) ** 2 / (A + E) ** 2
                    tan2 = (1 - U) / (W - Q)
            else:
                if pm_type == "eeo":
                    U = (A + B) ** 2 / C ** 2
                    S = (A + B) ** 2 / (D + E) ** 2
                    tan2 = (1 - U) / (U - S)
                elif pm_type == "oeo":
                    V = B ** 2 / (C - A) ** 2
                    Y = B ** 2 / E ** 2
                    tan2 = (1 - V) / (V - Y)
                else:  # eoo
                    Tt = A ** 2 / (C - B) ** 2
                    Z = A ** 2 / D ** 2
                    tan2 = (1 - Tt) / (Tt - Z)

        tan2 = np.asarray(tan2, dtype=float)
        theta = np.full(tan2.shape, np.nan) if tan2.ndim else np.nan
        valid = np.isfinite(tan2) & (tan2 >= 0)
        if tan2.ndim:
            theta[valid] = np.degrees(np.arctan(np.sqrt(tan2[valid])))
        elif valid:
            theta = float(np.degrees(np.arctan(np.sqrt(tan2))))
        return theta
