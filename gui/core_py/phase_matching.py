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
