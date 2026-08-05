"""
ConfigBuilder
=============
Translates GUI parameters into the JSON config consumed by the C++ engine.

All optical constants (n, β₁, β₂, β₃, Δk) are computed here via
SellmeierFormula so that the C++ engine stays Sellmeier-free.
"""

import math
import numpy as np

from .crystal_db import CrystalDB, get_db
from .sellmeier import SellmeierFormula


_C = 299.792_458  # μm/ps


def _idler_wavelength(lp: float, ls: float) -> float:
    """Energy conservation: 1/λi = 1/λp − 1/λs  →  λi = λp·λs / (λs − λp)."""
    inv_li = 1.0 / lp - 1.0 / ls
    if inv_li <= 0:
        raise ValueError(
            f"Unphysical idler: λp={lp:.4f} μm must be shorter than λs={ls:.4f} μm")
    return 1.0 / inv_li


def _dk_qpm(k_p: float, k_s: float, k_i: float, Lambda_um: float) -> float:
    """Δk_eff = (kp − ks − ki) − 2π/Λ  [μm⁻¹]."""
    dk_bare = k_p - k_s - k_i
    return dk_bare - (2.0 * math.pi / Lambda_um) if Lambda_um > 0 else dk_bare


def build_config(params: dict, db: CrystalDB | None = None) -> dict:
    """
    Build a complete engine JSON config from GUI params.

    Parameters
    ----------
    params  : dict from SimulationTab._collect_params()
    db      : CrystalDB (uses singleton if None)

    Returns
    -------
    dict — JSON-serializable, ready to pass to EngineRunner
    """
    if db is None:
        db = get_db()

    process      = params["process"]           # "SHG" | "SFG" | "OPG" | "DFG"
    crystal_name = params["crystal"]
    T_C          = float(params["temperature"])
    Lambda_um    = float(params["grating_um"])
    Lcr_mm       = float(params["length_mm"])

    # ── Wavelengths ────────────────────────────────────────────────────────────
    lp_gui = float(params["pump"]["lambda_um"])    # fundamental for SHG, pump for rest
    ls_gui = float(params["signal"]["lambda_um"])
    li_gui = float(params["idler"]["lambda_um"])
    degenerate = bool(params["idler"]["degenerate"])

    if process == "SHG":
        # Engine convention: pump = fundamental(ω), signal = idler = SH(2ω)
        pump_um   = lp_gui
        signal_um = lp_gui / 2.0
        idler_um  = lp_gui / 2.0
        degenerate = True
    elif process in ("OPG", "SFG", "DFG"):
        pump_um   = lp_gui
        signal_um = ls_gui
        if degenerate:
            idler_um = ls_gui
        else:
            try:
                idler_um = _idler_wavelength(pump_um, signal_um)
            except ValueError:
                idler_um = li_gui  # fall back to user value
    else:
        pump_um   = lp_gui
        signal_um = ls_gui
        idler_um  = li_gui

    # ── Sellmeier ──────────────────────────────────────────────────────────────
    cr = db.get(crystal_name)
    if cr is None:
        raise ValueError(f"Crystal '{crystal_name}' not found in database")

    sf = db.sellmeier(cr, axis="e")

    np_val  = float(sf.n(pump_um,   T_C))
    ns_val  = float(sf.n(signal_um, T_C))
    ni_val  = float(sf.n(idler_um,  T_C))

    vp = float(sf.GV(pump_um,   T_C))
    vs = float(sf.GV(signal_um, T_C))
    vi = float(sf.GV(idler_um,  T_C))

    b2p = float(sf.beta2(pump_um,   T_C))
    b2s = float(sf.beta2(signal_um, T_C))
    b2i = float(sf.beta2(idler_um,  T_C))

    b3p = float(sf.beta3(pump_um,   T_C))
    b3s = float(sf.beta3(signal_um, T_C))
    b3i = float(sf.beta3(idler_um,  T_C))

    # ── Phase mismatch ─────────────────────────────────────────────────────────
    kp = float(sf.k(pump_um,   T_C))
    ks = float(sf.k(signal_um, T_C))
    ki = float(sf.k(idler_um,  T_C))

    if process == "SHG":
        # Δk = k_SH − 2·k_fund − 2π/Λ
        # signal_um = SH, pump_um = fundamental
        dk = _dk_qpm(ks, kp, kp, Lambda_um)   # k_SH − k_fund − k_fund
    else:
        # Standard: Δk = kp − ks − ki − 2π/Λ
        dk = _dk_qpm(kp, ks, ki, Lambda_um)

    # Allow explicit Δk override (used in parameter sweep mode)
    _dk_override = params.get("dk")
    if _dk_override is not None:
        dk = float(_dk_override)

    # ── Transverse grid size ───────────────────────────────────────────────────
    waist_um = float(params["pump"]["waist_um"])
    if "lx_mm" in params and float(params["lx_mm"]) > 0:
        LX_mm = float(params["lx_mm"])
    else:
        half_mm = max(6.0 * waist_um * 1e-3, 0.5)
        LX_mm   = round(2.0 * half_mm, 3)
    if "ly_mm" in params and float(params["ly_mm"]) > 0:
        LY_mm = float(params["ly_mm"])
    else:
        LY_mm = LX_mm

    # ── Pump mode string ───────────────────────────────────────────────────────
    if "mode" in params["pump"]:
        pump_mode = params["pump"]["mode"]   # e.g. "focused-cw" — set directly by the Field mode combo
    else:
        regime  = params["pump"]["regime"]   # "CW" | "Pulsed"
        profile = params["pump"]["profile"]  # "Focused Gaussian" | "Plane wave"
        _pulsed  = regime == "Pulsed"
        _focused = "Focused" in profile
        if _focused and not _pulsed:
            pump_mode = "focused-cw"
        elif _focused and _pulsed:
            pump_mode = "focused-pulsed"
        elif not _focused and not _pulsed:
            pump_mode = "waveplane-cw"
        else:
            pump_mode = "waveplane-pulsed"

    # ── Time window ────────────────────────────────────────────────────────────
    # Pulsed: use 10× FWHM by default; caller may override via "t_window_ps".
    # CW: 1 ps placeholder (NT=1 and dispersion disabled in CW runs).
    fwhm_ps = float(params["pump"].get("fwhm_ps", 0.2))
    _t_win_override = params.get("t_window_ps")
    if _pulsed and _t_win_override is not None and float(_t_win_override) > 0:
        t_window = float(_t_win_override)
    else:
        t_window = max(10.0 * fwhm_ps, 1.0) if _pulsed else 1.0

    # ── Assemble JSON ──────────────────────────────────────────────────────────
    cfg = {
        "crystal": {
            "name": crystal_name,
            "type": "QPM",
            "geometry": {
                "Lcr_mm": Lcr_mm,
                "LX_mm":  LX_mm,
                "LY_mm":  LY_mm,
            },
            "process":    process,
            "degenerate": degenerate,
            "wavelengths": {
                "pump_um":   pump_um,
                "signal_um": signal_um,
                "idler_um":  idler_um,
                "dk_um-1":   dk,
            },
            "optics": {
                "np": np_val, "ns": ns_val, "ni": ni_val,
                "vp_um_ps":  vp,  "vs_um_ps":  vs,  "vi_um_ps":  vi,
                "b2p_ps2_um": b2p, "b2s_ps2_um": b2s, "b2i_ps2_um": b2i,
                "b3p_ps3_um": b3p, "b3s_ps3_um": b3s, "b3i_ps3_um": b3i,
            },
            "nonlinear": {
                # Only deff stored in database; dQ = 2×deff/π for QPM is applied in C++ code
                "deff_pm_V": float(cr["deff"]),
                "alpha_crp_cm-1": float(cr["alpha_p"]) if params.get("alpha_p_active", True) else 0.0,
                "alpha_crs_cm-1": float(cr["alpha_s"]) if params.get("alpha_s_active", True) else 0.0,
                "alpha_cri_cm-1": float(cr["alpha_i"]) if params.get("alpha_i_active", True) else 0.0,
                "beta_crp_um_W":  float(cr.get("beta_p", 0.0)) if params.get("beta_p_active", False) else 0.0,
                "beta_crs_um_W":  float(cr.get("beta_s", 0.0)) if params.get("beta_s_active", False) else 0.0,
                "beta_cri_um_W":  float(cr.get("beta_i", 0.0)) if params.get("beta_i_active", False) else 0.0,
                "rho_p_rad":      float(cr.get("rho_p", 0.0)) if params.get("rho_p_active", False) else 0.0,
                "rho_s_rad":      float(cr.get("rho_s", 0.0)) if params.get("rho_s_active", False) else 0.0,
                "rho_i_rad":      float(cr.get("rho_i", 0.0)) if params.get("rho_i_active", False) else 0.0,
            },
            "qpm": {
                "lambda0_um":   Lambda_um,
                "T0_C":         T_C,
                "alpha_th_K-1": float(cr["alpha_th"]),
            },
            "thermal_props": {
                "kappa_W_mK": float(cr["kappa"]),
                "cp_J_kgK":   float(cr["cp"]),
                "rho_kg_m3":  float(cr["rho"]),
            },
        },
        "fields": {
            "pump": {
                "power_W":        float(params["pump"]["power_W"]),
                "waist_um":       waist_um,
                "fwhm_ps":        fwhm_ps,
                "focal_point_um": Lcr_mm * 500.0,  # crystal centre in μm
                "mode":           pump_mode,
            },
            "signal": {
                "noise": params["signal"].get("mode", "noise") == "noise",
                "power_W": float(params["signal"].get("power_W", 0.0)),
            },
            "t_window_ps": t_window,
        },
        "save_mode": {
            "save_pump":                   True,
            "save_signal":                 True,
            "save_idler":                  True,
            "save_in_pump":                True,
            "save_propagation":            True,
            "save_time_and_freq_vectors":  True,
        },
        "dispersion": _pulsed,
        "undepleted_pump": bool(params.get("undepleted_pump", False)),
    }

    # Thermal section — pass through with correct key names for the C++ engine
    th = params.get("thermal", {})
    if th.get("enabled", False):
        cfg["thermal"] = {
            "enabled":        True,
            "T_oven_C":       float(th.get("T_oven_C",       25.0)),
            "T_ambient_C":    float(th.get("T_ambient_C",    25.0)),
            "h_conv_W_m2K":   float(th.get("h_conv_W_m2K",  10.0)),
            "max_iter":       int(th.get("max_iter",         100000)),
            "tol":            float(th.get("tol",            5e-4)),
            "max_outer_iter": int(th.get("max_outer_iter",   10)),
            "outer_tol":      float(th.get("outer_tol",      0.001)),
        }
    else:
        cfg["thermal"] = {"enabled": False}

    return cfg


def config_summary(cfg: dict) -> str:
    """One-line summary of the config for the log."""
    cr  = cfg["crystal"]
    wl  = cr["wavelengths"]
    opt = cr["optics"]
    dk  = wl["dk_um-1"]
    return (
        f"  Crystal : {cr['name']}  L={cr['geometry']['Lcr_mm']} mm\n"
        f"  Process : {cr['process']}  degenerate={cr['degenerate']}\n"
        f"  λ_pump  : {wl['pump_um']:.4f} μm  λ_signal: {wl['signal_um']:.4f} μm\n"
        f"  n_pump  : {opt['np']:.5f}  n_signal: {opt['ns']:.5f}\n"
        f"  Δk_eff  : {dk:.4f} μm⁻¹  (Δk·L = {dk * cr['geometry']['Lcr_mm'] * 1e3:.2f} rad)\n"
        f"  GVD_p   : {opt['b2p_ps2_um']*1e9:.2f} fs²/mm  "
        f"GVD_s: {opt['b2s_ps2_um']*1e9:.2f} fs²/mm"
    )
