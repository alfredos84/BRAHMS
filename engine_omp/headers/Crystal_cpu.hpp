#pragma once
/*
 * Crystal_cpu.hpp
 * ===============
 * Identical logic to Crystal.cuh — reads all optical constants from JSON.
 * No Sellmeier equations; Python GUI computes n, β1, β2, β3, Δk via SymPy.
 */

#include "Common_cpu.hpp"
#include <string>

struct Crystal {
    // Geometry
    real_t Lcr, LX, LY;
    real_t dz, dx, dy;

    // Process
    int  process;
    bool degenerate;
    bool undepleted;  // if true, dAp/dz = 0 (undepleted pump approximation)

    // Wavelengths [μm]
    real_t lp, ls, li;

    // Refractive indices
    real_t np, ns, ni;

    // Group velocities [μm/ps]
    real_t vp, vs, vi;

    // GVD [ps²/μm]
    real_t b2p, b2s, b2i;

    // TOD [ps³/μm]
    real_t b3p, b3s, b3i;

    // Phase mismatch [μm⁻¹]
    real_t dk;

    // Nonlinear
    real_t dQ;                    // d_eff [pm/V]
    real_t alpha_crp, alpha_crs, alpha_cri;  // absorption [μm⁻¹]
    real_t beta_crp,  beta_crs,  beta_cri;   // TPA coefficients [μm/W]

    // QPM grating
    real_t Lambda0, T0, alpha_th;

    // Thermal props
    real_t kappa_th, cp_th, rho_th;

    // ── Constructor ─────────────────────────────────────────────────────────
    Crystal(const json& cfg) {
        const auto& cr  = cfg["crystal"];
        const auto& geo = cr["geometry"];
        const auto& wl  = cr["wavelengths"];
        const auto& opt = cr["optics"];
        const auto& nl  = cr["nonlinear"];

        Lcr = geo["Lcr_mm"].get<real_t>() * 1000.0f;   // mm → μm
        LX  = geo["LX_mm"].get<real_t>()  * 1000.0f;  // mm → μm
        LY  = geo["LY_mm"].get<real_t>()  * 1000.0f;  // mm → μm
        dz  = Lcr / static_cast<real_t>(NZ);           // μm/step
        dx  = LX  / static_cast<real_t>(NX);           // μm/pixel
        dy  = LY  / static_cast<real_t>(NY);           // μm/pixel

        std::string ps = cr["process"].get<std::string>();
        if      (ps == "SHG") process = PROC_SHG;
        else if (ps == "SFG") process = PROC_SFG;
        else if (ps == "OPG") process = PROC_OPG;
        else if (ps == "DFG") process = PROC_DFG;
        else                  process = PROC_SHG;
        degenerate = cr.value("degenerate", true);
        undepleted = cfg.value("undepleted_pump", false);

        lp = wl["pump_um"].get<real_t>();
        ls = wl["signal_um"].get<real_t>();
        li = wl["idler_um"].get<real_t>();
        dk = wl["dk_um-1"].get<real_t>();

        np  = opt["np"].get<real_t>();
        ns  = opt["ns"].get<real_t>();
        ni  = opt["ni"].get<real_t>();
        vp  = opt["vp_um_ps"].get<real_t>();
        vs  = opt["vs_um_ps"].get<real_t>();
        vi  = opt["vi_um_ps"].get<real_t>();
        b2p = opt["b2p_ps2_um"].get<real_t>();
        b2s = opt["b2s_ps2_um"].get<real_t>();
        b2i = opt["b2i_ps2_um"].get<real_t>();
        b3p = opt.value("b3p_ps3_um", 0.0f);
        b3s = opt.value("b3s_ps3_um", 0.0f);
        b3i = opt.value("b3i_ps3_um", 0.0f);

        dQ        = nl["deff_pm_V"].get<real_t>();

        // For QPM crystals: dQ_eff = 2 * deff / π (effective modulation)
        if (cr.contains("qpm")) {
            dQ = 2.0f * dQ / PI;
        }

        alpha_crp = nl["alpha_crp_cm-1"].get<real_t>() * 1e-4f;
        alpha_crs = nl["alpha_crs_cm-1"].get<real_t>() * 1e-4f;
        alpha_cri = nl["alpha_cri_cm-1"].get<real_t>() * 1e-4f;
        beta_crp  = nl.value("beta_crp_um_W", 0.0f);
        beta_crs  = nl.value("beta_crs_um_W", 0.0f);
        beta_cri  = nl.value("beta_cri_um_W", 0.0f);

        if (cr.contains("qpm")) {
            const auto& q = cr["qpm"];
            Lambda0  = q.value("lambda0_um",   0.0f);
            T0       = q.value("T0_C",          27.0f);
            alpha_th = q.value("alpha_th_K-1", 1.48e-5f);
        } else { Lambda0 = 0.0f; T0 = 27.0f; alpha_th = 0.0f; }

        if (cr.contains("thermal_props")) {
            const auto& th = cr["thermal_props"];
            kappa_th = th.value("kappa_W_mK",  4.6f);
            cp_th    = th.value("cp_J_kgK",  628.0f);
            rho_th   = th.value("rho_kg_m3", 4640.0f);
        } else { kappa_th = 4.6f; cp_th = 628.0f; rho_th = 4640.0f; }
    }

    void print() const {
        const char* proc_name[] = {"SHG", "SFG", "OPG", "DFG"};
        print_line();
        std::cout << "  Process : " << proc_name[process] << "\n"
                  << "  Crystal : L=" << Lcr*1e-3f << " mm"
                  << "   LX=" << LX*1e-3f << " mm  LY=" << LY*1e-3f << " mm\n"
                  << "  Wavelengths (μm) :\n"
                  << "    λ_p=" << lp  << "   λ_s=" << ls  << "   λ_i=" << li  << "\n"
                  << "  Refractive indices :\n"
                  << "    n_p=" << np  << "   n_s=" << ns  << "   n_i=" << ni  << "\n"
                  << "  Linear absorption (cm⁻¹) :\n"
                  << "    α_p=" << alpha_crp*1e4f
                  <<  "   α_s=" << alpha_crs*1e4f
                  <<  "   α_i=" << alpha_cri*1e4f << "\n"
                  << "  TPA (cm/GW) :\n"
                  << "    β_p=" << beta_crp*1e5f
                  <<  "   β_s=" << beta_crs*1e5f
                  <<  "   β_i=" << beta_cri*1e5f << "\n"
                  << "  Phase matching :\n"
                  << "    Δk=" << dk << " μm⁻¹   d_eff=" << dQ << " pm/V\n"
                  << "  GVD (fs²/mm) :\n"
                  << "    β2_p=" << b2p*1e9f
                  <<  "   β2_s=" << b2s*1e9f
                  <<  "   β2_i=" << b2i*1e9f << "\n"
                  << "  TOD (fs³/mm) :\n"
                  << "    β3_p=" << b3p*1e12f
                  <<  "   β3_s=" << b3s*1e12f
                  <<  "   β3_i=" << b3i*1e12f << "\n";
        print_line();
    }
};
