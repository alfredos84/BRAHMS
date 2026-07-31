#ifndef _CRYSTAL_CUH
#define _CRYSTAL_CUH

/*
 * Crystal
 * =======
 * All optical, nonlinear and thermal properties of the crystal are read from
 * the JSON configuration file.  No Sellmeier equation lives in C++; the Python
 * GUI computes n(λ,T), β₁, β₂, β₃ via SymPy and embeds them in the JSON.
 *
 * JSON schema (crystal section):
 * {
 *   "crystal": {
 *     "name":      "MgO:PPLN",
 *     "type":      "QPM",           // "QPM" | "Birefringent"
 *     "geometry":  { "Lcr_mm": 10.0, "LX_mm": 2.0, "LY_mm": 1.0 },
 *     "process":   "SHG",           // "SHG" | "SFG" | "OPG" | "DFG"
 *     "degenerate": true,           // signal == idler (SHG / degenerate OPG)
 *     "wavelengths": {
 *       "pump_um":   1.064,
 *       "signal_um": 0.532,
 *       "idler_um":  0.532,
 *       "dk_um-1":   0.0            // phase mismatch [μm⁻¹], precomputed by Python
 *     },
 *     "optics": {
 *       "np": 2.14883, "ns": 2.22515, "ni": 2.22515,
 *       "vp_um_ps": 136.06, "vs_um_ps": 122.81, "vi_um_ps": 122.81,
 *       "b2p_ps2_um": 2.345e-7, "b2s_ps2_um": 0.0, "b2i_ps2_um": 0.0,
 *       "b3p_ps3_um": 0.0,       "b3s_ps3_um": 0.0, "b3i_ps3_um": 0.0
 *     },
 *     "nonlinear": {
 *       "deff_pm_V": 17.0,
 *       "alpha_crp_cm-1": 0.002,
 *       "alpha_crs_cm-1": 0.025,
 *       "alpha_cri_cm-1": 0.025
 *     },
 *     "qpm": {
 *       "lambda0_um": 6.99,
 *       "T0_C": 27.0,
 *       "alpha_th_K-1": 1.48e-5
 *     },
 *     "thermal": {
 *       "kappa_W_mK": 4.6,
 *       "cp_J_kgK":  628.0,
 *       "rho_kg_m3": 4640.0
 *     }
 *   }
 * }
 */

struct Crystal {
    // ── Geometry ─────────────────────────────────────────────────────────────
    real_t Lcr;          // crystal length [μm]
    real_t LX, LY;      // transverse size [μm]
    real_t dz;          // longitudinal step [μm]
    real_t dx, dy;      // transverse steps [μm]

    // ── Process ──────────────────────────────────────────────────────────────
    int    process;     // PROC_SHG / PROC_SFG / PROC_OPG / PROC_DFG
    bool   degenerate;  // signal == idler
    bool   undepleted;  // if true, dAp/dz = 0 (undepleted pump approximation)

    // ── Wavelengths [μm] ─────────────────────────────────────────────────────
    real_t lp, ls, li;

    // ── Refractive indices ───────────────────────────────────────────────────
    real_t np, ns, ni;

    // ── Group velocities [μm/ps] ─────────────────────────────────────────────
    real_t vp, vs, vi;

    // ── GVD [ps²/μm] ────────────────────────────────────────────────────────
    real_t b2p, b2s, b2i;

    // ── TOD [ps³/μm] ────────────────────────────────────────────────────────
    real_t b3p, b3s, b3i;

    // ── Phase mismatch [μm⁻¹] (Python-computed) ─────────────────────────────
    real_t dk;

    // ── Nonlinear ────────────────────────────────────────────────────────────
    real_t dQ;           // effective d [pm/V] (d_eff or d33·2/π for QPM)
    real_t alpha_crp;    // pump absorption [μm⁻¹]   (converted from cm⁻¹)
    real_t alpha_crs;    // signal absorption [μm⁻¹]
    real_t alpha_cri;    // idler  absorption [μm⁻¹]
    real_t beta_crp;     // pump TPA coefficient [μm/W]
    real_t beta_crs;     // signal TPA coefficient [μm/W]
    real_t beta_cri;     // idler TPA coefficient [μm/W]
    real_t rho_p;        // pump   spatial walk-off angle [rad]
    real_t rho_s;        // signal spatial walk-off angle [rad]
    real_t rho_i;        // idler  spatial walk-off angle [rad]

    // ── QPM grating ──────────────────────────────────────────────────────────
    real_t Lambda0;      // grating period [μm]
    real_t T0;           // reference temperature [°C]
    real_t alpha_th;     // thermal expansion coefficient [K⁻¹]

    // ── Thermal material props ────────────────────────────────────────────────
    real_t kappa_th;     // thermal conductivity [W/(m·K)]
    real_t cp_th;        // heat capacity [J/(kg·K)]
    real_t rho_th;       // density [kg/m³]

    // ── Constructor: populate everything from JSON ────────────────────────────
    Crystal(const json& cfg) {
        const auto& cr  = cfg["crystal"];
        const auto& geo = cr["geometry"];
        const auto& wl  = cr["wavelengths"];
        const auto& opt = cr["optics"];
        const auto& nl  = cr["nonlinear"];

        // geometry — store in μm for consistency with all kernels (dk, kp, etc.)
        Lcr = geo["Lcr_mm"].get<real_t>() * 1000.0f;   // mm → μm
        LX  = geo["LX_mm"].get<real_t>()  * 1000.0f;
        LY  = geo["LY_mm"].get<real_t>()  * 1000.0f;
        dz  = Lcr / static_cast<real_t>(NZ);            // μm/step
        dx  = LX  / static_cast<real_t>(NX);            // μm/pixel
        dy  = LY  / static_cast<real_t>(NY);

        // process
        std::string proc_str = cr["process"].get<std::string>();
        if      (proc_str == "SHG") process = PROC_SHG;
        else if (proc_str == "SFG") process = PROC_SFG;
        else if (proc_str == "OPG") process = PROC_OPG;
        else if (proc_str == "DFG") process = PROC_DFG;
        else {
            std::cerr << "[Crystal] Unknown process: " << proc_str << "\n";
            process = PROC_SHG;
        }
        degenerate = cr.value("degenerate", true);
        undepleted = cfg.value("undepleted_pump", false);

        // wavelengths
        lp = wl["pump_um"].get<real_t>();
        ls = wl["signal_um"].get<real_t>();
        li = wl["idler_um"].get<real_t>();
        dk = wl["dk_um-1"].get<real_t>();

        // optical properties (from Python SymPy — no Sellmeier here)
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

        // nonlinear
        dQ        = nl["deff_pm_V"].get<real_t>();  // pm/V

        // For QPM crystals: dQ_eff = 2 * deff / π (effective modulation)
        if (cr.contains("qpm")) {
            dQ = 2.0f * dQ / M_PI;
        }

        // convert absorption from cm⁻¹ to μm⁻¹  (1 cm⁻¹ = 1e-4 μm⁻¹)
        alpha_crp = nl["alpha_crp_cm-1"].get<real_t>() * 1e-4f;
        alpha_crs = nl["alpha_crs_cm-1"].get<real_t>() * 1e-4f;
        alpha_cri = nl["alpha_cri_cm-1"].get<real_t>() * 1e-4f;
        // TPA coefficients [μm/W] — zero by default (off)
        beta_crp  = nl.value("beta_crp_um_W", 0.0f);
        beta_crs  = nl.value("beta_crs_um_W", 0.0f);
        beta_cri  = nl.value("beta_cri_um_W", 0.0f);
        // Spatial walk-off angles [rad] — zero by default (off)
        rho_p     = nl.value("rho_p_rad", 0.0f);
        rho_s     = nl.value("rho_s_rad", 0.0f);
        rho_i     = nl.value("rho_i_rad", 0.0f);

        // QPM grating
        if (cr.contains("qpm")) {
            const auto& q = cr["qpm"];
            Lambda0  = q.value("lambda0_um", 0.0f);
            T0       = q.value("T0_C",       27.0f);
            alpha_th = q.value("alpha_th_K-1", 1.48e-5f);
        } else {
            Lambda0  = 0.0f;
            T0       = 27.0f;
            alpha_th = 0.0f;
        }

        // thermal material
        if (cr.contains("thermal_props")) {
            const auto& th = cr["thermal_props"];
            kappa_th = th.value("kappa_W_mK",  4.6f);
            cp_th    = th.value("cp_J_kgK",  628.0f);
            rho_th   = th.value("rho_kg_m3", 4640.0f);
        } else {
            kappa_th = 4.6f;
            cp_th    = 628.0f;
            rho_th   = 4640.0f;
        }
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

#endif // _CRYSTAL_CUH
