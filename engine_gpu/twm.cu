/*
 * twm.cu — Unified Three-Wave Mixing engine
 * ==========================================
 * GPU-accelerated (3+1)D nonlinear optics simulator.
 * Supports SHG, SFG, OPG, DFG with optional:
 *   - Dispersion (SSFM)
 *   - Diffraction (beam propagation)
 *   - Thermal model (3D heat equation + QPM Δk(T))
 *   - Multipass
 *
 * Usage:
 *   twm config.json [output_dir]
 *
 * Compilation:
 *   make NX=128 NY=128 NZ=100 NT=512
 *
 * All optical parameters (n, β1, β2, β3, Δk) are precomputed by the Python
 * GUI via SymPy and embedded in the JSON config — no Sellmeier C++ code here.
 */

#include "headers/Libraries.cuh"
#include "headers/PackageLibraries.cuh"


// ── JSON helpers ──────────────────────────────────────────────────────────────

json load_config(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[twm] Cannot open config: " << path << "\n";
        exit(1);
    }
    json cfg;
    f >> cfg;
    return cfg;
}


// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    auto t_start = std::chrono::high_resolution_clock::now();
    // Disable stdout buffering so QProcess (pipe) sees output line-by-line
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    welcome();

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " config.json [output_dir]\n";
        return 1;
    }

    std::string config_path = argv[1];
    std::string out_dir     = (argc >= 3) ? argv[2] : ".";

    // ── Load configuration ────────────────────────────────────────────────────
    json cfg = load_config(config_path);
    is_gpu_available();
    print_grid();

    // ── Crystal ───────────────────────────────────────────────────────────────
    Crystal Cr(cfg);
    Cr.print();

    // ── Fields ────────────────────────────────────────────────────────────────
    const auto& fld = cfg["fields"];
    real_t pump_power = fld["pump"]["power_W"].get<real_t>();
    real_t pump_waist = fld["pump"]["waist_um"].get<real_t>();
    real_t pump_fwhm  = fld["pump"].value("fwhm_ps", 1.0f);
    real_t pump_fp    = fld["pump"].value("focal_point_um", 0.0f);
    std::string pump_mode = fld["pump"].value("mode", "waveplane-cw");
    real_t t_window   = fld.value("t_window_ps", 10.0f);

    EFields A(pump_power, pump_waist, &Cr);
    A.set_time_freq_vectors(t_window);
    A.set_pump_field(pump_power, pump_fwhm, pump_waist, pump_fp, pump_mode);

    // ── Initialise signal / idler ─────────────────────────────────────────────
    bool signal_noise = fld["signal"].value("noise", true);
    if (signal_noise) {
        A.noise_generator(A.As);
        A.noise_generator(A.Ai);
    } else {
        // DFG / SFG: injected signal — initialise with a CW Gaussian
        // (power from JSON; waist same as pump for now)
        real_t sig_pwr = fld["signal"].value("power_W", 0.0f);
        // TODO: add dedicated signal-field kernel if needed
        A.noise_generator(A.As);  // placeholder until signal init kernel added
        A.noise_generator(A.Ai);
    }
    if (Cr.degenerate) A.Ai = A.As;  // SHG / degenerate OPG

    // ── Thermal (optional) ────────────────────────────────────────────────────
    bool   thermal_on  = cfg.contains("thermal") && cfg["thermal"].value("enabled", false);
    real_t T_oven      = thermal_on ? cfg["thermal"].value("T_oven_C",     Cr.T0) : Cr.T0;
    real_t T_ambient   = thermal_on ? cfg["thermal"].value("T_ambient_C",  25.0f) : 25.0f;
    real_t h_conv      = thermal_on ? cfg["thermal"].value("h_conv_W_m2K", 10.0f) : 10.0f;
    int    th_max_iter = thermal_on ? cfg["thermal"].value("max_iter",      2000)  : 2000;
    real_t th_tol      = thermal_on ? cfg["thermal"].value("tol",           1e-4f) : 1e-4f;
    int    outer_iter  = thermal_on ? cfg["thermal"].value("max_outer_iter",  10)  : 1;
    real_t outer_tol   = thermal_on ? cfg["thermal"].value("outer_tol",    0.1f)   : 0.1f;

    // h_conv: W/(m²·K) → W/(μm²·K)  (×1e-12)
    // kappa:  W/(m·K)  → W/(μm·K)   (×1e-6)
    h_conv   *= 1e-12f;
    real_t kappa_um = Cr.kappa_th * 1e-6f;

    Tfield        Tf;
    PhaseMatching PM;

    // Initialise PhaseMatching with constant dk, then compute cumulative integral.
    // In the thermal loop, DK3D is rebuilt from T(x,y,z) each outer iteration.
    PM.set_constant(Cr.dk);
    PM.integrate(Cr.dz);
    PM.compute_cumulative(Cr.dz);   // DKcum — used by dAdz kernel

    if (thermal_on) {
        std::cout << "  Thermal model: enabled  T_oven=" << T_oven
                  << " °C  T_amb=" << T_ambient << " °C\n";
        Tf.set_temperature(T_oven);
        Tf.set_bottom_oven(T_oven);
        Tf.zero_Q();
        PM.thermal = true;
    }

    // ── Solver ────────────────────────────────────────────────────────────────
    // Pass &Tf only in thermal mode — Solver::run() updates Q[x,y,z] slice-by-slice.
    Solver S(&Cr, &A, cfg, &PM, thermal_on ? &Tf : nullptr);

    if (!thermal_on) {
        S.run();
    } else {
        // ── Autoconsistent thermal loop ────────────────────────────────────────
        // Convergence criterion: ΔT_rms between consecutive outer iterations < outer_tol [K]
        //
        //  Pass 0 (seed):
        //    Propagate with T = T_oven (uniform, Q = 0) → initial Q[x,y,z]
        //
        //  Outer iterations:
        //    1. Snapshot T for convergence comparison
        //    2. Solve ∇²T = −Q/κ → T(x,y,z) with heating from current Q
        //    3. Compute ΔT_rms between new T and previous T
        //    4. Rebuild Δk(x,y,z) from T(x,y,z) via QPM grating thermal expansion
        //    5. Re-propagate optical fields → new Q[x,y,z]
        //    6. Convergence: ΔT_rms < outer_tol → stop

        const real_t PI_ = 3.14159265f;
        real_t dk_bare   = Cr.dk + 2.0f * PI_ / Cr.Lambda0;

        // Seed pass: propagate with T = T_oven (uniform), Q = 0
        std::cout << "  [Thermal] Seed pass  (T = " << T_oven << " °C uniform, Q = 0)...\n";
        A.noise_generator(A.As);
        A.noise_generator(A.Ai);
        if (Cr.degenerate) A.Ai = A.As;
        Tf.zero_Q();
        S.run();   // Q[x,y,z] populated from seed optical intensities

        for (int out = 0; out < outer_iter; ++out) {
            std::cout << "\n  [Thermal] Outer iteration " << out + 1
                      << " / " << outer_iter << "\n";

            // Step 1: Snapshot T before solving (used for ΔT_rms)
            Tf.snapshot_outer();

            // Step 2: Solve ∇²T = −Q/κ with Q from previous optical pass
            std::cout << "    Solving heat equation (max_iter=" << th_max_iter
                      << ", tol=" << th_tol << ")...\n";
            Tf.solve(Cr.dx, Cr.dy, Cr.dz, kappa_um, h_conv, T_ambient, T_oven,
                     th_max_iter, th_tol);

            // Step 3: ΔT_rms between new T and previous-iteration T
            real_t dT_rms = Tf.outer_T_rms();

            // Step 4: Rebuild Δk(x,y,z) from T(x,y,z)
            PM.update_from_temperature(Tf.Tf_ptr(), dk_bare, Cr.Lambda0, Cr.alpha_th, Cr.T0);
            PM.integrate(Cr.dz);
            PM.compute_cumulative(Cr.dz);

            // Monitor Δk_mean (informational)
            real_t dk_sum  = thrust::reduce(PM.DKint.begin(), PM.DKint.end(), 0.0f);
            real_t dk_mean = dk_sum / ((real_t)(NX * NY) * Cr.Lcr);

            // Estimate temperature range
            real_t T_max = *thrust::max_element(Tf.Tf.begin(), Tf.Tf.end());
            real_t T_min = *thrust::min_element(Tf.Tf.begin(), Tf.Tf.end());

            std::cout << "    T_min=" << T_min << " °C  T_max=" << T_max
                      << " °C  ΔT_max=" << (T_max - T_oven)
                      << " K\n";
            std::cout << "    ΔT_rms (vs prev iter) = " << dT_rms
                      << " K  (tol = " << outer_tol << " K)\n";
            std::cout << "    Δk_mean = " << dk_mean << " μm⁻¹\n";

            // Step 5: Re-propagate optical fields with new Δk(x,y,z)
            std::cout << "    Re-propagating optical fields...\n";
            A.noise_generator(A.As);
            A.noise_generator(A.Ai);
            if (Cr.degenerate) A.Ai = A.As;
            Tf.zero_Q();
            S.run();   // Q re-accumulated with thermally corrected DK

            // Step 6: Convergence on ΔT_rms
            if (out > 0 && dT_rms < outer_tol) {
                std::cout << "\n  [Thermal] Converged at iteration " << out + 1
                          << "  (ΔT_rms = " << dT_rms << " K < " << outer_tol << " K)\n";
                break;
            }
        }
        std::cout << "\n";
    }

    // ── Save outputs ──────────────────────────────────────────────────────────
    if (out_dir != ".") {
        // Create output directory (suppress unused-result warning)
        { volatile int _r = std::system(("mkdir -p " + out_dir).c_str()); (void)_r; }
    }

    if (thermal_on) {
        rVech_t Th_host(Tf.Tf.begin(), Tf.Tf.end());
        save_thermal_h5(Th_host, out_dir + "/thermal_output.h5");
        std::cout << "  Thermal field saved → thermal_output.h5\n";
    }

    save_input_pump_slices_XY(&A, cfg, out_dir);

    {
        cVech_t Aph(TENSOR_SIZE), Ash(TENSOR_SIZE), Aih(TENSOR_SIZE);
        thrust::copy(A.Ap.begin(), A.Ap.end(), Aph.begin());
        thrust::copy(A.As.begin(), A.As.end(), Ash.begin());
        thrust::copy(A.Ai.begin(), A.Ai.end(), Aih.begin());

        const auto& sm = cfg["save_mode"];
        if (sm.value("save_pump",   true))
            save_matrix_complex_h5_time(Aph, out_dir + "/pump_output.h5");
        if (sm.value("save_signal", true))
            save_matrix_complex_h5_time(Ash, out_dir + "/signal_output.h5");
        if (sm.value("save_idler",  true))
            save_matrix_complex_h5_time(Aih, out_dir + "/idler_output.h5");
    }
    if (cfg["save_mode"].value("save_time_and_freq_vectors", true)) {
        rVech_t th = A.t, fh = A.F;
        save_vector_real_h5(th, out_dir + "/time.h5");
        save_vector_real_h5(fh, out_dir + "/frequency.h5");
    }

    // Save intensity volumes |A(x, y, z)|²  →  (NZ, NY, NX)
    if (cfg["save_mode"].value("save_propagation", false)) {
        const auto& sm = cfg["save_mode"];
        if (sm.value("save_pump",   true))
            save_volume_h5(S.vol_p, out_dir + "/pump_volume.h5");
        if (sm.value("save_signal", true))
            save_volume_h5(S.vol_s, out_dir + "/signal_volume.h5");
        if (sm.value("save_idler",  true))
            save_volume_h5(S.vol_i, out_dir + "/idler_volume.h5");
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "\n  Output written to: " << out_dir << "/\n";
    std::cout << "  Execution time:   " << std::fixed << std::setprecision(2)
              << elapsed << " s\n";
    print_line();
    return 0;
}
