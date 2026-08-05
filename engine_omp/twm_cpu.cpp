/*
 * twm_cpu.cpp — BRAHMS CPU backend
 * =================================
 * OpenMP-parallelised (3+1)D three-wave mixing solver.
 * Identical JSON config as the GPU engine (./twm).
 *
 * Usage:
 *   twm_cpu config.json [output_dir]
 *
 * Compilation:
 *   make -C engine_cpu NX=64 NY=64 NZ=100 NT=256
 */

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <chrono>
#include <iomanip>

// System headers needed before project headers
#include <nlohmann/json.hpp>
#include "H5Cpp.h"
#include <omp.h>

#include "headers/DataTypes_cpu.hpp"
#include "headers/Common_cpu.hpp"
#include "headers/Crystal_cpu.hpp"
#include "headers/EFields_cpu.hpp"
#include "headers/Tfield_cpu.hpp"
#include "headers/Solver_cpu.hpp"
#include "headers/SaveOutputs_cpu.hpp"


json load_config(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[twm_cpu] Cannot open config: " << path << "\n";
        std::exit(1);
    }
    json cfg; f >> cfg;
    return cfg;
}


int main(int argc, char* argv[]) {
    auto t_start = std::chrono::high_resolution_clock::now();
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    welcome_cpu();

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " config.json [output_dir]\n";
        return 1;
    }

    std::string config_path = argv[1];
    std::string out_dir     = (argc >= 3) ? argv[2] : ".";

    json cfg = load_config(config_path);
    print_engine_info();
    print_grid();

    // ── Crystal ───────────────────────────────────────────────────────────────
    Crystal Cr(cfg);
    Cr.print();

    // ── Fields ────────────────────────────────────────────────────────────────
    const auto& fld       = cfg["fields"];
    real_t pump_power     = fld["pump"]["power_W"].get<real_t>();
    real_t pump_waist     = fld["pump"]["waist_um"].get<real_t>();
    real_t pump_fwhm      = fld["pump"].value("fwhm_ps", 1.0f);
    real_t pump_fp        = fld["pump"].value("focal_point_um", 0.0f);
    std::string pump_mode = fld["pump"].value("mode", "waveplane-cw");
    real_t t_window       = fld.value("t_window_ps", 10.0f);

    EFields A(pump_power, pump_waist, &Cr);
    A.set_time_freq_vectors(t_window);
    A.set_pump_field(pump_power, pump_fwhm, pump_waist, pump_fp, pump_mode);

    // Signal / idler initialisation — each independently seeded from vacuum
    // noise or an injected field (same four modes as the pump). Re-seeded
    // before every optical pass (including each thermal outer iteration
    // below), so this is a lambda rather than a one-off block.
    auto seed_signal_idler = [&]() {
        if (fld["signal"].value("noise", true)) {
            A.noise_generator(A.As);
        } else {
            real_t sig_pwr  = fld["signal"].value("power_W", 0.0f);
            real_t sig_wst  = fld["signal"].value("waist_um", pump_waist);
            real_t sig_fwhm = fld["signal"].value("fwhm_ps", pump_fwhm);
            real_t sig_fp   = fld["signal"].value("focal_point_um", pump_fp);
            std::string sig_mode = fld["signal"].value("mode", "waveplane-cw");
            A.set_signal_field(sig_pwr, sig_fwhm, sig_wst, sig_fp, sig_mode);
        }
        if (fld["idler"].value("noise", true)) {
            A.noise_generator(A.Ai);
        } else {
            real_t idl_pwr  = fld["idler"].value("power_W", 0.0f);
            real_t idl_wst  = fld["idler"].value("waist_um", pump_waist);
            real_t idl_fwhm = fld["idler"].value("fwhm_ps", pump_fwhm);
            real_t idl_fp   = fld["idler"].value("focal_point_um", pump_fp);
            std::string idl_mode = fld["idler"].value("mode", "waveplane-cw");
            A.set_idler_field(idl_pwr, idl_fwhm, idl_wst, idl_fp, idl_mode);
        }
        if (Cr.degenerate) A.Ai = A.As;
    };
    seed_signal_idler();

    // ── Thermal (optional) ────────────────────────────────────────────────────
    bool   thermal_on  = cfg.contains("thermal") && cfg["thermal"].value("enabled", false);
    real_t T_oven      = thermal_on ? cfg["thermal"].value("T_oven_C",      Cr.T0) : Cr.T0;
    real_t T_ambient   = thermal_on ? cfg["thermal"].value("T_ambient_C",   25.0f) : 25.0f;
    real_t h_conv      = thermal_on ? cfg["thermal"].value("h_conv_W_m2K",  10.0f) : 10.0f;
    int    th_max_iter = thermal_on ? cfg["thermal"].value("max_iter",       2000)  : 2000;
    real_t th_tol      = thermal_on ? cfg["thermal"].value("tol",            5e-4f) : 5e-4f;
    int    outer_iter  = thermal_on ? cfg["thermal"].value("max_outer_iter",   10)  : 1;
    real_t outer_tol   = thermal_on ? cfg["thermal"].value("outer_tol",     0.1f)   : 0.1f;

    // h_conv: W/(m²·K) → W/(μm²·K)  (×1e-12)
    // kappa:  W/(m·K)  → W/(μm·K)   (×1e-6)
    h_conv   *= 1e-12f;
    real_t kappa_um = Cr.kappa_th * 1e-6f;

    Tfield_cpu Tf;

    if (thermal_on) {
        std::cout << "  Thermal model: enabled  T_oven=" << T_oven
                  << " °C  T_amb=" << T_ambient << " °C\n";
        Tf.set_temperature(T_oven);
        Tf.set_bottom_oven(T_oven);
        Tf.zero_Q();
    }

    // ── Multipass ─────────────────────────────────────────────────────────────
    bool multipass = cfg.contains("multipass") && cfg["multipass"].value("enabled", false);
    int  npasses   = multipass ? cfg["multipass"].value("npasses", 1) : 1;

    // ── Solver ────────────────────────────────────────────────────────────────
    Solver S(&Cr, &A, cfg, thermal_on ? &Tf : nullptr);

    if (!thermal_on) {
        if (npasses > 1) S.run_multipass(npasses);
        else             S.run_single_pass();
    } else {
        // ── Autoconsistent thermal loop ────────────────────────────────────────
        const real_t PI_ = 3.14159265f;
        real_t dk_bare   = Cr.dk + 2.0f * PI_ / Cr.Lambda0;

        // Seed pass: propagate with T = T_oven (uniform, Q = 0)
        std::cout << "  [Thermal] Seed pass  (T = " << T_oven << " °C uniform, Q = 0)...\n";
        seed_signal_idler();
        Tf.zero_Q();
        S.run_single_pass();

        for (int out = 0; out < outer_iter; ++out) {
            std::cout << "\n  [Thermal] Outer iteration " << out + 1
                      << " / " << outer_iter << "\n";

            // Step 1: Snapshot T before solving
            Tf.snapshot_outer();

            // Step 2: Solve ∇²T = −Q/κ
            std::cout << "    Solving heat equation (max_iter=" << th_max_iter
                      << ", tol=" << th_tol << ")...\n";
            Tf.solve(Cr.dx, Cr.dy, Cr.dz, kappa_um, h_conv, T_ambient, T_oven,
                     th_max_iter, th_tol);

            // Step 3: ΔT_rms between new T and previous outer T
            real_t dT_rms = Tf.outer_T_rms();

            // Step 4: Rebuild Δk(z) from T(x,y,z) — xy-mean as in GPU
            S.update_dk_from_T(Tf.Tf);
            // (update_dk_from_T also calls build_cumulative internally)

            real_t T_max = *std::max_element(Tf.Tf.begin(), Tf.Tf.end());
            real_t T_min = *std::min_element(Tf.Tf.begin(), Tf.Tf.end());
            std::cout << "    T_min=" << T_min << " °C  T_max=" << T_max
                      << " °C  ΔT_max=" << (T_max - T_oven) << " K\n";
            std::cout << "    ΔT_rms (vs prev iter) = " << dT_rms
                      << " K  (tol = " << outer_tol << " K)\n";

            // Step 5: Re-propagate optical fields with new Δk(z)
            std::cout << "    Re-propagating optical fields...\n";
            seed_signal_idler();
            Tf.zero_Q();
            S.run_single_pass();

            // Step 6: Convergence
            if (out > 0 && dT_rms < outer_tol) {
                std::cout << "\n  [Thermal] Converged at iteration " << out + 1
                          << "  (ΔT_rms = " << dT_rms << " K < " << outer_tol << " K)\n";
                break;
            }
        }
        std::cout << "\n";
    }

    // ── Save outputs ──────────────────────────────────────────────────────────
    if (out_dir != ".")
        std::filesystem::create_directories(out_dir);

    if (thermal_on) {
        // Save T[NZ,NY,NX] in the same layout as the GPU engine (thermal_output.h5)
        H5::H5File file(out_dir + "/thermal_output.h5", H5F_ACC_TRUNC);
        hsize_t dims[3] = { NZ, NY, NX };
        H5::DataSpace ds(3, dims);
        H5::DataSet   dset = file.createDataSet(
            "T", H5::PredType::NATIVE_FLOAT, ds);
        // GPU stores T in IDX3(ix,iy,iz) = iz*NX*NY + iy*NX + ix order
        // which maps to [NZ][NY][NX] HDF5 layout — same for CPU
        dset.write(Tf.Tf.data(), H5::PredType::NATIVE_FLOAT);
        std::cout << "  Thermal field saved → thermal_output.h5\n";
    }

    const auto& sm = cfg["save_mode"];
    if (sm.value("save_in_pump", true))
        save_complex_h5(A.Api, out_dir + "/pump_input_XY.h5");
    if (sm.value("save_pump",   true))
        save_complex_h5(A.Ap, out_dir + "/pump_output.h5");
    if (sm.value("save_signal", true))
        save_complex_h5(A.As, out_dir + "/signal_output.h5");
    if (sm.value("save_idler",  true))
        save_complex_h5(A.Ai, out_dir + "/idler_output.h5");
    if (sm.value("save_time_and_freq_vectors", true)) {
        save_vector_h5(A.t, out_dir + "/time.h5");
        save_vector_h5(A.F, out_dir + "/frequency.h5");
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    if      (elapsed < 60.0)   std::cout << "Elapsed: " << elapsed        << " s\n";
    else if (elapsed < 3600.0) std::cout << "Elapsed: " << elapsed/60.0  << " min\n";
    else                       std::cout << "Elapsed: " << elapsed/3600.0 << " h\n";
    std::cout << "  Output written to: " << out_dir << "/\n";
    std::cout << "  Execution time:   " << std::fixed << std::setprecision(2)
              << elapsed << " s\n";
    print_line();
    std::cout << "[OK] Simulation complete.\n";
    return 0;
}
