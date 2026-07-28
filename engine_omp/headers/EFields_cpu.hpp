#pragma once
/*
 * EFields_cpu.hpp
 * ===============
 * Pump/signal/idler electric field envelopes on CPU.
 * FFTs use FFTW3 single-precision (fftwf_*).
 * Same conventions as EFields.cuh (same JSON config).
 *
 * Field storage: [NT][NY][NX], index = IDX(x,y,t) = t*NX*NY + y*NX + x
 * Dispersion buffer: [NX*NY][NT], index = J2(x,y,t)  (stride-1 over t for FFTW)
 */

#include "Common_cpu.hpp"
#include "Crystal_cpu.hpp"
#include <fftw3.h>
#include <random>
#include <cmath>
#include <algorithm>
#include <omp.h>

struct EFields {
    cVec_t Ap, As, Ai, Api;   // pump (initial copy), signal, idler
    rVec_t t, F;               // time [ps] and frequency [THz] vectors
    rVec_t _w;                 // angular frequency [rad/ps] in FFT order (fftshift)
    real_t dk;

    // FFTW3 plans (reused across all z-steps)
    fftwf_plan _plan_disp_fwd,  _plan_disp_inv;   // 1D over t  (dispersion)
    fftwf_plan _plan_diff_fwd,  _plan_diff_inv;   // 2D over xy (diffraction)

    cVec_t _buf_disp;          // [NX*NY*NT] in J2 layout for dispersion FFT
    real_t _dt;                // time step [ps]
    real_t _kp, _ks, _ki;     // wavenumbers at pump/signal/idler [μm⁻¹]
    const Crystal* _Cr;

    EFields(real_t /*power*/, real_t /*waist*/, Crystal* Cr)
        : Ap(SIZE, {0,0}), As(SIZE, {0,0}), Ai(SIZE, {0,0}), Api(SIZE, {0,0}),
          t(NT, 0), F(NT, 0), _w(NT, 0),
          _buf_disp(SIZE, {0,0}),
          _dt(0), _kp(0), _ks(0), _ki(0), _Cr(Cr)
    {
        dk  = Cr->dk;
        _kp = 2.0f * PI * Cr->np / Cr->lp;
        _ks = 2.0f * PI * Cr->ns / Cr->ls;
        _ki = 2.0f * PI * Cr->ni / Cr->li;

        // FFTW plans ─────────────────────────────────────────────────────────
        // Dispersion: batched 1D FFT of length NT, NX*NY batches, stride 1
        // Buffer layout [NX*NY][NT] → each row is contiguous in t
        int nt = NT, nxy = NX * NY;
        _plan_disp_fwd = fftwf_plan_many_dft(
            1, &nt, nxy,
            reinterpret_cast<fftwf_complex*>(_buf_disp.data()), nullptr, 1, NT,
            reinterpret_cast<fftwf_complex*>(_buf_disp.data()), nullptr, 1, NT,
            FFTW_FORWARD, FFTW_MEASURE);
        _plan_disp_inv = fftwf_plan_many_dft(
            1, &nt, nxy,
            reinterpret_cast<fftwf_complex*>(_buf_disp.data()), nullptr, 1, NT,
            reinterpret_cast<fftwf_complex*>(_buf_disp.data()), nullptr, 1, NT,
            FFTW_BACKWARD, FFTW_MEASURE);

        // Diffraction: batched 2D FFT of size NY×NX, NT batches
        // Field layout [NT][NY][NX]: each t-slice is contiguous
        int dims2[2] = {NY, NX};
        _plan_diff_fwd = fftwf_plan_many_dft(
            2, dims2, NT,
            reinterpret_cast<fftwf_complex*>(Ap.data()), nullptr, 1, NX*NY,
            reinterpret_cast<fftwf_complex*>(Ap.data()), nullptr, 1, NX*NY,
            FFTW_FORWARD, FFTW_MEASURE);
        _plan_diff_inv = fftwf_plan_many_dft(
            2, dims2, NT,
            reinterpret_cast<fftwf_complex*>(Ap.data()), nullptr, 1, NX*NY,
            reinterpret_cast<fftwf_complex*>(Ap.data()), nullptr, 1, NX*NY,
            FFTW_BACKWARD, FFTW_MEASURE);
    }

    ~EFields() {
        fftwf_destroy_plan(_plan_disp_fwd);
        fftwf_destroy_plan(_plan_disp_inv);
        fftwf_destroy_plan(_plan_diff_fwd);
        fftwf_destroy_plan(_plan_diff_inv);
    }

    // ── Time / frequency vectors ──────────────────────────────────────────────
    // Same convention as GPU/pycuTWM: linspace(-T/2, T/2, NT) → dt = T/(NT-1)
    void set_time_freq_vectors(real_t t_window_ps) {
        _dt = t_window_ps / static_cast<real_t>(NT - 1);
        real_t t0  = -t_window_ps / 2.0f;
        real_t F0  = -0.5f * static_cast<real_t>(NT) / t_window_ps;
        real_t dF  = static_cast<real_t>(NT) / (t_window_ps * static_cast<real_t>(NT - 1));
        for (uint32_t i = 0; i < NT; ++i) {
            t[i] = t0 + i * _dt;
            F[i] = F0 + i * dF;
        }
        // w = 2π × fftshift(F)  — same construction as GPU
        for (uint32_t i = 0; i < NT; ++i) {
            uint32_t j = (i + NT/2) % NT;
            _w[i] = 2.0f * PI * F[j];
        }
    }

    // ── Noise initialisation ──────────────────────────────────────────────────
    void noise_generator(cVec_t& A) {
        std::mt19937 rng(std::random_device{}());
        std::normal_distribution<float> dist(0.0f, 1e-15f);
        for (auto& z : A) z = {dist(rng), dist(rng)};
    }

    // ── Pump field initialisation ─────────────────────────────────────────────
    void set_pump_field(real_t power_W, real_t fwhm_ps, real_t waist_um,
                        real_t fp_um, const std::string& mode)
    {
        // Voltage amplitude [V/μm] — same convention as GPU engine
        real_t A0   = std::sqrt(4.0f * power_W / (EPS0 * C * PI * _Cr->np * waist_um * waist_um));
        real_t sig  = fwhm_ps / (2.0f * std::sqrt(2.0f * std::log(2.0f)));
        bool pulsed  = (mode.find("pulsed") != std::string::npos);
        bool focused = (mode.find("focused") != std::string::npos);
        real_t w2   = waist_um * waist_um;

        // Focused: MX = 1 - i·η, η = fp/z_R
        // The SSFM diffraction operator exp(-ikx²dz/2k) decreases q by dz each step.
        // For the waist to form at z=fp, we need q(z=0) = +fp + iz_R, which gives MX = 1 - i·η.
        complex_t MX{1.0f, 0.0f};
        if (focused) {
            real_t zR = _Cr->np * PI * w2 / _Cr->lp;
            MX = complex_t{1.0f, -fp_um / zR};
        }

        real_t cx = (_Cr->LX / 2.0f);
        real_t cy = (_Cr->LY / 2.0f);

        #pragma omp parallel for collapse(3)
        for (uint32_t it = 0; it < NT; ++it)
        for (uint32_t iy = 0; iy < NY; ++iy)
        for (uint32_t ix = 0; ix < NX; ++ix) {
            real_t x  = ix * _Cr->dx - cx;
            real_t y  = iy * _Cr->dy - cy;
            real_t r2 = x*x + y*y;

            complex_t val;
            if (focused) {
                // Complex Gaussian: converging wavefront, focus at fp_um
                val = (complex_t{A0, 0.0f} / MX) * std::exp(complex_t{-r2/w2, 0.0f} / MX);
            } else {
                // Plane wave: real Gaussian at crystal face (matches GPU setWavePlaneCW)
                val = {A0 * std::exp(-r2 / w2), 0.0f};
            }
            if (pulsed) val *= std::exp(-t[it]*t[it] / (2.0f*sig*sig));
            Ap[IDX(ix,iy,it)] = val;
        }
        Api = Ap;   // save initial pump

        print_line();
        std::cout << "  Pump field :\n"
                  << "    mode  = " << mode << "\n"
                  << "    Power = " << power_W   << " W\n"
                  << "    waist = " << waist_um  << " μm\n";
        if (focused) {
            real_t zR = _Cr->np * PI * w2 / _Cr->lp;
            real_t eta = fp_um / zR;
            real_t xi  = (_Cr->Lcr * _Cr->lp) / (2.0f * PI * _Cr->np * w2);
            std::cout << "    focal point = " << fp_um << " μm\n"
                      << "    z_R  = " << zR  << " μm\n"
                      << "    η    = " << eta << "  (fp/z_R)\n"
                      << "    ξ    = " << xi  << "  (L·λ_p / 2π·n_p·w²)\n";
        }
        if (pulsed)
            std::cout << "    FWHM  = " << fwhm_ps << " ps\n";
        print_line();
    }

    // ── Dispersion step ───────────────────────────────────────────────────────
    // Applies linear dispersion in frequency domain for one full z-step `dz`.
    // Reference frame: signal group velocity vs.
    void apply_dispersion(cVec_t& Afield, real_t b2, real_t b3, real_t gvm, real_t dz)
    {
        if (NT <= 1) return;   // CW: no dispersion

        // Reorder to J2 layout [NX*NY][NT]
        #pragma omp parallel for collapse(2)
        for (uint32_t iy = 0; iy < NY; ++iy)
        for (uint32_t ix = 0; ix < NX; ++ix)
            for (uint32_t it = 0; it < NT; ++it)
                _buf_disp[J2(ix,iy,it)] = Afield[IDX(ix,iy,it)];

        // Backward FFT first (matches GPU: CUFFT_INVERSE → multiply → CUFFT_FORWARD)
        fftwf_execute_dft(_plan_disp_inv,
            reinterpret_cast<fftwf_complex*>(_buf_disp.data()),
            reinterpret_cast<fftwf_complex*>(_buf_disp.data()));

        // Multiply by dispersion propagator — same w as GPU (fftshift of linspace)
        real_t norm = 1.0f / static_cast<real_t>(NT);
        #pragma omp parallel for collapse(2)
        for (uint32_t iy = 0; iy < NY; ++iy)
        for (uint32_t ix = 0; ix < NX; ++ix) {
            for (uint32_t it = 0; it < NT; ++it) {
                real_t ww  = _w[it];
                real_t phi = dz * ww * (gvm + 0.5f*b2*ww + (1.0f/6.0f)*b3*ww*ww);
                _buf_disp[J2(ix,iy,it)] *= CpxExp(phi) * norm;
            }
        }

        // Forward FFT
        fftwf_execute_dft(_plan_disp_fwd,
            reinterpret_cast<fftwf_complex*>(_buf_disp.data()),
            reinterpret_cast<fftwf_complex*>(_buf_disp.data()));

        // Reorder back to IDX layout
        #pragma omp parallel for collapse(2)
        for (uint32_t iy = 0; iy < NY; ++iy)
        for (uint32_t ix = 0; ix < NX; ++ix)
            for (uint32_t it = 0; it < NT; ++it)
                Afield[IDX(ix,iy,it)] = _buf_disp[J2(ix,iy,it)];
    }

    // ── Diffraction step ──────────────────────────────────────────────────────
    // 2D paraxial propagation: exp(-i kperp²/(2k) dz) in kx-ky space.
    void apply_diffraction(cVec_t& Afield, real_t k0, real_t dz)
    {
        if (NX <= 1 && NY <= 1) return;  // 1D: no diffraction

        // In-place 2D FFT (plans were created for Ap.data() but pointer is same layout)
        fftwf_execute_dft(_plan_diff_fwd,
            reinterpret_cast<fftwf_complex*>(Afield.data()),
            reinterpret_cast<fftwf_complex*>(Afield.data()));

        real_t norm = 1.0f / static_cast<real_t>(NX * NY);
        #pragma omp parallel for collapse(3)
        for (uint32_t it = 0; it < NT; ++it)
        for (uint32_t iy = 0; iy < NY; ++iy)
        for (uint32_t ix = 0; ix < NX; ++ix) {
            int mx = (ix < NX/2) ? (int)ix : (int)ix - NX;
            int my = (iy < NY/2) ? (int)iy : (int)iy - NY;
            real_t kx = 2.0f * PI * mx / (_Cr->LX);
            real_t ky = 2.0f * PI * my / (_Cr->LY);
            real_t phi = -(kx*kx + ky*ky) / (2.0f * k0) * dz;
            Afield[IDX(ix,iy,it)] *= CpxExp(phi) * norm;
        }

        fftwf_execute_dft(_plan_diff_inv,
            reinterpret_cast<fftwf_complex*>(Afield.data()),
            reinterpret_cast<fftwf_complex*>(Afield.data()));
    }
};
