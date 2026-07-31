#pragma once
/*
 * Solver_cpu.hpp
 * ==============
 * SSFM + RK4 solver for CPU.
 * Same split-step scheme as Solver.cuh:
 *   RK4(dz/2) → dispersion → diffraction → RK4(dz/2)
 *
 * dAdz convention (identical to GPU engine):
 *   SHG:  Ap = fundamental(ω), As = Ai = SH(2ω)
 *   Other: Ap = pump, As = signal, Ai = idler
 */

#include "EFields_cpu.hpp"
#include "Tfield_cpu.hpp"
#include <omp.h>
#include <iostream>

// ── dAdz nonlinear coupling ───────────────────────────────────────────────────
// dk_phase = accumulated phase at this (iz, dz_frac) position:
//   dk_phase = dkcum_z[iz] + dz_frac · dk_z[iz]
// For uniform dk: dk_phase = dk · iz · dz + dz_frac · dk = dk · z  (same as before)
static void dAdz(
    cVec_t& dAp, cVec_t& dAs, cVec_t& dAi,
    const cVec_t& Ap, const cVec_t& As, const cVec_t& Ai,
    real_t kappa_p, real_t kappa_s, real_t kappa_i,
    real_t alpha_p, real_t alpha_s, real_t alpha_i,
    real_t np, real_t ns, real_t ni,
    real_t beta_p, real_t beta_s, real_t beta_i,
    real_t dk_phase, int process, int dir, bool undepleted)
{
    const real_t Ip_fac = 0.5f * EPS0 * C * np;
    const real_t Is_fac = 0.5f * EPS0 * C * ns;
    const real_t Ii_fac = 0.5f * EPS0 * C * ni;

    #pragma omp parallel for collapse(3) schedule(static)
    for (uint32_t it = 0; it < NT; ++it)
    for (uint32_t iy = 0; iy < NY; ++iy)
    for (uint32_t ix = 0; ix < NX; ++ix) {
        uint32_t I   = IDX(ix, iy, it);
        complex_t Ap_ = Ap[I], As_ = As[I], Ai_ = Ai[I];
        real_t    ph  = dir * dk_phase;

        if (process == PROC_SHG) {
            dAp[I] =      Im * kappa_p * As_ * std::conj(Ap_) * CpxExp(+ph)
                        - 0.5f * (alpha_p + beta_p * Ip_fac * std::norm(Ap_)) * Ap_;
            dAs[I] = 0.5f*Im * kappa_s * Ap_ * Ap_             * CpxExp(-ph)
                        - 0.5f * (alpha_s + beta_s * Is_fac * std::norm(As_)) * As_;
            dAi[I] = 0.5f*Im * kappa_i * Ap_ * Ap_             * CpxExp(-ph)
                        - 0.5f * (alpha_i + beta_i * Ii_fac * std::norm(Ai_)) * Ai_;
        } else {
            dAp[I] = Im * kappa_p * As_ * Ai_              * CpxExp(-ph)
                       - 0.5f * (alpha_p + beta_p * Ip_fac * std::norm(Ap_)) * Ap_;
            dAs[I] = Im * kappa_s * Ap_ * std::conj(Ai_)   * CpxExp(+ph)
                       - 0.5f * (alpha_s + beta_s * Is_fac * std::norm(As_)) * As_;
            dAi[I] = Im * kappa_i * Ap_ * std::conj(As_)   * CpxExp(+ph)
                       - 0.5f * (alpha_i + beta_i * Ii_fac * std::norm(Ai_)) * Ai_;
        }
        if (undepleted) dAp[I] = complex_t{0.0f, 0.0f};
    }
}

// ── RK4 half-step (advances the field by dz/2) ───────────────────────────────
// iz, dkcum_z, dk_z — z-varying phase mismatch (xy-uniform approximation for CPU)
// RK4 sub-positions (dz_frac): k1→0, k2→dz/4, k3→dz/4, k4→dz/2
static void rk4_half(
    cVec_t& Ap, cVec_t& As, cVec_t& Ai,
    real_t kappa_p, real_t kappa_s, real_t kappa_i,
    real_t alpha_p, real_t alpha_s, real_t alpha_i,
    real_t np, real_t ns, real_t ni,
    real_t beta_p, real_t beta_s, real_t beta_i,
    const std::vector<real_t>& dkcum_z, const std::vector<real_t>& dk_z,
    uint32_t iz, int process, int dir, real_t dz, bool undepleted)
{
    real_t h      = dz / 2.0f;
    real_t cum    = dkcum_z[iz];
    real_t dk_loc = dk_z[iz];

    cVec_t k1p(SIZE), k1s(SIZE), k1i(SIZE);
    cVec_t k2p(SIZE), k2s(SIZE), k2i(SIZE);
    cVec_t k3p(SIZE), k3s(SIZE), k3i(SIZE);
    cVec_t k4p(SIZE), k4s(SIZE), k4i(SIZE);
    cVec_t tp(SIZE),  ts(SIZE),  ti(SIZE);

    // k1 at sub-position 0
    dAdz(k1p,k1s,k1i, Ap,As,Ai,
         kappa_p,kappa_s,kappa_i, alpha_p,alpha_s,alpha_i,
         np,ns,ni, beta_p,beta_s,beta_i,
         cum, process, dir, undepleted);

    // k2 at sub-position dz/4
    #pragma omp parallel for
    for (uint32_t I = 0; I < SIZE; ++I) {
        tp[I] = Ap[I] + (h/2.0f)*k1p[I];
        ts[I] = As[I] + (h/2.0f)*k1s[I];
        ti[I] = Ai[I] + (h/2.0f)*k1i[I];
    }
    dAdz(k2p,k2s,k2i, tp,ts,ti,
         kappa_p,kappa_s,kappa_i, alpha_p,alpha_s,alpha_i,
         np,ns,ni, beta_p,beta_s,beta_i,
         cum + (h/2.0f)*dk_loc, process, dir, undepleted);

    // k3 at sub-position dz/4
    #pragma omp parallel for
    for (uint32_t I = 0; I < SIZE; ++I) {
        tp[I] = Ap[I] + (h/2.0f)*k2p[I];
        ts[I] = As[I] + (h/2.0f)*k2s[I];
        ti[I] = Ai[I] + (h/2.0f)*k2i[I];
    }
    dAdz(k3p,k3s,k3i, tp,ts,ti,
         kappa_p,kappa_s,kappa_i, alpha_p,alpha_s,alpha_i,
         np,ns,ni, beta_p,beta_s,beta_i,
         cum + (h/2.0f)*dk_loc, process, dir, undepleted);

    // k4 at sub-position dz/2
    #pragma omp parallel for
    for (uint32_t I = 0; I < SIZE; ++I) {
        tp[I] = Ap[I] + h*k3p[I];
        ts[I] = As[I] + h*k3s[I];
        ti[I] = Ai[I] + h*k3i[I];
    }
    dAdz(k4p,k4s,k4i, tp,ts,ti,
         kappa_p,kappa_s,kappa_i, alpha_p,alpha_s,alpha_i,
         np,ns,ni, beta_p,beta_s,beta_i,
         cum + h*dk_loc, process, dir, undepleted);

    real_t c = h / 6.0f;
    #pragma omp parallel for
    for (uint32_t I = 0; I < SIZE; ++I) {
        Ap[I] += c * (k1p[I] + 2.0f*k2p[I] + 2.0f*k3p[I] + k4p[I]);
        As[I] += c * (k1s[I] + 2.0f*k2s[I] + 2.0f*k3s[I] + k4s[I]);
        Ai[I] += c * (k1i[I] + 2.0f*k2i[I] + 2.0f*k3i[I] + k4i[I]);
    }
}

// ── Main Solver ───────────────────────────────────────────────────────────────
struct Solver {
    Crystal*      _Cr;
    EFields*      _A;
    Tfield_cpu*   _Tf;    // null → non-thermal; non-null → accumulate Q
    real_t        _kappa_p, _kappa_s, _kappa_i;
    int           _dir;

    // z-varying phase mismatch (xy-mean): dk_z[iz] and prefix-sum dkcum_z[iz]
    // Initialised to uniform Cr.dk; updated by update_dk_from_T() for thermal mode.
    std::vector<real_t> dk_z;
    std::vector<real_t> dkcum_z;

    Solver(Crystal* Cr, EFields* A, const json& cfg, Tfield_cpu* tf = nullptr)
        : _Cr(Cr), _A(A), _Tf(tf), _dir(1),
          dk_z(NZ, Cr->dk), dkcum_z(NZ, 0.0f)
    {
        _kappa_p = PI * Cr->dQ * 1e-6f / (Cr->np * Cr->lp);
        _kappa_s = PI * Cr->dQ * 1e-6f / (Cr->ns * Cr->ls);
        _kappa_i = PI * Cr->dQ * 1e-6f / (Cr->ni * Cr->li);
        build_cumulative();
    }

    // Exclusive prefix-sum of dk_z * dz
    void build_cumulative() {
        real_t acc = 0.0f;
        for (uint32_t iz = 0; iz < NZ; ++iz) {
            dkcum_z[iz] = acc;
            acc += dk_z[iz] * _Cr->dz;
        }
    }

    // Update z-varying dk from a 3D temperature field T[NZ×NY×NX] (row-major IDX3).
    // Uses the xy-mean of Δk(T) at each z-slice — same QPM formula as GPU engine.
    void update_dk_from_T(const std::vector<real_t>& T3D) {
        const real_t PI_  = 3.14159265f;
        real_t dk_bare = _Cr->dk + 2.0f * PI_ / _Cr->Lambda0;
        for (uint32_t iz = 0; iz < NZ; ++iz) {
            real_t sum = 0.0f;
            for (uint32_t iy = 0; iy < NY; ++iy)
            for (uint32_t ix = 0; ix < NX; ++ix) {
                real_t T_loc = T3D[IDX3(ix, iy, iz, NX, NY)];
                real_t Lam_T = _Cr->Lambda0 * (1.0f + _Cr->alpha_th * (T_loc - _Cr->T0));
                sum += dk_bare - 2.0f * PI_ / Lam_T;
            }
            dk_z[iz] = sum / (real_t)(NX * NY);
        }
        build_cumulative();
    }

    void run_single_pass() {
        Crystal& Cr = *_Cr;
        EFields& A  = *_A;
        int proc = Cr.process;

        real_t GVM_p = 1.0f/Cr.vp - 1.0f/Cr.vs;
        real_t GVM_s = 0.0f;
        real_t GVM_i = 1.0f/Cr.vi - 1.0f/Cr.vs;

        std::cout << "  Solver ready.\n";

        for (int iz = 0; iz < NZ; ++iz) {
            if (iz % std::max(1, NZ/10) == 0)
                print_progress(iz, NZ);

            // 1. Half-step nonlinear (RK4)
            rk4_half(A.Ap, A.As, A.Ai,
                     _kappa_p, _kappa_s, _kappa_i,
                     Cr.alpha_crp, Cr.alpha_crs, Cr.alpha_cri,
                     Cr.np, Cr.ns, Cr.ni,
                     Cr.beta_crp, Cr.beta_crs, Cr.beta_cri,
                     dkcum_z, dk_z, (uint32_t)iz, proc, _dir, Cr.dz, Cr.undepleted);

            // 2. Dispersion (full step dz, reference frame vs)
            A.apply_dispersion(A.Ap, Cr.b2p, Cr.b3p, GVM_p, Cr.dz);
            A.apply_dispersion(A.As, Cr.b2s, Cr.b3s, GVM_s, Cr.dz);
            A.apply_dispersion(A.Ai, Cr.b2i, Cr.b3i, GVM_i, Cr.dz);

            // 3. Diffraction (full step dz)
            A.apply_diffraction(A.Ap, A._kp, A._rho_p, Cr.dz);
            A.apply_diffraction(A.As, A._ks, A._rho_s, Cr.dz);
            A.apply_diffraction(A.Ai, A._ki, A._rho_i, Cr.dz);

            // 4. Second half-step nonlinear
            rk4_half(A.Ap, A.As, A.Ai,
                     _kappa_p, _kappa_s, _kappa_i,
                     Cr.alpha_crp, Cr.alpha_crs, Cr.alpha_cri,
                     Cr.np, Cr.ns, Cr.ni,
                     Cr.beta_crp, Cr.beta_crs, Cr.beta_cri,
                     dkcum_z, dk_z, (uint32_t)iz, proc, _dir, Cr.dz, Cr.undepleted);

            // 5. Accumulate heat source Q at this z-slice (thermal mode only)
            if (_Tf)
                _Tf->update_Q_slice((uint32_t)iz, A.Ap, A.As, A.Ai,
                                    Cr.alpha_crp, Cr.alpha_crs, Cr.alpha_cri,
                                    Cr.np, Cr.ns, Cr.ni);
        }

        print_progress(NZ, NZ);
        printf("\n"); fflush(stdout);
    }

    void run_multipass(int npasses) {
        for (int p = 0; p < npasses; ++p) {
            std::cout << "  Pass " << p+1 << "/" << npasses << "\n";
            run_single_pass();
            _A->Ap = _A->Api;
        }
    }
};
