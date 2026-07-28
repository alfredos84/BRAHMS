#pragma once
/*
 * Tfield_cpu.hpp
 * ==============
 * CPU mirror of engine_gpu/headers/Tfield.cuh.
 * Steady-state 3-D heat equation solved with Gauss-Seidel (Jacobi sweep).
 *
 *   ∇²T = −Q / κ
 *
 * Boundary conditions:
 *   y = 0          : Dirichlet  T = T_oven  (bottom face, oven contact)
 *   all other faces: Robin (convective)  −κ ∂T/∂n = h·(T − T_∞)
 *
 * Heat source [W/μm³]:
 *   Q = α_p·(½ε₀ c n_p)|E_p|² + α_s·(½ε₀ c n_s)|E_s|² + α_i·(½ε₀ c n_i)|E_i|²
 *
 * Convergence criterion identical to GPU engine:
 *   ||Tf − Ti||₂ = sqrt(Σ|ΔT|²) < tol   (default tol = 5e-4)
 */

#include "Common_cpu.hpp"
#include "DataTypes_cpu.hpp"
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <numeric>

class Tfield_cpu {
public:
    std::vector<real_t> Ti;       // T at sweep n
    std::vector<real_t> Tf;       // T at sweep n+1
    std::vector<real_t> Q;        // heat source [W/μm³]
    std::vector<real_t> T_outer;  // snapshot for outer-loop convergence

    static constexpr uint32_t N3 = (uint32_t)NX * NY * NZ;

    Tfield_cpu()
        : Ti(N3, 0.0f), Tf(N3, 0.0f),
          Q(N3, 0.0f), T_outer(N3, 0.0f) {}

    void set_temperature(real_t T0) {
        std::fill(Ti.begin(), Ti.end(), T0);
        std::fill(Tf.begin(), Tf.end(), T0);
    }

    void set_bottom_oven(real_t T_oven) {
        for (uint32_t ix = 0; ix < NX; ++ix)
            for (uint32_t iz = 0; iz < NZ; ++iz)
                Ti[IDX3(ix, 0, iz, NX, NY)] =
                Tf[IDX3(ix, 0, iz, NX, NY)] = T_oven;
    }

    void zero_Q() { std::fill(Q.begin(), Q.end(), 0.0f); }

    // Accumulate Q at z-slice iz from optical field amplitudes (t-averaged intensity).
    // Ap/As/Ai are the full field arrays of size SIZE = NX*NY*NT.
    void update_Q_slice(uint32_t iz,
                        const std::vector<complex_t>& Ap,
                        const std::vector<complex_t>& As,
                        const std::vector<complex_t>& Ai,
                        real_t alpha_crp, real_t alpha_crs, real_t alpha_cri,
                        real_t np, real_t ns, real_t ni)
    {
        const real_t fp = 0.5f * EPS0 * C * np;
        const real_t fs = 0.5f * EPS0 * C * ns;
        const real_t fi = 0.5f * EPS0 * C * ni;

        #pragma omp parallel for collapse(2) schedule(static)
        for (uint32_t iy = 0; iy < NY; ++iy)
        for (uint32_t ix = 0; ix < NX; ++ix) {
            // Time-average: Σ_t |A(x,y,t)|² / NT
            real_t sumP = 0.0f, sumS = 0.0f, sumI = 0.0f;
            for (uint32_t it = 0; it < NT; ++it) {
                uint32_t I = IDX(ix, iy, it);
                sumP += std::norm(Ap[I]);
                sumS += std::norm(As[I]);
                sumI += std::norm(Ai[I]);
            }
            real_t Ip = fp * sumP / (real_t)NT;
            real_t Is = fs * sumS / (real_t)NT;
            real_t Ii = fi * sumI / (real_t)NT;
            Q[IDX3(ix, iy, iz, NX, NY)] = alpha_crp * Ip + alpha_crs * Is + alpha_cri * Ii;
        }
    }

    // One Jacobi sweep of ∇²T = −Q/κ.
    // Uses Ti as input (previous sweep), writes to Tf.
    // dx, dy, dz in [μm]; kappa in [W/(μm·K)]; h in [W/(μm²·K)].
    void sweep(real_t dx, real_t dy, real_t dz,
               real_t kappa, real_t h_conv, real_t T_inf, real_t T_oven)
    {
        const real_t dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
        const real_t L2  = 1.0f / (2.0f / dx2 + 2.0f / dy2 + 2.0f / dz2);
        const real_t Bix = h_conv * dx / kappa;
        const real_t Biy = h_conv * dy / kappa;
        const real_t Biz = h_conv * dz / kappa;

        #pragma omp parallel for collapse(2) schedule(static)
        for (uint32_t iy = 0; iy < NY; ++iy)
        for (uint32_t ix = 0; ix < NX; ++ix) {
            for (uint32_t iz = 0; iz < NZ; ++iz) {
                bool bx0 = (ix == 0),       bxN = (ix == NX - 1);
                bool by0 = (iy == 0),       byN = (iy == NY - 1);
                bool bz0 = (iz == 0),       bzN = (iz == NZ - 1);
                bool inner = !bx0 && !bxN && !by0 && !byN && !bz0 && !bzN;

                uint32_t c = IDX3(ix, iy, iz, NX, NY);

                // y=0: Dirichlet — oven face
                if (by0) { Tf[c] = T_oven; continue; }

                if (inner) {
                    Tf[c] = L2 * (
                        (Ti[IDX3(ix+1, iy,   iz,   NX, NY)] + Ti[IDX3(ix-1, iy,   iz,   NX, NY)]) / dx2 +
                        (Ti[IDX3(ix,   iy+1, iz,   NX, NY)] + Ti[IDX3(ix,   iy-1, iz,   NX, NY)]) / dy2 +
                        (Ti[IDX3(ix,   iy,   iz+1, NX, NY)] + Ti[IDX3(ix,   iy,   iz-1, NX, NY)]) / dz2 +
                        Q[c] / kappa
                    );
                } else {
                    real_t ax = dy2 / dx2, bx_ = dz2 / dx2;
                    real_t ay = dx2 / dy2, by_ = dz2 / dy2;
                    real_t az = dx2 / dz2, bz_ = dy2 / dz2;
                    real_t denom, rhs = 0.0f;

                    // Safe neighbour access (clamp to boundary on missing side)
                    auto xp = [&]{ return Ti[IDX3(ix < NX-1 ? ix+1 : ix, iy, iz, NX, NY)]; };
                    auto xm = [&]{ return Ti[IDX3(ix > 0    ? ix-1 : ix, iy, iz, NX, NY)]; };
                    auto yp = [&]{ return Ti[IDX3(ix, iy < NY-1 ? iy+1 : iy, iz, NX, NY)]; };
                    auto ym = [&]{ return Ti[IDX3(ix, iy > 0    ? iy-1 : iy, iz, NX, NY)]; };
                    auto zp = [&]{ return Ti[IDX3(ix, iy, iz < NZ-1 ? iz+1 : iz, NX, NY)]; };
                    auto zm = [&]{ return Ti[IDX3(ix, iy, iz > 0    ? iz-1 : iz, NX, NY)]; };

                    if (bx0 && !byN && !bz0 && !bzN) {
                        denom = 1.0f + 2.0f*ax + 2.0f*bx_ + Bix;
                        rhs   = xp() + ax*(yp()+ym()) + bx_*(zp()+zm()) + Bix*T_inf;
                    } else if (bxN && !byN && !bz0 && !bzN) {
                        denom = 1.0f + 2.0f*ax + 2.0f*bx_ + Bix;
                        rhs   = xm() + ax*(yp()+ym()) + bx_*(zp()+zm()) + Bix*T_inf;
                    } else if (byN && !bx0 && !bxN && !bz0 && !bzN) {
                        denom = 1.0f + 2.0f*ay + 2.0f*by_ + Biy;
                        rhs   = ym() + ay*(xp()+xm()) + by_*(zp()+zm()) + Biy*T_inf;
                    } else if (bz0 && !bx0 && !bxN && !byN) {
                        denom = 1.0f + 2.0f*az + 2.0f*bz_ + Biz;
                        rhs   = zp() + az*(xp()+xm()) + bz_*(yp()+ym()) + Biz*T_inf;
                    } else if (bzN && !bx0 && !bxN && !byN) {
                        denom = 1.0f + 2.0f*az + 2.0f*bz_ + Biz;
                        rhs   = zm() + az*(xp()+xm()) + bz_*(yp()+ym()) + Biz*T_inf;
                    } else {
                        Tf[c] = T_inf;
                        continue;
                    }
                    Tf[c] = rhs / denom;
                }
            }
        }
    }

    // L2 norm ||Tf − Ti||₂ (identical criterion to GPU engine: raw sum, no /N).
    real_t check_convergence() const {
        real_t sum = 0.0f;
        #pragma omp parallel for reduction(+:sum) schedule(static)
        for (uint32_t i = 0; i < N3; ++i) {
            real_t d = Tf[i] - Ti[i];
            sum += d * d;
        }
        return std::sqrt(sum);
    }

    // Ti ← Tf
    void advance() { Ti = Tf; }

    // Gauss-Seidel iterations until convergence.
    void solve(real_t dx, real_t dy, real_t dz,
               real_t kappa, real_t h_conv, real_t T_inf, real_t T_oven,
               int max_iter = 2000, real_t tol = 5e-4f)
    {
        int it = 0;
        for (; it < max_iter; ++it) {
            sweep(dx, dy, dz, kappa, h_conv, T_inf, T_oven);
            real_t err = check_convergence();
            advance();
            if (err < tol) break;
            if (it % 10000 == 9999)
                std::printf("      [Heat eq] iter %d  ||ΔT||₂ = %.4e  (tol %.4e)\n",
                            it + 1, err, tol);
        }
        real_t T_max = *std::max_element(Tf.begin(), Tf.end());
        real_t T_min = *std::min_element(Tf.begin(), Tf.end());
        std::printf("      [Heat eq] converged in %d iters  T=[%.2f, %.2f] °C\n",
                    it, T_min, T_max);
        std::fflush(stdout);
    }

    void snapshot_outer() { T_outer = Tf; }

    // RMS(Tf − T_outer) for outer-loop convergence.
    real_t outer_T_rms() const {
        real_t sum = 0.0f;
        #pragma omp parallel for reduction(+:sum) schedule(static)
        for (uint32_t i = 0; i < N3; ++i) {
            real_t d = Tf[i] - T_outer[i];
            sum += d * d;
        }
        return std::sqrt(sum / (real_t)N3);
    }

    // Return raw pointer to Tf (for update_dk_from_T compatibility).
    const real_t* Tf_ptr() const { return Tf.data(); }
};
