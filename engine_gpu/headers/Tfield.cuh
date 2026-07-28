#ifndef _TFIELD_CUH
#define _TFIELD_CUH

/*
 * Tfield
 * ======
 * 3-D thermal field on the same NX×NY×NZ grid as the optical fields.
 * Solves the steady-state heat equation iteratively (Gauss-Seidel):
 *
 *   ∇²T = −Q / κ
 *
 * Boundary conditions:
 *   y = 0          : Dirichlet  T = T_oven   (bottom face, oven contact)
 *   all other faces: Robin (convective)  −κ ∂T/∂n = h·(T − T_∞)
 *
 * Heat source per unit volume [W/μm³]:
 *   Q = α_p · (½ε₀ c n_p) |E_p|²
 *     + α_s · (½ε₀ c n_s) |E_s|²
 *     + α_i · (½ε₀ c n_i) |E_i|²
 *
 * Ported and generalised from cuSHG/src/headers/Tfield.h.
 * All material constants are passed as kernel parameters (not #define macros).
 */

// ── Auxiliary: |T_new − T_old|² ───────────────────────────────────────────────
__global__ void kernelDiffSq(real_t *Dif, const real_t *Tnew, const real_t *Told) {
    uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= NX || iy >= NY) return;
    for (uint32_t iz = 0; iz < NZ; ++iz) {
        uint32_t i = IDX3(ix, iy, iz, NX, NY);
        real_t d = Tnew[i] - Told[i];
        Dif[i] = d * d;
    }
}

// ── Initialise T = T0 everywhere ──────────────────────────────────────────────
__global__ void kernelFillTemp(real_t *T, real_t T0) {
    uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= NX || iy >= NY) return;
    for (uint32_t iz = 0; iz < NZ; ++iz)
        T[IDX3(ix, iy, iz, NX, NY)] = T0;
}

// ── Apply Dirichlet BC at y = 0 (oven bottom) ─────────────────────────────────
__global__ void kernelSetBottomBC(real_t *T, real_t T_oven) {
    uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= NX) return;
    for (uint32_t iz = 0; iz < NZ; ++iz)
        T[IDX3(ix, 0, iz, NX, NY)] = T_oven;
}

// ── Initialise heat source Q = 0 ──────────────────────────────────────────────
__global__ void kernelZeroQ(real_t *Q) {
    uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= NX || iy >= NY) return;
    for (uint32_t iz = 0; iz < NZ; ++iz)
        Q[IDX3(ix, iy, iz, NX, NY)] = 0.0f;
}

// ── Update Q from current optical fields (complex amplitudes) ─────────────────
// s = current z-index; Ep/Es/Ei are 2-D slices [NX×NY] of each field tensor.
__global__ void kernelUpdateQ(real_t *Q, uint32_t s,
                               const complex_t *Ep, const complex_t *Es, const complex_t *Ei,
                               real_t alpha_crp, real_t alpha_crs, real_t alpha_cri,
                               real_t np, real_t ns, real_t ni)
{
    uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= NX || iy >= NY) return;

    uint32_t slice2d = IDX(ix, iy);
    real_t Ip = 0.5f * EPS0 * C * np * CpxAbs2(Ep[slice2d]);
    real_t Is = 0.5f * EPS0 * C * ns * CpxAbs2(Es[slice2d]);
    real_t Ii = 0.5f * EPS0 * C * ni * CpxAbs2(Ei[slice2d]);

    Q[IDX3(ix, iy, s, NX, NY)] = alpha_crp * Ip + alpha_crs * Is + alpha_cri * Ii;
}

// ── Update Q from pre-computed intensity slices [W/μm²] ──────────────────────
// Ip/Is/Ii are already ½·n·ε₀·c·|E|² — avoids re-computing from complex fields.
// Layout: Ip[iy*NX + ix]  (same as Solver::Ip_buf).
__global__ void kernelUpdateQFromIntensity(real_t *Q, uint32_t iz,
                                            const real_t *Ip, const real_t *Is, const real_t *Ii,
                                            real_t alpha_crp, real_t alpha_crs, real_t alpha_cri)
{
    uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= NX || iy >= NY) return;

    uint32_t i2d = iy * NX + ix;
    Q[IDX3(ix, iy, iz, NX, NY)] = alpha_crp * Ip[i2d] + alpha_crs * Is[i2d] + alpha_cri * Ii[i2d];
}

// ── One Gauss-Seidel sweep of the heat equation ───────────────────────────────
// BC: y=0 Dirichlet (enforced by kernelSetBottomBC), all others convective.
// κ = kappa_th [W/(μm·K)], h = h_conv [W/(μm²·K)]
// Grid steps dx, dy, dz in [mm] — NOTE: heat equation is in mm units (Q in W/mm³).
// Caller must ensure consistent unit system: all W, mm, K.
__global__ void kernelHeatEquation(real_t *Tnew, const real_t *Told,
                                    const real_t *Q,
                                    real_t dx, real_t dy, real_t dz,
                                    real_t kappa, real_t h, real_t T_inf)
{
    uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= NX || iy >= NY) return;

    // Precompute Biot numbers (dimensionless)
    real_t Bix = h * dx / kappa;
    real_t Biy = h * dy / kappa;
    real_t Biz = h * dz / kappa;
    real_t L2  = 1.0f / (2.0f / (dx * dx) + 2.0f / (dy * dy) + 2.0f / (dz * dz));

    for (uint32_t iz = 0; iz < NZ; ++iz) {
        bool bx0 = (ix == 0),      bxN = (ix == NX - 1);
        bool by0 = (iy == 0),      byN = (iy == NY - 1);
        bool bz0 = (iz == 0),      bzN = (iz == NZ - 1);
        bool inner = !bx0 && !bxN && !by0 && !byN && !bz0 && !bzN;

        uint32_t c   = IDX3(ix,   iy,   iz,   NX, NY);
        uint32_t xp  = IDX3(ix+1, iy,   iz,   NX, NY);
        uint32_t xm  = IDX3(ix-1, iy,   iz,   NX, NY);
        uint32_t yp  = IDX3(ix,   iy+1, iz,   NX, NY);
        uint32_t ym  = IDX3(ix,   iy-1, iz,   NX, NY);
        uint32_t zp  = IDX3(ix,   iy,   iz+1, NX, NY);
        uint32_t zm  = IDX3(ix,   iy,   iz-1, NX, NY);

        // y=0: Dirichlet — do not update here (set by kernelSetBottomBC)
        if (by0) { Tnew[c] = Told[c]; continue; }

        if (inner) {
            Tnew[c] = L2 * (
                (Told[xp] + Told[xm]) / (dx * dx) +
                (Told[yp] + Told[ym]) / (dy * dy) +
                (Told[zp] + Told[zm]) / (dz * dz) +
                Q[c] / kappa
            );
        } else {
            // Robin BC (convective) on active boundary faces.
            // Simple second-order approximation:
            real_t ax = dy * dy / (dx * dx), bx = dz * dz / (dx * dx);
            real_t ay = dx * dx / (dy * dy), by_ = dz * dz / (dy * dy);
            real_t az = dx * dx / (dz * dz), bz = dy * dy / (dz * dz);

            real_t denom, rhs = 0.0f;

            if (bx0 && !byN && !bz0 && !bzN) {
                // Left face x=0
                denom = 1.0f + 2.0f * ax + 2.0f * bx + Bix;
                rhs = Told[xp] + ax * (Told[yp] + Told[ym]) + bx * (Told[zp] + Told[zm]) + Bix * T_inf;
                Tnew[c] = rhs / denom;
            } else if (bxN && !byN && !bz0 && !bzN) {
                // Right face x=Lx
                denom = 1.0f + 2.0f * ax + 2.0f * bx + Bix;
                rhs = Told[xm] + ax * (Told[yp] + Told[ym]) + bx * (Told[zp] + Told[zm]) + Bix * T_inf;
                Tnew[c] = rhs / denom;
            } else if (byN && !bx0 && !bxN && !bz0 && !bzN) {
                // Top face y=Ly
                denom = 1.0f + 2.0f * ay + 2.0f * by_ + Biy;
                rhs = Told[ym] + ay * (Told[xp] + Told[xm]) + by_ * (Told[zp] + Told[zm]) + Biy * T_inf;
                Tnew[c] = rhs / denom;
            } else if (bz0 && !bx0 && !bxN && !byN) {
                // Back face z=0
                denom = 1.0f + 2.0f * az + 2.0f * bz + Biz;
                rhs = Told[zp] + az * (Told[xp] + Told[xm]) + bz * (Told[yp] + Told[ym]) + Biz * T_inf;
                Tnew[c] = rhs / denom;
            } else if (bzN && !bx0 && !bxN && !byN) {
                // Front face z=Lcr
                denom = 1.0f + 2.0f * az + 2.0f * bz + Biz;
                rhs = Told[zm] + az * (Told[xp] + Told[xm]) + bz * (Told[yp] + Told[ym]) + Biz * T_inf;
                Tnew[c] = rhs / denom;
            } else {
                // Edges and corners: use ambient temperature
                Tnew[c] = T_inf;
            }
        }
    }
}

// ── Host-side manager ─────────────────────────────────────────────────────────
class Tfield {
public:
    rVecd_t Ti;       // T at inner iteration n     [NX × NY × NZ]
    rVecd_t Tf;       // T at inner iteration n+1
    rVecd_t Taux;     // scratch for convergence check
    rVecd_t Q;        // heat source [W/μm³]
    rVecd_t T_outer;  // T snapshot from previous outer iteration

    Tfield() : Ti(NX * NY * NZ, 0.0f),
               Tf(NX * NY * NZ, 0.0f),
               Taux(NX * NY * NZ, 0.0f),
               Q(NX * NY * NZ, 0.0f),
               T_outer(NX * NY * NZ, 0.0f) {}

    real_t* Ti_ptr()      { return thrust::raw_pointer_cast(Ti.data());      }
    real_t* Tf_ptr()      { return thrust::raw_pointer_cast(Tf.data());      }
    real_t* Taux_ptr()    { return thrust::raw_pointer_cast(Taux.data());    }
    real_t* Q_ptr()       { return thrust::raw_pointer_cast(Q.data());       }
    real_t* T_outer_ptr() { return thrust::raw_pointer_cast(T_outer.data()); }

    void set_temperature(real_t T0) {
        dim3 block(BLKX, BLKY), grid((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
        kernelFillTemp<<<grid, block>>>(Ti_ptr(), T0);
        kernelFillTemp<<<grid, block>>>(Tf_ptr(), T0);
        cudaDeviceSynchronize();
    }

    void set_bottom_oven(real_t T_oven) {
        dim3 block(BLKX, 1), grid((NX+BLKX-1)/BLKX, 1);
        kernelSetBottomBC<<<grid, block>>>(Ti_ptr(), T_oven);
        kernelSetBottomBC<<<grid, block>>>(Tf_ptr(), T_oven);
        cudaDeviceSynchronize();
    }

    void zero_Q() {
        dim3 block(BLKX, BLKY), grid((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
        kernelZeroQ<<<grid, block>>>(Q_ptr());
        cudaDeviceSynchronize();
    }

    // Accumulate Q from one z-slice of the optical fields (complex amplitudes).
    // Ep, Es, Ei point to 2-D slices (NX×NY) at the current z-step.
    void update_Q_slice(uint32_t iz,
                        const complex_t *Ep, const complex_t *Es, const complex_t *Ei,
                        real_t alpha_crp, real_t alpha_crs, real_t alpha_cri,
                        real_t np, real_t ns, real_t ni)
    {
        dim3 block(BLKX, BLKY), grid((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
        kernelUpdateQ<<<grid, block>>>(Q_ptr(), iz, Ep, Es, Ei,
                                       alpha_crp, alpha_crs, alpha_cri,
                                       np, ns, ni);
        cudaDeviceSynchronize();
    }

    // Accumulate Q from pre-computed intensity slices [W/μm²] — Ip = ½·n·ε₀·c·|E|².
    // Called once per z-step inside Solver::run() with the Ip_buf/Is_buf/Ii_buf device arrays.
    void update_Q_from_intensity(uint32_t iz,
                                  const real_t *Ip, const real_t *Is, const real_t *Ii,
                                  real_t alpha_crp, real_t alpha_crs, real_t alpha_cri)
    {
        dim3 block(BLKX, BLKY), grid((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
        kernelUpdateQFromIntensity<<<grid, block>>>(Q_ptr(), iz, Ip, Is, Ii,
                                                    alpha_crp, alpha_crs, alpha_cri);
        cudaDeviceSynchronize();
    }

    // One Gauss-Seidel sweep; applies Dirichlet BC afterwards.
    void sweep(real_t dx, real_t dy, real_t dz,
               real_t kappa, real_t h_conv, real_t T_inf, real_t T_oven)
    {
        dim3 block(BLKX, BLKY), grid((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
        kernelHeatEquation<<<grid, block>>>(Tf_ptr(), Ti_ptr(), Q_ptr(),
                                            dx, dy, dz, kappa, h_conv, T_inf);
        cudaDeviceSynchronize();
        kernelSetBottomBC<<<grid, block>>>(Tf_ptr(), T_oven);
        cudaDeviceSynchronize();
    }

    // Returns L2 norm ||Tf - Ti||₂ = sqrt(Σ|ΔT|²).
    // Convergence criterion matches cuSHG: L2 < tol (default tol = 5e-4).
    // Do NOT divide by N here — that made the criterion 400000× too loose,
    // causing the Gauss-Seidel to exit after 1 sweep without converging.
    real_t check_convergence() {
        dim3 block(BLKX, BLKY), grid((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
        kernelDiffSq<<<grid, block>>>(Taux_ptr(), Tf_ptr(), Ti_ptr());
        cudaDeviceSynchronize();
        thrust::device_ptr<real_t> d = thrust::device_pointer_cast(Taux_ptr());
        real_t sum = thrust::reduce(d, d + NX * NY * NZ, 0.0f);
        return sqrtf(sum);
    }

    // Ti ← Tf (prepare next inner iteration)
    void advance() { Ti = Tf; }

    // Run heat-equation iterations until convergence.
    // Convergence: ||Tf - Ti||₂ = sqrt(Σ|ΔT|²) < tol  (matches cuSHG default: tol = 5e-4)
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
        real_t T_max = *thrust::max_element(Tf.begin(), Tf.end());
        real_t T_min = *thrust::min_element(Tf.begin(), Tf.end());
        std::printf("      [Heat eq] converged in %d iters  T=[%.2f, %.2f] °C\n",
                    it, T_min, T_max);
        std::fflush(stdout);
    }

    // Snapshot Tf into T_outer for outer-loop convergence tracking.
    // Call BEFORE the outer iteration modifies Tf.
    void snapshot_outer() { T_outer = Tf; }

    // RMS difference between current Tf and the outer snapshot [K].
    // Use to check convergence of the autoconsistent thermal loop:
    //   if outer_T_rms() < outer_tol → converged.
    real_t outer_T_rms() {
        dim3 block(BLKX, BLKY), grid((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
        kernelDiffSq<<<grid, block>>>(Taux_ptr(), Tf_ptr(), T_outer_ptr());
        cudaDeviceSynchronize();
        real_t sum = thrust::reduce(Taux.begin(), Taux.end(), 0.0f);
        return sqrtf(sum / static_cast<real_t>(NX * NY * NZ));
    }
};

#endif // _TFIELD_CUH
