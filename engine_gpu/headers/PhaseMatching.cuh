#ifndef _PHASEMATCHING_CUH
#define _PHASEMATCHING_CUH

/*
 * PhaseMatching
 * =============
 * Manages the spatially varying phase-mismatch Δk(x,y,z) for a QPM crystal
 * under a temperature distribution T(x,y,z).
 *
 * For constant temperature (no thermal model):
 *   Δk = const  (read from Crystal::dk)
 *
 * For thermal mode:
 *   Λ(T) = Λ₀ · [1 + α_th · (T − T₀)]
 *   Δk(x,y,z,T) = Δk_bare(λ_p, λ_s, λ_i, T=T₀) − 2π/Λ(T)
 *
 * In the solver the kernel multiplier is exp(±i·Δk·z), so we store Δk as a
 * 3-D device array DK[NX][NY][NZ] and its running integral DKint[NX][NY].
 *
 * Ported from cuSHG/src/headers/PhaseMatching.h and generalised.
 */

// ── Kernels ───────────────────────────────────────────────────────────────────

// Fill DK with a spatially-uniform phase mismatch (no thermal)
__global__ void kernelSetDKConstant(real_t *DK, real_t dk_val) {
    uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t iz = blockIdx.z;
    if (ix >= NX || iy >= NY || iz >= NZ) return;
    DK[IDX3(ix, iy, iz, NX, NY)] = dk_val;
}

// Fill DK from a 3-D temperature field:
//   Λ(T) = Λ₀ · (1 + α_th · (T − T₀))
//   Δk(T) = dk_bare − 2π/Λ(T)          (Λ in μm)
__global__ void kernelSetDKThermal(real_t *DK, const real_t *T3D,
                                   real_t dk_bare,
                                   real_t Lambda0, real_t alpha_th, real_t T0)
{
    uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t iz = blockIdx.z;
    if (ix >= NX || iy >= NY || iz >= NZ) return;

    uint32_t idx = IDX3(ix, iy, iz, NX, NY);
    real_t T_loc = T3D[idx];
    real_t Lam_T = Lambda0 * (1.0f + alpha_th * (T_loc - T0));
    DK[idx] = dk_bare - 2.0f * 3.14159265f / Lam_T;
}

// Total integral along z per column:
//   DKint[ix, iy] = Σ_{iz=0}^{NZ-1} DK[ix,iy,iz] · dz
// Used for convergence monitoring (not for in-kernel phase computation).
__global__ void kernelIntegrateDK(real_t *DKint, const real_t *DK, real_t dz) {
    uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= NX || iy >= NY) return;

    real_t acc = 0.0f;
    for (uint32_t iz = 0; iz < NZ; ++iz)
        acc += DK[IDX3(ix, iy, iz, NX, NY)];
    DKint[IDX(ix, iy)] = acc * dz;
}

// Exclusive prefix-sum along z:
//   DKcum[ix, iy, iz] = Σ_{k=0}^{iz-1} DK[ix,iy,k] · dz
// This is the accumulated phase mismatch from the crystal entrance UP TO (but not
// including) slice iz.  Used in dAdz as: phase = DKcum[iz] + dz_frac * DK[iz].
__global__ void kernelCumIntegrateDK(real_t *DKcum, const real_t *DK, real_t dz) {
    uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= NX || iy >= NY) return;

    real_t acc = 0.0f;
    for (uint32_t iz = 0; iz < NZ; ++iz) {
        DKcum[IDX3(ix, iy, iz, NX, NY)] = acc;           // phase before this slice
        acc += DK[IDX3(ix, iy, iz, NX, NY)] * dz;
    }
}


// ── Host-side manager ─────────────────────────────────────────────────────────
struct PhaseMatching {
    rVecd_t  DK;      // 3-D Δk(x,y,z)             [NX × NY × NZ]
    rVecd_t  DKcum;   // 3-D exclusive prefix sum   [NX × NY × NZ]  ← used in dAdz
    rVecd_t  DKint;   // 2-D total integral per col [NX × NY]        ← monitoring
    bool     thermal; // true → DK rebuilt from Tfield each iteration

    PhaseMatching() : DK(NX * NY * NZ, 0.0f),
                      DKcum(NX * NY * NZ, 0.0f),
                      DKint(NX * NY, 0.0f),
                      thermal(false) {}

    real_t* dk_ptr()    { return thrust::raw_pointer_cast(DK.data());    }
    real_t* dkcum_ptr() { return thrust::raw_pointer_cast(DKcum.data()); }
    real_t* dkint_ptr() { return thrust::raw_pointer_cast(DKint.data()); }

    // Initialise with a constant dk (non-thermal or initial guess)
    void set_constant(real_t dk_val) {
        dim3 block(BLKX, BLKY, 1);
        dim3 grid((NX + BLKX - 1) / BLKX, (NY + BLKY - 1) / BLKY, NZ);
        kernelSetDKConstant<<<grid, block>>>(dk_ptr(), dk_val);
        cudaDeviceSynchronize();
    }

    // Update from thermal field (call each outer thermal iteration)
    void update_from_temperature(const real_t *T3D,
                                 real_t dk_bare, real_t Lambda0,
                                 real_t alpha_th, real_t T0) {
        dim3 block(BLKX, BLKY, 1);
        dim3 grid((NX + BLKX - 1) / BLKX, (NY + BLKY - 1) / BLKY, NZ);
        kernelSetDKThermal<<<grid, block>>>(
            dk_ptr(), T3D, dk_bare, Lambda0, alpha_th, T0);
        cudaDeviceSynchronize();
    }

    // Total integral (for convergence monitoring); call after set_constant / update_from_temperature
    void integrate(real_t dz) {
        dim3 block(BLKX, BLKY, 1);
        dim3 grid((NX + BLKX - 1) / BLKX, (NY + BLKY - 1) / BLKY, 1);
        kernelIntegrateDK<<<grid, block>>>(dkint_ptr(), dk_ptr(), dz);
        cudaDeviceSynchronize();
    }

    // Exclusive prefix-sum used by dAdz kernel; call after set_constant / update_from_temperature
    void compute_cumulative(real_t dz) {
        dim3 block(BLKX, BLKY, 1);
        dim3 grid((NX + BLKX - 1) / BLKX, (NY + BLKY - 1) / BLKY, 1);
        kernelCumIntegrateDK<<<grid, block>>>(dkcum_ptr(), dk_ptr(), dz);
        cudaDeviceSynchronize();
    }
};

#endif // _PHASEMATCHING_CUH
