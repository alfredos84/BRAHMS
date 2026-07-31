#ifndef _SOLVER_CUH
#define _SOLVER_CUH

/*
 * Solver
 * ======
 * Split-Step Fourier Method (SSFM) + RK4 solver for the three-wave mixing CWEs.
 *
 * Key changes vs pycuTWM:
 *   1. dAdz kernel uses runtime `process` (PROC_SHG / PROC_SFG / PROC_OPG / PROC_DFG)
 *      — no #ifdef at compile time.
 *   2. Dispersion propagators use signal as the reference frame (standard convention):
 *        GVM_p = 1/vp − 1/vs,  GVM_s = 0,  GVM_i = 1/vi − 1/vs
 *      Optional TOD (β3) term included.
 *   3. Optional thermal model: outer convergence loop over Tfield iterations.
 */

// ── Dispersion 2D layout helper ───────────────────────────────────────────────
// Layout: [NX*NY][NT] row-major → index = batch*NT + t, batch = iy*NX + ix.
// (planDispersion sets stride=1, dist=NT per batch.)
__device__ inline uint32_t J2(uint32_t ix, uint32_t iy, uint32_t it) {
    return (iy * NX + ix) * NT + it;
}

// ── FFT scale kernels ──────────────────────────────────────────────────────────

__global__ void cuFFT2DScaleDisp(complex_t *Ap, complex_t *As, complex_t *Ai) {
    real_t s = static_cast<real_t>(NT);
    uint32_t ix = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t iy = threadIdx.y + blockDim.y * blockIdx.y;
    for (uint32_t it = 0; it < NT; ++it) {
        if (ix < NX && iy < NY) {
            uint32_t j = J2(ix, iy, it);
            Ap[j] /= s; As[j] /= s; Ai[j] /= s;
        }
    }
}

__global__ void cuFFT2DScaleDiff(complex_t *Ap, complex_t *As, complex_t *Ai) {
    real_t s = static_cast<real_t>(NX * NY);
    uint32_t ix = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t iy = threadIdx.y + blockDim.y * blockIdx.y;
    for (uint32_t it = 0; it < NT; ++it) {
        if (ix < NX && iy < NY) {
            Ap[IDX(ix, iy, it)] /= s;
            As[IDX(ix, iy, it)] /= s;
            Ai[IDX(ix, iy, it)] /= s;
        }
    }
}

// ── Tensor reorder kernels ─────────────────────────────────────────────────────

__global__ void tensor3Dto2D(complex_t *Ap2, complex_t *Ap3,
                              complex_t *As2, complex_t *As3,
                              complex_t *Ai2, complex_t *Ai3) {
    uint32_t ix = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t iy = threadIdx.y + blockDim.y * blockIdx.y;
    for (uint32_t it = 0; it < NT; ++it) {
        if (ix < NX && iy < NY) {
            uint32_t j2 = J2(ix, iy, it);
            uint32_t j3 = IDX(ix, iy, it);
            Ap2[j2] = Ap3[j3]; As2[j2] = As3[j3]; Ai2[j2] = Ai3[j3];
        }
    }
}

__global__ void tensor2Dto3D(complex_t *Ap3, complex_t *Ap2,
                              complex_t *As3, complex_t *As2,
                              complex_t *Ai3, complex_t *Ai2) {
    uint32_t ix = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t iy = threadIdx.y + blockDim.y * blockIdx.y;
    for (uint32_t it = 0; it < NT; ++it) {
        if (ix < NX && iy < NY) {
            uint32_t j2 = J2(ix, iy, it);
            uint32_t j3 = IDX(ix, iy, it);
            Ap3[j3] = Ap2[j2]; As3[j3] = As2[j2]; Ai3[j3] = Ai2[j2];
        }
    }
}

// ── Dispersion propagator setup ────────────────────────────────────────────────
// Reference frame: signal group velocity vs.
// eiLz = exp(i·dir·dz·ω·[GVM + ½·ω·β2 + (1/6)·ω²·β3])
__global__ void kernelDispPropagators(complex_t *eLp, complex_t *eLs, complex_t *eLi,
                                       real_t vp, real_t vs, real_t vi,
                                       real_t b2p, real_t b2s, real_t b2i,
                                       real_t b3p, real_t b3s, real_t b3i,
                                       real_t dz, const real_t *w, int dir) {
    uint32_t iw = threadIdx.x + blockDim.x * blockIdx.x;
    if (iw >= NT) return;
    real_t W = w[iw];
    eLp[iw] = CpxExp(dir * dz * W * ((1.0f/vp - 1.0f/vs) + 0.5f * W * b2p + (1.0f/6.0f) * W * W * b3p));
    eLs[iw] = CpxExp(dir * dz * W * (                       0.5f * W * b2s + (1.0f/6.0f) * W * W * b3s));
    eLi[iw] = CpxExp(dir * dz * W * ((1.0f/vi - 1.0f/vs) + 0.5f * W * b2i + (1.0f/6.0f) * W * W * b3i));
}

__global__ void kernelDispProduct(complex_t *Awp_pr, complex_t *Awp, complex_t *eLp,
                                  complex_t *Aws_pr, complex_t *Aws, complex_t *eLs,
                                  complex_t *Awi_pr, complex_t *Awi, complex_t *eLi) {
    uint32_t ix = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t iy = threadIdx.y + blockDim.y * blockIdx.y;
    for (uint32_t it = 0; it < NT; ++it) {
        if (ix < NX && iy < NY) {
            uint32_t j = J2(ix, iy, it);  // [NX*NY][NT] dispersion layout
            Awp_pr[j] = Awp[j] * eLp[it];
            Aws_pr[j] = Aws[j] * eLs[it];
            Awi_pr[j] = Awi[j] * eLi[it];
        }
    }
}

// ── Diffraction propagator setup ──────────────────────────────────────────────
// Includes an optional spatial walk-off term along x, linear in the spatial
// frequency fx (analogous to the temporal walk-off term in the dispersion
// propagator, linear in ω): the extra phase is dir·dz·2π·qx·tan(ρ_λ), with
// ρ_λ the walk-off angle of wave λ (zero ⇒ no walk-off, recovers prior behaviour).
__global__ void kernelDiffPropagators(complex_t *eQp, complex_t *eQs, complex_t *eQi,
                                       real_t uX, real_t uY, real_t dx, real_t dy, real_t dz,
                                       real_t kp, real_t ks, real_t ki,
                                       real_t rho_p, real_t rho_s, real_t rho_i, int dir) {
    uint32_t ix = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t iy = threadIdx.y + blockDim.y * blockIdx.y;
    if (ix >= NX || iy >= NY) return;
    real_t dfX = 1.0f / (dx * NX), dfY = 1.0f / (dy * NY);
    real_t qx = dfX*(ix-uX), qy = dfY*(iy-uY);
    real_t q2 = qx*qx + qy*qy;
    real_t twoPiQx = 2.0f * PI * qx;
    eQp[IDX(ix,iy,0)] = CpxExp(dir * dz * (2.0f * PI * PI / kp * q2 + twoPiQx * tanf(rho_p)));
    eQs[IDX(ix,iy,0)] = CpxExp(dir * dz * (2.0f * PI * PI / ks * q2 + twoPiQx * tanf(rho_s)));
    eQi[IDX(ix,iy,0)] = CpxExp(dir * dz * (2.0f * PI * PI / ki * q2 + twoPiQx * tanf(rho_i)));
}

__global__ void kernelDiffProduct(complex_t *AQp_pr, complex_t *eQp, complex_t *AQp,
                                  complex_t *AQs_pr, complex_t *eQs, complex_t *AQs,
                                  complex_t *AQi_pr, complex_t *eQi, complex_t *AQi) {
    uint32_t ix = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t iy = threadIdx.y + blockDim.y * blockIdx.y;
    for (uint32_t it = 0; it < NT; ++it) {
        if (ix < NX && iy < NY) {
            uint32_t j3 = IDX(ix, iy, it), j2 = IDX(ix, iy, 0);
            AQp_pr[j3] = eQp[j2] * AQp[j3];
            AQs_pr[j3] = eQs[j2] * AQs[j3];
            AQi_pr[j3] = eQi[j2] * AQi[j3];
        }
    }
}

// ── Coupled-wave equations kernel (runtime process selection) ──────────────────
//
// Phase mismatch is now spatially varying: DK3D[NX×NY×NZ] holds Δk(x,y,z),
// and DKcum3D[NX×NY×NZ] holds the exclusive prefix-sum (accumulated phase up
// to but not including slice iz, in μm⁻¹·μm = rad).
// Within an RK4 half-step, dz_frac encodes the intra-slice sub-position:
//   k1 → 0,     k2 → dz/4,  k3 → dz/4,  k4 → dz/2
//
//   phase(x,y) = DKcum3D[ix,iy,iz] + dz_frac · DK3D[ix,iy,iz]
//
// For a uniform-dk crystal: DKcum[iz] = dk·iz·dz → identical to prior dk·z.
//
// SHG  (process == PROC_SHG):
//   Ap = fundamental (ω), As = Ai = SH (2ω)
//   dAp/dz = i·κ_p · As · Ap* · exp(+dir·phase) − ½·α_p · Ap
//   dAs/dz = ½·i·κ_s · Ap² · exp(−dir·phase)    − ½·α_s · As
//   dAi/dz = ½·i·κ_i · Ap² · exp(−dir·phase)    − ½·α_i · Ai
//
// OPG / SFG / DFG  (all others):
//   Ap = pump (ω_p), As = signal (ω_s), Ai = idler (ω_i)
//   dAp/dz = i·κ_p · As · Ai  · exp(−dir·phase) − ½·α_p · Ap
//   dAs/dz = i·κ_s · Ap · Ai* · exp(+dir·phase) − ½·α_s · As
//   dAi/dz = i·κ_i · Ap · As* · exp(+dir·phase) − ½·α_i · Ai
//
// `dir` = ±1 for multipass (forward / backward pass)
__global__ void dAdz(complex_t *dAp, complex_t *dAs, complex_t *dAi,
                     const complex_t *Ap, const complex_t *As, const complex_t *Ai,
                     real_t kappa_p, real_t kappa_s, real_t kappa_i,
                     const real_t *DK3D, const real_t *DKcum3D,
                     uint32_t iz, real_t dz_frac,
                     real_t alpha_p, real_t alpha_s, real_t alpha_i,
                     real_t np, real_t ns, real_t ni,
                     real_t beta_p, real_t beta_s, real_t beta_i,
                     int process, int dir, bool undepleted)
{
    complex_t Im; Im.x = 0.0f; Im.y = 1.0f;
    uint32_t ix = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t iy = threadIdx.y + blockDim.y * blockIdx.y;

    // Precompute half-intensity prefactors for TPA: 0.5·ε₀·c·n  [W/μm² per |A|²]
    const real_t Ip_fac = 0.5f * EPS0 * C * np;
    const real_t Is_fac = 0.5f * EPS0 * C * ns;
    const real_t Ii_fac = 0.5f * EPS0 * C * ni;

    for (uint32_t it = 0; it < NT; ++it) {
        if (ix < NX && iy < NY) {
            uint32_t I   = IDX(ix, iy, it);
            uint32_t i3d = IDX3(ix, iy, iz, NX, NY);
            complex_t Ap_ = Ap[I], As_ = As[I], Ai_ = Ai[I];

            // Spatially-varying accumulated phase at (x, y, z = iz·dz + dz_frac)
            real_t phase = DKcum3D[i3d] + dz_frac * DK3D[i3d];

            if (process == PROC_SHG) {
                dAp[I] =      Im * kappa_p * As_ * CpxConj(Ap_) * CpxExp(+dir * phase)
                         - 0.5f * (alpha_p + beta_p * Ip_fac * CpxAbs2(Ap_)) * Ap_;
                dAs[I] = 0.5f * Im * kappa_s * Ap_ * Ap_ * CpxExp(-dir * phase)
                         - 0.5f * (alpha_s + beta_s * Is_fac * CpxAbs2(As_)) * As_;
                dAi[I] = 0.5f * Im * kappa_i * Ap_ * Ap_ * CpxExp(-dir * phase)
                         - 0.5f * (alpha_i + beta_i * Ii_fac * CpxAbs2(Ai_)) * Ai_;
            } else {
                // OPG / SFG / DFG — identical CWE form
                dAp[I] = Im * kappa_p * As_ * Ai_           * CpxExp(-dir * phase)
                         - 0.5f * (alpha_p + beta_p * Ip_fac * CpxAbs2(Ap_)) * Ap_;
                dAs[I] = Im * kappa_s * Ap_ * CpxConj(Ai_) * CpxExp(+dir * phase)
                         - 0.5f * (alpha_s + beta_s * Is_fac * CpxAbs2(As_)) * As_;
                dAi[I] = Im * kappa_i * Ap_ * CpxConj(As_) * CpxExp(+dir * phase)
                         - 0.5f * (alpha_i + beta_i * Ii_fac * CpxAbs2(Ai_)) * Ai_;
            }
            if (undepleted) { dAp[I].x = 0.0f; dAp[I].y = 0.0f; }
        }
    }
}

// ── Linear combination  aux = A + s·k ─────────────────────────────────────────
__global__ void kernelLinComb(complex_t *auxp, complex_t *auxs, complex_t *auxi,
                               const complex_t *Ap, const complex_t *As, const complex_t *Ai,
                               const complex_t *kp, const complex_t *ks, const complex_t *ki,
                               real_t s) {
    uint32_t ix = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t iy = threadIdx.y + blockDim.y * blockIdx.y;
    for (uint32_t it = 0; it < NT; ++it) {
        if (ix < NX && iy < NY) {
            uint32_t I = IDX(ix, iy, it);
            auxp[I] = Ap[I] + kp[I] * s;
            auxs[I] = As[I] + ks[I] * s;
            auxi[I] = Ai[I] + ki[I] * s;
        }
    }
}

// ── RK4 final update ──────────────────────────────────────────────────────────
__global__ void rk4(complex_t *Ap, complex_t *As, complex_t *Ai,
                    const complex_t *k1p, const complex_t *k1s, const complex_t *k1i,
                    const complex_t *k2p, const complex_t *k2s, const complex_t *k2i,
                    const complex_t *k3p, const complex_t *k3s, const complex_t *k3i,
                    const complex_t *k4p, const complex_t *k4s, const complex_t *k4i,
                    real_t dz) {
    uint32_t ix = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t iy = threadIdx.y + blockDim.y * blockIdx.y;
    for (uint32_t it = 0; it < NT; ++it) {
        if (ix < NX && iy < NY) {
            uint32_t I = IDX(ix, iy, it);
            Ap[I] += (k1p[I] + 2.0f*k2p[I] + 2.0f*k3p[I] + k4p[I]) * (dz / 6.0f);
            As[I] += (k1s[I] + 2.0f*k2s[I] + 2.0f*k3s[I] + k4s[I]) * (dz / 6.0f);
            Ai[I] += (k1i[I] + 2.0f*k2i[I] + 2.0f*k3i[I] + k4i[I]) * (dz / 6.0f);
        }
    }
}

// ── Intensity slice: out[iy*NX + ix] = ½·n·ε₀·c · Σ_t |A(ix, iy, it)|² ───────
__global__ void _computeIntensity2D(real_t *out, const complex_t *A, real_t I_fac) {
    uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= NX || iy >= NY) return;
    real_t sum = 0.0f;
    for (uint32_t it = 0; it < NT; ++it) {
        complex_t v = A[IDX(ix, iy, it)];
        sum += v.x * v.x + v.y * v.y;
    }
    out[iy * NX + ix] = I_fac * sum;
}

// ── Solver class ──────────────────────────────────────────────────────────────
class Solver {
public:
    // RK4 stage arrays
    cVecd_t k1p, k2p, k3p, k4p;
    cVecd_t k1s, k2s, k3s, k4s;
    cVecd_t k1i, k2i, k3i, k4i;
    cVecd_t auxp, auxs, auxi;

    // Per-wave intensity slices [NX*NY, device] — computed at each z-step
    rVecd_t Ip_buf, Is_buf, Ii_buf;

    // Volume intensity buffers: I(x, y, z) [W/μm²]  →  shape (NZ, NY, NX), host
    std::vector<real_t> vol_p, vol_s, vol_i;

    Crystal       *Cr;
    EFields       *A;
    PhaseMatching *PM;        // spatially-varying Δk and cumulative phase arrays
    Tfield        *Tf_therm;  // null → non-thermal; non-null → update Q each z-step
    json           config;
    int            process;

    // pm   — PhaseMatching holding DK3D and DKcum3D (must have compute_cumulative called)
    // tf   — Tfield for autoconsistent thermal mode (nullptr for non-thermal)
    Solver(Crystal *cr, EFields *a, const json& cfg,
           PhaseMatching *pm, Tfield *tf = nullptr)
        : Cr(cr), A(a), config(cfg), process(cr->process),
          PM(pm), Tf_therm(tf)
    {
        k1p.resize(SIZE); k2p.resize(SIZE); k3p.resize(SIZE); k4p.resize(SIZE);
        k1s.resize(SIZE); k2s.resize(SIZE); k3s.resize(SIZE); k4s.resize(SIZE);
        k1i.resize(SIZE); k2i.resize(SIZE); k3i.resize(SIZE); k4i.resize(SIZE);
        auxp.resize(SIZE); auxs.resize(SIZE); auxi.resize(SIZE);
        Ip_buf.resize((size_t)NX * NY, 0.0f);
        Is_buf.resize((size_t)NX * NY, 0.0f);
        Ii_buf.resize((size_t)NX * NY, 0.0f);
        vol_p.resize((size_t)NZ * NY * NX, 0.0f);
        vol_s.resize((size_t)NZ * NY * NX, 0.0f);
        vol_i.resize((size_t)NZ * NY * NX, 0.0f);
        std::cout << "  Solver ready.\n";
    }

    ~Solver() {}

    void set_disp_propagators();
    void dispersion();
    void set_diff_propagators();
    void diffraction();
    void solver_rk4(uint32_t iz);   // iz = z-slice index; dz_frac handled internally
    void run();
};

// ── Implementations ───────────────────────────────────────────────────────────

void Solver::set_disp_propagators() {
    dim3 block(BLKT), grid((NT + BLKT - 1) / BLKT);
    real_t *w_ptr = thrust::raw_pointer_cast(A->w.data());
    kernelDispPropagators<<<grid, block>>>(
        thrust::raw_pointer_cast(A->eiLz_p.data()),
        thrust::raw_pointer_cast(A->eiLz_s.data()),
        thrust::raw_pointer_cast(A->eiLz_i.data()),
        A->vp, A->vs, A->vi,
        A->b2p, A->b2s, A->b2i,
        A->b3p, A->b3s, A->b3i,
        A->dz, w_ptr, 1);
    CHECK(cudaDeviceSynchronize());
}

void Solver::dispersion() {
    dim3 block(BLKX, BLKY), grid((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
    auto *Ap  = thrust::raw_pointer_cast(A->Ap.data());
    auto *As  = thrust::raw_pointer_cast(A->As.data());
    auto *Ai  = thrust::raw_pointer_cast(A->Ai.data());
    auto *aAp = thrust::raw_pointer_cast(A->Aux_Ap.data());
    auto *aAs = thrust::raw_pointer_cast(A->Aux_As.data());
    auto *aAi = thrust::raw_pointer_cast(A->Aux_Ai.data());
    auto *awp = thrust::raw_pointer_cast(A->Aux_Awp.data());
    auto *aws = thrust::raw_pointer_cast(A->Aux_Aws.data());
    auto *awi = thrust::raw_pointer_cast(A->Aux_Awi.data());
    auto *awpp= thrust::raw_pointer_cast(A->Aux_Awp_prop.data());
    auto *awsp= thrust::raw_pointer_cast(A->Aux_Aws_prop.data());
    auto *awip= thrust::raw_pointer_cast(A->Aux_Awi_prop.data());
    auto *eLp = thrust::raw_pointer_cast(A->eiLz_p.data());
    auto *eLs = thrust::raw_pointer_cast(A->eiLz_s.data());
    auto *eLi = thrust::raw_pointer_cast(A->eiLz_i.data());

    tensor3Dto2D<<<grid,block>>>(aAp,Ap, aAs,As, aAi,Ai);
    cufftExecC2C(A->planDispersion, aAp, awp, CUFFT_INVERSE);
    cufftExecC2C(A->planDispersion, aAs, aws, CUFFT_INVERSE);
    cufftExecC2C(A->planDispersion, aAi, awi, CUFFT_INVERSE);
    cuFFT2DScaleDisp<<<grid,block>>>(awp, aws, awi);
    kernelDispProduct<<<grid,block>>>(awpp,awp,eLp, awsp,aws,eLs, awip,awi,eLi);
    cufftExecC2C(A->planDispersion, awpp, aAp, CUFFT_FORWARD);
    cufftExecC2C(A->planDispersion, awsp, aAs, CUFFT_FORWARD);
    cufftExecC2C(A->planDispersion, awip, aAi, CUFFT_FORWARD);
    tensor2Dto3D<<<grid,block>>>(Ap,aAp, As,aAs, Ai,aAi);
}

void Solver::set_diff_propagators() {
    dim3 block(BLKX, BLKY), grid((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
    kernelDiffPropagators<<<grid, block>>>(
        thrust::raw_pointer_cast(A->eiQz_p.data()),
        thrust::raw_pointer_cast(A->eiQz_s.data()),
        thrust::raw_pointer_cast(A->eiQz_i.data()),
        0.5f*NX, 0.5f*NY, A->dx, A->dy, A->dz,
        A->kp, A->ks, A->ki,
        A->rho_p, A->rho_s, A->rho_i, -1);
    CHECK(cudaDeviceSynchronize());
    A->fftShift2D(A->eiQz_p);
    A->fftShift2D(A->eiQz_s);
    A->fftShift2D(A->eiQz_i);
}

void Solver::diffraction() {
    dim3 block(BLKX, BLKY), grid((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
    auto *Ap  = thrust::raw_pointer_cast(A->Ap.data());
    auto *As  = thrust::raw_pointer_cast(A->As.data());
    auto *Ai  = thrust::raw_pointer_cast(A->Ai.data());
    auto *AQp = thrust::raw_pointer_cast(A->AQp.data());
    auto *AQs = thrust::raw_pointer_cast(A->AQs.data());
    auto *AQi = thrust::raw_pointer_cast(A->AQi.data());
    auto *AQpp= thrust::raw_pointer_cast(A->AuxQp.data());
    auto *AQsp= thrust::raw_pointer_cast(A->AuxQs.data());
    auto *AQip= thrust::raw_pointer_cast(A->AuxQi.data());
    auto *eQp = thrust::raw_pointer_cast(A->eiQz_p.data());
    auto *eQs = thrust::raw_pointer_cast(A->eiQz_s.data());
    auto *eQi = thrust::raw_pointer_cast(A->eiQz_i.data());

    cufftExecC2C(A->planDiffraction, Ap, AQp, CUFFT_FORWARD);
    cufftExecC2C(A->planDiffraction, As, AQs, CUFFT_FORWARD);
    cufftExecC2C(A->planDiffraction, Ai, AQi, CUFFT_FORWARD);
    kernelDiffProduct<<<grid,block>>>(AQpp,eQp,AQp, AQsp,eQs,AQs, AQip,eQi,AQi);
    CHECK(cudaMemcpy(AQp, AQpp, nBytes3Dc, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(AQs, AQsp, nBytes3Dc, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(AQi, AQip, nBytes3Dc, cudaMemcpyDeviceToDevice));
    cufftExecC2C(A->planDiffraction, AQp, Ap, CUFFT_INVERSE);
    cufftExecC2C(A->planDiffraction, AQs, As, CUFFT_INVERSE);
    cufftExecC2C(A->planDiffraction, AQi, Ai, CUFFT_INVERSE);
    cuFFT2DScaleDiff<<<grid,block>>>(Ap, As, Ai);
}

void Solver::solver_rk4(uint32_t iz) {
    dim3 block(BLKX, BLKY), grid((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
    auto *Ap  = thrust::raw_pointer_cast(A->Ap.data());
    auto *As  = thrust::raw_pointer_cast(A->As.data());
    auto *Ai  = thrust::raw_pointer_cast(A->Ai.data());
    auto *ap  = thrust::raw_pointer_cast(auxp.data());
    auto *as_ = thrust::raw_pointer_cast(auxs.data());
    auto *ai  = thrust::raw_pointer_cast(auxi.data());
    auto *k1p = thrust::raw_pointer_cast(this->k1p.data());
    auto *k2p = thrust::raw_pointer_cast(this->k2p.data());
    auto *k3p = thrust::raw_pointer_cast(this->k3p.data());
    auto *k4p = thrust::raw_pointer_cast(this->k4p.data());
    auto *k1s = thrust::raw_pointer_cast(this->k1s.data());
    auto *k2s = thrust::raw_pointer_cast(this->k2s.data());
    auto *k3s = thrust::raw_pointer_cast(this->k3s.data());
    auto *k4s = thrust::raw_pointer_cast(this->k4s.data());
    auto *k1i = thrust::raw_pointer_cast(this->k1i.data());
    auto *k2i = thrust::raw_pointer_cast(this->k2i.data());
    auto *k3i = thrust::raw_pointer_cast(this->k3i.data());
    auto *k4i = thrust::raw_pointer_cast(this->k4i.data());

    real_t dz  = Cr->dz;
    real_t kp  = A->kappa_p, ks = A->kappa_s, ki = A->kappa_i;
    real_t alp = Cr->alpha_crp, als = Cr->alpha_crs, ali = Cr->alpha_cri;
    real_t bnp = A->np,       bns = A->ns,       bni = A->ni;
    real_t btp = Cr->beta_crp, bts = Cr->beta_crs, bti = Cr->beta_cri;
    bool   undepleted = Cr->undepleted;

    // Spatially-varying Δk arrays from PhaseMatching
    auto *DK3D  = PM->dk_ptr();
    auto *DKcum = PM->dkcum_ptr();

    // RK4 stages — dz_frac encodes sub-position within the half-step:
    //   k1: at z=iz·dz           → dz_frac = 0
    //   k2: at z=iz·dz + dz/4   → dz_frac = dz/4
    //   k3: at z=iz·dz + dz/4   → dz_frac = dz/4
    //   k4: at z=iz·dz + dz/2   → dz_frac = dz/2
    dAdz<<<grid,block>>>(k1p,k1s,k1i, Ap,As,Ai, kp,ks,ki,
                         DK3D,DKcum, iz, 0.0f, alp,als,ali,
                         bnp,bns,bni, btp,bts,bti, process,1,undepleted);
    kernelLinComb<<<grid,block>>>(ap,as_,ai, Ap,As,Ai, k1p,k1s,k1i, dz*0.25f);
    dAdz<<<grid,block>>>(k2p,k2s,k2i, ap,as_,ai, kp,ks,ki,
                         DK3D,DKcum, iz, dz*0.25f, alp,als,ali,
                         bnp,bns,bni, btp,bts,bti, process,1,undepleted);
    kernelLinComb<<<grid,block>>>(ap,as_,ai, Ap,As,Ai, k2p,k2s,k2i, dz*0.25f);
    dAdz<<<grid,block>>>(k3p,k3s,k3i, ap,as_,ai, kp,ks,ki,
                         DK3D,DKcum, iz, dz*0.25f, alp,als,ali,
                         bnp,bns,bni, btp,bts,bti, process,1,undepleted);
    kernelLinComb<<<grid,block>>>(ap,as_,ai, Ap,As,Ai, k3p,k3s,k3i, dz*0.5f);
    dAdz<<<grid,block>>>(k4p,k4s,k4i, ap,as_,ai, kp,ks,ki,
                         DK3D,DKcum, iz, dz*0.5f,  alp,als,ali,
                         bnp,bns,bni, btp,bts,bti, process,1,undepleted);
    // A ← A + (k1+2k2+2k3+k4)·dz/6  (half-step split)
    rk4<<<grid,block>>>(Ap,As,Ai,
                        k1p,k1s,k1i, k2p,k2s,k2i,
                        k3p,k3s,k3i, k4p,k4s,k4i, dz*0.5f);
    CHECK(cudaDeviceSynchronize());
}

void Solver::run() {
    double t0 = seconds();
    bool dispersion_on = config.value("dispersion", true);

    A->set_plan_diffraction();
    A->set_plan_dispersion();
    A->Ap = A->Api;          // reset to initial pump
    set_disp_propagators();
    set_diff_propagators();

    dim3 blk2(BLKX, BLKY), grd2((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
    real_t *ibp = thrust::raw_pointer_cast(Ip_buf.data());
    real_t *ibs = thrust::raw_pointer_cast(Is_buf.data());
    real_t *ibi = thrust::raw_pointer_cast(Ii_buf.data());
    int last_pct = -1;

    for (uint32_t sz = 0; sz < NZ; ++sz) {
        // SSFM: NL(dz/2) → L(dz) → NL(dz/2)  using slice index sz
        solver_rk4(sz);
        if (dispersion_on) dispersion();
        diffraction();
        solver_rk4(sz);

        // Compute per-wave intensity slices: Ix = ½·n·ε₀·c·Σ_t|A(x,y,t)|²  [W/μm²]
        _computeIntensity2D<<<grd2,blk2>>>(ibp, thrust::raw_pointer_cast(A->Ap.data()), 0.5f*EPS0*C*A->np);
        _computeIntensity2D<<<grd2,blk2>>>(ibs, thrust::raw_pointer_cast(A->As.data()), 0.5f*EPS0*C*A->ns);
        _computeIntensity2D<<<grd2,blk2>>>(ibi, thrust::raw_pointer_cast(A->Ai.data()), 0.5f*EPS0*C*A->ni);
        CHECK(cudaDeviceSynchronize());

        // Update heat source Q at this z-slice (autoconsistent thermal model).
        // Q[x,y,sz] = α_p·Ip + α_s·Is + α_i·Ii  [W/μm³]
        // This Q will be used in the NEXT outer thermal iteration to solve ∇²T = -Q/κ.
        if (Tf_therm)
            Tf_therm->update_Q_from_intensity(sz, ibp, ibs, ibi,
                                               Cr->alpha_crp, Cr->alpha_crs, Cr->alpha_cri);

        // Save intensity volumes to host: vol_x[sz*NY*NX + iy*NX + ix]
        size_t offset = (size_t)sz * NY * NX;
        thrust::copy(Ip_buf.begin(), Ip_buf.end(), vol_p.begin() + offset);
        thrust::copy(Is_buf.begin(), Is_buf.end(), vol_s.begin() + offset);
        thrust::copy(Ii_buf.begin(), Ii_buf.end(), vol_i.begin() + offset);

        int pct = (int)((sz + 1) * 100.0 / NZ);
        if (pct % 10 == 0 && pct != last_pct) {
            std::cout << "\r  Completed: " << pct << " %" << std::flush;
            last_pct = pct;
        }
    }
    std::cout << "\n" << std::flush;

    A->destroy_cufft_plans();
    timing_code(seconds() - t0);
}

#endif // _SOLVER_CUH
