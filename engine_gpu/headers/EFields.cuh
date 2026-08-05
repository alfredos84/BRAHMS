#ifndef _EFIELDS_CUH
#define _EFIELDS_CUH

/*
 * EFields
 * =======
 * (3+1)-D optical fields: Ap (pump), As (signal), Ai (idler).
 * Adapted from pycuTWM EFields.cuh.  Key change: the constructor now takes
 * a Crystal* for all optical parameters; no Sellmeier evaluation here.
 *
 * Convention (same as pycuTWM):
 *   SHG:      Ap = fundamental (ω),  As = Ai = SH (2ω)
 *   OPG/SFG/DFG: Ap = pump (high-ω), As = signal, Ai = idler
 */

// ── Helper kernels (unchanged from pycuTWM) ───────────────────────────────────

__global__ void cuFFT1D_Scale(complex_t *A) {
    real_t s = static_cast<real_t>(NT);
    uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < TENSOR_SIZE) A[i] = A[i] / s;
}

__global__ void kernelGetSlice(complex_t *Dst, const complex_t *Src, int sl) {
    uint32_t ix = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t iy = threadIdx.y + blockDim.y * blockIdx.y;
    if (ix < NX && iy < NY)
        Dst[IDX(ix, iy, 0)] = Src[IDX(ix, iy, sl)];
}

__global__ void kernelSetSlice(complex_t *Dst, const complex_t *Src, int sl) {
    uint32_t ix = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t iy = threadIdx.y + blockDim.y * blockIdx.y;
    if (ix < NX && iy < NY)
        Dst[IDX(ix, iy, sl)] = Src[IDX(ix, iy, 0)];
}

template<typename T>
void fftshift(T& V_flip, const T& V) {
    int c = V.size() / 2;
    for (int i = 0; i < c; ++i) {
        V_flip[i + c] = V[i];
        V_flip[i]     = V[i + c];
    }
}

// ── Pump initialisation kernels (unchanged from pycuTWM) ──────────────────────

__global__ void setWavePlaneCW(complex_t *Ap, real_t Ap0, real_t w, real_t uX, real_t uY, real_t dx, real_t dy) {
    uint32_t ix = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t iy = threadIdx.y + blockDim.y * blockIdx.y;
    for (uint32_t it = 0; it < NT; ++it) {
        if (ix < NX && iy < NY) {
            real_t ex = (ix - uX) * dx, ey = (iy - uY) * dy;
            real_t r2 = ex*ex + ey*ey;
            Ap[IDX(ix, iy, it)].x = Ap0 * expf(-r2 / (w * w));
            Ap[IDX(ix, iy, it)].y = 0.0f;
        }
    }
}

__global__ void setWavePlanePulsed(complex_t *Ap, const real_t *t, real_t Ap0, real_t w,
                                   real_t tau, real_t uX, real_t uY, real_t dx, real_t dy) {
    uint32_t ix = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t iy = threadIdx.y + blockDim.y * blockIdx.y;
    for (uint32_t it = 0; it < NT; ++it) {
        if (ix < NX && iy < NY) {
            real_t ex = (ix - uX) * dx, ey = (iy - uY) * dy;
            real_t r2 = ex*ex + ey*ey;
            real_t tt = t[it] / tau;
            Ap[IDX(ix, iy, it)].x = Ap0 * expf(-(tt*tt) - r2 / (w * w));
            Ap[IDX(ix, iy, it)].y = 0.0f;
        }
    }
}

__global__ void setFocusedCW(complex_t *Ap, real_t Ap0, complex_t MX, real_t w,
                              real_t uX, real_t uY, real_t dx, real_t dy) {
    uint32_t ix = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t iy = threadIdx.y + blockDim.y * blockIdx.y;
    for (uint32_t it = 0; it < NT; ++it) {
        if (ix < NX && iy < NY) {
            real_t ex = (ix - uX) * dx, ey = (iy - uY) * dy;
            real_t r2 = ex*ex + ey*ey;
            Ap[IDX(ix, iy, it)] = (Ap0 / MX) * CpxExp(-r2 / (w * w * MX));
        }
    }
}

__global__ void setFocusedPulsed(complex_t *Ap, const real_t *t, real_t Ap0, complex_t MX,
                                  real_t w, real_t tau, real_t uX, real_t uY, real_t dx, real_t dy) {
    uint32_t ix = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t iy = threadIdx.y + blockDim.y * blockIdx.y;
    for (uint32_t it = 0; it < NT; ++it) {
        if (ix < NX && iy < NY) {
            real_t ex = (ix - uX) * dx, ey = (iy - uY) * dy;
            real_t r2 = ex*ex + ey*ey;
            real_t tt = t[it] / tau;
            Ap[IDX(ix, iy, it)] = (Ap0 / MX) * expf(-(tt*tt))
                                   * CpxExp(-r2 / (w * w * MX));
        }
    }
}

// ── fftShift 2-D helper kernels ────────────────────────────────────────────────
__global__ void fftShift2DH(complex_t *F, complex_t *aux) {
    uint32_t ix = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t iy = threadIdx.y + blockDim.y * blockIdx.y;
    uint32_t c  = NX / 2;
    if (ix < c && iy < NY) {
        F[IDX(ix + c, iy, 0)] = aux[IDX(ix, iy, 0)];
        F[IDX(ix,     iy, 0)] = aux[IDX(ix + c, iy, 0)];
    }
}
__global__ void fftShift2DV(complex_t *F, complex_t *aux) {
    uint32_t ix = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t iy = threadIdx.y + blockDim.y * blockIdx.y;
    uint32_t r  = NY / 2;
    if (iy < r && ix < NX) {
        F[IDX(ix, iy + r, 0)] = aux[IDX(ix, iy,     0)];
        F[IDX(ix, iy,     0)] = aux[IDX(ix, iy + r, 0)];
    }
}

// ── EFields class ──────────────────────────────────────────────────────────────
class EFields {
public:
    // ── Device arrays ────────────────────────────────────────────────────────
    cVecd_t Api;           // initial pump (saved for multipass reset)
    cVecd_t Ap, As, Ai;   // current fields
    cVecd_t Awp, Aws, Awi;
    cVecd_t AQp, AQs, AQi;
    cVecd_t eiQz_p, eiQz_s, eiQz_i;   // diffraction propagators [NX×NY]
    cVecd_t eiLz_p, eiLz_s, eiLz_i;   // dispersion propagators  [NT]
    cVecd_t AuxQp, AuxQs, AuxQi;
    cVecd_t Aux_Ap,  Aux_As,  Aux_Ai;
    cVecd_t Aux_Awp, Aux_Aws, Aux_Awi;
    cVecd_t Aux_Awp_prop, Aux_Aws_prop, Aux_Awi_prop;
    rVecd_t t, F, w;      // time, frequency, angular frequency vectors

    // ── Optical parameters (from Crystal) ────────────────────────────────────
    real_t lp, ls, li;
    real_t np, ns, ni;
    real_t vp, vs, vi;
    real_t b2p, b2s, b2i;
    real_t b3p, b3s, b3i;
    real_t alpha_crp, alpha_crs, alpha_cri;
    real_t rho_p, rho_s, rho_i;      // spatial walk-off angles [rad]
    real_t kp, ks, ki;              // wavevectors [μm⁻¹]
    real_t kappa_p, kappa_s, kappa_i; // coupling coefficients [1/(V·μm)] — see note
    real_t dx, dy, dz;
    real_t dk;
    real_t Power, waist;
    real_t Lcr;

    cufftHandle planDiffraction, planDispersion;

    // ── Constructor ───────────────────────────────────────────────────────────
    EFields(real_t _Power, real_t _waist, Crystal *Cr)
        : Power(_Power), waist(_waist)
    {
        Api.resize(TENSOR_SIZE);
        Ap.resize(TENSOR_SIZE);  As.resize(TENSOR_SIZE);  Ai.resize(TENSOR_SIZE);
        Awp.resize(TENSOR_SIZE); Aws.resize(TENSOR_SIZE); Awi.resize(TENSOR_SIZE);
        AQp.resize(TENSOR_SIZE); AQs.resize(TENSOR_SIZE); AQi.resize(TENSOR_SIZE);
        AuxQp.resize(TENSOR_SIZE); AuxQs.resize(TENSOR_SIZE); AuxQi.resize(TENSOR_SIZE);
        Aux_Ap.resize(TENSOR_SIZE); Aux_As.resize(TENSOR_SIZE); Aux_Ai.resize(TENSOR_SIZE);
        Aux_Awp.resize(TENSOR_SIZE); Aux_Aws.resize(TENSOR_SIZE); Aux_Awi.resize(TENSOR_SIZE);
        Aux_Awp_prop.resize(TENSOR_SIZE); Aux_Aws_prop.resize(TENSOR_SIZE); Aux_Awi_prop.resize(TENSOR_SIZE);
        eiQz_p.resize(NX * NY); eiQz_s.resize(NX * NY); eiQz_i.resize(NX * NY);
        eiLz_p.resize(NT); eiLz_s.resize(NT); eiLz_i.resize(NT);
        t.resize(NT); F.resize(NT); w.resize(NT);

        // Copy optical props from Crystal
        lp = Cr->lp; ls = Cr->ls; li = Cr->li;
        np = Cr->np; ns = Cr->ns; ni = Cr->ni;
        vp = Cr->vp; vs = Cr->vs; vi = Cr->vi;
        b2p = Cr->b2p; b2s = Cr->b2s; b2i = Cr->b2i;
        b3p = Cr->b3p; b3s = Cr->b3s; b3i = Cr->b3i;
        alpha_crp = Cr->alpha_crp;
        alpha_crs = Cr->alpha_crs;
        alpha_cri = Cr->alpha_cri;
        rho_p = Cr->rho_p; rho_s = Cr->rho_s; rho_i = Cr->rho_i;
        dx = Cr->dx; dy = Cr->dy; dz = Cr->dz;
        dk = Cr->dk;
        Lcr = Cr->Lcr;

        // Wavevectors [μm⁻¹]
        kp = 2.0f * PI * np / lp;
        ks = 2.0f * PI * ns / ls;
        ki = 2.0f * PI * ni / li;

        // Coupling coefficients κ_j = π d_eff / (n_j λ_j)  [μm/V / μm = 1/V]
        // Convention matches cuSHG (validated experimentally).
        // d_eff: pm/V × 1e-6 → μm/V;  λ in μm → κ in 1/V
        real_t dQ = Cr->dQ * 1e-6f;   // pm/V → μm/V
        kappa_p = PI * dQ / (np * lp);
        kappa_s = PI * dQ / (ns * ls);
        kappa_i = PI * dQ / (ni * li);

    }

    ~EFields() {}

    // ── Methods ───────────────────────────────────────────────────────────────
    void set_pump_field(real_t Power, real_t FWHM, real_t waist, real_t focalpoint, const std::string& mode);
    void set_signal_field(real_t Power, real_t FWHM, real_t waist, real_t focalpoint, const std::string& mode);
    void set_idler_field(real_t Power, real_t FWHM, real_t waist, real_t focalpoint, const std::string& mode);
    void noise_generator(cVecd_t& Vec);
    void set_time_freq_vectors(real_t t_window);
    void fftShift2D(cVecd_t& propagator);
    void set_plan_diffraction();
    void set_plan_dispersion();
    void destroy_cufft_plans();

private:
    // Shared implementation for set_{pump,signal,idler}_field: initialises
    // `target` (Api for the pump, As/Ai directly for signal/idler — these
    // are not reset per multipass the way the pump is, so no separate
    // "initial" buffer is needed for them) with a wave-plane or focused
    // Gaussian beam, CW or pulsed, at the given wavelength/index.
    void init_field(cVecd_t& target, real_t n_val, real_t lambda_val,
                     real_t Pwr, real_t FWHM, real_t wst, real_t fp,
                     const std::string& mode, const char* label);
};

// ── Method implementations ─────────────────────────────────────────────────────

void EFields::init_field(cVecd_t& target, real_t n_val, real_t lambda_val,
                          real_t Pwr, real_t FWHM, real_t wst, real_t fp,
                          const std::string& mode, const char* label) {
    real_t tau  = FWHM * sqrtf(2.0f) / (2.0f * sqrtf(2.0f * logf(2.0f)));
    real_t w02  = wst * wst;
    real_t zR   = PI * n_val * w02 / lambda_val;
    real_t eta  = fp / zR;
    real_t A0   = sqrtf(4.0f * Pwr / (EPS0 * C * PI * n_val * w02));
    complex_t Im; Im.x = 0; Im.y = 1;
    complex_t MX = (1.0f - Im * eta);  // -i·η → converging beam, waist at z=fp

    complex_t *ptr = thrust::raw_pointer_cast(target.data());
    real_t *t_ptr  = thrust::raw_pointer_cast(this->t.data());
    dim3 block(BLKX, BLKY), grid((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);

    if      (mode == "waveplane-cw")
        setWavePlaneCW<<<grid, block>>>(ptr, A0, wst, 0.5f*NX, 0.5f*NY, dx, dy);
    else if (mode == "waveplane-pulsed")
        setWavePlanePulsed<<<grid, block>>>(ptr, t_ptr, A0, wst, tau, 0.5f*NX, 0.5f*NY, dx, dy);
    else if (mode == "focused-cw")
        setFocusedCW<<<grid, block>>>(ptr, A0, MX, wst, 0.5f*NX, 0.5f*NY, dx, dy);
    else if (mode == "focused-pulsed")
        setFocusedPulsed<<<grid, block>>>(ptr, t_ptr, A0, MX, wst, tau, 0.5f*NX, 0.5f*NY, dx, dy);
    else
        std::cerr << "[EFields] Unknown " << label << " mode: " << mode << "\n";

    cudaDeviceSynchronize();
    print_line();
    std::cout << "  " << label << " field :\n"
              << "    mode  = " << mode << "\n"
              << "    Power = " << Pwr  << " W\n"
              << "    waist = " << wst  << " μm\n";
    if (mode.find("focused") != std::string::npos) {
        real_t xi = (Lcr * lambda_val) / (2.0f * PI * n_val * w02);
        std::cout << "    focal point = " << fp << " μm\n"
                  << "    z_R  = " << zR  << " μm\n"
                  << "    η    = " << eta << "  (fp/z_R)\n"
                  << "    ξ    = " << xi  << "  (L·λ / 2π·n·w²)\n";
    }
    if (mode.find("pulsed") != std::string::npos)
        std::cout << "    FWHM  = " << FWHM << " ps\n";
    print_line();
}

void EFields::set_pump_field(real_t Pwr, real_t FWHM, real_t wst, real_t fp, const std::string& mode) {
    init_field(Api, np, lp, Pwr, FWHM, wst, fp, mode, "Pump");
}

void EFields::set_signal_field(real_t Pwr, real_t FWHM, real_t wst, real_t fp, const std::string& mode) {
    init_field(As, ns, ls, Pwr, FWHM, wst, fp, mode, "Signal");
}

void EFields::set_idler_field(real_t Pwr, real_t FWHM, real_t wst, real_t fp, const std::string& mode) {
    init_field(Ai, ni, li, Pwr, FWHM, wst, fp, mode, "Idler");
}

void EFields::noise_generator(cVecd_t& Vec) {
    uint32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::normal_distribution<float> dist(0.0f, 1e-15f);
    cVech_t h(Vec.size());
    for (size_t i = 0; i < h.size(); ++i) {
        h[i].x = dist(gen);
        h[i].y = dist(gen);
    }
    thrust::copy(h.begin(), h.end(), Vec.begin());
}

void EFields::set_time_freq_vectors(real_t t_win) {
    this->t = linspace<rVecd_t>(-0.5f * t_win, 0.5f * t_win, NT);
    this->F = linspace<rVecd_t>(-0.5f * NT / t_win, 0.5f * NT / t_win, NT);
    fftshift<rVecd_t>(this->w, this->F);
    this->w *= (2.0f * PI);
}

void EFields::fftShift2D(cVecd_t& prop) {
    complex_t *ptr = thrust::raw_pointer_cast(prop.data());
    complex_t *aux; CHECK(cudaMalloc((void**)&aux, nBytes2Dc));
    dim3 block(BLKX, BLKY), grid((NX+BLKX-1)/BLKX, (NY+BLKY-1)/BLKY);
    CHECK(cudaMemcpy(aux, ptr, nBytes2Dc, cudaMemcpyDeviceToDevice));
    fftShift2DV<<<grid, block>>>(ptr, aux);
    cudaDeviceSynchronize();
    CHECK(cudaMemcpy(aux, ptr, nBytes2Dc, cudaMemcpyDeviceToDevice));
    fftShift2DH<<<grid, block>>>(ptr, aux);
    cudaDeviceSynchronize();
    CHECK(cudaFree(aux));
}

void EFields::set_plan_diffraction() {
    int rank = 2, n[2] = {NY, NX};
    int idist = NX * NY, odist = NX * NY;
    int inembed[] = {NY, NX}, onembed[] = {NY, NX};
    cufftPlanMany(&planDiffraction, rank, n, inembed, 1, idist, onembed, 1, odist, CUFFT_C2C, NT);
}

void EFields::set_plan_dispersion() {
    int NT_c = NT;
    cufftPlanMany(&planDispersion, 1, &NT_c, nullptr, 1, NT_c, nullptr, 1, NT_c, CUFFT_C2C, NX * NY);
}

void EFields::destroy_cufft_plans() {
    cufftDestroy(planDiffraction);
    cufftDestroy(planDispersion);
}

#endif // _EFIELDS_CUH
