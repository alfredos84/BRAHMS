#ifndef _DATATYPES_CUH
#define _DATATYPES_CUH

// ── Scalar types ──────────────────────────────────────────────────────────────
using real_t    = float;
using complex_t = cufftComplex;

// ── Thrust vector aliases ─────────────────────────────────────────────────────
using rVech_t = thrust::host_vector<real_t>;
using rVecd_t = thrust::device_vector<real_t>;
using cVech_t = thrust::host_vector<complex_t>;
using cVecd_t = thrust::device_vector<complex_t>;

using json = nlohmann::json;

// ── Grid dimensions (injected at compile time via -D flags) ──────────────────
// Defaults (overridden by Makefile):
#ifndef NX
#define NX   128
#endif
#ifndef NY
#define NY   128
#endif
#ifndef NZ
#define NZ   100
#endif
#ifndef NT
#define NT   512
#endif
#ifndef BLKX
#define BLKX 16
#endif
#ifndef BLKY
#define BLKY 16
#endif
#ifndef BLKT
#define BLKT 128
#endif

static const uint32_t SIZE = NT * NX * NY;

// ── Memory sizes ──────────────────────────────────────────────────────────────
static const size_t nBytes1Dr = sizeof(real_t)    * NT;
static const size_t nBytes1Dc = sizeof(complex_t) * NT;
static const size_t nBytes2Dr = sizeof(real_t)    * NX * NY;
static const size_t nBytes2Dc = sizeof(complex_t) * NX * NY;
static const size_t nBytes3Dr = sizeof(real_t)    * SIZE;
static const size_t nBytes3Dc = sizeof(complex_t) * SIZE;

// ── Physical constants ────────────────────────────────────────────────────────
static constexpr real_t PI   = 3.14159265358979323846f;
static constexpr real_t C    = 299.792458f;           // speed of light [μm/ps]
static constexpr real_t EPS0 = 8.8541878128e-12f * 1e12f / 1e6f; // [W·ps/(V²·μm)]

// ── Process codes ─────────────────────────────────────────────────────────────
enum Process { PROC_SHG = 0, PROC_SFG = 1, PROC_OPG = 2, PROC_DFG = 3 };

#endif // _DATATYPES_CUH
