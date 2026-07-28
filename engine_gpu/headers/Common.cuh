#ifndef _COMMON_CUH
#define _COMMON_CUH

#include <sys/time.h>
#include <iomanip>

// ── CUDA error-checking macros ────────────────────────────────────────────────
#define CHECK(call)                                                         \
{                                                                           \
    const cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                               \
        fprintf(stderr,"Error: %s:%d  code=%d  %s\n",                      \
                __FILE__, __LINE__, err, cudaGetErrorString(err));          \
        exit(1);                                                            \
    }                                                                       \
}

#define CHECK_CUFFT(call)                                                   \
{                                                                           \
    cufftResult err;                                                        \
    if ((err = (call)) != CUFFT_SUCCESS) {                                  \
        fprintf(stderr,"CUFFT error %d at %s:%d\n", err, __FILE__, __LINE__);\
        exit(1);                                                            \
    }                                                                       \
}

#define CHECK_CURAND(call)                                                  \
{                                                                           \
    curandStatus_t err;                                                     \
    if ((err = (call)) != CURAND_STATUS_SUCCESS) {                          \
        fprintf(stderr,"CURAND error %d at %s:%d\n", err, __FILE__, __LINE__);\
        exit(1);                                                            \
    }                                                                       \
}

// ── Timing ────────────────────────────────────────────────────────────────────
inline double seconds() {
    static auto t0 = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now - t0).count();
}

inline void timing_code(double dt) {
    if      (dt < 60.0)   std::cout << "\nElapsed: " << dt       << " s\n";
    else if (dt < 3600.0) std::cout << "\nElapsed: " << dt/60.0  << " min\n";
    else                  std::cout << "\nElapsed: " << dt/3600.0 << " h\n";
}

// ── Indexing ──────────────────────────────────────────────────────────────────
__host__ __device__ inline uint32_t IDX(uint32_t x, uint32_t y, uint32_t t)
{   // (x, y, t) → linear index in NX×NY×NT tensor (t is the time/z index)
    return (t * NX * NY) + (y * NX) + x;
}
__host__ __device__ inline uint32_t IDX(uint32_t x, uint32_t y)
{   // 2-D index (NX×NY)
    return (y * NX) + x;
}
__host__ __device__ inline uint32_t IDX3(uint32_t x, uint32_t y, uint32_t z,
                                          uint32_t nx, uint32_t ny)
{   // General 3-D index for thermal grid (may differ from optical grid)
    return (z * nx * ny) + (y * nx) + x;
}

// ── GPU info ──────────────────────────────────────────────────────────────────
inline void is_gpu_available() {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    std::cout << "GPU: " << p.name
              << "  [CC " << p.major << "." << p.minor << "]\n";
    cudaSetDevice(dev);
}

inline void print_line() {
    std::cout << "*------------------------------------------------------------*\n";
}

inline void print_grid() {
    print_line();
    std::cout << "  Grid  (NT, NX, NY, NZ) = ("
              << NT << ", " << NX << ", " << NY << ", " << NZ << ")\n"
              << "  Block (BLKT, BLKX, BLKY) = ("
              << BLKT << ", " << BLKX << ", " << BLKY << ")\n"
              << "  Full tensor size NT*NX*NY = " << SIZE << "\n";
    print_line();
}

inline void welcome() {
    std::cout << R"(
*----------------------------------------------*
*                                              *
*   BBBBB  RRRR   AAAA  HH  HH MM   MM  SSS   *
*   BB  BB RR RR AA  AA HH  HH MMM MMM SS      *
*   BBBBB  RRRR  AAAAAA HHHHHH MM M MM  SSS    *
*   BB  BB RR RR AA  AA HH  HH MM   MM    SS   *
*   BBBBB  RR RR AA  AA HH  HH MM   MM  SSS    *
*                                              *
*   Three-Wave Mixing Simulator  (3+1)D GPU    *
*----------------------------------------------*
)" << "\n";
}

// ── linspace helpers ──────────────────────────────────────────────────────────
template<typename T>
void linspace(T& v, real_t xmin, real_t xmax) {
    uint32_t n = v.size();
    for (uint32_t i = 0; i < n; ++i)
        v[i] = xmin + i * (xmax - xmin) / (n - 1);
}

template<typename T>
T linspace(real_t xmin, real_t xmax, uint32_t n) {
    T v(n);
    for (uint32_t i = 0; i < n; ++i)
        v[i] = xmin + i * (xmax - xmin) / (n - 1);
    return v;
}

#endif // _COMMON_CUH
