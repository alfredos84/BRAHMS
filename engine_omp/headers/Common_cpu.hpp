#pragma once
/*
 * Common_cpu.hpp
 * ==============
 * Index helpers, complex arithmetic, print utilities.
 */

#include "DataTypes_cpu.hpp"
#include <iostream>
#include <cmath>
#include <omp.h>

// ── Field indexing — identical convention to the GPU engine ──────────────────
// Main 3D field: [NT][NY][NX]  →  IDX(x,y,t) = t*NX*NY + y*NX + x
inline uint32_t IDX(uint32_t x, uint32_t y, uint32_t t) {
    return t * (uint32_t)NX * NY + y * (uint32_t)NX + x;
}
// 2D slice: [NY][NX]
inline uint32_t IDX(uint32_t x, uint32_t y) {
    return y * (uint32_t)NX + x;
}
// 3D temperature grid: [NZ][NY][NX]
inline uint32_t IDX3(uint32_t x, uint32_t y, uint32_t z, uint32_t nx, uint32_t ny) {
    return z * nx * ny + y * nx + x;
}
// Dispersion FFT layout [NX*NY][NT]
inline uint32_t J2(uint32_t ix, uint32_t iy, uint32_t it) {
    return (iy * (uint32_t)NX + ix) * (uint32_t)NT + it;
}

// ── Complex helpers ───────────────────────────────────────────────────────────
static const complex_t Im = {0.0f, 1.0f};

inline complex_t CpxExp(real_t phi) {
    return {cosf(phi), sinf(phi)};
}
inline real_t CpxAbs2(complex_t z) {
    return z.real()*z.real() + z.imag()*z.imag();
}

// ── Print utilities ───────────────────────────────────────────────────────────
inline void print_line() {
    std::cout << "*------------------------------------------------------------*\n";
}

inline void print_grid() {
    print_line();
    std::cout << "  Grid  (NT, NX, NY, NZ) = ("
              << NT << ", " << NX << ", " << NY << ", " << NZ << ")\n"
              << "  Threads (OMP) = " << omp_get_max_threads() << "\n"
              << "  Full tensor size NT*NX*NY = " << SIZE << "\n";
    print_line();
}

inline void print_engine_info() {
    std::cout << "CPU: OpenMP  [threads: " << omp_get_max_threads() << "]\n";
}

inline void welcome_cpu() {
    std::cout << R"(
*----------------------------------------------*
*                                              *
*   BBBBB  RRRR   AAAA  HH  HH MM   MM  SSS   *
*   BB  BB RR RR AA  AA HH  HH MMM MMM SS      *
*   BBBBB  RRRR  AAAAAA HHHHHH MM M MM  SSS    *
*   BB  BB RR RR AA  AA HH  HH MM   MM    SS   *
*   BBBBB  RR RR AA  AA HH  HH MM   MM  SSS    *
*                                              *
*  Three-Wave Mixing Simulator  (3+1)D  CPU    *
*----------------------------------------------*
)" << "\n";
}

inline void print_progress(int iz, int nz) {
    int pct = (iz * 100) / nz;
    if (pct % 10 == 0) {
        printf("\r  Completed: %d %%", pct);
        fflush(stdout);
    }
}
