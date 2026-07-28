#pragma once
/*
 * DataTypes_cpu.hpp
 * =================
 * CPU counterpart of DataTypes.cuh.
 * Uses std::complex<float> instead of cufftComplex.
 * Grid dimensions are still injected as -D macros (same Makefile pattern).
 */

#include <complex>
#include <vector>
#include <cstdint>
#include <nlohmann/json.hpp>

using real_t    = float;
using complex_t = std::complex<float>;
using rVec_t    = std::vector<real_t>;
using cVec_t    = std::vector<complex_t>;
using json      = nlohmann::json;

#ifndef NX
#define NX 128
#endif
#ifndef NY
#define NY 128
#endif
#ifndef NZ
#define NZ 100
#endif
#ifndef NT
#define NT 512
#endif

static constexpr uint32_t SIZE = (uint32_t)NX * NY * NT;

static constexpr real_t C    = 299.792458f;       // μm/ps
static constexpr real_t EPS0 = 8.8541878128e-12f * 1e12f / 1e6f;
static constexpr real_t PI   = 3.14159265358979f;

enum Process { PROC_SHG = 0, PROC_SFG = 1, PROC_OPG = 2, PROC_DFG = 3 };
