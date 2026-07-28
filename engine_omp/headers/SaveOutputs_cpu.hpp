#pragma once
/*
 * SaveOutputs_cpu.hpp
 * ===================
 * HDF5 output — same file format as the GPU engine.
 * Fields saved as [NT, NY, NX] with datasets "real" and "imag".
 * Vectors saved as 1D "data" dataset.
 */

#include "Common_cpu.hpp"
#include "H5Cpp.h"
#include <string>

void save_complex_h5(const cVec_t& field, const std::string& filename) {
    using namespace H5;
    hsize_t dims[3] = { NT, NY, NX };

    std::vector<float> arr_r(SIZE), arr_i(SIZE);
    for (uint32_t t = 0; t < NT; ++t)
    for (uint32_t y = 0; y < NY; ++y)
    for (uint32_t x = 0; x < NX; ++x) {
        uint32_t flat = IDX(x, y, t);
        uint32_t h5   = t*(NY*NX) + y*NX + x;
        arr_r[h5] = field[flat].real();
        arr_i[h5] = field[flat].imag();
    }

    H5File file(filename, H5F_ACC_TRUNC);
    DataSpace ds(3, dims);
    file.createDataSet("real", PredType::NATIVE_FLOAT, ds).write(arr_r.data(), PredType::NATIVE_FLOAT);
    file.createDataSet("imag", PredType::NATIVE_FLOAT, ds).write(arr_i.data(), PredType::NATIVE_FLOAT);
    file.close();
}

void save_vector_h5(const rVec_t& v, const std::string& filename) {
    using namespace H5;
    hsize_t dim = v.size();
    H5File file(filename, H5F_ACC_TRUNC);
    DataSpace ds(1, &dim);
    file.createDataSet("data", PredType::NATIVE_FLOAT, ds)
        .write(v.data(), PredType::NATIVE_FLOAT);
    file.close();
}
