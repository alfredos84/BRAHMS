/*---------------------------------------------------------------------------*/
// * This file set the policy for output files
// * the optical cavity as well as some intracavity elements.
/*---------------------------------------------------------------------------*/


#ifndef _SAVEOUTPUTS
#define _SAVEOUTPUTS


void init_full_field_h5(const std::string& filename, uint32_t NZtot) {
    using namespace H5;
    hsize_t dims[3] = { NX, NY, NZtot };

    H5File file(filename, H5F_ACC_TRUNC); // Crea o sobreescribe
    DataSpace dataspace(3, dims);

    file.createDataSet("real", PredType::NATIVE_FLOAT, dataspace);
    file.createDataSet("imag", PredType::NATIVE_FLOAT, dataspace);

    file.close();
}


void write_zslice_h5(const cVech_t& field, const std::string& filename, uint32_t z_slice) {
    using namespace H5;

    // Open file in read-write mode
    H5File file(filename, H5F_ACC_RDWR);
    DataSet dset_real = file.openDataSet("real");
    DataSet dset_imag = file.openDataSet("imag");

    hsize_t dims[3] = { NX, NY, 1 }; 

    // Define "hyperslab": [NX, NY, 1] in position z_slice
    hsize_t offset[3] = { 0, 0, static_cast<hsize_t>(z_slice) };
    DataSpace filespace = dset_real.getSpace();
    filespace.selectHyperslab(H5S_SELECT_SET, dims, offset);

    // Prepare slice real/imag plane [NX, NY]
    std::vector<real_t> arr_real(NX * NY), arr_imag(NX * NY);
    for (uint32_t y = 0; y < NY; ++y)
        for (uint32_t x = 0; x < NX; ++x) {
            uint32_t idx = x + NX * y;
            arr_real[idx] = field[idx].x;
            arr_imag[idx] = field[idx].y;
        }

    // Memory dataspace for 1 slice
    DataSpace memspace(3, dims);

    dset_real.write(arr_real.data(), PredType::NATIVE_FLOAT, memspace, filespace);
    dset_imag.write(arr_imag.data(), PredType::NATIVE_FLOAT, memspace, filespace);

    file.close();
}


void save_matrix_complex_h5_time(const cVech_t& field, const std::string& filename)
{
    using namespace H5;

    // Queremos el orden [t, y, x] = [NT, NY, NX]
    hsize_t dims[3] = {
        static_cast<hsize_t>(NT),
        static_cast<hsize_t>(NY),
        static_cast<hsize_t>(NX)
    };

    H5File file(filename, H5F_ACC_TRUNC);
    DataSpace dataspace(3, dims);

    DataSet dset_real = file.createDataSet("real", PredType::NATIVE_FLOAT, dataspace);
    DataSet dset_imag = file.createDataSet("imag", PredType::NATIVE_FLOAT, dataspace);

    std::vector<real_t> arr_real(NX * NY * NT);
    std::vector<real_t> arr_imag(NX * NY * NT);

    for (uint32_t t = 0; t < NT; ++t) {
        for (uint32_t y = 0; y < NY; ++y) {
            for (uint32_t x = 0; x < NX; ++x) {

                // Índice en tu buffer original
                uint32_t idx_flat = IDX(x, y, t);   // = t*(NX*NY) + y*NX + x

                // Índice en el array HDF5 si lo interpretamos como [t, y, x]
                // Para dims = {NT, NY, NX}:
                // idx_h5(t,y,x) = t*(NY*NX) + y*NX + x
                uint32_t idx_h5  = t*(NY*NX) + y*NX + x;

                arr_real[idx_h5] = field[idx_flat].x;
                arr_imag[idx_h5] = field[idx_flat].y;
            }
        }
    }

    dset_real.write(arr_real.data(), PredType::NATIVE_FLOAT);
    dset_imag.write(arr_imag.data(), PredType::NATIVE_FLOAT);

    file.close();
}


// Save thermal field T[NZ,NY,NX] — layout IDX3(ix,iy,iz)=iz*NX*NY+iy*NX+ix
void save_thermal_h5(const rVech_t& T_host, const std::string& filename) {
    using namespace H5;
    hsize_t dims[3] = { NZ, NY, NX };
    H5File    file(filename, H5F_ACC_TRUNC);
    DataSpace ds(3, dims);
    DataSet   dset = file.createDataSet("T", PredType::NATIVE_FLOAT, ds);
    dset.write(T_host.data(), PredType::NATIVE_FLOAT);
    file.close();
}


// Save (NZ, NY, NX) intensity volume — layout: vol[iz*NY*NX + iy*NX + ix]
void save_volume_h5(const std::vector<real_t>& vol, const std::string& filename) {
    using namespace H5;
    hsize_t dims[3] = { NZ, NY, NX };
    H5File    file(filename, H5F_ACC_TRUNC);
    DataSpace ds(3, dims);
    DataSet   dset = file.createDataSet("intensity", PredType::NATIVE_FLOAT, ds);
    dset.write(vol.data(), PredType::NATIVE_FLOAT);
    file.close();
}


void save_output_slices_XY(EFields *A, json config) {
    bool save_XY = config["save_mode"]["save_fields_XY_times_slides"].get<bool>();
    bool save_p  = config["save_mode"]["save_pump"].get<bool>();
    bool save_s  = config["save_mode"]["save_signal"].get<bool>();
    bool save_i  = config["save_mode"]["save_idler"].get<bool>();

    if (save_XY && save_p)
        save_matrix_complex_h5_time(A->Ap, "pump_output_XY.h5");
    if (save_XY && save_s)
        save_matrix_complex_h5_time(A->As, "signal_output_XY.h5");
    if (save_XY && save_i)
        save_matrix_complex_h5_time(A->Ai, "idler_output_XY.h5");
}


void save_input_pump_slices_XY (EFields *A, json config, const std::string& out_dir = ".")
{
    bool save_p = config["save_mode"].value("save_in_pump", false);
    if (save_p)
        save_matrix_complex_h5_time(A->Api, out_dir + "/pump_input_XY.h5");
}


// Save a real vector into an HDF5 file (1D dataset called "data")
void save_vector_real_h5(const rVech_t &V, const std::string &Filename)
{
    // Create HDF5 file (overwrite if it already exists)
    H5::H5File file(Filename, H5F_ACC_TRUNC);

    // Define the dataspace (1D array)
    hsize_t dim[] = { V.size() };
    H5::DataSpace dataspace(1, dim);

    // Data type
    H5::PredType datatype = H5::PredType::NATIVE_FLOAT;

    // Create dataset with the name "data"
    H5::DataSet dataset = file.createDataSet("data", datatype, dataspace);

    // Write the vector into the dataset
    dataset.write(V.data(), datatype);
}


// GPU → CPU → HDF5
void save_vector_real_h5_gpu(const rVecd_t &V, const std::string &Filename)
{
    rVech_t Vh = V;
    save_vector_real_h5(Vh, Filename);
}


// Save time.h5 and frequency.h5
void save_time_and_frequency_vectors_h5(EFields *A, json config)
{
    bool save = config["save_mode"]["save_time_and_freq_vectors"].get<bool>();

    if (save)
    {
        save_vector_real_h5_gpu(A->t, "time.h5");
        save_vector_real_h5_gpu(A->F, "frequency.h5");
    }
}


// Save all outputs
void save_all_outputs(EFields *A, json config)
{
    // Save input pump field
    save_input_pump_slices_XY(A, config);

    // Save output fields
    save_output_slices_XY(A, config);

    // Save time and frequency vectors
    save_time_and_frequency_vectors_h5(A, config);

    return;
}

#endif // -> #ifdef _SAVEOUTPUTS