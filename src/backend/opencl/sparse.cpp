/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <sparse.hpp>
#include <kernel/sparse.hpp>

#include <stdexcept>
#include <string>

#include <arith.hpp>
#include <cast.hpp>
#include <complex.hpp>
#include <err_common.hpp>
#include <lookup.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <reduce.hpp>
#include <where.hpp>

namespace opencl
{

using namespace common;

////////////////////////////////////////////////////////////////////////////////
// clSPARSE Setup and Teardown Manager
// This gets initialized in opencl/sparse.cpp
////////////////////////////////////////////////////////////////////////////////
class clSPARSEManager
{
    public:
    static clsparseControl control;

    clSPARSEManager()
    {
        CLSPARSE_CHECK(clsparseSetup());
        clsparseCreateResult createResult = clsparseCreateControl(getQueue()());
        control = (createResult.status == clsparseSuccess) ? createResult.control : nullptr;
    }

    ~clSPARSEManager()
    {
        CLSPARSE_CHECK(clsparseReleaseControl(control));
        control = nullptr;
        CLSPARSE_CHECK(clsparseTeardown());
    }
};
////////////////////////////////////////////////////////////////////////////////

clsparseControl clSPARSEManager::control = nullptr;

// Instantiate clSPARSEManager
void clSPARSEInit()
{
    static clSPARSEManager manager = clSPARSEManager();
}

clsparseControl getControl()
{
    return clSPARSEManager::control;
}

////////////////////////////////////////////////////////////////////////////////
#define SPARSE_FUNC_DEF(NAME)                                               \
template<typename T>                                                        \
struct NAME##_func;

#define SPARSE_FUNC(NAME, TYPE, PREFIX)                                     \
template<>                                                                  \
struct NAME##_func<TYPE>                                                    \
{                                                                           \
    template<typename... Args>                                              \
    clsparseStatus                                                          \
    operator() (Args... args) { return clsparse##PREFIX##NAME(args...); }   \
};

// Dense -> CSR
SPARSE_FUNC_DEF(dense2csr)
SPARSE_FUNC(dense2csr, float,      S)
SPARSE_FUNC(dense2csr, double,     D)
// TODO
// Fix this. clSPARSE does not have functions for C and Z
SPARSE_FUNC(dense2csr, cfloat,     S)
SPARSE_FUNC(dense2csr, cdouble,    D)

// CSR -> Dense
SPARSE_FUNC_DEF(csr2dense)
SPARSE_FUNC(csr2dense, float,      S)
SPARSE_FUNC(csr2dense, double,     D)
// TODO
// Fix this. clSPARSE does not have functions for C and Z
SPARSE_FUNC(csr2dense, cfloat,     S)
SPARSE_FUNC(csr2dense, cdouble,    D)

// CSR -> COO
SPARSE_FUNC_DEF(csr2coo)
SPARSE_FUNC(csr2coo, float,      S)
SPARSE_FUNC(csr2coo, double,     D)
// TODO
// Fix this. clSPARSE does not have functions for C and Z
SPARSE_FUNC(csr2coo, cfloat,     S)
SPARSE_FUNC(csr2coo, cdouble,    D)

// COO -> CSR
SPARSE_FUNC_DEF(coo2csr)
SPARSE_FUNC(coo2csr, float,      S)
SPARSE_FUNC(coo2csr, double,     D)
// TODO
// Fix this. clSPARSE does not have functions for C and Z
SPARSE_FUNC(coo2csr, cfloat,     S)
SPARSE_FUNC(coo2csr, cdouble,    D)

#undef SPARSE_FUNC_DEF
#undef SPARSE_FUNC

// Partial template specialization of sparseConvertDenseToStorage for COO
// However, template specialization is not allowed
template<typename T>
SparseArray<T> sparseConvertDenseToCOO(const Array<T> &in)
{
    in.eval();

    Array<uint> nonZeroIdx_ = where<T>(in);
    Array<int> nonZeroIdx = cast<int, uint>(nonZeroIdx_);

    dim_t nNZ = nonZeroIdx.elements();

    Array<int> constNNZ = createValueArray<int>(dim4(nNZ), nNZ);
    constNNZ.eval();

    Array<int> rowIdx = arithOp<int, af_mod_t>(nonZeroIdx, constNNZ, nonZeroIdx.dims());
    Array<int> colIdx = arithOp<int, af_div_t>(nonZeroIdx, constNNZ, nonZeroIdx.dims());
    Array<T>   values = lookup<T, int>(in, nonZeroIdx, 0);

    return createArrayDataSparseArray<T>(in.dims(), values, rowIdx, colIdx, AF_STORAGE_COO);
}

template<typename T, af_storage stype>
SparseArray<T> sparseConvertDenseToStorage(const Array<T> &in_)
{
    clSPARSEInit();

    in_.eval();

    uint nNZ = reduce_all<af_notzero_t, T, uint>(in_);

    SparseArray<T> sparse_ = createEmptySparseArray<T>(in_.dims(), nNZ, stype);
    sparse_.eval();

    // Assign to clSparse Dense
    cldenseMatrix clDenseMat;
    CLSPARSE_CHECK(cldenseInitMatrix(&clDenseMat));
    clDenseMat.values = (*in_.get())();
    clDenseMat.num_rows = in_.dims()[0];
    clDenseMat.num_cols = in_.dims()[1];
    clDenseMat.lead_dim = in_.strides()[1];

    // Assign to clSparse CSR
    clsparseCsrMatrix clSparseMat;
    CLSPARSE_CHECK(clsparseInitCsrMatrix(&clSparseMat));

    clSparseMat.values      = (*sparse_.getValues().get())();
    clSparseMat.row_pointer = (*sparse_.getRowIdx().get())();
    clSparseMat.col_indices = (*sparse_.getColIdx().get())();
    clSparseMat.num_rows = in_.dims()[0];
    clSparseMat.num_cols = in_.dims()[1];
    clSparseMat.num_nonzeros = nNZ;

    if(stype == AF_STORAGE_CSR)
        CLSPARSE_CHECK(dense2csr_func<T>()(&clDenseMat, &clSparseMat, getControl()));
    else
        AF_ERROR("OpenCL Backend only supports Dense to CSR or COO", AF_ERR_NOT_SUPPORTED);

    return sparse_;
}

// Partial template specialization of sparseConvertStorageToDense for COO
// However, template specialization is not allowed
template<typename T>
Array<T> sparseConvertCOOToDense(const SparseArray<T> &in)
{
    in.eval();

    Array<T> dense = createValueArray<T>(in.dims(), scalar<T>(0));
    dense.eval();

    const Array<T>   values = in.getValues();
    const Array<int> rowIdx = in.getRowIdx();
    const Array<int> colIdx = in.getColIdx();

    kernel::coo2dense<T>(dense, values, rowIdx, colIdx);

    return dense;
}

template<typename T, af_storage stype>
Array<T> sparseConvertStorageToDense(const SparseArray<T> &in_)
{
    clSPARSEInit();

    if(stype != AF_STORAGE_CSR)
        AF_ERROR("OpenCL Backend only supports CSR or COO to Dense", AF_ERR_NOT_SUPPORTED);

    in_.eval();

    Array<T> dense_ = createValueArray<T>(in_.dims(), scalar<T>(0));
    dense_.eval();

    // Assign to clSparse CSR
    clsparseCsrMatrix clSparseMat;
    CLSPARSE_CHECK(clsparseInitCsrMatrix(&clSparseMat));

    clSparseMat.values      = (*in_.getValues().get())();
    clSparseMat.row_pointer = (*in_.getRowIdx().get())();
    clSparseMat.col_indices = (*in_.getColIdx().get())();
    clSparseMat.num_rows = in_.dims()[0];
    clSparseMat.num_cols = in_.dims()[1];
    clSparseMat.num_nonzeros = in_.getNNZ();

    // Assign to clSparse Dense
    cldenseMatrix clDenseMat;
    CLSPARSE_CHECK(cldenseInitMatrix(&clDenseMat));
    clDenseMat.values = (*dense_.get())();
    clDenseMat.num_rows = dense_.dims()[0];
    clDenseMat.num_cols = dense_.dims()[1];
    clDenseMat.lead_dim = dense_.strides()[1];

    if(stype == AF_STORAGE_CSR)
        CLSPARSE_CHECK(csr2dense_func<T>()(&clSparseMat, &clDenseMat, getControl()));
    else
        AF_ERROR("OpenCL Backend only supports CSR or COO to Dense", AF_ERR_NOT_SUPPORTED);

    return dense_;
}

template<typename T, af_storage src, af_storage dest>
SparseArray<T> sparseConvertStorageToStorage(const SparseArray<T> &in)
{
    // TODO
    // Convert CSR <-> CSC <-> COO <-> CSR
    // Currently supports CSR <-> COO

    // If src and dest are the same, simply return.
    if(src == dest)
        return in;

    clSPARSEInit();

    in.eval();

    SparseArray<T> out = createEmptySparseArray<T>(in.dims(), (int)in.getNNZ(), dest);
    out.eval();

    clsparseCsrMatrix csrMat;
    CLSPARSE_CHECK(clsparseInitCsrMatrix(&csrMat));

    clsparseCooMatrix cooMat;
    CLSPARSE_CHECK(clsparseInitCooMatrix(&cooMat));

    csrMat.num_rows = in.dims()[0];
    csrMat.num_cols = in.dims()[1];
    csrMat.num_nonzeros = in.getNNZ();

    cooMat.num_rows = in.dims()[0];
    cooMat.num_cols = in.dims()[1];
    cooMat.num_nonzeros = in.getNNZ();

    if(src == AF_STORAGE_CSR && dest == AF_STORAGE_COO) {
        csrMat.values      = (*in.getValues().get())();
        csrMat.row_pointer = (*in.getRowIdx().get())();
        csrMat.col_indices = (*in.getColIdx().get())();

        cooMat.values      = (*out.getValues().get())();
        cooMat.row_indices = (*out.getRowIdx().get())();
        cooMat.col_indices = (*out.getColIdx().get())();

        CLSPARSE_CHECK(csr2coo_func<T>()(&csrMat, &cooMat, getControl()));
    } else if(src == AF_STORAGE_COO && dest == AF_STORAGE_CSR) {
        cooMat.values      = (*in.getValues().get())();
        cooMat.row_indices = (*in.getRowIdx().get())();
        cooMat.col_indices = (*in.getColIdx().get())();

        csrMat.values      = (*out.getValues().get())();
        csrMat.row_pointer = (*out.getRowIdx().get())();
        csrMat.col_indices = (*out.getColIdx().get())();

        CLSPARSE_CHECK(coo2csr_func<T>()(&cooMat, &csrMat, getControl()));
    }

    return out;
}


#define INSTANTIATE_TO_STORAGE(T, S)                                                                        \
    template SparseArray<T> sparseConvertStorageToStorage<T, S, AF_STORAGE_CSR>(const SparseArray<T> &in);  \
    template SparseArray<T> sparseConvertStorageToStorage<T, S, AF_STORAGE_CSC>(const SparseArray<T> &in);  \
    template SparseArray<T> sparseConvertStorageToStorage<T, S, AF_STORAGE_COO>(const SparseArray<T> &in);  \

#define INSTANTIATE_COO_SPECIAL(T)                                                                      \
    template<> SparseArray<T> sparseConvertDenseToStorage<T, AF_STORAGE_COO>(const Array<T> &in)        \
    { return sparseConvertDenseToCOO<T>(in); }                                                          \
    template<> Array<T> sparseConvertStorageToDense<T, AF_STORAGE_COO>(const SparseArray<T> &in)        \
    { return sparseConvertCOOToDense<T>(in); }                                                          \

#define INSTANTIATE_SPARSE(T)                                                                           \
    template SparseArray<T> sparseConvertDenseToStorage<T, AF_STORAGE_CSR>(const Array<T> &in);         \
    template SparseArray<T> sparseConvertDenseToStorage<T, AF_STORAGE_CSC>(const Array<T> &in);         \
                                                                                                        \
    template Array<T> sparseConvertStorageToDense<T, AF_STORAGE_CSR>(const SparseArray<T> &in);         \
    template Array<T> sparseConvertStorageToDense<T, AF_STORAGE_CSC>(const SparseArray<T> &in);         \
                                                                                                        \
    INSTANTIATE_COO_SPECIAL(T)                                                                          \
                                                                                                        \
    INSTANTIATE_TO_STORAGE(T, AF_STORAGE_CSR)                                                           \
    INSTANTIATE_TO_STORAGE(T, AF_STORAGE_CSC)                                                           \
    INSTANTIATE_TO_STORAGE(T, AF_STORAGE_COO)                                                           \


INSTANTIATE_SPARSE(float)
INSTANTIATE_SPARSE(double)
INSTANTIATE_SPARSE(cfloat)
INSTANTIATE_SPARSE(cdouble)

#undef INSTANTIATE_TO_STORAGE
#undef INSTANTIATE_COO_SPECIAL
#undef INSTANTIATE_SPARSE

}
