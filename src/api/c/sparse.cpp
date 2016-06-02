/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/sparse.h>
#include <af/array.h>
#include <af/algorithm.h>
#include <sparse_handle.hpp>
#include <sparse.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <err_common.hpp>
#include <arith.hpp>
#include <lookup.hpp>
#include <platform.hpp>

using namespace detail;
using namespace common;
using af::dim4;

const SparseArrayBase& getSparseArrayBase(const af_array in, bool device_check)
{
    const SparseArrayBase *base = static_cast<SparseArrayBase*>(reinterpret_cast<void *>(in));

    if(!base->isSparse()) {
        AF_ERROR("Input is not a SparseArray and cannot be used in Sparse functions",
                 AF_ERR_ARG);
    }

    if (device_check && base->getDevId() != detail::getActiveDeviceId()) {
        AF_ERROR("Input Array not created on current device", AF_ERR_DEVICE);
    }

    return *base;
}

////////////////////////////////////////////////////////////////////////////////
// Sparse Creation
////////////////////////////////////////////////////////////////////////////////
template<typename T>
af_array createSparseArray(const af::dim4 &dims, const af_array values,
                           const af_array rowIdx, const af_array colIdx,
                           const af::sparseStorage storage)
{
    SparseArray<T> sparse = common::createArrayDataSparseArray(
                            dims, getArray<T>(values),
                            getArray<int>(rowIdx), getArray<int>(colIdx),
                            storage);
    return getHandle(sparse);
}

af_err af_create_sparse_array(
                 af_array *out,
                 const dim_t nRows, const dim_t nCols, const dim_t nNZ,
                 const af_array values, const af_array rowIdx, const af_array colIdx,
                 const af_sparse_storage storage)
{
    try {
        // Checks:
        // rowIdx and colIdx arrays are of s32 type
        // values is of floating point type
        // if COO, rowIdx, colIdx and values should have same dims
        // if CRS, colIdx and values should have same dims, rowIdx.dims = nRows
        // if CRC, rowIdx and values should have same dims, colIdx.dims = nCols
        // storage is within acceptable range
        // type is floating type

        if(!(storage == AF_SPARSE_CSR
          || storage == AF_SPARSE_CSC
          || storage == AF_SPARSE_COO)) {
            AF_ERROR("Storage type is out of range/unsupported", AF_ERR_ARG);
        }

        ArrayInfo vInfo = getInfo(values);
        ArrayInfo rInfo = getInfo(rowIdx);
        ArrayInfo cInfo = getInfo(colIdx);

        TYPE_ASSERT(vInfo.isFloating());
        DIM_ASSERT(4, vInfo.isLinear());
        DIM_ASSERT(5, rInfo.isLinear());
        DIM_ASSERT(6, cInfo.isLinear());

        af_array output = 0;

        af::dim4 dims(nRows, nCols);

        switch(vInfo.getType()) {
            case f32: output = createSparseArray<float  >(dims, values, rowIdx, colIdx, storage); break;
            case f64: output = createSparseArray<double >(dims, values, rowIdx, colIdx, storage); break;
            case c32: output = createSparseArray<cfloat >(dims, values, rowIdx, colIdx, storage); break;
            case c64: output = createSparseArray<cdouble>(dims, values, rowIdx, colIdx, storage); break;
            default : TYPE_ERROR(1, vInfo.getType());
        }
        std::swap(*out, output);

    } CATCHALL;

    return AF_SUCCESS;
}

template<typename T>
af_array createSparseArrayFromPtr(
        const af::dim4 &dims, const dim_t nNZ,
        const T * const values, const int * const rowIdx, const int * const colIdx,
        const af::sparseStorage storage, const af::source source)
{
    SparseArray<T> sparse = createEmptySparseArray<T>(dims, nNZ, storage);

    if(source == afHost)
        sparse = common::createHostDataSparseArray(
                         dims, nNZ, values, rowIdx, colIdx, storage);
    else if (source == afDevice)
        sparse = common::createDeviceDataSparseArray(
                         dims, nNZ, values, rowIdx, colIdx, storage);

    return getHandle(sparse);
}

af_err af_create_sparse_array_from_ptr(
                 af_array *out,
                 const dim_t nRows, const dim_t nCols, const dim_t nNZ,
                 const void * const values, const int * const rowIdx, const int * const colIdx,
                 const af_dtype type, const af_sparse_storage storage,
                 const af_source source)
{
    try {
        // Checks:
        // rowIdx and colIdx arrays are of s32 type
        // values is of floating point type
        // if COO, rowIdx, colIdx and values should have same dims
        // if CRS, colIdx and values should have same dims, rowIdx.dims = nRows
        // if CRC, rowIdx and values should have same dims, colIdx.dims = nCols
        // storage is within acceptable range
        // type is floating type
        if(!(storage == AF_SPARSE_CSR
          || storage == AF_SPARSE_CSC
          || storage == AF_SPARSE_COO)) {
            AF_ERROR("Storage type is out of range/unsupported", AF_ERR_ARG);
        }

        TYPE_ASSERT(type == f32 || type == f64
                 || type == c32 || type == c64);


        af_array output = 0;

        af::dim4 dims(nRows, nCols);

        switch(type) {
            case f32: output = createSparseArrayFromPtr<float  >
                               (dims, nNZ, static_cast<const float  *>(values), rowIdx, colIdx, storage, source);
                      break;
            case f64: output = createSparseArrayFromPtr<double >
                               (dims, nNZ, static_cast<const double *>(values), rowIdx, colIdx, storage, source);
                      break;
            case c32: output = createSparseArrayFromPtr<cfloat >
                               (dims, nNZ, static_cast<const cfloat *>(values), rowIdx, colIdx, storage, source);
                      break;
            case c64: output = createSparseArrayFromPtr<cdouble>
                               (dims, nNZ, static_cast<const cdouble *>(values), rowIdx, colIdx, storage, source);
                      break;
            default : TYPE_ERROR(1, type);
        }
        std::swap(*out, output);

    } CATCHALL;

    return AF_SUCCESS;
}

template<typename T>
af_array createSparseArrayFromDense(
        const af::dim4 &dims, const af_array _in,
        const af_sparse_storage storage)
{
    const Array<T> in = getArray<T>(_in);

    switch(storage) {
        case AF_SPARSE_CSR:
            return getHandle(sparseConvertDenseToStorage<T, AF_SPARSE_CSR>(in));
        case AF_SPARSE_CSC:
            return getHandle(sparseConvertDenseToStorage<T, AF_SPARSE_CSC>(in));
        case AF_SPARSE_COO:
            return getHandle(sparseConvertDenseToStorage<T, AF_SPARSE_COO>(in));
        default: AF_ERROR("Storage type is out of range/unsupported", AF_ERR_ARG);
    }
}

af_err af_create_sparse_array_from_dense(af_array *out, const af_array in,
                                         const af_sparse_storage storage)
{
    try {
        // Checks:
        // storage is within acceptable range
        // values is of floating point type

        ArrayInfo info = getInfo(in);

        if(!(storage == AF_SPARSE_CSR
          || storage == AF_SPARSE_CSC
          || storage == AF_SPARSE_COO)) {
            AF_ERROR("Storage type is out of range/unsupported", AF_ERR_ARG);
        }

        // Only matrices allowed
        DIM_ASSERT(1, info.ndims() == 2);

        TYPE_ASSERT(info.isFloating());

        af::dim4 dims(info.dims()[0], info.dims()[1]);

        af_array output = 0;

        switch(info.getType()) {
            case f32: output = createSparseArrayFromDense<float  >(dims, in, storage); break;
            case f64: output = createSparseArrayFromDense<double >(dims, in, storage); break;
            case c32: output = createSparseArrayFromDense<cfloat >(dims, in, storage); break;
            case c64: output = createSparseArrayFromDense<cdouble>(dims, in, storage); break;
            default: TYPE_ERROR(1, info.getType());
        }
        std::swap(*out, output);

    } CATCHALL;

    return AF_SUCCESS;
}

template<typename T>
af_array sparseConvertStorage(const af_array in_, const af_sparse_storage destStorage)
{
    const SparseArray<T> in = getSparseArray<T>(in_);

    // Only destStorage == AF_SPARSE_DENSE is supported
    // All the other calls are for future when conversions are supported in
    // the backend
    if(destStorage == AF_SPARSE_DENSE) {
        // Returns a regular af_array, not sparse
        switch(in.getStorage()) {
            case AF_SPARSE_CSR:
                return getHandle(detail::sparseConvertStorageToDense<T, AF_SPARSE_CSR>(in));
            case AF_SPARSE_CSC:
                return getHandle(detail::sparseConvertStorageToDense<T, AF_SPARSE_CSC>(in));
            case AF_SPARSE_COO:
                return getHandle(detail::sparseConvertStorageToDense<T, AF_SPARSE_COO>(in));
            default:
                AF_ERROR("Invalid storage type of input array", AF_ERR_ARG);
        }
    } else if(destStorage == AF_SPARSE_CSR) {
        // Returns a sparse af_array
        switch(in.getStorage()) {
            case AF_SPARSE_CSR:
                return retainSparseHandle<T>(in_);
            case AF_SPARSE_CSC:
                return getHandle(detail::sparseConvertStorageToStorage<T, AF_SPARSE_CSR, AF_SPARSE_CSC>(in));
            case AF_SPARSE_COO:
                return getHandle(detail::sparseConvertStorageToStorage<T, AF_SPARSE_CSR, AF_SPARSE_COO>(in));
            default:
                AF_ERROR("Invalid storage type of input array", AF_ERR_ARG);
        }
    } else if(destStorage == AF_SPARSE_CSC) {
        // Returns a sparse af_array
        switch(in.getStorage()) {
            case AF_SPARSE_CSR:
                return getHandle(detail::sparseConvertStorageToStorage<T, AF_SPARSE_CSC, AF_SPARSE_CSR>(in));
            case AF_SPARSE_CSC:
                return retainSparseHandle<T>(in_);
            case AF_SPARSE_COO:
                return getHandle(detail::sparseConvertStorageToStorage<T, AF_SPARSE_CSC, AF_SPARSE_COO>(in));
            default:
                AF_ERROR("Invalid storage type of input array", AF_ERR_ARG);
        }
    } else if(destStorage == AF_SPARSE_COO) {
        // Returns a sparse af_array
        switch(in.getStorage()) {
            case AF_SPARSE_CSR:
                return getHandle(detail::sparseConvertStorageToStorage<T, AF_SPARSE_COO, AF_SPARSE_CSR>(in));
            case AF_SPARSE_CSC:
                return getHandle(detail::sparseConvertStorageToStorage<T, AF_SPARSE_COO, AF_SPARSE_CSC>(in));
            case AF_SPARSE_COO:
                return retainSparseHandle<T>(in_);
            default:
                AF_ERROR("Invalid storage type of input array", AF_ERR_ARG);
        }
    }

    // Shoud never come here
    return NULL;
}

af_err af_sparse_convert_storage(af_array *out, const af_array in,
                           const af_sparse_storage destStorage)
{
    // Right now dest_storage can only be AF_SPARSE_DENSE
    try {
        af_array output = 0;

        const SparseArrayBase base = getSparseArrayBase(in);

        // Dense not allowed as input -> Should never happen
        // To convert from dense to type, use the create* functions
        ARG_ASSERT(1, base.getStorage() != AF_SPARSE_DENSE);

        // Right now dest_storage can only be AF_SPARSE_DENSE
        // TODO: Add support for [CSR, CSC, COO] <-> [CSR, CSC, COO] in backends
        ARG_ASSERT(1, destStorage == AF_SPARSE_DENSE);

        if(base.getStorage() == destStorage) {
            // Return a reference
            AF_CHECK(af_retain_array(out, in));
            return AF_SUCCESS;
        }

        switch(base.getType()) {
            case f32: output = sparseConvertStorage<float  >(in, destStorage); break;
            case f64: output = sparseConvertStorage<double >(in, destStorage); break;
            case c32: output = sparseConvertStorage<cfloat >(in, destStorage); break;
            case c64: output = sparseConvertStorage<cdouble>(in, destStorage); break;
            default : AF_ERROR("Output storage type is not valid", AF_ERR_ARG);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
// Get Functions
////////////////////////////////////////////////////////////////////////////////
template<typename T>
af_array getSparseValues(const af_array in)
{
    return getHandle(getSparseArray<T>(in).getValues());
}

af_err af_sparse_get_arrays(af_array *values, af_array *rows, af_array *cols,
                      const af_array in)
{
    try {
        if(values != NULL) AF_CHECK(af_sparse_get_values(values, in));
        if(rows   != NULL) AF_CHECK(af_sparse_get_row_idx(rows  , in));
        if(cols   != NULL) AF_CHECK(af_sparse_get_col_idx(cols  , in));
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_sparse_get_values(af_array *out, const af_array in)
{
    try{
        const SparseArrayBase base = getSparseArrayBase(in);

        af_array output = 0;

        switch(base.getType()) {
            case f32: output = getSparseValues<float  >(in); break;
            case f64: output = getSparseValues<double >(in); break;
            case c32: output = getSparseValues<cfloat >(in); break;
            case c64: output = getSparseValues<cdouble>(in); break;
            default : TYPE_ERROR(1, base.getType());
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_sparse_get_row_idx(af_array *out, const af_array in)
{
    try {
        const SparseArrayBase base = getSparseArrayBase(in);
        *out = getHandle(base.getRowIdx());
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_sparse_get_col_idx(af_array *out, const af_array in)
{
    try {
        const SparseArrayBase base = getSparseArrayBase(in);
        *out = getHandle(base.getColIdx());
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_sparse_get_num_nonzero(dim_t *out, const af_array in)
{
    try {
        const SparseArrayBase base = getSparseArrayBase(in);
        *out = base.getNNZ();
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_sparse_get_storage(af_sparse_storage *out, const af_array in)
{
    try {
        const SparseArrayBase base = getSparseArrayBase(in);
        *out = base.getStorage();
    } CATCHALL;
    return AF_SUCCESS;
}
