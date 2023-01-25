/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arith.hpp>
#include <backend.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <lookup.hpp>
#include <platform.hpp>
#include <sparse.hpp>
#include <sparse_handle.hpp>
#include <af/algorithm.h>
#include <af/array.h>
#include <af/sparse.h>

using af::dim4;
using arrayfire::getSparseArray;
using arrayfire::retainSparseHandle;
using arrayfire::common::createArrayDataSparseArray;
using arrayfire::common::createDeviceDataSparseArray;
using arrayfire::common::createEmptySparseArray;
using arrayfire::common::createHostDataSparseArray;
using arrayfire::common::SparseArray;
using arrayfire::common::SparseArrayBase;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::sparseConvertDenseToStorage;

namespace arrayfire {

const SparseArrayBase &getSparseArrayBase(const af_array in,
                                          bool device_check) {
    const SparseArrayBase *base =
        static_cast<SparseArrayBase *>(static_cast<void *>(in));

    if (!base->isSparse()) {
        AF_ERROR(
            "Input is not a SparseArray and cannot be used in Sparse functions",
            AF_ERR_ARG);
    }

    if (device_check &&
        base->getDevId() != static_cast<int>(detail::getActiveDeviceId())) {
        AF_ERROR("Input Array not created on current device", AF_ERR_DEVICE);
    }

    return *base;
}

////////////////////////////////////////////////////////////////////////////////
// Sparse Creation
////////////////////////////////////////////////////////////////////////////////
template<typename T>
af_array createSparseArrayFromData(const dim4 &dims, const af_array values,
                                   const af_array rowIdx, const af_array colIdx,
                                   const af::storage stype) {
    SparseArray<T> sparse = createArrayDataSparseArray(
        dims, getArray<T>(values), getArray<int>(rowIdx), getArray<int>(colIdx),
        stype);
    return getHandle(sparse);
}

template<typename T>
af_array createSparseArrayFromPtr(const af::dim4 &dims, const dim_t nNZ,
                                  const T *const values,
                                  const int *const rowIdx,
                                  const int *const colIdx,
                                  const af::storage stype,
                                  const af::source source) {
    if (nNZ) {
        switch (source) {
            case afHost:
                return getHandle(createHostDataSparseArray(
                    dims, nNZ, values, rowIdx, colIdx, stype));
                break;
            case afDevice:
                return getHandle(createDeviceDataSparseArray(
                    dims, nNZ, const_cast<T *>(values),
                    const_cast<int *>(rowIdx), const_cast<int *>(colIdx),
                    stype));
                break;
        }
    }

    return getHandle(createEmptySparseArray<T>(dims, nNZ, stype));
}

template<typename T>
af_array createSparseArrayFromDense(const af_array _in,
                                    const af_storage stype) {
    const Array<T> in = getArray<T>(_in);

    switch (stype) {
        case AF_STORAGE_CSR:
            return getHandle(
                sparseConvertDenseToStorage<T, AF_STORAGE_CSR>(in));
        case AF_STORAGE_COO:
            return getHandle(
                sparseConvertDenseToStorage<T, AF_STORAGE_COO>(in));
        case AF_STORAGE_CSC:
            // return getHandle(sparseConvertDenseToStorage<T,
            // AF_STORAGE_CSC>(in));
        default:
            AF_ERROR("Storage type is out of range/unsupported", AF_ERR_ARG);
    }
}

template<typename T>
af_array sparseConvertStorage(const af_array in_,
                              const af_storage destStorage) {
    const SparseArray<T> in = getSparseArray<T>(in_);

    if (destStorage == AF_STORAGE_DENSE) {
        // Returns a regular af_array, not sparse
        switch (in.getStorage()) {
            case AF_STORAGE_CSR:
                return getHandle(
                    detail::sparseConvertStorageToDense<T, AF_STORAGE_CSR>(in));
            case AF_STORAGE_COO:
                return getHandle(
                    detail::sparseConvertStorageToDense<T, AF_STORAGE_COO>(in));
            default:
                AF_ERROR("Invalid storage type of input array", AF_ERR_ARG);
        }
    } else if (destStorage == AF_STORAGE_CSR) {
        // Returns a sparse af_array
        switch (in.getStorage()) {
            case AF_STORAGE_CSR: return retainSparseHandle<T>(in_);
            case AF_STORAGE_COO:
                return getHandle(
                    detail::sparseConvertStorageToStorage<T, AF_STORAGE_CSR,
                                                          AF_STORAGE_COO>(in));
            default:
                AF_ERROR("Invalid storage type of input array", AF_ERR_ARG);
        }
    } else if (destStorage == AF_STORAGE_COO) {
        // Returns a sparse af_array
        switch (in.getStorage()) {
            case AF_STORAGE_CSR:
                return getHandle(
                    detail::sparseConvertStorageToStorage<T, AF_STORAGE_COO,
                                                          AF_STORAGE_CSR>(in));
            case AF_STORAGE_COO: return retainSparseHandle<T>(in_);
            default:
                AF_ERROR("Invalid storage type of input array", AF_ERR_ARG);
        }
    }

    // Shoud never come here
    return NULL;
}

////////////////////////////////////////////////////////////////////////////////
// Get Functions
////////////////////////////////////////////////////////////////////////////////
template<typename T>
af_array getSparseValues(const af_array in) {
    return getHandle(getSparseArray<T>(in).getValues());
}

}  // namespace arrayfire

using arrayfire::createSparseArrayFromData;
using arrayfire::createSparseArrayFromDense;
using arrayfire::createSparseArrayFromPtr;
using arrayfire::getSparseArrayBase;
using arrayfire::getSparseValues;
using arrayfire::sparseConvertStorage;

af_err af_create_sparse_array(af_array *out, const dim_t nRows,
                              const dim_t nCols, const af_array values,
                              const af_array rowIdx, const af_array colIdx,
                              const af_storage stype) {
    try {
        // Checks:
        // rowIdx and colIdx arrays are of s32 type
        // values is of floating point type
        // if COO, rowIdx, colIdx and values should have same dims
        // if CRS, colIdx and values should have same dims, rowIdx.dims = nRows
        // if CRC, rowIdx and values should have same dims, colIdx.dims = nCols
        // stype is within acceptable range
        // type is floating type

        if (!(stype == AF_STORAGE_CSR || stype == AF_STORAGE_CSC ||
              stype == AF_STORAGE_COO)) {
            AF_ERROR("Storage type is out of range/unsupported", AF_ERR_ARG);
        }

        const ArrayInfo &vInfo = getInfo(values);
        const ArrayInfo &rInfo = getInfo(rowIdx);
        const ArrayInfo &cInfo = getInfo(colIdx);

        TYPE_ASSERT(vInfo.isFloating());
        DIM_ASSERT(3, vInfo.isLinear());
        ARG_ASSERT(4, rInfo.getType() == s32);
        DIM_ASSERT(4, rInfo.isLinear());
        ARG_ASSERT(5, cInfo.getType() == s32);
        DIM_ASSERT(5, cInfo.isLinear());

        const dim_t nNZ = vInfo.elements();
        if (stype == AF_STORAGE_COO) {
            DIM_ASSERT(4, rInfo.elements() == nNZ);
            DIM_ASSERT(5, cInfo.elements() == nNZ);
        } else if (stype == AF_STORAGE_CSR) {
            DIM_ASSERT(4, (dim_t)rInfo.elements() == nRows + 1);
            DIM_ASSERT(5, cInfo.elements() == nNZ);
        } else if (stype == AF_STORAGE_CSC) {
            DIM_ASSERT(4, rInfo.elements() == nNZ);
            DIM_ASSERT(5, (dim_t)cInfo.elements() == nCols + 1);
        }

        af_array output = nullptr;

        dim4 dims(nRows, nCols);

        switch (vInfo.getType()) {
            case f32:
                output = createSparseArrayFromData<float>(dims, values, rowIdx,
                                                          colIdx, stype);
                break;
            case f64:
                output = createSparseArrayFromData<double>(dims, values, rowIdx,
                                                           colIdx, stype);
                break;
            case c32:
                output = createSparseArrayFromData<cfloat>(dims, values, rowIdx,
                                                           colIdx, stype);
                break;
            case c64:
                output = createSparseArrayFromData<cdouble>(
                    dims, values, rowIdx, colIdx, stype);
                break;
            default: TYPE_ERROR(1, vInfo.getType());
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_create_sparse_array_from_ptr(
    af_array *out, const dim_t nRows, const dim_t nCols, const dim_t nNZ,
    const void *const values, const int *const rowIdx, const int *const colIdx,
    const af_dtype type, const af_storage stype, const af_source source) {
    try {
        // Checks:
        // rowIdx and colIdx arrays are of s32 type
        // values is of floating point type
        // if COO, rowIdx, colIdx and values should have same dims
        // if CRS, colIdx and values should have same dims, rowIdx.dims = nRows
        // if CRC, rowIdx and values should have same dims, colIdx.dims = nCols
        // stype is within acceptable range
        // type is floating type
        if (!(stype == AF_STORAGE_CSR || stype == AF_STORAGE_CSC ||
              stype == AF_STORAGE_COO)) {
            AF_ERROR("Storage type is out of range/unsupported", AF_ERR_ARG);
        }

        TYPE_ASSERT(type == f32 || type == f64 || type == c32 || type == c64);

        af_array output = nullptr;

        dim4 dims(nRows, nCols);

        switch (type) {
            case f32:
                output = createSparseArrayFromPtr<float>(
                    dims, nNZ, static_cast<const float *>(values), rowIdx,
                    colIdx, stype, source);
                break;
            case f64:
                output = createSparseArrayFromPtr<double>(
                    dims, nNZ, static_cast<const double *>(values), rowIdx,
                    colIdx, stype, source);
                break;
            case c32:
                output = createSparseArrayFromPtr<cfloat>(
                    dims, nNZ, static_cast<const cfloat *>(values), rowIdx,
                    colIdx, stype, source);
                break;
            case c64:
                output = createSparseArrayFromPtr<cdouble>(
                    dims, nNZ, static_cast<const cdouble *>(values), rowIdx,
                    colIdx, stype, source);
                break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_create_sparse_array_from_dense(af_array *out, const af_array in,
                                         const af_storage stype) {
    try {
        // Checks:
        // stype is within acceptable range
        // values is of floating point type

        const ArrayInfo &info = getInfo(in);

        if (!(stype == AF_STORAGE_CSR || stype == AF_STORAGE_CSC ||
              stype == AF_STORAGE_COO)) {
            AF_ERROR("Storage type is out of range/unsupported", AF_ERR_ARG);
        }

        // Only matrices allowed
        DIM_ASSERT(1, info.ndims() == 2);

        TYPE_ASSERT(info.isFloating());

        af_array output = 0;

        switch (info.getType()) {
            case f32:
                output = createSparseArrayFromDense<float>(in, stype);
                break;
            case f64:
                output = createSparseArrayFromDense<double>(in, stype);
                break;
            case c32:
                output = createSparseArrayFromDense<cfloat>(in, stype);
                break;
            case c64:
                output = createSparseArrayFromDense<cdouble>(in, stype);
                break;
            default: TYPE_ERROR(1, info.getType());
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_sparse_convert_to(af_array *out, const af_array in,
                            const af_storage destStorage) {
    try {
        // Handle dense case
        const ArrayInfo &info = getInfo(in, false, true);
        if (!info.isSparse()) {  // If input is dense
            return af_create_sparse_array_from_dense(out, in, destStorage);
        }

        af_array output = nullptr;

        const SparseArrayBase &base = getSparseArrayBase(in);

        // Dense not allowed as input -> Should never happen with
        // SparseArrayBase CSC is currently not supported
        ARG_ASSERT(1, base.getStorage() != AF_STORAGE_DENSE &&
                          base.getStorage() != AF_STORAGE_CSC);

        // Conversion to and from CSC is not supported
        ARG_ASSERT(2, destStorage != AF_STORAGE_CSC);

        if (base.getStorage() == destStorage) {
            // Return a reference
            AF_CHECK(af_retain_array(out, in));
            return AF_SUCCESS;
        }

        switch (base.getType()) {
            case f32:
                output = sparseConvertStorage<float>(in, destStorage);
                break;
            case f64:
                output = sparseConvertStorage<double>(in, destStorage);
                break;
            case c32:
                output = sparseConvertStorage<cfloat>(in, destStorage);
                break;
            case c64:
                output = sparseConvertStorage<cdouble>(in, destStorage);
                break;
            default: AF_ERROR("Output storage type is not valid", AF_ERR_ARG);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_sparse_to_dense(af_array *out, const af_array in) {
    try {
        af_array output = nullptr;

        const SparseArrayBase &base = getSparseArrayBase(in);

        // Dense not allowed as input -> Should never happen
        // To convert from dense to type, use the create* functions
        ARG_ASSERT(1, base.getStorage() != AF_STORAGE_DENSE);

        switch (base.getType()) {
            case f32:
                output = sparseConvertStorage<float>(in, AF_STORAGE_DENSE);
                break;
            case f64:
                output = sparseConvertStorage<double>(in, AF_STORAGE_DENSE);
                break;
            case c32:
                output = sparseConvertStorage<cfloat>(in, AF_STORAGE_DENSE);
                break;
            case c64:
                output = sparseConvertStorage<cdouble>(in, AF_STORAGE_DENSE);
                break;
            default: AF_ERROR("Output storage type is not valid", AF_ERR_ARG);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_sparse_get_info(af_array *values, af_array *rows, af_array *cols,
                          af_storage *stype, const af_array in) {
    try {
        if (values != NULL) { AF_CHECK(af_sparse_get_values(values, in)); }
        if (rows != NULL) { AF_CHECK(af_sparse_get_row_idx(rows, in)); }
        if (cols != NULL) { AF_CHECK(af_sparse_get_col_idx(cols, in)); }
        if (stype != NULL) { AF_CHECK(af_sparse_get_storage(stype, in)); }
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_sparse_get_values(af_array *out, const af_array in) {
    try {
        const SparseArrayBase base = getSparseArrayBase(in);

        af_array output = nullptr;

        switch (base.getType()) {
            case f32: output = getSparseValues<float>(in); break;
            case f64: output = getSparseValues<double>(in); break;
            case c32: output = getSparseValues<cfloat>(in); break;
            case c64: output = getSparseValues<cdouble>(in); break;
            default: TYPE_ERROR(1, base.getType());
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_sparse_get_row_idx(af_array *out, const af_array in) {
    try {
        const SparseArrayBase base = getSparseArrayBase(in);
        *out                       = getHandle(base.getRowIdx());
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_sparse_get_col_idx(af_array *out, const af_array in) {
    try {
        const SparseArrayBase base = getSparseArrayBase(in);
        *out                       = getHandle(base.getColIdx());
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_sparse_get_nnz(dim_t *out, const af_array in) {
    try {
        const SparseArrayBase base = getSparseArrayBase(in);
        *out                       = base.getNNZ();
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_sparse_get_storage(af_storage *out, const af_array in) {
    try {
        const SparseArrayBase base = getSparseArrayBase(in);
        *out                       = base.getStorage();
    }
    CATCHALL;
    return AF_SUCCESS;
}
