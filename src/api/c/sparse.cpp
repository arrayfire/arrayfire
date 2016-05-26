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

        if(!(storage == AF_SPARSE_COO
          || storage == AF_SPARSE_CSR
          || storage == AF_SPARSE_CSC)) {
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
        if(!(storage == AF_SPARSE_COO
          || storage == AF_SPARSE_CSR
          || storage == AF_SPARSE_CSC)) {
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
        const af::dim4 &dims, const dim_t nNZ,
        const af_array _in, const af_array _nonZeroIdx,
        const af_sparse_storage storage)
{
    Array<int> nonZeroIdx = castArray<int>(_nonZeroIdx);
    const Array<T> in = getArray<T>(_in);

    Array<int> constNNZ = createValueArray<int>(dim4(nNZ), nNZ);

    Array<int> rowIdx = *initArray<int>();
    Array<int> colIdx = *initArray<int>();
    Array<T>   values = *initArray<T>();

    if(storage == AF_SPARSE_COO) {
        rowIdx = arithOp<int, af_mod_t>(nonZeroIdx, constNNZ, nonZeroIdx.dims());
        colIdx = arithOp<int, af_div_t>(nonZeroIdx, constNNZ, nonZeroIdx.dims());
        values = lookup<T, int>(in, nonZeroIdx, 0);
    } else if(storage == AF_SPARSE_CSR) {
        dense2storage<T, AF_SPARSE_CSR>(values, rowIdx, colIdx, in);
    } else if(storage == AF_SPARSE_CSC) {
        dense2storage<T, AF_SPARSE_CSC>(values, rowIdx, colIdx, in);
    }

    SparseArray<T> sparse = common::createArrayDataSparseArray(
                            dims, values, rowIdx, colIdx, storage);

    return getHandle(sparse);
}

af_err af_create_sparse_array_from_dense(af_array *out, const af_array in,
                                         const af_sparse_storage storage)
{
    try {
        // Checks:
        // storage is within acceptable range
        // values is of floating point type

        ArrayInfo info = getInfo(in);

        if(!(storage == AF_SPARSE_COO
          || storage == AF_SPARSE_CSR
          || storage == AF_SPARSE_CSC)) {
            AF_ERROR("Storage type is out of range/unsupported", AF_ERR_ARG);
        }

        // Only matrices allowed
        DIM_ASSERT(1, info.ndims() == 2);

        TYPE_ASSERT(info.isFloating());

        af_array nonZeroIdx = 0; // Yes I know how this looks
        AF_CHECK(af_where(&nonZeroIdx, in));

        ArrayInfo nNZInfo = getInfo(nonZeroIdx);
        dim_t nNZ = nNZInfo.elements();

        af::dim4 dims(info.dims()[0], info.dims()[1]);

        af_array output = 0;

        switch(info.getType()) {
            case f32: output = createSparseArrayFromDense<float  >(dims, nNZ, in, nonZeroIdx, storage); break;
            case f64: output = createSparseArrayFromDense<double >(dims, nNZ, in, nonZeroIdx, storage); break;
            case c32: output = createSparseArrayFromDense<cfloat >(dims, nNZ, in, nonZeroIdx, storage); break;
            case c64: output = createSparseArrayFromDense<cdouble>(dims, nNZ, in, nonZeroIdx, storage); break;
            default: TYPE_ERROR(1, info.getType());
        }
        std::swap(*out, output);

        if(nonZeroIdx != 0) AF_CHECK(af_release_array(nonZeroIdx));
    } CATCHALL;

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
    CATCHALL
    return AF_SUCCESS;
}

af_err af_sparse_get_rows(af_array *out, const af_array in)
{
    try {
        const SparseArrayBase base = getSparseArrayBase(in);
        *out = getHandle(base.getRows());
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_sparse_get_cols(af_array *out, const af_array in)
{
    try {
        const SparseArrayBase base = getSparseArrayBase(in);
        *out = getHandle(base.getColumns());
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_sparse_get_num_values(dim_t *out, const af_array in)
{
    try {
        const SparseArrayBase base = getSparseArrayBase(in);
        *out = base.getNNZ();
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_sparse_get_num_rows(dim_t *out, const af_array in)
{
    try {
        const SparseArrayBase base = getSparseArrayBase(in);
        *out = base.getRows().elements();
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_sparse_get_num_cols(dim_t *out, const af_array in)
{
    try {
        const SparseArrayBase base = getSparseArrayBase(in);
        *out = base.getColumns().elements();
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
