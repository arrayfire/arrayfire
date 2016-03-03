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
#include <sparse_t.hpp>
#include <sparse.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <err_common.hpp>
#include <arith.hpp>
#include <lookup.hpp>

using namespace detail;
using af::dim4;

af_sparse_array getSparseHandle(const af_sparse_t sparse)
{
    af_sparse_t *sparseHandle = new af_sparse_t;
    *sparseHandle = sparse;
    return (af_sparse_array)sparseHandle;
}

af_sparse_t getSparse(const af_sparse_array sparseHandle)
{
    return *(af_sparse_t *)sparseHandle;
}

af_err af_release_sparse_array(af_sparse_array arr)
{

    try {
        af_sparse_t sparse = *(af_sparse_t *)arr;
        if (sparse.storage > 0) {
            if (sparse.rowIdx != 0)     AF_CHECK(af_release_array(sparse.rowIdx));
            if (sparse.colIdx != 0)     AF_CHECK(af_release_array(sparse.colIdx));
            if (sparse.values != 0)     AF_CHECK(af_release_array(sparse.values));
        }
        delete (af_sparse_t *)arr;
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_retain_sparse_array(af_sparse_array *out, const af_sparse_array in)
{
    try {
        af_sparse_t input = getSparse(in);
        af_sparse_t output;

        output.storage  = input.storage;
        output.nRows    = input.nRows;
        output.nCols    = input.nCols;
        output.nNZ      = input.nNZ;

        AF_CHECK(af_retain_array(&output.values, input.values));
        AF_CHECK(af_retain_array(&output.rowIdx, input.rowIdx));
        AF_CHECK(af_retain_array(&output.colIdx, input.colIdx));

        *out = getSparseHandle(output);
    }
    CATCHALL;
    return AF_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
// Sparse Creation
////////////////////////////////////////////////////////////////////////////////
af_err af_create_sparse_array(
                 af_sparse_array *out,
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

        af_sparse_t sparse;
        sparse.storage = storage;
        sparse.nRows = nRows;
        sparse.nCols = nCols;
        sparse.nNZ = nNZ;

        AF_CHECK(af_retain_array(&sparse.rowIdx, rowIdx));
        AF_CHECK(af_retain_array(&sparse.colIdx, colIdx));
        AF_CHECK(af_retain_array(&sparse.values, values));

        *out = getSparseHandle(sparse);
    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_create_sparse_array_from_host(
                 af_sparse_array *out,
                 const dim_t nRows, const dim_t nCols, const dim_t nNZ,
                 const void * const values,
                 const int * const rowIdx, const int * const colIdx,
                 const af_dtype type, const af_sparse_storage storage)
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

        af_sparse_t sparse;
        sparse.storage = storage;
        sparse.nRows = nRows;
        sparse.nCols = nCols;
        sparse.nNZ = nNZ;

        AF_CHECK(af_create_array(&sparse.values, values, 1, &nNZ, type));

        if(storage == AF_SPARSE_COO) {
            AF_CHECK(af_create_array(&sparse.rowIdx, rowIdx, 1, &nNZ, s32));
            AF_CHECK(af_create_array(&sparse.colIdx, colIdx, 1, &nNZ, s32));
        } else if(storage == AF_SPARSE_CSR) {
            AF_CHECK(af_create_array(&sparse.rowIdx, rowIdx, 1, &nRows, s32));
            AF_CHECK(af_create_array(&sparse.colIdx, colIdx, 1, &nNZ, s32));
        } else if(storage == AF_SPARSE_CSC) {
            AF_CHECK(af_create_array(&sparse.rowIdx, rowIdx, 1, &nNZ, s32));
            AF_CHECK(af_create_array(&sparse.colIdx, colIdx, 1, &nCols, s32));
        }

        *out = getSparseHandle(sparse);
    } CATCHALL;

    return AF_SUCCESS;
}

template<typename T>
void create_sparse_array_from_dense(af_sparse_t *out, const af_array in_,
                                    const af_array nonZeroIdx_, const af_sparse_storage storage)
{
    Array<int> nonZeroIdx = castArray<int>(nonZeroIdx_);

    const Array<T> in = getArray<T>(in_);

    dim_t nNZ = nonZeroIdx.elements();
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

    out->rowIdx = getHandle(rowIdx);
    out->colIdx = getHandle(colIdx);
    out->values = getHandle(values);

}

af_err af_create_sparse_array_from_dense(af_sparse_array *out, const af_array in,
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

        TYPE_ASSERT(info.isFloating());

        af_sparse_t sparse;

        af_array nonZeroIdx = 0; // Yes I know how this looks
        AF_CHECK(af_where(&nonZeroIdx, in));

        ArrayInfo nNZInfo = getInfo(nonZeroIdx);
        dim_t nNZ = nNZInfo.elements();

        sparse.storage = storage;
        sparse.nRows = info.dims()[0];
        sparse.nCols = info.dims()[1];
        sparse.nNZ = nNZ;

        switch(info.getType()) {
            case f32: create_sparse_array_from_dense<float  >(&sparse, in, nonZeroIdx, storage); break;
            case f64: create_sparse_array_from_dense<double >(&sparse, in, nonZeroIdx, storage); break;
            case c32: create_sparse_array_from_dense<cfloat >(&sparse, in, nonZeroIdx, storage); break;
            case c64: create_sparse_array_from_dense<cdouble>(&sparse, in, nonZeroIdx, storage); break;
            default: TYPE_ERROR(1, info.getType());
        }

        // Call the conversion in the backend here

        *out = getSparseHandle(sparse);

        if(nonZeroIdx != 0) AF_CHECK(af_release_array(nonZeroIdx));
    } CATCHALL;

    return AF_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
// Get Functions
////////////////////////////////////////////////////////////////////////////////
af_err af_sparse_get_values(af_array *out, const af_sparse_array in)
{
    try {
        af_sparse_t sparse = getSparse(in);
        *out = sparse.values;
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_sparse_get_rows(af_array *out, const af_sparse_array in)
{
    try {
        af_sparse_t sparse = getSparse(in);
        *out = sparse.rowIdx;
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_sparse_get_cols(af_array *out, const af_sparse_array in)
{
    try {
        af_sparse_t sparse = getSparse(in);
        *out = sparse.colIdx;
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_sparse_get_num_values(dim_t *out, const af_sparse_array in)
{
    try {
        af_sparse_t sparse = getSparse(in);
        *out = sparse.nNZ;
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_sparse_get_num_rows(dim_t *out, const af_sparse_array in)
{
    try {
        af_sparse_t sparse = getSparse(in);
        *out = sparse.nRows;
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_sparse_get_num_cols(dim_t *out, const af_sparse_array in)
{
    try {
        af_sparse_t sparse = getSparse(in);
        *out = sparse.nCols;
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_sparse_get_storage(af_sparse_storage *out, const af_sparse_array in)
{
    try {
        af_sparse_t sparse = getSparse(in);
        *out = sparse.storage;
    } CATCHALL;
    return AF_SUCCESS;
}
