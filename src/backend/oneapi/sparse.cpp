/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel/sparse.hpp>
#include <sparse.hpp>

#include <arith.hpp>
#include <common/cast.hpp>
#include <common/moddims.hpp>
#include <complex.hpp>
#include <copy.hpp>
#include <err_oneapi.hpp>
#include <lookup.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <range.hpp>
#include <reduce.hpp>
#include <where.hpp>

#include <stdexcept>
#include <string>

namespace arrayfire {
namespace oneapi {

using namespace common;

// Partial template specialization of sparseConvertDenseToStorage for COO
// However, template specialization is not allowed
template<typename T>
SparseArray<T> sparseConvertDenseToCOO(const Array<T> &in) {
    ONEAPI_NOT_SUPPORTED("sparseConvertDenseToCOO Not supported");
    // in.eval();

    // Array<uint> nonZeroIdx_ = where<T>(in);
    // Array<int> nonZeroIdx   = cast<int, uint>(nonZeroIdx_);

    // dim_t nNZ = nonZeroIdx.elements();

    // Array<int> constDim = createValueArray<int>(dim4(nNZ), in.dims()[0]);
    // constDim.eval();

    // Array<int> rowIdx =
    //     arithOp<int, af_mod_t>(nonZeroIdx, constDim, nonZeroIdx.dims());
    // Array<int> colIdx =
    //     arithOp<int, af_div_t>(nonZeroIdx, constDim, nonZeroIdx.dims());

    // Array<T> values = copyArray<T>(in);
    // values          = modDims(values, dim4(values.elements()));
    // values          = lookup<T, int>(values, nonZeroIdx, 0);

    // return createArrayDataSparseArray<T>(in.dims(), values, rowIdx, colIdx,
    //                                      AF_STORAGE_COO);
}

template<typename T, af_storage stype>
SparseArray<T> sparseConvertDenseToStorage(const Array<T> &in_) {
    // ONEAPI_NOT_SUPPORTED("sparseConvertDenseToStorage Not supported");
    in_.eval();

    printf("FFFUUU %d\n", __LINE__);
    auto tmp = reduce_all<af_notzero_t, T, uint>(in_);
    printf("FFFUUU %d\n", __LINE__);
    uint nNZ = getScalar<uint>(tmp);

    printf("got ... %d\n", nNZ);

    printf("FFFUUU %d\n", __LINE__);
    SparseArray<T> sparse_ = createEmptySparseArray<T>(in_.dims(), nNZ, stype);
    printf("FFFUUU %d\n", __LINE__);
    sparse_.eval();
    printf("FFFUUU %d\n", __LINE__);

    printf("FFFUUU %d\n", __LINE__);
    Array<T> &values = sparse_.getValues();
    printf("FFFUUU %d\n", __LINE__);
    Array<int> &rowIdx = sparse_.getRowIdx();
    printf("FFFUUU %d\n", __LINE__);
    Array<int> &colIdx = sparse_.getColIdx();
    printf("FFFUUU %d\n", __LINE__);

    kernel::dense2csr<T>(values, rowIdx, colIdx, in_);
    printf("FFFUUU %d\n", __LINE__);

    return sparse_;
}

// Partial template specialization of sparseConvertStorageToDense for COO
// However, template specialization is not allowed
template<typename T>
Array<T> sparseConvertCOOToDense(const SparseArray<T> &in) {
    ONEAPI_NOT_SUPPORTED("sparseConvertCOOToDense Not supported");
    //    in.eval();
    //
    //    Array<T> dense = createValueArray<T>(in.dims(), scalar<T>(0));
    //    dense.eval();
    //
    //    const Array<T> values   = in.getValues();
    //    const Array<int> rowIdx = in.getRowIdx();
    //    const Array<int> colIdx = in.getColIdx();

    // kernel::coo2dense<T>(dense, values, rowIdx, colIdx);

    // return dense;
}

template<typename T, af_storage stype>
Array<T> sparseConvertStorageToDense(const SparseArray<T> &in_) {
    ONEAPI_NOT_SUPPORTED("sparseConvertStorageToDense Not supported");
    //
    //    if (stype != AF_STORAGE_CSR) {
    //        AF_ERROR("OpenCL Backend only supports CSR or COO to Dense",
    //                 AF_ERR_NOT_SUPPORTED);
    //    }
    //
    //    in_.eval();
    //
    //    Array<T> dense_ = createValueArray<T>(in_.dims(), scalar<T>(0));
    //    dense_.eval();
    //
    //    const Array<T> &values   = in_.getValues();
    //    const Array<int> &rowIdx = in_.getRowIdx();
    //    const Array<int> &colIdx = in_.getColIdx();
    //
    //    if (stype == AF_STORAGE_CSR) {
    //        // kernel::csr2dense<T>(dense_, values, rowIdx, colIdx);
    //    } else {
    //        AF_ERROR("OpenCL Backend only supports CSR or COO to Dense",
    //                 AF_ERR_NOT_SUPPORTED);
    //    }
    //
    //    return dense_;
}

template<typename T, af_storage dest, af_storage src>
SparseArray<T> sparseConvertStorageToStorage(const SparseArray<T> &in) {
    ONEAPI_NOT_SUPPORTED("sparseConvertStorageToStorage Not supported");
    // in.eval();

    // SparseArray<T> converted = createEmptySparseArray<T>(
    //     in.dims(), static_cast<int>(in.getNNZ()), dest);
    // converted.eval();

    // if (src == AF_STORAGE_CSR && dest == AF_STORAGE_COO) {
    //     Array<int> index = range<int>(in.getNNZ(), 0);
    //     index.eval();

    //    Array<T> &ovalues         = converted.getValues();
    //    Array<int> &orowIdx       = converted.getRowIdx();
    //    Array<int> &ocolIdx       = converted.getColIdx();
    //    const Array<T> &ivalues   = in.getValues();
    //    const Array<int> &irowIdx = in.getRowIdx();
    //    const Array<int> &icolIdx = in.getColIdx();

    //    // kernel::csr2coo<T>(ovalues, orowIdx, ocolIdx, ivalues, irowIdx,
    //    // icolIdx,
    //    //                    index);

    //} else if (src == AF_STORAGE_COO && dest == AF_STORAGE_CSR) {
    //    Array<int> index = range<int>(in.getNNZ(), 0);
    //    index.eval();

    //    Array<T> &ovalues         = converted.getValues();
    //    Array<int> &orowIdx       = converted.getRowIdx();
    //    Array<int> &ocolIdx       = converted.getColIdx();
    //    const Array<T> &ivalues   = in.getValues();
    //    const Array<int> &irowIdx = in.getRowIdx();
    //    const Array<int> &icolIdx = in.getColIdx();

    //    Array<int> rowCopy = copyArray<int>(irowIdx);
    //    rowCopy.eval();

    //    kernel::coo2csr<T>(ovalues, orowIdx, ocolIdx, ivalues, irowIdx,
    //    icolIdx,
    //                       index, rowCopy, in.dims()[0]);

    //} else {
    //    // Should never come here
    //    AF_ERROR("OpenCL Backend invalid conversion combination",
    //             AF_ERR_NOT_SUPPORTED);
    //}

    // return converted;
}

#define INSTANTIATE_TO_STORAGE(T, S)                     \
    template SparseArray<T>                              \
    sparseConvertStorageToStorage<T, S, AF_STORAGE_CSR>( \
        const SparseArray<T> &in);                       \
    template SparseArray<T>                              \
    sparseConvertStorageToStorage<T, S, AF_STORAGE_CSC>( \
        const SparseArray<T> &in);                       \
    template SparseArray<T>                              \
    sparseConvertStorageToStorage<T, S, AF_STORAGE_COO>( \
        const SparseArray<T> &in);

#define INSTANTIATE_COO_SPECIAL(T)                                 \
    template<>                                                     \
    SparseArray<T> sparseConvertDenseToStorage<T, AF_STORAGE_COO>( \
        const Array<T> &in) {                                      \
        return sparseConvertDenseToCOO<T>(in);                     \
    }                                                              \
    template<>                                                     \
    Array<T> sparseConvertStorageToDense<T, AF_STORAGE_COO>(       \
        const SparseArray<T> &in) {                                \
        return sparseConvertCOOToDense<T>(in);                     \
    }

#define INSTANTIATE_SPARSE(T)                                               \
    template SparseArray<T> sparseConvertDenseToStorage<T, AF_STORAGE_CSR>( \
        const Array<T> &in);                                                \
    template SparseArray<T> sparseConvertDenseToStorage<T, AF_STORAGE_CSC>( \
        const Array<T> &in);                                                \
                                                                            \
    template Array<T> sparseConvertStorageToDense<T, AF_STORAGE_CSR>(       \
        const SparseArray<T> &in);                                          \
    template Array<T> sparseConvertStorageToDense<T, AF_STORAGE_CSC>(       \
        const SparseArray<T> &in);                                          \
                                                                            \
    INSTANTIATE_COO_SPECIAL(T)                                              \
                                                                            \
    INSTANTIATE_TO_STORAGE(T, AF_STORAGE_CSR)                               \
    INSTANTIATE_TO_STORAGE(T, AF_STORAGE_CSC)                               \
    INSTANTIATE_TO_STORAGE(T, AF_STORAGE_COO)

INSTANTIATE_SPARSE(float)
INSTANTIATE_SPARSE(double)
INSTANTIATE_SPARSE(cfloat)
INSTANTIATE_SPARSE(cdouble)

#undef INSTANTIATE_TO_STORAGE
#undef INSTANTIATE_COO_SPECIAL
#undef INSTANTIATE_SPARSE

}  // namespace oneapi
}  // namespace arrayfire
