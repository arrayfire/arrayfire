/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel/sparse.hpp>
#include <sparse.hpp>

#include <stdexcept>
#include <string>

#include <arith.hpp>
#include <common/cast.hpp>
#include <common/complex.hpp>
#include <common/err_common.hpp>
#include <complex.hpp>
#include <copy.hpp>
#include <lookup.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <reduce.hpp>
#include <where.hpp>

#include <functional>

using arrayfire::common::cast;
using std::function;

namespace arrayfire {
namespace cpu {

using arrayfire::common::createArrayDataSparseArray;
using arrayfire::common::createEmptySparseArray;
using arrayfire::common::SparseArray;

template<typename T, af_storage stype>
SparseArray<T> sparseConvertDenseToStorage(const Array<T> &in) {
    if (stype == AF_STORAGE_CSR) {
        uint nNZ = reduce_all<af_notzero_t, T, uint>(in);

        auto sparse = createEmptySparseArray<T>(in.dims(), nNZ, stype);
        sparse.eval();

        Array<T> values   = sparse.getValues();
        Array<int> rowIdx = sparse.getRowIdx();
        Array<int> colIdx = sparse.getColIdx();

        getQueue().enqueue(kernel::dense2csr<T>, values, rowIdx, colIdx, in);

        return sparse;
    } else if (stype == AF_STORAGE_COO) {
        auto nonZeroIdx = cast<int, uint>(where<T>(in));

        dim_t nNZ = nonZeroIdx.elements();

        auto cnst = createValueArray<int>(dim4(nNZ), in.dims()[0]);

        auto rowIdx =
            arithOp<int, af_mod_t>(nonZeroIdx, cnst, nonZeroIdx.dims());
        auto colIdx =
            arithOp<int, af_div_t>(nonZeroIdx, cnst, nonZeroIdx.dims());

        Array<T> values = copyArray<T>(in);
        values.modDims(dim4(values.elements()));
        values = lookup<T, int>(values, nonZeroIdx, 0);

        return createArrayDataSparseArray<T>(in.dims(), values, rowIdx, colIdx,
                                             stype);
    } else {
        AF_ERROR("CPU Backend only supports Dense to CSR or COO",
                 AF_ERR_NOT_SUPPORTED);
    }
}

template<typename T, af_storage stype>
Array<T> sparseConvertStorageToDense(const SparseArray<T> &in) {
    Array<T> dense = createValueArray<T>(in.dims(), scalar<T>(0));

    Array<T> values   = in.getValues();
    Array<int> rowIdx = in.getRowIdx();
    Array<int> colIdx = in.getColIdx();

    if (stype == AF_STORAGE_CSR) {
        getQueue().enqueue(kernel::csr2dense<T>, dense, values, rowIdx, colIdx);
    } else if (stype == AF_STORAGE_COO) {
        getQueue().enqueue(kernel::coo2dense<T>, dense, values, rowIdx, colIdx);
    } else {
        AF_ERROR("CPU Backend only supports CSR or COO to Dense",
                 AF_ERR_NOT_SUPPORTED);
    }

    return dense;
}

template<typename T, af_storage dest, af_storage src>
SparseArray<T> sparseConvertStorageToStorage(const SparseArray<T> &in) {
    in.eval();

    auto converted = createEmptySparseArray<T>(
        in.dims(), static_cast<int>(in.getNNZ()), dest);
    converted.eval();

    function<void(Param<T>, Param<int>, Param<int>, CParam<T>, CParam<int>,
                  CParam<int>)>
        converter;

    if (src == AF_STORAGE_CSR && dest == AF_STORAGE_COO) {
        converter = kernel::csr2coo<T>;
    } else if (src == AF_STORAGE_COO && dest == AF_STORAGE_CSR) {
        converter = kernel::coo2csr<T>;
    } else {
        // Should never come here
        AF_ERROR("CPU Backend invalid conversion combination",
                 AF_ERR_NOT_SUPPORTED);
    }
    getQueue().enqueue(converter, converted.getValues(), converted.getRowIdx(),
                       converted.getColIdx(), in.getValues(), in.getRowIdx(),
                       in.getColIdx());
    return converted;
}

#define INSTANTIATE_TO_STORAGE(T, S)                     \
    template SparseArray<T>                              \
    sparseConvertStorageToStorage<T, S, AF_STORAGE_CSR>( \
        const SparseArray<T> &);                         \
    template SparseArray<T>                              \
    sparseConvertStorageToStorage<T, S, AF_STORAGE_CSC>( \
        const SparseArray<T> &);                         \
    template SparseArray<T>                              \
    sparseConvertStorageToStorage<T, S, AF_STORAGE_COO>( \
        const SparseArray<T> &);

#define INSTANTIATE_SPARSE(T)                                               \
    template SparseArray<T> sparseConvertDenseToStorage<T, AF_STORAGE_CSR>( \
        const Array<T> &in);                                                \
    template SparseArray<T> sparseConvertDenseToStorage<T, AF_STORAGE_CSC>( \
        const Array<T> &in);                                                \
    template SparseArray<T> sparseConvertDenseToStorage<T, AF_STORAGE_COO>( \
        const Array<T> &in);                                                \
    template Array<T> sparseConvertStorageToDense<T, AF_STORAGE_CSR>(       \
        const SparseArray<T> &in);                                          \
    template Array<T> sparseConvertStorageToDense<T, AF_STORAGE_CSC>(       \
        const SparseArray<T> &in);                                          \
    template Array<T> sparseConvertStorageToDense<T, AF_STORAGE_COO>(       \
        const SparseArray<T> &in);                                          \
                                                                            \
    INSTANTIATE_TO_STORAGE(T, AF_STORAGE_CSR)                               \
    INSTANTIATE_TO_STORAGE(T, AF_STORAGE_CSC)                               \
    INSTANTIATE_TO_STORAGE(T, AF_STORAGE_COO)

INSTANTIATE_SPARSE(float)
INSTANTIATE_SPARSE(double)
INSTANTIATE_SPARSE(cfloat)
INSTANTIATE_SPARSE(cdouble)

#undef INSTANTIATE_TO_STORAGE
#undef INSTANTIATE_SPARSE

}  // namespace cpu
}  // namespace arrayfire
