/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#if defined(AF_USE_NEW_CUSPARSE_API)
// CUDA Toolkit 10.0 or later

#include <common/unique_handle.hpp>
#include <cusparse.hpp>

namespace cuda {

template<typename T>
common::unique_handle<cusparseSpMatDescr_t> csrMatDescriptor(
    const common::SparseArray<T> &in) {
    auto dims                   = in.dims();
    cusparseSpMatDescr_t resMat = NULL;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &resMat, dims[0], dims[1], in.getNNZ(), (void *)(in.getRowIdx().get()),
        (void *)(in.getColIdx().get()), (void *)(in.getValues().get()),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        getType<T>()));
    return common::unique_handle<cusparseSpMatDescr_t>(resMat);
}

template<typename T>
common::unique_handle<cusparseDnVecDescr_t> denVecDescriptor(
    const Array<T> &in) {
    auto dims                   = in.dims();
    cusparseDnVecDescr_t resVec = NULL;
    CUSPARSE_CHECK(cusparseCreateDnVec(&resVec, dims.elements(),
                                       (void *)(in.get()), getType<T>()));
    return common::unique_handle<cusparseDnVecDescr_t>(resVec);
}

template<typename T>
common::unique_handle<cusparseDnMatDescr_t> denMatDescriptor(
    const Array<T> &in) {
    auto dims                   = in.dims();
    cusparseDnMatDescr_t resMat = NULL;
    CUSPARSE_CHECK(cusparseCreateDnMat(&resMat, dims[0], dims[1], dims[0],
                                       (void *)(in.get()), getType<T>(),
                                       CUSPARSE_ORDER_COL));
    return common::unique_handle<cusparseDnMatDescr_t>(resMat);
}

}  // namespace cuda

#endif
