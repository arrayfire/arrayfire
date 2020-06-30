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

#include <utility>

namespace cuda {

template<typename T>
auto csrMatDescriptor(const common::SparseArray<T> &in) {
    auto dims = in.dims();
    return common::make_handle<cusparseSpMatDescr_t>(
        dims[0], dims[1], in.getNNZ(), (void *)(in.getRowIdx().get()),
        (void *)(in.getColIdx().get()), (void *)(in.getValues().get()),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        getType<T>());
}

template<typename T>
auto denVecDescriptor(const Array<T> &in) {
    return common::make_handle<cusparseDnVecDescr_t>(
        in.elements(), (void *)(in.get()), getType<T>());
}

template<typename T>
auto denMatDescriptor(const Array<T> &in) {
    auto dims = in.dims();
    return common::make_handle<cusparseDnMatDescr_t>(
        dims[0], dims[1], dims[0], (void *)(in.get()), getType<T>(),
        CUSPARSE_ORDER_COL);
}

}  // namespace cuda

#endif
