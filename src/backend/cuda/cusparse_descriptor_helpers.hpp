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
#include <cudaDataType.hpp>
#include <cusparse.hpp>

#include <utility>

namespace arrayfire {
namespace cuda {

template<typename T>
auto cusparseDescriptor(const common::SparseArray<T> &in) {
    auto dims = in.dims();

    return common::make_handle<cusparseSpMatDescr_t>(in);
}

template<typename T>
auto denVecDescriptor(const Array<T> &in) {
    return common::make_handle<cusparseDnVecDescr_t>(
        in.elements(), (void *)(in.get()), getType<T>());
}

template<typename T>
auto denMatDescriptor(const Array<T> &in) {
    auto dims    = in.dims();
    auto strides = in.strides();
    return common::make_handle<cusparseDnMatDescr_t>(
        dims[0], dims[1], strides[1], (void *)in.get(), getType<T>(),
        CUSPARSE_ORDER_COL);
}

}  // namespace cuda
}  // namespace arrayfire

#endif
