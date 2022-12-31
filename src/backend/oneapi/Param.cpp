/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>
#include <kernel/KParam.hpp>
#include <platform.hpp>
#include <af/defines.h>

namespace arrayfire {
namespace oneapi {

template<typename T>
Param<T> makeParam(sycl::buffer<T> &mem, int off, const int dims[4],
                   const int strides[4]) {
    Param<T> out;
    out.data        = &mem;
    out.info.offset = off;
    for (int i = 0; i < 4; i++) {
        out.info.dims[i]    = dims[i];
        out.info.strides[i] = strides[i];
    }
    return out;
}

}  // namespace oneapi
}  // namespace arrayfire
