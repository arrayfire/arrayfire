/*******************************************************
 * Copyright (c) 2014, ArrayFire
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

namespace opencl {
Param::Param() : data(nullptr), info{{0, 0, 0, 0}, {0, 0, 0, 0}, 0} {}
Param::Param(cl::Buffer *data_, KParam info_) : data(data_), info(info_) {}

Param makeParam(cl_mem mem, int off, const int dims[4], const int strides[4]) {
    Param out;
    out.data        = new cl::Buffer(mem);
    out.info.offset = off;
    for (int i = 0; i < 4; i++) {
        out.info.dims[i]    = dims[i];
        out.info.strides[i] = strides[i];
    }
    return out;
}
}  // namespace opencl
