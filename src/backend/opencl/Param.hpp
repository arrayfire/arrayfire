/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <platform.hpp>
#include <kernel/KParam.hpp>

namespace opencl
{

    struct Param
    {
        cl::Buffer *data;
        KParam info;
        Param& operator=(const Param& other) = default;
        Param(const Param& other) = default;
        Param(Param&& other) = default;

        // DEPRECATED("Use Array<T>")
        Param();
        // DEPRECATED("Use Array<T>")
        Param(cl::Buffer *data_, KParam info_);
        ~Param() = default;
    };

    // DEPRECATED("Use Array<T>")
    Param makeParam(cl_mem mem, int off, int dims[4], int strides[4]);
}
