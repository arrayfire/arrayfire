/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <CL/sycl.hpp>
#include <kernel/KParam.hpp>

namespace oneapi {

template<typename T>
struct Param {
    sycl::buffer<T>* data;
    KParam info;
    Param& operator=(const Param& other) = default;
    Param(const Param& other)            = default;
    Param(Param&& other)                 = default;

    // AF_DEPRECATED("Use Array<T>")
    Param() : data(nullptr), info{{0, 0, 0, 0}, {0, 0, 0, 0}, 0} {}

    // AF_DEPRECATED("Use Array<T>")
    Param(sycl::buffer<T> *data_, KParam info_) : data(data_), info(info_) {}

    ~Param() = default;
};

// AF_DEPRECATED("Use Array<T>")
template<typename T>
Param<T> makeParam(sycl::buffer<T>& mem, int off, const int dims[4],
                   const int strides[4]);
}  // namespace oneapi
