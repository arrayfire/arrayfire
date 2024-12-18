/*******************************************************
 * Copyright (c) 2023, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <sycl/sycl.hpp>

#include <array>

namespace arrayfire {
namespace oneapi {

typedef struct {
    int offs[4];
    int strds[4];
    int steps[4];
    bool isSeq[4];
    std::array<sycl::accessor<unsigned int, 1, sycl::access::mode::read,
                              sycl::access::target::device>,
               4>
        ptr;

} AssignKernelParam;

using IndexKernelParam = AssignKernelParam;

}  // namespace oneapi
}  // namespace arrayfire
