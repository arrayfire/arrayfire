/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <utility.hpp>

#include <err_cuda.hpp>

namespace arrayfire {
namespace cuda {

int interpOrder(const af_interp_type p) noexcept {
    int order = 1;
    switch (p) {
        case AF_INTERP_NEAREST:
        case AF_INTERP_LOWER: order = 1; break;
        case AF_INTERP_LINEAR:
        case AF_INTERP_BILINEAR:
        case AF_INTERP_LINEAR_COSINE:
        case AF_INTERP_BILINEAR_COSINE: order = 2; break;
        case AF_INTERP_CUBIC:
        case AF_INTERP_BICUBIC:
        case AF_INTERP_CUBIC_SPLINE:
        case AF_INTERP_BICUBIC_SPLINE: order = 3; break;
    }
    return order;
}

}  // namespace cuda
}  // namespace arrayfire
