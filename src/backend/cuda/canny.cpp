/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <canny.hpp>
#include <err_cuda.hpp>
#include <kernel/canny.hpp>

using af::dim4;

namespace arrayfire {
namespace cuda {
Array<float> nonMaximumSuppression(const Array<float>& mag,
                                   const Array<float>& gx,
                                   const Array<float>& gy) {
    Array<float> out = createValueArray<float>(mag.dims(), 0);
    kernel::nonMaxSuppression<float>(out, mag, gx, gy);
    return out;
}

Array<char> edgeTrackingByHysteresis(const Array<char>& strong,
                                     const Array<char>& weak) {
    Array<char> out = createValueArray<char>(strong.dims(), 0);
    kernel::edgeTrackingHysteresis<char>(out, strong, weak);
    return out;
}
}  // namespace cuda
}  // namespace arrayfire
