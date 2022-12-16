/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <canny.hpp>

#include <Array.hpp>
#include <Param.hpp>
#include <kernel/canny.hpp>
#include <platform.hpp>
#include <queue.hpp>

namespace arrayfire {
namespace cpu {
Array<float> nonMaximumSuppression(const Array<float>& mag,
                                   const Array<float>& gx,
                                   const Array<float>& gy) {
    Array<float> out = createValueArray<float>(mag.dims(), 0);

    getQueue().enqueue(kernel::nonMaxSuppression<float>, out, mag, gx, gy);

    return out;
}

Array<char> edgeTrackingByHysteresis(const Array<char>& strong,
                                     const Array<char>& weak) {
    Array<char> out = createValueArray<char>(strong.dims(), 0);

    getQueue().enqueue(kernel::edgeTrackingHysteresis<char>, out, strong, weak);

    return out;
}
}  // namespace cpu
}  // namespace arrayfire
