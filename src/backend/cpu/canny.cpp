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
#include <platform.hpp>
#include <queue.hpp>
#include <kernel/canny.hpp>

using af::dim4;

namespace cpu
{
Array<float> nonMaximumSuppression(const Array<float>& mag,
                                   const Array<float>& gx, const Array<float>& gy)
{
    mag.eval();
    gx.eval();
    gy.eval();

    Array<float> out = createValueArray<float>(mag.dims(), 0);
    out.eval();

    getQueue().enqueue(kernel::nonMaxSuppression<float>, out, mag, gx, gy);

    return out;
}

Array<char> edgeTrackingByHysteresis(const Array<char>& strong, const Array<char>& weak)
{
    strong.eval();
    weak.eval();

    Array<char> out = createValueArray<char>(strong.dims(), 0);
    out.eval();

    getQueue().enqueue(kernel::edgeTrackingHysteresis<char>, out, strong, weak);

    return out;
}
}
