/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace arrayfire {
namespace oneapi {
Array<float> nonMaximumSuppression(const Array<float>& mag,
                                   const Array<float>& gx,
                                   const Array<float>& gy);

Array<char> edgeTrackingByHysteresis(const Array<char>& strong,
                                     const Array<char>& weak);
}  // namespace oneapi
}  // namespace arrayfire
