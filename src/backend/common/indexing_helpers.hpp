/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Array.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>

#include <array>

namespace arrayfire {
namespace common {

// will generate indexes to flip input array
// of size original dims according to axes specified in flip
template<typename T>
static detail::Array<T> flip(const detail::Array<T>& in,
                             const std::array<bool, AF_MAX_DIMS> flip) {
    std::vector<af_seq> index(4, af_span);
    const af::dim4& dims = in.dims();

    for (int i = 0; i < AF_MAX_DIMS; ++i) {
        if (flip[i]) {
            index[i] = {static_cast<double>(dims[i] - 1), 0.0, -1.0};
        }
    }
    return createSubArray(in, index);
}

}  // namespace common
}  // namespace arrayfire
