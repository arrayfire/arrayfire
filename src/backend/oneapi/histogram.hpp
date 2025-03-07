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
template<typename T>
Array<uint> histogram(const Array<T> &in, const unsigned &nbins,
                      const double &minval, const double &maxval,
                      const bool isLinear);
}  // namespace oneapi
}  // namespace arrayfire
