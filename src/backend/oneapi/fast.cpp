/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_oneapi.hpp>
#include <af/dim4.hpp>
#include <af/features.h>

using af::dim4;
using af::features;

namespace arrayfire {
namespace oneapi {

template<typename T>
unsigned fast(Array<float> &x_out, Array<float> &y_out, Array<float> &score_out,
              const Array<T> &in, const float thr, const unsigned arc_length,
              const bool non_max, const float feature_ratio,
              const unsigned edge) {
    ONEAPI_NOT_SUPPORTED("");
    return 0;
}

#define INSTANTIATE(T)                                                        \
    template unsigned fast<T>(                                                \
        Array<float> & x_out, Array<float> & y_out, Array<float> & score_out, \
        const Array<T> &in, const float thr, const unsigned arc_length,       \
        const bool nonmax, const float feature_ratio, const unsigned edge);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(char)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace oneapi
}  // namespace arrayfire
