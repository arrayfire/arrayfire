/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <af/features.h>

using af::features;

namespace arrayfire {
namespace oneapi {

template<typename T, typename convAccT>
unsigned orb(Array<float> &x, Array<float> &y, Array<float> &score,
             Array<float> &orientation, Array<float> &size,
             Array<unsigned> &desc, const Array<T> &image, const float fast_thr,
             const unsigned max_feat, const float scl_fctr,
             const unsigned levels, const bool blur_img);

}  // namespace oneapi
}  // namespace arrayfire
