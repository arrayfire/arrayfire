/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/image.h>
#include "error.hpp"

namespace af {
array canny(const array& in, const cannyThreshold ctType, const float ltr,
            const float htr, const unsigned sW, const bool isFast) {
    af_array temp = 0;
    AF_THROW(af_canny(&temp, in.get(), ctType, ltr, htr, sW, isFast));
    return array(temp);
}
}  // namespace af
