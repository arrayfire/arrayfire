/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/vision.h>
#include <af/array.h>
#include "error.hpp"

namespace af
{

features harris(const array& in, const unsigned max_corners,
                const float min_response, const float sigma,
                const unsigned block_size, const float k_thr)
{
    af_features temp;
    AF_THROW(af_harris(&temp, in.get(), max_corners,
                       min_response, sigma, block_size, k_thr));
    return features(temp);
}

}
