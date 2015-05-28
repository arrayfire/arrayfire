/*******************************************************
 * Copyright (c) 2014, ArrayFire
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

features fast(const array& in, const float thr, const unsigned arc_length,
                const bool non_max, const float feature_ratio,
                const unsigned edge)
{
    af_features temp;
    AF_THROW(af_fast(&temp, in.get(), thr, arc_length,
                     non_max, feature_ratio, edge));
    return features(temp);
}

}
