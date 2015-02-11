/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/image.h>
#include "error.hpp"

namespace af
{

features fast(const array& in, const float thr, const unsigned arc_length,
                const bool non_max, const float feature_ratio)
{
    af_features temp;
    AF_THROW(af_fast(&temp, in.get(), thr, arc_length, non_max, feature_ratio));
    features out(temp);

    af_destroy_array(temp.x);
    af_destroy_array(temp.y);
    af_destroy_array(temp.score);
    af_destroy_array(temp.orientation);
    af_destroy_array(temp.size);

    return out;
}

}
