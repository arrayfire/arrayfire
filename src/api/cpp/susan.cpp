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

features susan(const array& in, const unsigned radius, const float diff_thr, const float geom_thr,
               const float feature_ratio, const unsigned edge)
{
    af_features temp;
    AF_THROW(af_susan(&temp, in.get(), radius, diff_thr, geom_thr, feature_ratio, edge));
    return features(temp);
}

}
