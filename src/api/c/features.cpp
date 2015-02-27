/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/features.h>
#include <af/array.h>
#include <handle.hpp>

af_err af_destroy_features(af_features feat)
{

    try {
        if (feat.n > 0) {
            if (feat.x != 0)           AF_CHECK(af_destroy_array(feat.x));
            if (feat.y != 0)           AF_CHECK(af_destroy_array(feat.y));
            if (feat.score != 0)       AF_CHECK(af_destroy_array(feat.score));
            if (feat.orientation != 0) AF_CHECK(af_destroy_array(feat.orientation));
            if (feat.size != 0)        AF_CHECK(af_destroy_array(feat.size));
            feat.n = 0;
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}
