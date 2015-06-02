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

void orb(features& feat, array& desc, const array& in,
         const float fast_thr, const unsigned max_feat,
         const float scl_fctr, const unsigned levels,
         const bool blur_img)
{
    af_features temp_feat;
    af_array temp_desc = 0;
    AF_THROW(af_orb(&temp_feat, &temp_desc, in.get(), fast_thr,
                    max_feat, scl_fctr, levels, blur_img));

    dim_t num = 0;
    AF_THROW(af_get_features_num(&num,  temp_feat));
    feat = features(temp_feat);
    desc = array(temp_desc);
}

}
