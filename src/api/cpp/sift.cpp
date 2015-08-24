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

void sift(features& feat, array& desc, const array& in,
          const unsigned n_layers, const float contrast_thr,
          const float edge_thr, const float init_sigma,
          const bool double_input, const float img_scale,
          const float feature_ratio)
{
    af_features temp_feat;
    af_array temp_desc = 0;
    AF_THROW(af_sift(&temp_feat, &temp_desc, in.get(), n_layers, contrast_thr,
                     edge_thr, init_sigma, double_input, img_scale, feature_ratio));

    dim_t num = 0;
    AF_THROW(af_get_features_num(&num, temp_feat));
    feat = features(temp_feat);
    desc = array(temp_desc);
}

}
