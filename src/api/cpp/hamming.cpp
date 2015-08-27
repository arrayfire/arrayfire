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

void hammingMatcher(array& idx, array& dist,
                     const array& query, const array& train,
                     const dim_t dist_dim, const unsigned n_dist)
{
    af_array temp_idx  = 0;
    af_array temp_dist = 0;
    AF_THROW(af_nearest_neighbour(&temp_idx, &temp_dist, query.get(), train.get(), dist_dim, n_dist, AF_SHD));
    idx  = array(temp_idx);
    dist = array(temp_dist);
}

}
