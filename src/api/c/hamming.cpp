/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/vision.h>

af_err af_hamming_matcher(af_array* idx, af_array* dist, const af_array query, const af_array train,
        const dim_t dist_dim, const unsigned n_dist)
{
    return af_nearest_neighbour(idx, dist, query, train, dist_dim, n_dist, AF_SHD);
}
